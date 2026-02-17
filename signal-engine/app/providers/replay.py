from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

from pydantic import BaseModel


class ReplayDatasetError(RuntimeError):
    pass


class ReplayCandle(BaseModel):
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime


class ReplayKlineSnapshot(BaseModel):
    symbol: str
    interval: str
    price: float
    volume: float
    kline_open_time_ms: int
    kline_close_time_ms: int
    kline_is_closed: bool
    candle: ReplayCandle
    candles: list[ReplayCandle]


@dataclass
class ReplayCursor:
    index: int = 0


class ReplayProvider:
    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._cache: dict[tuple[str, str], list[ReplayCandle]] = {}
        self._cursor: dict[tuple[str, str], ReplayCursor] = {}

    def _path_for(self, symbol: str, interval: str) -> tuple[Path, str]:
        symbol_dir = self._base / symbol.upper()
        csv_path = symbol_dir / f"{interval}.csv"
        jsonl_path = symbol_dir / f"{interval}.jsonl"
        if csv_path.exists():
            return csv_path, "csv"
        if jsonl_path.exists():
            return jsonl_path, "jsonl"
        raise ReplayDatasetError(
            f"replay_dataset_missing symbol={symbol.upper()} interval={interval} expected={csv_path} or {jsonl_path}"
        )

    def _parse_ts(self, value: object) -> datetime:
        if isinstance(value, (int, float)):
            v = float(value)
            if v > 10_000_000_000:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ReplayDatasetError(f"invalid_timestamp value={value!r}") from exc
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def _load_series(self, symbol: str, interval: str) -> list[ReplayCandle]:
        key = (symbol.upper(), interval)
        if key in self._cache:
            return self._cache[key]
        path, kind = self._path_for(symbol, interval)
        candles: list[ReplayCandle] = []
        if kind == "csv":
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    open_time = self._parse_ts(row.get("timestamp"))
                    close_time = self._parse_ts(row.get("close_time") or row.get("timestamp"))
                    candles.append(
                        ReplayCandle(
                            open_time=open_time,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row.get("volume", 0.0) or 0.0),
                            close_time=close_time,
                        )
                    )
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    open_time = self._parse_ts(row.get("timestamp"))
                    close_time = self._parse_ts(row.get("close_time") or row.get("timestamp"))
                    candles.append(
                        ReplayCandle(
                            open_time=open_time,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row.get("volume", 0.0) or 0.0),
                            close_time=close_time,
                        )
                    )
        if not candles:
            raise ReplayDatasetError(f"replay_dataset_empty symbol={symbol} interval={interval} path={path}")
        candles.sort(key=lambda c: c.close_time)
        self._cache[key] = candles
        self._cursor.setdefault(key, ReplayCursor(index=0))
        return candles

    def validate_dataset(self, symbol: str, interval: str) -> dict[str, object]:
        candles = self._load_series(symbol, interval)
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "total_bars": len(candles),
            "start_ts": candles[0].close_time.isoformat(),
            "end_ts": candles[-1].close_time.isoformat(),
        }

    def reset_cursor(self, symbol: str, interval: str) -> None:
        key = (symbol.upper(), interval)
        self._cursor[key] = ReplayCursor(index=0)

    def status(self, symbol: str, interval: str) -> dict[str, object]:
        candles = self._load_series(symbol, interval)
        key = (symbol.upper(), interval)
        cursor = self._cursor.setdefault(key, ReplayCursor(index=max(0, min(len(candles) - 1, limit - 1))))
        idx = min(max(cursor.index, 0), len(candles) - 1)
        ts = candles[idx].close_time
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "bar_index": idx,
            "total_bars": len(candles),
            "ts": ts.isoformat(),
            "progress_pct": round(((idx + 1) / len(candles)) * 100.0, 4),
        }

    async def fetch_symbol_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        speed: float = 1.0,
        start_ts: str | None = None,
        end_ts: str | None = None,
    ) -> ReplayKlineSnapshot:
        candles = self._load_series(symbol, interval)
        if start_ts or end_ts:
            start_dt = self._parse_ts(start_ts) if start_ts else None
            end_dt = self._parse_ts(end_ts) if end_ts else None
            candles = [c for c in candles if (start_dt is None or c.close_time >= start_dt) and (end_dt is None or c.close_time <= end_dt)]
            if not candles:
                raise ReplayDatasetError(f"replay_filter_empty symbol={symbol} interval={interval}")
        key = (symbol.upper(), interval)
        cursor = self._cursor.setdefault(key, ReplayCursor(index=0))
        if cursor.index == 0:
            cursor.index = max(0, min(len(candles) - 1, limit - 1))
        step = max(1, int(round(speed)))
        cursor.index = min(len(candles) - 1, cursor.index + step)
        window_start = max(0, cursor.index - max(2, limit) + 1)
        window = candles[window_start : cursor.index + 1]
        candle = window[-1]
        return ReplayKlineSnapshot(
            symbol=symbol.upper(),
            interval=interval,
            price=candle.close,
            volume=candle.volume,
            kline_open_time_ms=int(candle.open_time.timestamp() * 1000),
            kline_close_time_ms=int(candle.close_time.timestamp() * 1000),
            kline_is_closed=True,
            candle=candle,
            candles=window,
        )
