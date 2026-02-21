from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import inspect
import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel


logger = logging.getLogger(__name__)


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
    cursor_index: int = 0
    total_bars: int = 0
    source_file_path: str = ""


@dataclass
class ReplayCursor:
    index: int = 0


class ReplayProvider:
    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._cache: dict[tuple[str, str], list[ReplayCandle]] = {}
        self._cursor: dict[tuple[str, str], ReplayCursor] = {}
        self._log_every_bars = 250
        self._state_file = self._base / "replay_runtime_state.json"
        self._start_resolution: dict[tuple[str, str], dict[str, object]] = {}


    def _state_key(self, symbol: str, interval: str) -> str:
        return f"{symbol.upper()}:{interval}"

    def _load_resume_state(self) -> dict[str, object]:
        if not self._state_file.exists():
            return {}
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_resume_state(self, payload: dict[str, object]) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _first_index_at_or_after(self, candles: list[ReplayCandle], ts: datetime) -> int | None:
        for idx, candle in enumerate(candles):
            if candle.close_time >= ts:
                return idx
        return None

    def _resolve_start_index(self, *, symbol: str, interval: str, candles: list[ReplayCandle], start_ts: str | None, limit: int, history_limit: int | None) -> int:
        key = (symbol.upper(), interval)
        reason = "default"
        candidate_index = 0
        if start_ts:
            start_dt = self._parse_ts(start_ts)
            idx = self._first_index_at_or_after(candles, start_dt)
            if idx is None:
                raise ReplayDatasetError(f"replay_start_ts_out_of_range start={start_ts}")
            candidate_index = idx
            reason = "env(REPLAY_START_TS)"
        state = self._load_resume_state()
        resume_entry = state.get(self._state_key(symbol, interval)) if isinstance(state, dict) else None
        resume_idx = None
        if isinstance(resume_entry, dict) and resume_entry.get("last_processed_ts"):
            try:
                resume_ts = self._parse_ts(resume_entry["last_processed_ts"])
                resume_idx = self._first_index_at_or_after(candles, resume_ts)
            except ReplayDatasetError:
                resume_idx = None
        if resume_idx is not None:
            candidate_index = resume_idx
            reason = "resume_state_override"
            logger.warning("replay_start_precedence resume_state_active symbol=%s interval=%s", symbol.upper(), interval)
        warmup_floor = max(0, (max(2, int(history_limit)) if history_limit and history_limit > 0 else max(2, limit)) - 1)
        warmup_overrode = False
        if candidate_index < warmup_floor:
            candidate_index = warmup_floor
            warmup_overrode = True
        self._start_resolution[key] = {
            "effective_replay_start_ts": candles[candidate_index].close_time.isoformat(),
            "why_start_ts_overridden": "warmup_floor_enforced" if warmup_overrode else ("resume_state_override" if reason == "resume_state_override" else "none"),
            "selection_logic": "resume cursor/state > REPLAY_START_TS > CSV bounds/warmup",
            "source": reason,
            "resume_state_file": str(self._state_file),
        }
        return max(0, candidate_index - 1)

    def debug_start_resolution(self, symbol: str, interval: str) -> dict[str, object]:
        return dict(self._start_resolution.get((symbol.upper(), interval), {}))

    def _debug_trace_enabled(self) -> bool:
        return os.getenv("DEBUG_REPLAY_TRACE", "false").strip().lower() in {"1", "true", "yes", "on"}

    def _debug_trace_every(self) -> int:
        raw = os.getenv("DEBUG_REPLAY_TRACE_EVERY", "250").strip()
        return max(1, int(raw)) if raw.isdigit() else 250

    def _debug_trace(self, *, event: str, symbol: str, interval: str, cursor_index: int, candles_len: int, total_bars: int, source_file_path: str, slice_note: str) -> None:
        if not self._debug_trace_enabled() or (cursor_index % self._debug_trace_every() != 0):
            return
        caller = inspect.stack()[1]
        logger.info(
            "replay_trace event=%s symbol=%s interval=%s cursor_index=%s candles_len=%s total_bars=%s source=%s caller=%s:%s slice=%s",
            event,
            symbol.upper(),
            interval,
            cursor_index,
            candles_len,
            total_bars,
            source_file_path,
            caller.filename,
            caller.lineno,
            slice_note,
        )

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

    def validate_range(self, symbol: str, interval: str, start_ts: str | None, end_ts: str | None) -> dict[str, object]:
        dataset = self.validate_dataset(symbol, interval)
        candles = self._load_series(symbol, interval)
        start_dt = self._parse_ts(start_ts) if start_ts else candles[0].close_time
        end_dt = self._parse_ts(end_ts) if end_ts else candles[-1].close_time
        if start_dt > end_dt:
            raise ReplayDatasetError("replay_range_invalid start_ts must be <= end_ts")
        dataset_start = candles[0].close_time
        dataset_end = candles[-1].close_time
        if start_dt < dataset_start or start_dt > dataset_end:
            raise ReplayDatasetError(
                f"replay_start_ts_out_of_range start={start_dt.isoformat()} dataset=[{dataset_start.isoformat()}..{dataset_end.isoformat()}]"
            )
        if end_dt < dataset_start or end_dt > dataset_end:
            raise ReplayDatasetError(
                f"replay_end_ts_out_of_range end={end_dt.isoformat()} dataset=[{dataset_start.isoformat()}..{dataset_end.isoformat()}]"
            )
        filtered_bars = len([c for c in candles if start_dt <= c.close_time <= end_dt])
        if filtered_bars <= 0:
            raise ReplayDatasetError("replay_range_empty no candles in configured timestamp range")
        dataset["configured_start_ts"] = start_dt.isoformat()
        dataset["configured_end_ts"] = end_dt.isoformat()
        dataset["configured_total_bars"] = filtered_bars
        return dataset

    def reset_cursor(self, symbol: str, interval: str) -> None:
        key = (symbol.upper(), interval)
        self._cursor[key] = ReplayCursor(index=0)

    def status(self, symbol: str, interval: str) -> dict[str, object]:
        candles = self._load_series(symbol, interval)
        path, _ = self._path_for(symbol, interval)
        key = (symbol.upper(), interval)
        cursor = self._cursor.setdefault(key, ReplayCursor(index=0))
        idx = min(max(cursor.index, 0), len(candles) - 1)
        ts = candles[idx].close_time
        return {
            "symbol": symbol.upper(),
            "file_path": str(path),
            "exists": path.exists(),
            "interval": interval,
            "bar_index": idx,
            "bars_processed": idx + 1,
            "total_bars": len(candles),
            "row_count": len(candles),
            "first_ts": candles[0].close_time.isoformat(),
            "last_ts": candles[-1].close_time.isoformat(),
            "current_ts": ts.isoformat(),
            "progress_pct": round(((idx + 1) / len(candles)) * 100.0, 4),
            "resume_state_file": str(self._state_file),
            "start_resolution": self._start_resolution.get(key, {}),
        }

    async def fetch_symbol_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        speed: float = 1.0,
        start_ts: str | None = None,
        end_ts: str | None = None,
        history_limit: int | None = None,
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
        if key not in self._start_resolution:
            cursor.index = min(len(candles) - 1, self._resolve_start_index(symbol=symbol, interval=interval, candles=candles, start_ts=start_ts, limit=limit, history_limit=history_limit))
        _ = speed  # speed is scheduler pacing only; replay data advances one candle at a time for deterministic outcomes.
        cursor.index = min(len(candles) - 1, cursor.index + 1)
        if cursor.index % self._log_every_bars == 0:
            logger.info(
                "replay_cursor_progress symbol=%s interval=%s next_index=%s total=%s ts=%s",
                symbol.upper(),
                interval,
                cursor.index,
                len(candles),
                candles[cursor.index].close_time.isoformat(),
            )
        path, _ = self._path_for(symbol, interval)
        effective_history_limit = max(2, int(history_limit)) if history_limit and history_limit > 0 else max(2, limit)
        window_start = max(0, cursor.index - effective_history_limit + 1)
        window = candles[window_start : cursor.index + 1]
        self._debug_trace(
            event="snapshot_window",
            symbol=symbol,
            interval=interval,
            cursor_index=cursor.index,
            candles_len=len(window),
            total_bars=len(candles),
            source_file_path=str(path),
            slice_note=f"window_start={window_start};cursor={cursor.index};history_limit={effective_history_limit};limit={limit}",
        )
        candle = window[-1]
        state = self._load_resume_state()
        state[self._state_key(symbol, interval)] = {
            "cursor_index": cursor.index,
            "last_processed_ts": candle.close_time.isoformat(),
            **self._start_resolution.get(key, {}),
        }
        self._save_resume_state(state)
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
            cursor_index=cursor.index,
            total_bars=len(candles),
            source_file_path=str(path),
        )
