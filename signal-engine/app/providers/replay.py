from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from pydantic import BaseModel

from ..utils.intervals import interval_to_ms


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
        self._base.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, str], list[ReplayCandle]] = {}
        self._cursor: dict[tuple[str, str], ReplayCursor] = {}

    def _path_for(self, symbol: str, interval: str) -> Path:
        return self._base / f"{symbol}_{interval}.json"

    def _generate_series(self, symbol: str, interval: str, count: int = 500) -> list[ReplayCandle]:
        step = interval_to_ms(interval)
        now = datetime.now(timezone.utc)
        start = now - timedelta(milliseconds=step * count)
        price = 100000.0 if symbol.startswith("BTC") else 3500.0
        candles: list[ReplayCandle] = []
        for idx in range(count):
            t0 = start + timedelta(milliseconds=idx * step)
            drift = ((idx % 24) - 12) * 0.00015
            open_price = price
            close_price = max(1.0, price * (1 + drift))
            high = max(open_price, close_price) * 1.0006
            low = min(open_price, close_price) * 0.9994
            volume = 10 + (idx % 9)
            t1 = t0 + timedelta(milliseconds=step)
            candles.append(
                ReplayCandle(
                    open_time=t0,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                    close_time=t1,
                )
            )
            price = close_price
        self._write_file(symbol, interval, candles)
        return candles

    def _write_file(self, symbol: str, interval: str, candles: list[ReplayCandle]) -> None:
        path = self._path_for(symbol, interval)
        path.write_text(json.dumps([c.model_dump(mode="json") for c in candles], indent=2), encoding="utf-8")

    def _load_series(self, symbol: str, interval: str) -> list[ReplayCandle]:
        key = (symbol, interval)
        if key in self._cache:
            return self._cache[key]
        path = self._path_for(symbol, interval)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            candles = [ReplayCandle(**row) for row in raw]
            if candles:
                now = datetime.now(timezone.utc)
                if (now - candles[-1].close_time).total_seconds() > 120:
                    candles = self._generate_series(symbol, interval)
        else:
            candles = self._generate_series(symbol, interval)
        self._cache[key] = candles
        self._cursor.setdefault(key, ReplayCursor(index=max(0, len(candles) - 2)))
        return candles


    def _next_candle(self, previous: ReplayCandle, interval: str, index: int, symbol: str) -> ReplayCandle:
        step_ms = interval_to_ms(interval)
        open_price = previous.close
        drift = ((index % 24) - 12) * 0.00015
        close_price = max(1.0, open_price * (1 + drift))
        high = max(open_price, close_price) * 1.0006
        low = min(open_price, close_price) * 0.9994
        volume = 10 + (index % 9)
        t0 = previous.close_time
        t1 = t0 + timedelta(milliseconds=step_ms)
        return ReplayCandle(
            open_time=t0,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume,
            close_time=t1,
        )

    async def fetch_symbol_klines(self, symbol: str, interval: str, limit: int, speed: float = 1.0) -> ReplayKlineSnapshot:
        candles = self._load_series(symbol, interval)
        key = (symbol, interval)
        cursor = self._cursor.setdefault(key, ReplayCursor(index=max(0, len(candles) - limit)))
        step = max(1, int(round(speed)))
        cursor.index = min(len(candles) - 1, cursor.index + step)
        if cursor.index >= len(candles) - 1:
            candles.append(self._next_candle(candles[-1], interval, cursor.index + 1, symbol))
            cursor.index = len(candles) - 1
            self._cache[key] = candles
            self._write_file(symbol, interval, candles)
        window_start = max(0, cursor.index - max(2, limit) + 1)
        window = candles[window_start : cursor.index + 1]
        candle = window[-1]
        return ReplayKlineSnapshot(
            symbol=symbol,
            interval=interval,
            price=candle.close,
            volume=candle.volume,
            kline_open_time_ms=int(candle.open_time.timestamp() * 1000),
            kline_close_time_ms=int(candle.close_time.timestamp() * 1000),
            kline_is_closed=True,
            candle=candle,
            candles=window,
        )
