from __future__ import annotations

from datetime import datetime, timezone

import httpx
from pydantic import BaseModel

BINANCE_BASE_URL = "https://api.binance.com"


class BinanceCandle(BaseModel):
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime


class BinanceKlineSnapshot(BaseModel):
    symbol: str
    interval: str
    price: float
    volume: float
    kline_open_time_ms: int
    kline_close_time_ms: int
    kline_is_closed: bool
    candle: BinanceCandle


def _parse_kline(raw: list) -> BinanceCandle:
    open_time = datetime.fromtimestamp(raw[0] / 1000, tz=timezone.utc)
    close_time = datetime.fromtimestamp(raw[6] / 1000, tz=timezone.utc)
    return BinanceCandle(
        open_time=open_time,
        open=float(raw[1]),
        high=float(raw[2]),
        low=float(raw[3]),
        close=float(raw[4]),
        volume=float(raw[5]),
        close_time=close_time,
    )


async def fetch_btcusdt_klines(interval: str = "15m") -> BinanceKlineSnapshot:
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": 1}
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    if not data:
        raise ValueError("No kline data returned from Binance")
    candle = _parse_kline(data[-1])
    open_time_ms = int(data[-1][0])
    close_time_ms = int(data[-1][6])
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return BinanceKlineSnapshot(
        symbol="BTCUSDT",
        interval=interval,
        price=candle.close,
        volume=candle.volume,
        kline_open_time_ms=open_time_ms,
        kline_close_time_ms=close_time_ms,
        kline_is_closed=now_ms >= close_time_ms,
        candle=candle,
    )
