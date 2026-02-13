from __future__ import annotations

from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, Field

from ..utils.intervals import interval_to_ms


OKX_BASE_URL = "https://www.okx.com"


class OkxCandle(BaseModel):
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime


class OkxKlineSnapshot(BaseModel):
    symbol: str
    interval: str
    price: float
    volume: float
    kline_open_time_ms: int
    kline_close_time_ms: int
    kline_is_closed: bool
    candle: OkxCandle
    candles: list[OkxCandle] = Field(default_factory=list)


def _to_okx_inst(symbol: str) -> str:
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}-USDT-SWAP"
    return symbol


def _to_okx_bar(interval: str) -> str:
    mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }
    return mapping.get(interval.lower(), "1m")


def _parse_row(raw: list[str], interval: str) -> OkxCandle:
    open_time_ms = int(raw[0])
    close_time_ms = open_time_ms + interval_to_ms(interval)
    return OkxCandle(
        open_time=datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc),
        open=float(raw[1]),
        high=float(raw[2]),
        low=float(raw[3]),
        close=float(raw[4]),
        volume=float(raw[5]),
        close_time=datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc),
    )


async def fetch_symbol_klines(symbol: str, interval: str = "1m", limit: int = 120) -> OkxKlineSnapshot:
    url = f"{OKX_BASE_URL}/api/v5/market/history-candles"
    params = {
        "instId": _to_okx_inst(symbol),
        "bar": _to_okx_bar(interval),
        "limit": max(2, min(limit, 300)),
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise ValueError("No kline data returned from OKX")
    rows = list(reversed(data))
    candles = [_parse_row(row, interval) for row in rows]
    candle = candles[-1]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    open_ms = int(rows[-1][0])
    close_ms = open_ms + interval_to_ms(interval)
    return OkxKlineSnapshot(
        symbol=symbol,
        interval=interval,
        price=candle.close,
        volume=candle.volume,
        kline_open_time_ms=open_ms,
        kline_close_time_ms=close_ms,
        kline_is_closed=now_ms >= close_ms,
        candle=candle,
        candles=candles,
    )
