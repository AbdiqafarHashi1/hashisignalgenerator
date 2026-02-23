from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .bybit import BybitCandle, BybitKlineSnapshot


_GRANULARITY_MAP = {
    "1m": "M1",
    "5m": "M5",
    "1h": "H1",
}


def normalize_oanda_instrument(symbol: str) -> str:
    raw = symbol.strip().upper()
    if "_" in raw:
        return raw
    if len(raw) == 6:
        return f"{raw[:3]}_{raw[3:]}"
    return raw


class OandaPriceProvider:
    def __init__(self, *, api_token: str, account_id: str, environment: str = "practice") -> None:
        self.api_token = api_token.strip()
        self.account_id = account_id.strip()
        self.environment = (environment or "practice").strip().lower()
        host = "api-fxpractice.oanda.com" if self.environment != "live" else "api-fxtrade.oanda.com"
        self.rest_base = f"https://{host}/v3"

    def _headers(self) -> dict[str, str]:
        if not self.api_token:
            raise ValueError("oanda_api_token_missing")
        return {"Authorization": f"Bearer {self.api_token}"}

    async def fetch_candles(self, *, symbol: str, interval: str = "5m", limit: int = 120) -> BybitKlineSnapshot:
        granularity = _GRANULARITY_MAP.get(interval)
        if not granularity:
            raise ValueError(f"unsupported_oanda_interval:{interval}")
        instrument = normalize_oanda_instrument(symbol)
        params = {
            "price": "M",
            "granularity": granularity,
            "count": max(2, min(int(limit or 120), 5000)),
        }
        url = f"{self.rest_base}/instruments/{instrument}/candles"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, headers=self._headers(), params=params)
            response.raise_for_status()
            payload = response.json()

        rows = [item for item in payload.get("candles", []) if item.get("complete")]
        if not rows:
            raise ValueError(f"oanda_no_complete_candles:{instrument}")

        candles: list[BybitCandle] = []
        for row in rows:
            mid = row.get("mid") or {}
            close_time = _parse_ts(row.get("time"))
            open_time = _open_from_close(close_time, interval)
            candles.append(
                BybitCandle(
                    open_time=open_time,
                    open=float(mid.get("o", 0.0)),
                    high=float(mid.get("h", 0.0)),
                    low=float(mid.get("l", 0.0)),
                    close=float(mid.get("c", 0.0)),
                    volume=float(row.get("volume", 0.0) or 0.0),
                    close_time=close_time,
                )
            )

        candle = candles[-1]
        return BybitKlineSnapshot(
            symbol=symbol.upper().replace("_", ""),
            interval=interval,
            price=candle.close,
            volume=candle.volume,
            kline_open_time_ms=int(candle.open_time.timestamp() * 1000),
            kline_close_time_ms=int(candle.close_time.timestamp() * 1000),
            kline_is_closed=True,
            candle=candle,
            candles=candles,
            provider_name="oanda",
            provider_category="forex",
            provider_endpoint=url,
        )


def _parse_ts(value: Any) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _open_from_close(close_time: datetime, interval: str) -> datetime:
    delta_minutes = {"1m": 1, "5m": 5, "1h": 60}[interval]
    return close_time - timedelta(minutes=delta_minutes)
