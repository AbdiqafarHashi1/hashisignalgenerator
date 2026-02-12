from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

DEFAULT_REST_BASE = "https://api-testnet.bybit.com"
DEFAULT_WS_PUBLIC_LINEAR = "wss://stream-testnet.bybit.com/v5/public/linear"


class BybitCandle(BaseModel):
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime


class BybitKlineSnapshot(BaseModel):
    symbol: str
    interval: str
    price: float
    volume: float
    kline_open_time_ms: int
    kline_close_time_ms: int
    kline_is_closed: bool
    candle: BybitCandle
    candles: list[BybitCandle] = Field(default_factory=list)


class BybitClient:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        rest_base: str = DEFAULT_REST_BASE,
        recv_window: int = 5000,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.rest_base = rest_base.rstrip("/")
        self.recv_window = recv_window

    def _sign(self, timestamp: str, payload: str) -> str:
        raw = f"{timestamp}{self.api_key}{self.recv_window}{payload}"
        return hmac.new(self.api_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()

    def _signed_headers(self, payload: str) -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        signature = self._sign(timestamp, payload)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> dict[str, Any]:
        params = params or {}
        url = f"{self.rest_base}{path}"
        headers: dict[str, str] = {}
        if signed:
            # Bybit signs query string payload for GETs.
            payload = "&".join(f"{k}={params[k]}" for k in sorted(params.keys()))
            headers = self._signed_headers(payload)
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        if data.get("retCode") != 0:
            raise ValueError(f"Bybit error: {data.get('retMsg')} ({data.get('retCode')})")
        return data

    async def _post(self, path: str, body: dict[str, Any], signed: bool = True) -> dict[str, Any]:
        url = f"{self.rest_base}{path}"
        payload = json.dumps(body, separators=(",", ":"), sort_keys=True)
        headers = self._signed_headers(payload) if signed else {"Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, content=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        if data.get("retCode") != 0:
            raise ValueError(f"Bybit error: {data.get('retMsg')} ({data.get('retCode')})")
        return data

    async def fetch_candles(self, symbol: str, interval: str = "5m", limit: int = 120) -> BybitKlineSnapshot | None:
        requested_limit = max(2, min(limit, 1000))
        data = await self._get(
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": _to_bybit_interval(interval),
                "limit": requested_limit,
            },
        )
        rows = data.get("result", {}).get("list", [])
        if len(rows) < 1:
            raise ValueError("Bybit returned no klines")
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        rows = _sort_rows_oldest_first(rows)
        selected_row, selected_closed = _select_latest_closed_row(rows, interval, now_ms)
        if selected_row is None:
            return None
        selected_open_time_ms = _kline_start_time_ms(selected_row)
        selected_close_time_ms = _kline_close_time_ms(selected_row, interval)
        candles = [_parse_kline(raw, interval) for raw in rows if _is_row_closed(raw, interval, now_ms)]
        if not candles:
            candles = [_parse_kline(selected_row, interval)]
        candle = _parse_kline(selected_row, interval)
        return BybitKlineSnapshot(
            symbol=symbol,
            interval=interval,
            price=candle.close,
            volume=candle.volume,
            kline_open_time_ms=selected_open_time_ms,
            kline_close_time_ms=selected_close_time_ms,
            kline_is_closed=selected_closed,
            candle=candle,
            candles=candles,
        )

    async def fetch_kline_debug_rows(self, symbol: str, interval: str = "5m", limit: int = 3) -> dict[str, Any]:
        data = await self._get(
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": _to_bybit_interval(interval),
                "limit": max(1, min(limit, 10)),
            },
        )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        rows = _sort_rows_oldest_first(data.get("result", {}).get("list", []))
        sampled_rows = rows[-3:]
        return {
            "symbol": symbol,
            "interval": interval,
            "now_ms": now_ms,
            "rows": [
                {
                    "raw": row,
                    "start_time_ms": _kline_start_time_ms(row),
                    "close_time_ms": _kline_close_time_ms(row, interval),
                    "confirm": _kline_confirm(row),
                    "is_closed": _is_row_closed(row, interval, now_ms),
                }
                for row in sampled_rows
            ],
        }

    async def fetch_ticker_price(self, symbol: str) -> float:
        data = await self._get(
            "/v5/market/tickers",
            {"category": "linear", "symbol": symbol},
        )
        rows = data.get("result", {}).get("list", [])
        if not rows:
            raise ValueError(f"No ticker returned for {symbol}")
        return float(rows[0].get("lastPrice", 0.0))

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        order_type: str = "Market",
        price: str | None = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
        }
        if order_type.lower() == "limit" and price is not None:
            body["price"] = price
        return await self._post("/v5/order/create", body, signed=True)

    async def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        return await self._post(
            "/v5/order/cancel",
            {"category": "linear", "symbol": symbol, "orderId": order_id},
            signed=True,
        )

    async def fetch_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        data = await self._get("/v5/position/list", params, signed=True)
        return data.get("result", {}).get("list", [])

    async def fetch_realized_pnl(self, symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "category": "linear",
            "limit": max(1, min(limit, 100)),
        }
        if symbol:
            params["symbol"] = symbol
        data = await self._get("/v5/position/closed-pnl", params, signed=True)
        return data.get("result", {}).get("list", [])

    async def set_leverage(self, symbol: str, buy_leverage: str, sell_leverage: str) -> dict[str, Any]:
        return await self._post(
            "/v5/position/set-leverage",
            {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": buy_leverage,
                "sellLeverage": sell_leverage,
            },
            signed=True,
        )

    async def set_isolated_margin(self, symbol: str, leverage: str) -> dict[str, Any]:
        return await self._post(
            "/v5/position/switch-isolated",
            {
                "category": "linear",
                "symbol": symbol,
                "tradeMode": 1,
                "buyLeverage": leverage,
                "sellLeverage": leverage,
            },
            signed=True,
        )


def _parse_kline(raw: Any, interval: str) -> BybitCandle:
    open_time_ms = _kline_start_time_ms(raw)
    close_time_ms = _kline_close_time_ms(raw, interval)
    return BybitCandle(
        open_time=datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc),
        open=_kline_float(raw, 1, "openPrice"),
        high=_kline_float(raw, 2, "highPrice"),
        low=_kline_float(raw, 3, "lowPrice"),
        close=_kline_float(raw, 4, "closePrice"),
        volume=_kline_float(raw, 5, "volume"),
        close_time=datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc),
    )


def _kline_float(raw: Any, index: int, key: str) -> float:
    if isinstance(raw, dict):
        return float(raw.get(key, 0.0))
    return float(raw[index])


def _to_bybit_interval(interval: str) -> str:
    interval = interval.lower().strip()
    mapping = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
    return mapping.get(interval, "5")


def _interval_to_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping.get(interval.lower(), 300_000)


def _sort_rows_oldest_first(rows: list[Any]) -> list[Any]:
    return sorted(rows, key=_kline_start_time_ms)


def _select_latest_closed_row(rows: list[Any], interval: str, now_ms: int) -> tuple[Any | None, bool]:
    for row in reversed(rows):
        if _is_row_closed(row, interval, now_ms):
            return row, True
    return None, False


def _is_row_closed(raw: Any, interval: str, now_ms: int) -> bool:
    confirm = _kline_confirm(raw)
    if confirm is not None:
        return confirm
    close_time_ms = _kline_close_time_ms(raw, interval)
    return now_ms >= close_time_ms


def _kline_start_time_ms(raw: Any) -> int:
    if isinstance(raw, dict):
        value = raw.get("startTime") or raw.get("openTime") or raw.get("start") or 0
        return _timestamp_to_ms(value)
    return _timestamp_to_ms(raw[0])


def _kline_close_time_ms(raw: Any, interval: str) -> int:
    if isinstance(raw, dict):
        close_value = raw.get("closeTime") or raw.get("endTime")
        if close_value is not None:
            return _timestamp_to_ms(close_value)
    return _kline_start_time_ms(raw) + _interval_to_ms(interval)


def _timestamp_to_ms(value: Any) -> int:
    timestamp = int(float(value))
    # Handle second-based timestamps from some payloads; normalize to ms.
    if abs(timestamp) < 10_000_000_000:
        return timestamp * 1000
    return timestamp


def _kline_confirm(raw: Any) -> bool | None:
    value = None
    if isinstance(raw, dict):
        value = raw.get("confirm")
    elif isinstance(raw, (list, tuple)) and len(raw) >= 9:
        value = raw[8]
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


async def fetch_symbol_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 120,
    rest_base: str = DEFAULT_REST_BASE,
) -> BybitKlineSnapshot | None:
    client = BybitClient(rest_base=rest_base)
    return await client.fetch_candles(symbol=symbol, interval=interval, limit=limit)
