from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
import asyncio
import random
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..utils.intervals import interval_to_ms
from .binance import fetch_symbol_klines as fetch_binance_symbol_klines
from .okx import fetch_symbol_klines as fetch_okx_symbol_klines
from .replay import ReplayDatasetError, ReplayProvider

DEFAULT_REST_BASE = "https://api-testnet.bybit.com"
DEFAULT_WS_PUBLIC_LINEAR = "wss://stream-testnet.bybit.com/v5/public/linear"
PRICE_SANITY_DEVIATION = 0.10


logger = logging.getLogger(__name__)
_replay_providers: dict[str, ReplayProvider] = {}


class BybitRateLimitError(ValueError):
    pass


class MarketDataBlockedError(ValueError):
    pass


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
    provider_name: str = "bybit"
    provider_category: str = "linear"
    provider_endpoint: str = "/v5/market/kline"


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
            ret_code = data.get("retCode")
            message = f"Bybit error: {data.get('retMsg')} ({ret_code})"
            if ret_code == 10006:
                raise BybitRateLimitError(message)
            raise ValueError(message)
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
            ret_code = data.get("retCode")
            message = f"Bybit error: {data.get('retMsg')} ({ret_code})"
            if ret_code == 10006:
                raise BybitRateLimitError(message)
            raise ValueError(message)
        return data

    async def fetch_candles(self, symbol: str, interval: str = "5m", limit: int = 120) -> BybitKlineSnapshot | None:
        requested_limit = max(2, min(limit, 1000))
        now_ms = await self._fetch_server_time_ms()
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
        if not _is_valid_candle(candle):
            logger.warning(
                "price_sanity_failed symbol=%s endpoint=%s reason=invalid_candle o=%s h=%s l=%s c=%s",
                symbol,
                f"{self.rest_base}/v5/market/kline",
                candle.open,
                candle.high,
                candle.low,
                candle.close,
            )
            return None
        if not _price_within_reasonable_band(candle.close, candles):
            median_close = _median_close(candles)
            logger.warning(
                "price_sanity_failed symbol=%s endpoint=%s reason=price_band close=%s median=%s limit_pct=%.2f",
                symbol,
                f"{self.rest_base}/v5/market/kline",
                candle.close,
                median_close,
                PRICE_SANITY_DEVIATION * 100,
            )
            return None
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
            provider_endpoint=f"{self.rest_base}/v5/market/kline",
        )

    async def fetch_kline_debug_rows(self, symbol: str, interval: str = "5m", limit: int = 3) -> dict[str, Any]:
        now_ms = await self._fetch_server_time_ms()
        interval_ms = interval_to_ms(interval)
        data = await self._get(
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": _to_bybit_interval(interval),
                "limit": max(1, min(limit, 10)),
            },
        )
        rows = _sort_rows_oldest_first(data.get("result", {}).get("list", []))
        sampled_rows = rows[-3:]
        return {
            "symbol": symbol,
            "interval": interval,
            "now_ms": now_ms,
            "interval_ms": interval_ms,
            "rows": [
                {
                    "raw": row,
                    "start_time_ms": _kline_start_time_ms(row),
                    "interval_ms": interval_ms,
                    "end_ms": _kline_start_time_ms(row) + interval_ms,
                    "close_time_ms": _kline_close_time_ms(row, interval),
                    "confirm": _kline_confirm(row),
                    "is_closed": _is_row_closed(row, interval, now_ms),
                }
                for row in sampled_rows
            ],
        }

    async def _fetch_server_time_ms(self) -> int:
        data = await self._get("/v5/market/time")
        result = data.get("result", {})
        if isinstance(result, dict):
            time_nano = result.get("timeNano")
            if time_nano is not None:
                return int(int(time_nano) / 1_000_000)
            time_second = result.get("timeSecond")
            if time_second is not None:
                return int(time_second) * 1000
            time_ms = result.get("time")
            if time_ms is not None:
                return _timestamp_to_ms(time_ms)
        raise ValueError("Bybit returned invalid server time payload")

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
    return interval_to_ms(interval)


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


def _is_valid_candle(candle: BybitCandle) -> bool:
    values = (candle.open, candle.high, candle.low, candle.close)
    if any(math.isnan(value) or math.isinf(value) for value in values):
        return False
    return candle.high >= candle.low


def _median_close(candles: list[BybitCandle]) -> float:
    closes = sorted(candle.close for candle in candles if not math.isnan(candle.close) and not math.isinf(candle.close))
    if not closes:
        return 0.0
    mid = len(closes) // 2
    if len(closes) % 2 == 1:
        return closes[mid]
    return (closes[mid - 1] + closes[mid]) / 2.0


def _price_within_reasonable_band(price: float, candles: list[BybitCandle]) -> bool:
    if math.isnan(price) or math.isinf(price):
        return False
    median_close = _median_close(candles[-20:])
    if median_close <= 0:
        return False
    return abs(price - median_close) / median_close <= PRICE_SANITY_DEVIATION


async def fetch_symbol_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 120,
    rest_base: str = DEFAULT_REST_BASE,
    provider: str = "bybit",
    fallback_provider: str = "binance,okx",
    failover_threshold: int = 3,
    backoff_base_ms: int = 500,
    backoff_max_ms: int = 15000,
    replay_path: str = "data/replay",
    replay_speed: float = 1.0,
    replay_start_ts: str | None = None,
    replay_end_ts: str | None = None,
) -> BybitKlineSnapshot | None:
    provider_normalized = (provider or "bybit").lower()
    fallback_order = _parse_fallbacks(fallback_provider)
    if provider_normalized == "replay":
        return await _fetch_from_replay(symbol=symbol, interval=interval, limit=limit, replay_path=replay_path, replay_speed=replay_speed, replay_start_ts=replay_start_ts, replay_end_ts=replay_end_ts)
    if provider_normalized == "binance":
        return await _fetch_from_binance(symbol=symbol, interval=interval, limit=limit)
    if provider_normalized == "okx":
        return await _fetch_from_okx(symbol=symbol, interval=interval, limit=limit)
    if provider_normalized != "bybit":
        raise ValueError(f"unsupported_market_data_provider:{provider_normalized}")

    client = BybitClient(rest_base=rest_base)
    threshold = max(1, int(failover_threshold or 1))
    backoff_base = max(50, int(backoff_base_ms or 500))
    backoff_cap = max(backoff_base, int(backoff_max_ms or 15000))

    for attempt in range(1, threshold + 1):
        try:
            snapshot = await client.fetch_candles(symbol=symbol, interval=interval, limit=limit)
            if snapshot is not None:
                snapshot.provider_name = "bybit"
            return snapshot
        except Exception as exc:
            reason = _bybit_error_reason(exc)
            blocked = _is_blocking_error(exc)
            logger.warning(
                "market_data_primary_error provider=bybit symbol=%s interval=%s attempt=%s reason=%s",
                symbol,
                interval,
                attempt,
                reason,
            )
            if blocked:
                return await _try_fallback_chain(
                    fallback_order,
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    replay_path=replay_path,
                    replay_speed=replay_speed,
                    replay_start_ts=replay_start_ts,
                    replay_end_ts=replay_end_ts,
                    trigger_reason=reason,
                )
            if attempt >= threshold:
                break
            delay_ms = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
            jitter = random.uniform(0, delay_ms * 0.2)
            await asyncio.sleep((delay_ms + jitter) / 1000)

    return await _try_fallback_chain(
        fallback_order,
        symbol=symbol,
        interval=interval,
        limit=limit,
        replay_path=replay_path,
        replay_speed=replay_speed,
        replay_start_ts=replay_start_ts,
        replay_end_ts=replay_end_ts,
        trigger_reason="threshold_exceeded",
    )


async def _try_fallback_chain(
    fallbacks: list[str],
    *,
    symbol: str,
    interval: str,
    limit: int,
    replay_path: str,
    replay_speed: float,
    replay_start_ts: str | None,
    replay_end_ts: str | None,
    trigger_reason: str,
) -> BybitKlineSnapshot:
    logger.warning("market_data_failover_trigger symbol=%s reason=%s chain=%s", symbol, trigger_reason, ",".join(fallbacks))
    errors: list[str] = []
    for fallback in fallbacks:
        try:
            if fallback == "binance":
                return await _fetch_from_binance(symbol=symbol, interval=interval, limit=limit)
            if fallback == "okx":
                return await _fetch_from_okx(symbol=symbol, interval=interval, limit=limit)
            if fallback == "replay":
                return await _fetch_from_replay(symbol=symbol, interval=interval, limit=limit, replay_path=replay_path, replay_speed=replay_speed, replay_start_ts=replay_start_ts, replay_end_ts=replay_end_ts)
        except Exception as exc:
            errors.append(f"{fallback}:{type(exc).__name__}")
            continue
    raise MarketDataBlockedError("MARKET_DATA_BLOCKED:" + "|".join(errors or ["no_fallback"]))


def _parse_fallbacks(value: str) -> list[str]:
    raw = (value or "binance,okx,replay").strip()
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _is_blocking_error(exc: Exception) -> bool:
    if isinstance(exc, BybitRateLimitError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {401, 403, 418, 429}
    message = str(exc)
    return any(code in message for code in ("10006", "401", "403", "418", "429", "ProxyError", "Forbidden"))


def _bybit_error_reason(exc: Exception) -> str:
    if isinstance(exc, BybitRateLimitError):
        return "BYBIT_RATE_LIMIT_10006"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP_{exc.response.status_code}"
    if isinstance(exc, httpx.ProxyError):
        return "HTTP_PROXY_403"
    txt = str(exc)
    if "10006" in txt:
        return "BYBIT_RATE_LIMIT_10006"
    return type(exc).__name__


async def _fetch_from_binance(symbol: str, interval: str, limit: int) -> BybitKlineSnapshot:
    fallback = await fetch_binance_symbol_klines(symbol=symbol, interval=interval, limit=limit)
    candles = [
        BybitCandle(
            open_time=item.open_time,
            open=item.open,
            high=item.high,
            low=item.low,
            close=item.close,
            volume=item.volume,
            close_time=item.close_time,
        )
        for item in fallback.candles
    ]
    candle = candles[-1]
    return BybitKlineSnapshot(
        symbol=fallback.symbol,
        interval=fallback.interval,
        price=fallback.price,
        volume=fallback.volume,
        kline_open_time_ms=fallback.kline_open_time_ms,
        kline_close_time_ms=fallback.kline_close_time_ms,
        kline_is_closed=fallback.kline_is_closed,
        candle=candle,
        candles=candles,
        provider_name="binance",
        provider_category="spot",
        provider_endpoint="/api/v3/klines",
    )


async def _fetch_from_okx(symbol: str, interval: str, limit: int) -> BybitKlineSnapshot:
    fallback = await fetch_okx_symbol_klines(symbol=symbol, interval=interval, limit=limit)
    candles = [
        BybitCandle(
            open_time=item.open_time,
            open=item.open,
            high=item.high,
            low=item.low,
            close=item.close,
            volume=item.volume,
            close_time=item.close_time,
        )
        for item in fallback.candles
    ]
    candle = candles[-1]
    return BybitKlineSnapshot(
        symbol=fallback.symbol,
        interval=fallback.interval,
        price=fallback.price,
        volume=fallback.volume,
        kline_open_time_ms=fallback.kline_open_time_ms,
        kline_close_time_ms=fallback.kline_close_time_ms,
        kline_is_closed=fallback.kline_is_closed,
        candle=candle,
        candles=candles,
        provider_name="okx",
        provider_category="swap",
        provider_endpoint="/api/v5/market/history-candles",
    )


async def _fetch_from_replay(symbol: str, interval: str, limit: int, replay_path: str, replay_speed: float, replay_start_ts: str | None = None, replay_end_ts: str | None = None) -> BybitKlineSnapshot:
    provider = _replay_providers.get(replay_path)
    if provider is None:
        provider = ReplayProvider(replay_path)
        _replay_providers[replay_path] = provider
    replay = await provider.fetch_symbol_klines(symbol=symbol, interval=interval, limit=limit, speed=replay_speed, start_ts=replay_start_ts, end_ts=replay_end_ts)
    candles = [
        BybitCandle(
            open_time=item.open_time,
            open=item.open,
            high=item.high,
            low=item.low,
            close=item.close,
            volume=item.volume,
            close_time=item.close_time,
        )
        for item in replay.candles
    ]
    candle = candles[-1]
    return BybitKlineSnapshot(
        symbol=replay.symbol,
        interval=replay.interval,
        price=replay.price,
        volume=replay.volume,
        kline_open_time_ms=replay.kline_open_time_ms,
        kline_close_time_ms=replay.kline_close_time_ms,
        kline_is_closed=True,
        candle=candle,
        candles=candles,
        provider_name="replay",
        provider_category="local",
        provider_endpoint="replay",
    )



def replay_status(replay_path: str, symbol: str, interval: str) -> dict[str, object]:
    provider = _replay_providers.get(replay_path)
    if provider is None:
        provider = ReplayProvider(replay_path)
        _replay_providers[replay_path] = provider
    return provider.status(symbol, interval)


def replay_validate_dataset(
    replay_path: str,
    symbol: str,
    interval: str,
    start_ts: str | None = None,
    end_ts: str | None = None,
) -> dict[str, object]:
    provider = _replay_providers.get(replay_path)
    if provider is None:
        provider = ReplayProvider(replay_path)
        _replay_providers[replay_path] = provider
    return provider.validate_range(symbol, interval, start_ts=start_ts, end_ts=end_ts)


def replay_reset(replay_path: str, symbol: str, interval: str) -> None:
    provider = _replay_providers.get(replay_path)
    if provider is None:
        provider = ReplayProvider(replay_path)
        _replay_providers[replay_path] = provider
    provider.reset_cursor(symbol, interval)


def replay_is_error(exc: Exception) -> bool:
    return isinstance(exc, ReplayDatasetError)
