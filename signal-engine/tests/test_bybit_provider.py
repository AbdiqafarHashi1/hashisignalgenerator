from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.providers.bybit import BybitClient, _select_latest_closed_row, _sort_rows_oldest_first


class _StubBybitClient(BybitClient):
    def __init__(self, rows, now_ms: int = 1_700_000_150_000):
        super().__init__(rest_base="https://example.com")
        self._rows = rows
        self._now_ms = now_ms
        self.last_params = None

    async def _get(self, *args, **kwargs):  # type: ignore[override]
        path = args[0] if args else kwargs.get("path")
        if path == "/v5/market/time":
            return {"retCode": 0, "result": {"timeNano": str(self._now_ms * 1_000_000)}}
        if len(args) >= 2:
            self.last_params = args[1]
        else:
            self.last_params = kwargs.get("params")
        return {"retCode": 0, "result": {"list": self._rows}}


def test_select_latest_closed_row_ignores_newest_if_forming() -> None:
    rows_newest_first = [
        ["1700000120000", "101", "102", "99", "101.5", "20", "0", "0", "false"],
        ["1700000060000", "100", "101", "98", "100.5", "10", "0", "0", "true"],
    ]
    rows = _sort_rows_oldest_first(rows_newest_first)

    selected, is_closed = _select_latest_closed_row(rows, interval="1m", now_ms=1700000150000)

    assert is_closed is True
    assert selected is not None
    assert int(selected[0]) == 1700000060000


def test_select_latest_closed_row_returns_not_ready_when_all_rows_open() -> None:
    rows_newest_first = [
        ["1700000180000", "102", "103", "101", "102.5", "22", "0", "0", "false"],
        ["1700000120000", "101", "102", "99", "101.5", "20", "0", "0", "false"],
    ]
    rows = _sort_rows_oldest_first(rows_newest_first)

    selected, is_closed = _select_latest_closed_row(rows, interval="1m", now_ms=1700000150000)

    assert selected is None
    assert is_closed is False


def test_fetch_candles_returns_none_when_previous_candle_not_closed() -> None:
    rows_newest_first = [
        ["1700000180000", "102", "103", "101", "102.5", "22", "0", "0", "false"],
        ["1700000120000", "101", "102", "99", "101.5", "20", "0", "0", "false"],
    ]
    client = _StubBybitClient(rows_newest_first)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=2)
        assert snapshot is None

    asyncio.run(run())


def test_fetch_candles_uses_latest_closed_candle_when_newest_is_closed() -> None:
    rows_newest_first = [
        ["1700000180000", "102", "103", "101", "102.5", "22", "0", "0", "true"],
        ["1700000120000", "101", "102", "99", "101.5", "20", "0", "0", "true"],
    ]
    client = _StubBybitClient(rows_newest_first)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=2)
        assert snapshot is not None
        assert snapshot.kline_open_time_ms == 1700000180000
        assert snapshot.kline_close_time_ms == 1700000240000

    asyncio.run(run())


def test_fetch_candles_normalizes_second_timestamps_and_string_confirm() -> None:
    rows_newest_first = [
        {"startTime": "180", "openPrice": "102", "highPrice": "103", "lowPrice": "101", "closePrice": "102.5", "volume": "22", "confirm": "false"},
        {"startTime": "120", "openPrice": "101", "highPrice": "102", "lowPrice": "99", "closePrice": "101.5", "volume": "20", "confirm": "true"},
    ]
    client = _StubBybitClient(rows_newest_first)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=2)
        assert snapshot is not None
        assert snapshot.kline_open_time_ms == 120000
        assert snapshot.kline_close_time_ms == 180000

    asyncio.run(run())


def test_fetch_candles_returns_none_when_now_is_inside_latest_candle() -> None:
    rows_newest_first = [
        ["1700000180000", "101", "102", "99", "101.5", "20", "0", "0"],
        ["1700000130000", "100", "101", "98", "100.5", "10", "0", "0"],
    ]
    client = _StubBybitClient(rows_newest_first, now_ms=1700000185000)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=2)
        assert snapshot is None

    asyncio.run(run())


def test_fetch_candles_returns_snapshot_when_now_is_past_candle_end() -> None:
    rows_newest_first = [
        ["1700000180000", "101", "102", "99", "101.5", "20", "0", "0"],
        ["1700000130000", "100", "101", "98", "100.5", "10", "0", "0"],
    ]
    client = _StubBybitClient(rows_newest_first, now_ms=1700000240000)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=2)
        assert snapshot is not None
        assert snapshot.kline_open_time_ms == 1700000180000
        assert snapshot.kline_close_time_ms == 1700000240000
        assert snapshot.kline_is_closed is True

    asyncio.run(run())

def test_fetch_candles_uses_last_closed_candle_for_snapshot() -> None:
    rows_newest_first = [
        ["1700000120000", "101", "102", "99", "101.5", "20", "0", "0", "false"],
        ["1700000060000", "100", "101", "98", "100.5", "10", "0", "0", "true"],
    ]
    client = _StubBybitClient(rows_newest_first)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=2)
        assert snapshot.kline_open_time_ms == 1700000060000
        assert snapshot.kline_close_time_ms == 1700000120000
        assert snapshot.kline_is_closed is True
        assert snapshot.candle.open_time == datetime.fromtimestamp(1700000060, tz=timezone.utc)
        assert len(snapshot.candles) == 1
        assert snapshot.candles[0].open_time == datetime.fromtimestamp(1700000060, tz=timezone.utc)
        assert client.last_params is not None
        assert client.last_params["limit"] == 2

    asyncio.run(run())


def test_fetch_candles_never_requests_limit_one() -> None:
    rows_newest_first = [
        ["1700000120000", "101", "102", "99", "101.5", "20", "0", "0", "false"],
        ["1700000060000", "100", "101", "98", "100.5", "10", "0", "0", "true"],
    ]
    client = _StubBybitClient(rows_newest_first)

    async def run():
        await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=1)
        assert client.last_params is not None
        assert client.last_params["limit"] == 2

    asyncio.run(run())


def test_fetch_candles_returns_none_when_price_band_is_unrealistic() -> None:
    rows_newest_first = [
        ["1700000180000", "30000", "35000", "29000", "34000", "20", "0", "0", "true"],
        ["1700000120000", "100", "101", "99", "100", "20", "0", "0", "true"],
        ["1700000060000", "101", "102", "100", "101", "20", "0", "0", "true"],
    ]
    client = _StubBybitClient(rows_newest_first, now_ms=1700000240000)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=3)
        assert snapshot is None

    asyncio.run(run())


def test_fetch_candles_returns_none_on_invalid_high_low() -> None:
    rows_newest_first = [
        ["1700000180000", "101", "99", "100", "101", "20", "0", "0", "true"],
        ["1700000120000", "100", "101", "99", "100", "20", "0", "0", "true"],
    ]
    client = _StubBybitClient(rows_newest_first, now_ms=1700000240000)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=2)
        assert snapshot is None

    asyncio.run(run())


def test_failover_triggers_after_threshold_on_403(monkeypatch) -> None:
    from app.providers import bybit as bybit_module

    attempts = {"count": 0}

    async def fake_fetch_candles(self, symbol: str, interval: str = "5m", limit: int = 120):
        attempts["count"] += 1
        raise bybit_module.MarketDataBlockedError("HTTP_403")

    async def fake_binance(symbol: str, interval: str = "5m", limit: int = 120):
        candle = bybit_module.BybitCandle(
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        )
        return bybit_module.BybitKlineSnapshot(
            symbol=symbol,
            interval=interval,
            price=100.5,
            volume=10.0,
            kline_open_time_ms=1,
            kline_close_time_ms=2,
            kline_is_closed=True,
            candle=candle,
            candles=[candle],
            provider_name="binance",
            provider_category="spot",
            provider_endpoint="/api/v3/klines",
        )

    monkeypatch.setattr(bybit_module.BybitClient, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(bybit_module, "_fetch_from_binance", fake_binance)

    async def run() -> None:
        snapshot = await bybit_module.fetch_symbol_klines(
            symbol="BTCUSDT",
            interval="1m",
            failover_threshold=3,
            backoff_base_ms=1,
            backoff_max_ms=2,
        )
        assert snapshot is not None
        assert snapshot.provider_name == "binance"
        assert attempts["count"] == 1

    asyncio.run(run())


def test_failover_not_triggered_on_single_transient_failure(monkeypatch) -> None:
    from app.providers import bybit as bybit_module

    attempts = {"count": 0}

    async def fake_fetch_candles(self, symbol: str, interval: str = "5m", limit: int = 120):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError("transient")
        candle = bybit_module.BybitCandle(
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        )
        return bybit_module.BybitKlineSnapshot(
            symbol=symbol,
            interval=interval,
            price=100.5,
            volume=10.0,
            kline_open_time_ms=1,
            kline_close_time_ms=2,
            kline_is_closed=True,
            candle=candle,
            candles=[candle],
            provider_name="bybit",
            provider_category="linear",
            provider_endpoint="/v5/market/kline",
        )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("fallback should not be called")

    monkeypatch.setattr(bybit_module.BybitClient, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(bybit_module, "_fetch_from_binance", fail_if_called)

    async def run() -> None:
        snapshot = await bybit_module.fetch_symbol_klines(
            symbol="BTCUSDT",
            interval="1m",
            failover_threshold=3,
            backoff_base_ms=1,
            backoff_max_ms=2,
        )
        assert snapshot is not None
        assert snapshot.provider_name == "bybit"
        assert attempts["count"] == 2

    asyncio.run(run())


def test_failover_triggers_after_threshold_for_consistent_failures(monkeypatch) -> None:
    from app.providers import bybit as bybit_module

    attempts = {"count": 0}

    async def fake_fetch_candles(self, symbol: str, interval: str = "5m", limit: int = 120):
        attempts["count"] += 1
        raise ValueError("upstream_timeout")

    async def fake_binance(symbol: str, interval: str = "5m", limit: int = 120):
        candle = bybit_module.BybitCandle(
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        )
        return bybit_module.BybitKlineSnapshot(
            symbol=symbol,
            interval=interval,
            price=100.5,
            volume=10.0,
            kline_open_time_ms=1,
            kline_close_time_ms=2,
            kline_is_closed=True,
            candle=candle,
            candles=[candle],
            provider_name="binance",
            provider_category="spot",
            provider_endpoint="/api/v3/klines",
        )

    monkeypatch.setattr(bybit_module.BybitClient, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(bybit_module, "_fetch_from_binance", fake_binance)

    async def run() -> None:
        snapshot = await bybit_module.fetch_symbol_klines(
            symbol="BTCUSDT",
            interval="1m",
            failover_threshold=3,
            backoff_base_ms=1,
            backoff_max_ms=2,
        )
        assert snapshot is not None
        assert snapshot.provider_name == "binance"
        assert attempts["count"] == 3

    asyncio.run(run())


def test_provider_chain_uses_okx_after_bybit_and_binance_fail(monkeypatch) -> None:
    from app.providers import bybit as bybit_module

    async def bybit_fail(self, symbol: str, interval: str = "5m", limit: int = 120):
        raise bybit_module.MarketDataBlockedError("HTTP_403")

    async def binance_fail(*args, **kwargs):
        raise ValueError("binance_down")

    async def okx_success(*args, **kwargs):
        candle = bybit_module.BybitCandle(
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=5.0,
        )
        return bybit_module.BybitKlineSnapshot(
            symbol="BTCUSDT",
            interval="1m",
            price=100.5,
            volume=5.0,
            kline_open_time_ms=1,
            kline_close_time_ms=2,
            kline_is_closed=True,
            candle=candle,
            candles=[candle],
            provider_name="okx",
            provider_category="swap",
            provider_endpoint="/api/v5/market/history-candles",
        )

    monkeypatch.setattr(bybit_module.BybitClient, "fetch_candles", bybit_fail)
    monkeypatch.setattr(bybit_module, "_fetch_from_binance", binance_fail)
    monkeypatch.setattr(bybit_module, "_fetch_from_okx", okx_success)

    async def run() -> None:
        snapshot = await bybit_module.fetch_symbol_klines(
            symbol="BTCUSDT",
            interval="1m",
            provider="bybit",
            fallback_provider="binance,okx",
            failover_threshold=3,
        )
        assert snapshot is not None
        assert snapshot.provider_name == "okx"

    asyncio.run(run())
