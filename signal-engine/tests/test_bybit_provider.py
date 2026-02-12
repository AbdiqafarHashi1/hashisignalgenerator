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
