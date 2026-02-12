from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.providers.bybit import BybitClient, _select_previous_closed_row, _sort_rows_oldest_first


class _StubBybitClient(BybitClient):
    def __init__(self, rows):
        super().__init__(rest_base="https://example.com")
        self._rows = rows
        self.last_params = None

    async def _get(self, *args, **kwargs):  # type: ignore[override]
        if len(args) >= 2:
            self.last_params = args[1]
        else:
            self.last_params = kwargs.get("params")
        return {"retCode": 0, "result": {"list": self._rows}}


def test_select_previous_closed_row_ignores_newest_even_if_forming() -> None:
    rows_newest_first = [
        ["120000", "101", "102", "99", "101.5", "20", "0", "0", "false"],
        ["60000", "100", "101", "98", "100.5", "10", "0", "0", "true"],
    ]
    rows = _sort_rows_oldest_first(rows_newest_first)

    selected, is_closed = _select_previous_closed_row(rows, interval="1m", now_ms=150000)

    assert is_closed is True
    assert int(selected[0]) == 60000


def test_fetch_candles_uses_last_closed_candle_for_snapshot() -> None:
    rows_newest_first = [
        ["120000", "101", "102", "99", "101.5", "20", "0", "0", "false"],
        ["60000", "100", "101", "98", "100.5", "10", "0", "0", "true"],
    ]
    client = _StubBybitClient(rows_newest_first)

    async def run():
        snapshot = await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=2)
        assert snapshot.kline_open_time_ms == 60000
        assert snapshot.kline_close_time_ms == 120000
        assert snapshot.kline_is_closed is True
        assert snapshot.candle.open_time == datetime.fromtimestamp(60, tz=timezone.utc)
        assert len(snapshot.candles) == 1
        assert snapshot.candles[0].open_time == datetime.fromtimestamp(60, tz=timezone.utc)
        assert client.last_params is not None
        assert client.last_params["limit"] == 2

    asyncio.run(run())


def test_fetch_candles_never_requests_limit_one() -> None:
    rows_newest_first = [
        ["120000", "101", "102", "99", "101.5", "20", "0", "0", "false"],
        ["60000", "100", "101", "98", "100.5", "10", "0", "0", "true"],
    ]
    client = _StubBybitClient(rows_newest_first)

    async def run():
        await client.fetch_candles(symbol="BTCUSDT", interval="1m", limit=1)
        assert client.last_params is not None
        assert client.last_params["limit"] == 2

    asyncio.run(run())
