from __future__ import annotations

import asyncio
from pathlib import Path

from app.providers.replay import ReplayProvider


def test_replay_provider_advances_cursor(tmp_path: Path) -> None:
    provider = ReplayProvider(str(tmp_path / "replay"))

    async def run() -> None:
        first = await provider.fetch_symbol_klines("BTCUSDT", "1m", limit=5, speed=1.0)
        second = await provider.fetch_symbol_klines("BTCUSDT", "1m", limit=5, speed=1.0)
        assert second.kline_close_time_ms > first.kline_close_time_ms
        assert len(second.candles) == 5

    asyncio.run(run())
