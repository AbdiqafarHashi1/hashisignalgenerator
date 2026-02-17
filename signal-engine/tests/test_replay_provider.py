from __future__ import annotations

import asyncio
from pathlib import Path

from app.providers.replay import ReplayProvider


def test_replay_provider_advances_cursor() -> None:
    replay_root = Path(__file__).parent / "fixtures" / "replay"
    provider = ReplayProvider(str(replay_root))

    async def run() -> None:
        first = await provider.fetch_symbol_klines("ETHUSDT", "3m", limit=5, speed=1.0)
        second = await provider.fetch_symbol_klines("ETHUSDT", "3m", limit=5, speed=1.0)
        assert second.kline_close_time_ms > first.kline_close_time_ms
        assert len(second.candles) == 5

    asyncio.run(run())
