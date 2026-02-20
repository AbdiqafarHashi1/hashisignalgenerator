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


def test_replay_status_progress_fields() -> None:
    replay_root = Path(__file__).parent / "fixtures" / "replay"
    provider = ReplayProvider(str(replay_root))
    status = provider.status("ETHUSDT", "3m")
    assert status["bars_processed"] >= 1
    assert status["total_bars"] >= status["bars_processed"]
    assert status["current_ts"]


def test_replay_provider_history_limit_overrides_fetch_limit() -> None:
    replay_root = Path(__file__).parent / "fixtures" / "replay"
    provider = ReplayProvider(str(replay_root))

    async def run() -> None:
        snapshot = await provider.fetch_symbol_klines("ETHUSDT", "3m", limit=5, speed=1.0, history_limit=12)
        assert len(snapshot.candles) == 6
        for _ in range(20):
            snapshot = await provider.fetch_symbol_klines("ETHUSDT", "3m", limit=5, speed=1.0, history_limit=12)
        assert len(snapshot.candles) == 12

    asyncio.run(run())
