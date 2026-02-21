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


def test_replay_resume_state_overrides_start_ts(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    symbol_dir = replay_root / "BTCUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    csv_path = symbol_dir / "5m.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume,close_time\n"
        "2024-01-01T00:00:00+00:00,1,1,1,1,1,2024-01-01T00:05:00+00:00\n"
        "2024-01-01T00:05:00+00:00,1,1,1,1,1,2024-01-01T00:10:00+00:00\n"
        "2024-01-01T00:10:00+00:00,1,1,1,1,1,2024-01-01T00:15:00+00:00\n",
        encoding="utf-8",
    )
    (replay_root / "replay_runtime_state.json").write_text(
        '{"BTCUSDT:5m":{"last_processed_ts":"2024-01-01T00:10:00+00:00"}}',
        encoding="utf-8",
    )
    provider = ReplayProvider(str(replay_root))

    async def run() -> None:
        snap = await provider.fetch_symbol_klines("BTCUSDT", "5m", limit=2, speed=1.0, start_ts="2024-01-01T00:00:00+00:00")
        assert snap.candle.close_time.isoformat() >= "2024-01-01T00:10:00+00:00"
        status = provider.status("BTCUSDT", "5m")
        assert status["start_resolution"]["why_start_ts_overridden"] in {"resume_state_override", "warmup_floor_enforced"}

    asyncio.run(run())
