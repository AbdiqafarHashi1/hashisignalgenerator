from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from app.providers.replay import ReplayDatasetError, ReplayProvider


def test_replay_provider_advances_cursor() -> None:
    replay_root = Path(__file__).parent / "fixtures" / "replay"
    provider = ReplayProvider(str(replay_root))

    async def run() -> None:
        first = await provider.fetch_symbol_klines("ETHUSDT", "3m", limit=5, speed=1.0)
        second = await provider.fetch_symbol_klines("ETHUSDT", "3m", limit=5, speed=1.0)
        assert second.kline_close_time_ms > first.kline_close_time_ms
        assert len(second.candles) >= 2

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
        assert len(snapshot.candles) >= 2
        for _ in range(20):
            snapshot = await provider.fetch_symbol_klines("ETHUSDT", "3m", limit=5, speed=1.0, history_limit=12)
        assert len(snapshot.candles) == 12

    asyncio.run(run())


def test_replay_resume_state_ignored_when_disabled(tmp_path: Path) -> None:
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
    provider = ReplayProvider(str(replay_root), resume_enabled=False)

    async def run() -> None:
        snap = await provider.fetch_symbol_klines("BTCUSDT", "5m", limit=2, speed=1.0, start_ts="2024-01-01T00:05:00+00:00")
        assert snap.candle.close_time.isoformat() == "2024-01-01T00:10:00+00:00"
        status = provider.status("BTCUSDT", "5m")
        assert status["start_resolution"]["why_start_ts_overridden"] == "start_ts_used"
        assert status["start_resolution"]["trade_start_ts"] == "2024-01-01T00:05:00+00:00"
        assert status["start_resolution"]["selection_logic"] == "REPLAY_START_TS >= first candle > csv_first_ts_fallback"

    asyncio.run(run())


def test_replay_resume_state_overrides_start_ts_when_enabled(tmp_path: Path) -> None:
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
    provider = ReplayProvider(str(replay_root), resume_enabled=True)

    async def run() -> None:
        snap = await provider.fetch_symbol_klines("BTCUSDT", "5m", limit=2, speed=1.0, start_ts="2024-01-01T00:05:00+00:00")
        assert snap.candle.close_time.isoformat() >= "2024-01-01T00:10:00+00:00"
        status = provider.status("BTCUSDT", "5m")
        assert status["start_resolution"]["why_start_ts_overridden"] == "resume_enabled"
        assert status["start_resolution"]["trade_start_ts"] == "2024-01-01T00:05:00+00:00"
        assert status["start_resolution"]["selection_logic"] == "resume cursor/state > REPLAY_START_TS >= first candle > csv_first_ts_fallback"

    asyncio.run(run())


def test_replay_start_ts_anchors_to_first_candle_at_or_after_requested_ts(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    symbol_dir = replay_root / "BTCUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "5m.csv").write_text(
        "timestamp,open,high,low,close,volume,close_time\n"
        "2024-06-01T00:00:00+00:00,1,1,1,1,1,2024-06-01T00:05:00+00:00\n"
        "2024-06-01T00:05:00+00:00,1,1,1,1,1,2024-06-01T00:10:00+00:00\n"
        "2024-06-01T00:10:00+00:00,1,1,1,1,1,2024-06-01T00:15:00+00:00\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(str(replay_root), resume_enabled=False)

    async def run() -> None:
        snap = await provider.fetch_symbol_klines("BTCUSDT", "5m", limit=2, speed=1.0, start_ts="2024-06-01T00:06:00Z")
        assert snap.cursor_index == 1
        assert snap.candle.close_time.isoformat() == "2024-06-01T00:10:00+00:00"

    asyncio.run(run())


def test_replay_start_ts_before_first_candle_falls_back_to_csv_start(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    symbol_dir = replay_root / "BTCUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "5m.csv").write_text(
        "timestamp,open,high,low,close,volume,close_time\n"
        "2024-06-01T00:00:00+00:00,1,1,1,1,1,2024-06-01T00:05:00+00:00\n"
        "2024-06-01T00:05:00+00:00,1,1,1,1,1,2024-06-01T00:10:00+00:00\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(str(replay_root), resume_enabled=False)

    async def run() -> None:
        await provider.fetch_symbol_klines("BTCUSDT", "5m", limit=2, speed=1.0, start_ts="2024-05-31T12:00:00Z")
        status = provider.status("BTCUSDT", "5m")
        resolution = status["start_resolution"]
        assert resolution["trade_start_ts"] == "2024-06-01T00:05:00+00:00"
        assert resolution["why_start_ts_overridden"] == "csv_first_ts_fallback"

    asyncio.run(run())


def test_replay_start_ts_after_last_candle_raises_clear_error(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    symbol_dir = replay_root / "BTCUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "5m.csv").write_text(
        "timestamp,open,high,low,close,volume,close_time\n"
        "2024-06-01T00:00:00+00:00,1,1,1,1,1,2024-06-01T00:05:00+00:00\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(str(replay_root), resume_enabled=False)

    async def run() -> None:
        with pytest.raises(ReplayDatasetError, match="replay_start_ts_out_of_range"):
            await provider.fetch_symbol_klines("BTCUSDT", "5m", limit=2, speed=1.0, start_ts="2024-06-01T01:00:00Z")

    asyncio.run(run())


def test_replay_warmup_reporting_keeps_trade_start_anchor(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    symbol_dir = replay_root / "BTCUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "5m.csv").write_text(
        "timestamp,open,high,low,close,volume,close_time\n"
        "2024-01-01T00:00:00+00:00,1,1,1,1,1,2024-01-01T00:05:00+00:00\n"
        "2024-01-01T00:05:00+00:00,1,1,1,1,1,2024-01-01T00:10:00+00:00\n"
        "2024-01-01T00:10:00+00:00,1,1,1,1,1,2024-01-01T00:15:00+00:00\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(str(replay_root), resume_enabled=False)

    async def run() -> None:
        await provider.fetch_symbol_klines("BTCUSDT", "5m", limit=2, speed=1.0, start_ts="2024-01-01T00:10:00+00:00", history_limit=4)
        status = provider.status("BTCUSDT", "5m")
        resolution = status["start_resolution"]
        assert resolution["trade_start_ts"] == "2024-01-01T00:10:00+00:00"
        assert resolution["warmup_ready_at_ts"] == "2024-01-01T00:15:00+00:00"
        assert resolution["history_preload_start_ts"] == "2024-01-01T00:05:00+00:00"

    asyncio.run(run())


def test_replay_provider_loads_forex_pair_when_dataset_exists(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    symbol_dir = replay_root / "EURUSD"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "5m.csv").write_text(
        "timestamp,open,high,low,close,volume,close_time\n"
        "2024-06-01T00:00:00+00:00,1.1000,1.1002,1.0998,1.1001,10,2024-06-01T00:05:00+00:00\n"
        "2024-06-01T00:05:00+00:00,1.1001,1.1003,1.0999,1.1002,10,2024-06-01T00:10:00+00:00\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(str(replay_root), resume_enabled=False)

    async def run() -> None:
        snap = await provider.fetch_symbol_klines("EURUSD", "5m", limit=2, speed=1.0)
        assert snap.candle.close > 0
        status = provider.status("EURUSD", "5m")
        assert status["total_bars"] >= 2

    asyncio.run(run())
