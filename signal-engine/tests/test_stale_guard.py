from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.services.stale_guard import compute_freshness_status


def test_replay_mode_defaults_to_fresh_against_wall_clock() -> None:
    cfg = Settings(run_mode="replay", replay_enforce_stale=False, market_data_allow_stale=60, _env_file=None)
    old_ts_ms = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    replay_now = datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc)

    freshness = compute_freshness_status(
        settings=cfg,
        interval="5m",
        last_candle_ts_ms=old_ts_ms,
        replay_now_ts=replay_now,
    )

    assert freshness.stale_clock_source == "replay_clock"
    assert freshness.stale_blocked is False


def test_replay_enforced_stale_uses_replay_clock() -> None:
    cfg = Settings(run_mode="replay", replay_enforce_stale=True, market_data_allow_stale=0, _env_file=None)
    candle_ts_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    replay_now = datetime(2024, 1, 1, 0, 20, tzinfo=timezone.utc)

    freshness = compute_freshness_status(
        settings=cfg,
        interval="5m",
        last_candle_ts_ms=candle_ts_ms,
        replay_now_ts=replay_now,
    )

    assert freshness.stale_clock_source == "replay_clock"
    assert freshness.last_candle_age_seconds == 1200
    assert freshness.stale_blocked is True
