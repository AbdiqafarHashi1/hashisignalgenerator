from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import Settings
from app.providers.bybit import replay_validate_dataset
from app.providers.replay import ReplayDatasetError
from app.services.prop_governor import GovernorState, PropRiskGovernor
from app.services.database import Database


def test_settings_reject_unknown_keys() -> None:
    with pytest.raises(Exception):
        Settings(_env_file=None, UNKNOWN_FLAG=True)


def test_replay_start_end_validation(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "replay"
    replay_path = tmp_path / "replay"
    replay_path.mkdir(parents=True, exist_ok=True)
    (replay_path / "ETHUSDT").mkdir(parents=True, exist_ok=True)
    (replay_path / "ETHUSDT" / "3m.csv").write_text((fixture / "ETHUSDT" / "3m.csv").read_text())

    with pytest.raises(ReplayDatasetError):
        replay_validate_dataset(
            str(replay_path),
            "ETHUSDT",
            "3m",
            start_ts="2030-01-01T00:00:00+00:00",
            end_ts="2030-01-02T00:00:00+00:00",
        )


def test_governor_day_rollover_uses_candle_timestamp(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path / 'gov.db'}")
    db = Database(settings)
    gov = PropRiskGovernor(settings=settings, db=db)

    st = GovernorState(risk_pct=settings.prop_risk_base_pct, day_key="2024-01-01", daily_trades=3, daily_losses=2, consecutive_losses=2)
    gov.save(st)

    rolled = gov.load(datetime(2024, 1, 2, 0, 1, tzinfo=timezone.utc))
    gov._roll_day(rolled, datetime(2024, 1, 2, 0, 1, tzinfo=timezone.utc))

    assert rolled.day_key == "2024-01-02"
    assert rolled.daily_trades == 0
    assert rolled.daily_losses == 0


def test_accounting_global_dd_uses_hwm() -> None:
    from signal_engine.accounting import compute_accounting_snapshot

    snap = compute_accounting_snapshot(
        equity_start=100_000,
        realized_pnl=-3_000,
        unrealized_pnl=0,
        fees=0,
        day_start_equity=100_000,
        hwm=110_000,
        trade_close_dates=[],
        profit_target_pct=0.08,
    )
    assert round(snap.global_dd_pct, 6) == round((97_000 - 110_000) / 110_000, 6)
