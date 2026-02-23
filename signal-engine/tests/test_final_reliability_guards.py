from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.services.dashboard_metrics import build_dashboard_metrics
from app.services.database import Database
from app.services.paper_trader import PaperTrader
from app.state import StateStore


def _setup(tmp_path):
    settings = Settings(
        _env_file=None,
        MODE="paper",
        ENGINE_MODE="paper",
        symbols=["ETHUSDT"],
        data_dir=str(tmp_path),
        account_size=1000,
    )
    db = Database(settings)
    trader = PaperTrader(settings, db)
    state = StateStore()
    return settings, db, trader, state


def _closed_trade(db: Database, *, opened_at: datetime, closed_at: datetime, pnl_net: float, fee: float) -> None:
    trade_id = db.open_trade("ETHUSDT", 100.0, 90.0, 120.0, 1.0, "long", opened_at, trade_mode="paper")
    db.close_trade(trade_id, 110.0, pnl_net, 1.0, closed_at, "tp_close", fees=fee)


def test_equity_identity_within_epsilon(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    db.set_runtime_state("accounting.challenge_start_ts", value_text=(now - timedelta(days=2)).isoformat())
    _closed_trade(db, opened_at=now - timedelta(hours=2), closed_at=now - timedelta(hours=1), pnl_net=8.0, fee=2.0)

    metrics = build_dashboard_metrics(settings, db, trader, state, now)

    lhs = metrics["equity_now"]
    rhs = metrics["equity_start"] + metrics["realized_net"] + metrics["unrealized_net"]
    assert abs(lhs - rhs) < 1e-9


def test_drawdown_moves_down_and_daily_anchor_resets_on_next_day(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    day1 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    db.set_runtime_state("accounting.challenge_start_ts", value_text=(day1 - timedelta(days=1)).isoformat())

    m0 = build_dashboard_metrics(settings, db, trader, state, day1)
    assert m0["daily_dd_pct"] == 0.0
    assert m0["global_dd_pct"] == 0.0

    _closed_trade(db, opened_at=day1 - timedelta(hours=1), closed_at=day1 - timedelta(minutes=30), pnl_net=-100.0, fee=0.0)
    m1 = build_dashboard_metrics(settings, db, trader, state, day1)
    assert m1["equity_now"] == 900.0
    assert m1["daily_dd_pct"] > 0.0
    assert m1["global_dd_pct"] > 0.0

    day2 = datetime(2024, 1, 2, 0, 1, tzinfo=timezone.utc)
    m2 = build_dashboard_metrics(settings, db, trader, state, day2)
    assert m2["daily_dd_pct"] == 0.0
    assert m2["global_dd_pct"] > 0.0


def test_fees_reconcile_with_trade_log(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    db.set_runtime_state("accounting.challenge_start_ts", value_text=(now - timedelta(days=2)).isoformat())

    _closed_trade(db, opened_at=now - timedelta(hours=3), closed_at=now - timedelta(hours=2), pnl_net=10.0, fee=1.25)
    _closed_trade(db, opened_at=now - timedelta(hours=2), closed_at=now - timedelta(hours=1), pnl_net=-5.0, fee=0.75)

    metrics = build_dashboard_metrics(settings, db, trader, state, now)
    trades = [t for t in metrics["trades"] if t.closed_at is not None]
    trade_fees = sum(float(t.fees or 0.0) for t in trades)

    assert abs(metrics["fees_total"] - trade_fees) < 1e-9
    assert abs(metrics["fees_today"] - trade_fees) < 1e-9
