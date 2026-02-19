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
        symbols=["ETHUSDT"],
        data_dir=str(tmp_path),
        account_size=1000,
        fee_rate_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
    )
    db = Database(settings)
    trader = PaperTrader(settings, db)
    state = StateStore()
    return settings, db, trader, state


def _closed_trade(db: Database, opened_at: datetime, closed_at: datetime, pnl_net: float, fee: float) -> None:
    trade_id = db.open_trade("ETHUSDT", 100.0, 90.0, 120.0, 1.0, "long", opened_at, trade_mode="paper")
    db.close_trade(trade_id, 110.0, pnl_net, 1.0, closed_at, "tp_close", fees=fee)


def test_equity_reconciliation_delta_is_zero(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    _closed_trade(db, now - timedelta(hours=2), now - timedelta(hours=1), pnl_net=8.0, fee=2.0)

    metrics = build_dashboard_metrics(settings, db, trader, state, now)

    assert metrics["equity_reconcile_delta"] == 0.0
    assert metrics["equity_now"] == metrics["equity_start"] + metrics["pnl_realized_total"] + metrics["pnl_unrealized"] - metrics["fees_total"]


def test_daily_rollover_resets_daily_but_not_global(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    day1 = datetime(2024, 1, 1, 23, 59, tzinfo=timezone.utc)
    _closed_trade(db, day1 - timedelta(hours=2), day1 - timedelta(minutes=1), pnl_net=10.0, fee=0.0)

    m1 = build_dashboard_metrics(settings, db, trader, state, day1)
    assert m1["pnl_realized_today"] == 10.0

    day2 = datetime(2024, 1, 2, 0, 1, tzinfo=timezone.utc)
    m2 = build_dashboard_metrics(settings, db, trader, state, day2)

    assert m2["pnl_realized_today"] == 0.0
    assert m2["trades_today"] == 0
    assert m2["equity_high_watermark"] >= m1["equity_high_watermark"]


def test_high_watermark_and_global_drawdown(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    now = datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc)

    db.set_runtime_state("accounting.equity_high_watermark", value_number=1200.0)
    db.set_runtime_state("accounting.max_global_dd_abs", value_number=0.0)
    db.set_runtime_state("accounting.max_global_dd_pct", value_number=0.0)

    m = build_dashboard_metrics(settings, db, trader, state, now)
    assert m["global_dd_abs"] == 200.0
    assert m["global_dd_pct"] == (200.0 / 1200.0)
    assert m["max_global_dd_abs"] == 200.0


def test_replay_determinism_metrics_independent_of_polling_pace(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    t0 = datetime(2024, 1, 4, 12, 0, tzinfo=timezone.utc)
    _closed_trade(db, t0 - timedelta(hours=3), t0 - timedelta(hours=2), pnl_net=5.0, fee=1.0)
    _closed_trade(db, t0 - timedelta(hours=2), t0 - timedelta(hours=1), pnl_net=-3.0, fee=1.0)

    fast = build_dashboard_metrics(settings, db, trader, state, t0)
    _ = build_dashboard_metrics(settings, db, trader, state, t0 - timedelta(minutes=30))
    slow = build_dashboard_metrics(settings, db, trader, state, t0)

    assert fast["equity_now"] == slow["equity_now"]
    assert fast["global_dd_pct"] == slow["global_dd_pct"]
    assert fast["pnl_realized_total"] == slow["pnl_realized_total"]
