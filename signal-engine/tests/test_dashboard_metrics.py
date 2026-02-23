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


def test_equity_reconciliation_uses_net_realized_identity(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    db.set_runtime_state("accounting.challenge_start_ts", value_text=(now - timedelta(days=1)).isoformat())
    _closed_trade(db, now - timedelta(hours=2), now - timedelta(hours=1), pnl_net=8.0, fee=2.0)

    metrics = build_dashboard_metrics(settings, db, trader, state, now)

    assert metrics["reconciliation_delta_usd"] == 0.0
    assert metrics["equity_now_usd"] == metrics["equity_start_usd"] + metrics["realized_net_usd"] + metrics["unrealized_usd"]
    assert metrics["realized_net_usd"] == metrics["realized_gross_usd"] - metrics["fees_total_usd"]


def test_challenge_window_filters_out_pre_reset_closed_trades(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    challenge_start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    db.set_runtime_state("accounting.challenge_start_ts", value_text=challenge_start.isoformat())

    _closed_trade(db, now - timedelta(days=2), now - timedelta(days=2, hours=-1), pnl_net=50.0, fee=1.0)
    _closed_trade(db, now - timedelta(hours=2), now - timedelta(hours=1), pnl_net=10.0, fee=1.0)

    metrics = build_dashboard_metrics(settings, db, trader, state, now)

    assert metrics["realized_net_usd"] == 10.0
    assert metrics["fees_total_usd"] == 1.0
    assert len([t for t in metrics["trades"] if t.closed_at is not None]) == 1


def test_daily_rollover_resets_daily_but_not_global(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    day1 = datetime(2024, 1, 1, 23, 59, tzinfo=timezone.utc)
    db.set_runtime_state("accounting.challenge_start_ts", value_text=(day1 - timedelta(days=1)).isoformat())
    _closed_trade(db, day1 - timedelta(hours=2), day1 - timedelta(minutes=1), pnl_net=10.0, fee=0.0)

    m1 = build_dashboard_metrics(settings, db, trader, state, day1)
    assert m1["pnl_realized_today"] == 10.0

    day2 = datetime(2024, 1, 2, 0, 1, tzinfo=timezone.utc)
    m2 = build_dashboard_metrics(settings, db, trader, state, day2)

    assert m2["pnl_realized_today"] == 0.0
    assert m2["trades_today"] == 0
    assert m2["equity_high_watermark"] >= m1["equity_high_watermark"]


def test_drawdown_percentages_and_peak_tracking_are_live(tmp_path):
    settings, db, trader, state = _setup(tmp_path)
    now = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    db.set_runtime_state("accounting.challenge_start_ts", value_text=(now - timedelta(days=1)).isoformat())

    m0 = build_dashboard_metrics(settings, db, trader, state, now)
    assert m0["daily_drawdown_pct"] == 0.0
    assert m0["global_drawdown_pct"] == 0.0

    # New equity peak (+100), then draw down to 1050 from 1100 peak.
    _closed_trade(db, now - timedelta(hours=2), now - timedelta(hours=1, minutes=30), pnl_net=100.0, fee=0.0)
    m_peak = build_dashboard_metrics(settings, db, trader, state, now)
    assert m_peak["equity_high_watermark"] == 1100.0

    _closed_trade(db, now - timedelta(hours=1), now - timedelta(minutes=30), pnl_net=-50.0, fee=0.0)
    m_dd = build_dashboard_metrics(settings, db, trader, state, now)

    assert m_dd["equity_now_usd"] == 1050.0
    assert m_dd["daily_drawdown_usd"] == 0.0
    assert m_dd["global_drawdown_usd"] == 50.0
    assert m_dd["daily_drawdown_pct"] == 0.0
    assert m_dd["global_drawdown_pct"] == 4.545454545454546
    # Legacy ratio fields remain for backward compatibility.
    assert m_dd["daily_dd_pct"] == 0.0
    assert m_dd["global_dd_pct"] == 0.045454545454545456
