from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.services.dashboard_metrics import build_dashboard_metrics
from app.services.database import Database
from app.services.paper_trader import PaperTrader
from app.state import StateStore


def _setup(tmp_path):
    cfg = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{tmp_path / 'acct.db'}", MODE="paper", ENGINE_MODE="paper")
    db = Database(cfg)
    trader = PaperTrader(cfg, db)
    state = StateStore()
    return cfg, db, trader, state


def test_equity_and_drawdown_with_fees(tmp_path):
    cfg, db, trader, state = _setup(tmp_path)
    now = datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc)
    tid = db.open_trade("ETHUSDT", entry=100.0, stop=95.0, take_profit=110.0, size=1.0, side="long", opened_at=now - timedelta(hours=2))
    db.close_trade(tid, exit_price=110.0, pnl_usd=9.0, pnl_r=2.0, fees=1.0, closed_at=now - timedelta(hours=1), result="tp_close")
    tid2 = db.open_trade("ETHUSDT", entry=100.0, stop=95.0, take_profit=110.0, size=1.0, side="long", opened_at=now - timedelta(minutes=30))
    db.close_trade(tid2, exit_price=96.0, pnl_usd=-5.8, pnl_r=-0.8, fees=0.8, closed_at=now - timedelta(minutes=10), result="sl_close")

    metrics = build_dashboard_metrics(cfg, db, trader, state, now)

    assert round(metrics["realized_net_usd"], 2) == 3.2
    assert round(metrics["fees_total"], 2) == 1.8
    assert round(metrics["equity_now"], 2) == round(cfg.account_size + 3.2, 2)
    assert metrics["daily_dd_pct"] >= 0
    assert metrics["global_dd_pct"] >= 0


def test_daily_rollover_resets_day_anchor(tmp_path):
    cfg, db, trader, state = _setup(tmp_path)
    day1 = datetime(2024, 1, 3, 23, 55, tzinfo=timezone.utc)
    day2 = datetime(2024, 1, 4, 0, 5, tzinfo=timezone.utc)

    metrics1 = build_dashboard_metrics(cfg, db, trader, state, day1)
    day1_anchor = metrics1["day_start_equity"]
    metrics2 = build_dashboard_metrics(cfg, db, trader, state, day2)

    assert metrics2["day_start_equity"] == metrics2["equity_now"]
    assert metrics2["day_start_equity"] == day1_anchor
