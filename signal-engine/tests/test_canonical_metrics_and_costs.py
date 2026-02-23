from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.services.cost_model import CryptoCostModel, ForexCostModel
from app.services.database import Database
from app.services.metrics import compute_metrics
from app.services.paper_trader import PaperTrader
from app.state import StateStore


def test_compute_metrics_identity(tmp_path):
    cfg = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{tmp_path / 'm.db'}", MODE="paper", ENGINE_MODE="paper", ACCOUNT_SIZE=1000)
    db = Database(cfg)
    trader = PaperTrader(cfg, db)
    state = StateStore()
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    tid = db.open_trade("BTCUSDT", 100.0, 90.0, 120.0, 1.0, "long", now, trade_mode="paper")
    db.close_trade(tid, 110.0, pnl_usd=8.0, pnl_r=1.0, closed_at=now, result="tp_close", fees=2.0)

    metrics = compute_metrics(cfg, db, trader, state, now)
    assert metrics["equity_now"] == metrics["equity_start"] + metrics["realized_net"] + metrics["unrealized_net"]


def test_cost_models_fee_math():
    crypto = CryptoCostModel(fee_rate_bps=10.0, spread_bps=0.0, slippage_bps=0.0)
    fee = crypto.fees(symbol="BTCUSDT", entry=100.0, exit_price=110.0, qty=1.0)
    assert abs(fee - 0.21) < 1e-9

    forex = ForexCostModel(spread_bps=1.0, commission_bps=2.0)
    fx_fee = forex.fees(symbol="EURUSD", entry=1.1000, exit_price=1.1010, qty=100000)
    assert fx_fee > 0


def test_fee_idempotent_from_db(tmp_path):
    cfg = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{tmp_path / 'id.db'}", MODE="paper", ENGINE_MODE="paper")
    db = Database(cfg)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tid = db.open_trade("BTCUSDT", 100.0, 90.0, 120.0, 1.0, "long", now, trade_mode="paper")
    db.close_trade(tid, 110.0, pnl_usd=8.0, pnl_r=1.0, closed_at=now, result="tp_close", fees=2.0)
    t1 = db.fetch_trades()[0]
    t2 = db.fetch_trades()[0]
    assert t1.fees == t2.fees == 2.0
