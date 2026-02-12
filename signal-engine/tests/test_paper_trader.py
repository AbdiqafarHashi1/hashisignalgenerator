from __future__ import annotations

from app.config import Settings
from app.models import Direction, Posture, Status, TradePlan
from app.services.database import Database
from app.services.paper_trader import PaperTrader


def _settings(tmp_path):
    return Settings(
        symbols=["ETHUSDT"],
        data_dir=str(tmp_path),
        account_size=1000,
        fee_rate_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        leverage_elevated=2.0,
        sweet8_enabled=False,
    )


def _plan() -> TradePlan:
    return TradePlan(
        status=Status.TRADE,
        direction=Direction.long,
        entry_zone=(2000.0, 2000.0),
        stop_loss=1990.0,
        take_profit=2010.0,
        risk_pct_used=0.01,
        position_size_usd=1000.0,
        signal_score=90,
        posture=Posture.NORMAL,
        rationale=["unit-test"],
        raw_input_snapshot={},
    )


def test_linear_usdt_pnl_is_realistic(tmp_path):
    settings = _settings(tmp_path)
    db = Database(settings)
    trader = PaperTrader(settings, db)

    trade_id = trader.maybe_open_trade("ETHUSDT", _plan())
    assert trade_id is not None

    result = trader.force_close_trades("ETHUSDT", 2010.0, reason="tp")
    assert len(result) == 1
    assert result[0].pnl_usd == 5.0


def test_unrealized_pnl_updates_with_mark_price(tmp_path):
    settings = _settings(tmp_path)
    db = Database(settings)
    trader = PaperTrader(settings, db)

    trade_id = trader.maybe_open_trade("ETHUSDT", _plan())
    assert trade_id is not None
    trader.update_mark_price("ETHUSDT", 2005.0)

    assert trader.total_unrealized_pnl_usd() == 2.5


def test_trade_rejected_when_margin_insufficient(tmp_path):
    settings = _settings(tmp_path)
    settings.account_size = 100.0
    settings.leverage_elevated = 1.0
    db = Database(settings)
    trader = PaperTrader(settings, db)

    rejected = trader.maybe_open_trade("ETHUSDT", _plan())
    assert rejected is None
