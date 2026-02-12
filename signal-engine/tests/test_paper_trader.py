from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def _short_plan() -> TradePlan:
    return TradePlan(
        status=Status.TRADE,
        direction=Direction.short,
        entry_zone=(2000.0, 2000.0),
        stop_loss=2010.0,
        take_profit=1900.0,
        risk_pct_used=0.01,
        position_size_usd=1000.0,
        signal_score=90,
        posture=Posture.NORMAL,
        rationale=["unit-test"],
        raw_input_snapshot={},
    )


def test_tp_cap_applies_for_long_and_short(tmp_path):
    settings = _settings(tmp_path)
    db = Database(settings)
    trader = PaperTrader(settings, db)

    capped_long_plan = _plan().model_copy(update={"take_profit": 2100.0})
    long_id = trader.maybe_open_trade("ETHUSDT", capped_long_plan, allow_multiple=True)
    assert long_id is not None
    long_trade = db.fetch_open_trades("ETHUSDT")[0]
    assert long_trade.take_profit == long_trade.entry * 1.02

    trader.force_close_trades("ETHUSDT", long_trade.entry, reason="manual_close")

    capped_short_plan = _short_plan().model_copy(update={"take_profit": 1900.0})
    short_id = trader.maybe_open_trade("ETHUSDT", capped_short_plan, allow_multiple=True)
    assert short_id is not None
    short_trade = db.fetch_open_trades("ETHUSDT")[0]
    assert short_trade.take_profit == short_trade.entry * 0.98


def test_reentry_cooldown_blocks_then_allows(tmp_path):
    settings = _settings(tmp_path)
    settings.reentry_cooldown_minutes = 30
    db = Database(settings)
    trader = PaperTrader(settings, db)

    first_id = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert first_id is not None
    trader.evaluate_open_trades("ETHUSDT", 2010.0)

    blocked = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert blocked is None

    closed_trade = db.fetch_trades(limit=1)[0]
    assert closed_trade.result == "tp_close"
    db.close_trade(
        trade_id=closed_trade.id,
        exit_price=closed_trade.exit or 2010.0,
        pnl_usd=closed_trade.pnl_usd or 0.0,
        pnl_r=closed_trade.pnl_r or 0.0,
        closed_at=datetime.now(timezone.utc) - timedelta(minutes=31),
        result="tp_close",
    )

    allowed = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert allowed is not None
