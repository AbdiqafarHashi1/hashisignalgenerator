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


def test_take_profit_is_not_capped_for_long_and_short(tmp_path):
    settings = _settings(tmp_path)
    db = Database(settings)
    trader = PaperTrader(settings, db)

    capped_long_plan = _plan().model_copy(update={"take_profit": 2100.0})
    long_id = trader.maybe_open_trade("ETHUSDT", capped_long_plan, allow_multiple=True)
    assert long_id is not None
    long_trade = db.fetch_open_trades("ETHUSDT")[0]
    assert long_trade.take_profit == 2100.0

    trader.force_close_trades("ETHUSDT", long_trade.entry, reason="manual_close")

    capped_short_plan = _short_plan().model_copy(update={"take_profit": 1900.0})
    short_id = trader.maybe_open_trade("ETHUSDT", capped_short_plan, allow_multiple=True)
    assert short_id is not None
    short_trade = db.fetch_open_trades("ETHUSDT")[0]
    assert short_trade.take_profit == 1900.0


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


def test_candle_cross_exit_realism_long_and_deterministic_conflict(tmp_path):
    settings = _settings(tmp_path)
    settings.reentry_cooldown_minutes = 0
    db = Database(settings)
    trader = PaperTrader(settings, db)

    plan = _plan().model_copy(update={"entry_zone": (2000.0, 2000.0), "stop_loss": 1995.0, "take_profit": 2010.0})
    trade_id = trader.maybe_open_trade("ETHUSDT", plan, allow_multiple=True)
    assert trade_id is not None

    not_hit = trader.evaluate_open_trades("ETHUSDT", 2005.0, candle_high=2009.0, candle_low=1999.0)
    assert not not_hit

    tp_hit = trader.evaluate_open_trades("ETHUSDT", 2005.0, candle_high=2011.0, candle_low=1999.0)
    assert len(tp_hit) == 1
    assert tp_hit[0].result == "tp_close"

    trade_id = trader.maybe_open_trade("ETHUSDT", plan, allow_multiple=True)
    assert trade_id is not None
    sl_hit = trader.evaluate_open_trades("ETHUSDT", 1997.0, candle_high=2000.0, candle_low=1994.0)
    assert len(sl_hit) == 1
    assert sl_hit[0].result == "sl_close"

    trade_id = trader.maybe_open_trade("ETHUSDT", plan, allow_multiple=True)
    assert trade_id is not None
    conflict_hit = trader.evaluate_open_trades("ETHUSDT", 2000.0, candle_high=2011.0, candle_low=1994.0)
    assert len(conflict_hit) == 1
    assert conflict_hit[0].result == "sl_close"


def test_scalp_mode_uses_tighter_reentry_cooldown(tmp_path):
    settings = _settings(tmp_path)
    settings.current_mode = "SCALP"
    settings.scalp_reentry_cooldown_minutes = 30
    db = Database(settings)
    trader = PaperTrader(settings, db)

    trade_id = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert trade_id is not None
    trader.evaluate_open_trades("ETHUSDT", 2011.0, candle_high=2011.0, candle_low=2000.0)

    blocked = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert blocked is None


def test_zero_global_reentry_cooldown_allows_immediate_reentry_in_scalp_mode(tmp_path):
    settings = _settings(tmp_path)
    settings.current_mode = "SCALP"
    settings.reentry_cooldown_minutes = 0
    settings.scalp_reentry_cooldown_minutes = 30
    db = Database(settings)
    trader = PaperTrader(settings, db)

    trade_id = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert trade_id is not None
    trader.evaluate_open_trades("ETHUSDT", 2011.0, candle_high=2011.0, candle_low=2000.0)

    reopened = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert reopened is not None


def test_risk_reduction_moves_stop_from_minus_1r_to_minus_target_r(tmp_path):
    settings = _settings(tmp_path)
    settings.risk_reduction_enabled = True
    settings.risk_reduction_trigger_r = 0.5
    settings.risk_reduction_target_r = 0.6
    db = Database(settings)
    trader = PaperTrader(settings, db)

    trade_id = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert trade_id is not None
    trade_before = db.fetch_open_trades("ETHUSDT")[0]
    assert trade_before.entry == 2000.0
    assert trade_before.stop == 1990.0

    trader.update_mark_price("ETHUSDT", 2005.0)
    moved = trader.adjust_stop_dynamic("ETHUSDT")
    assert moved == 1

    trade_after = db.fetch_open_trades("ETHUSDT")[0]
    assert trade_after.stop == 1994.0

    events = [event for event in db.fetch_events() if event["event_type"] == "risk_reduced"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["trade_id"] == trade_before.id
    assert payload["old_stop"] == 1990.0
    assert payload["new_stop"] == 1994.0
    assert payload["trigger_r"] == 0.5
    assert payload["target_r"] == 0.6


def test_risk_reduction_never_moves_stop_backward(tmp_path):
    settings = _settings(tmp_path)
    settings.risk_reduction_enabled = True
    settings.risk_reduction_trigger_r = 0.5
    settings.risk_reduction_target_r = 1.2
    db = Database(settings)
    trader = PaperTrader(settings, db)

    trade_id = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert trade_id is not None
    original_stop = db.fetch_open_trades("ETHUSDT")[0].stop

    trader.update_mark_price("ETHUSDT", 2005.0)
    moved = trader.adjust_stop_dynamic("ETHUSDT")

    assert moved == 0
    assert db.fetch_open_trades("ETHUSDT")[0].stop == original_stop
    assert not [event for event in db.fetch_events() if event["event_type"] == "risk_reduced"]


def test_risk_reduction_disabled_keeps_stop_unchanged(tmp_path):
    settings = _settings(tmp_path)
    settings.risk_reduction_enabled = False
    settings.risk_reduction_trigger_r = 0.5
    settings.risk_reduction_target_r = 0.6
    db = Database(settings)
    trader = PaperTrader(settings, db)

    trade_id = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert trade_id is not None
    original_stop = db.fetch_open_trades("ETHUSDT")[0].stop

    trader.update_mark_price("ETHUSDT", 2005.0)
    moved = trader.adjust_stop_dynamic("ETHUSDT")

    assert moved == 0
    assert db.fetch_open_trades("ETHUSDT")[0].stop == original_stop


def test_direction_limit_blocks_second_long_but_allows_short(tmp_path):
    settings = _settings(tmp_path)
    settings.max_open_positions_per_direction = 1
    settings.sweet8_enabled = False
    db = Database(settings)
    trader = PaperTrader(settings, db)

    first_long = trader.maybe_open_trade("ETHUSDT", _plan(), allow_multiple=True)
    assert first_long is not None

    second_long = trader.maybe_open_trade("BTCUSDT", _plan(), allow_multiple=True)
    assert second_long is None

    short_open = trader.maybe_open_trade("BTCUSDT", _short_plan(), allow_multiple=True)
    assert short_open is not None
