from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.models import BiasSignal, Candle, DecisionRequest, Direction, MarketSnapshot, SetupType, TradingViewPayload
from app.services.database import Database
from app.services.paper_trader import PaperTrader
from app.services.scheduler import DecisionScheduler
from app.state import StateStore
from app.strategy import scalper
from app.strategy.decision import decide


def _candle(o: float, h: float, l: float, c: float, v: float = 100.0) -> Candle:
    return Candle(open=o, high=h, low=l, close=c, volume=v)


def test_trend_regime_generates_long_entry() -> None:
    cfg = Settings(_env_file=None, trend_strength_min=0.1, scalp_trend_slope_min=0.00001, min_signal_score_trend=0, pullback_atr_mult=5.0, scalp_atr_pct_max=0.05)
    candles = [_candle(100 + i * 0.08, 100 + i * 0.1, 99.9 + i * 0.08, 100 + i * 0.09) for i in range(80)]
    candles[-2] = _candle(106.2, 106.3, 105.8, 106.0)
    candles[-1] = _candle(106.0, 106.9, 105.9, 106.7)
    sig = scalper.generate_regime_signal(candles, cfg)
    assert sig is not None
    assert sig.regime == "TRENDING"
    assert sig.direction == Direction.long


def test_range_regime_generates_mean_reversion_entry() -> None:
    cfg = Settings(_env_file=None, trend_strength_min=0.9, scalp_trend_slope_min=0.01, min_signal_score_range=0, dev_atr_mult=0.2, scalp_atr_pct_max=0.05)
    candles = [_candle(100, 100.6, 99.4, 100 + ((-1) ** i) * 0.2, 120) for i in range(90)]
    candles[-1] = _candle(102.8, 103.0, 102.5, 102.9, 150)
    sig = scalper.generate_regime_signal(candles, cfg)
    assert sig is not None
    assert sig.regime == "RANGING"
    assert sig.direction == Direction.short


def test_break_even_moves_stop() -> None:
    cfg = Settings(_env_file=None, data_dir="data/test_strategy_v2")
    db = Database(cfg)
    db.reset_all()
    trader = PaperTrader(cfg, db)
    trade_id = db.open_trade("BTCUSDT", entry=100.0, stop=99.0, take_profit=102.0, size=1.0, side="long", opened_at=datetime.now(timezone.utc))
    assert trade_id > 0
    trader.update_mark_price("BTCUSDT", 100.8)
    moved = trader.move_stop_to_breakeven("BTCUSDT", trigger_r=0.6)
    assert moved == 1
    trade = db.fetch_open_trades("BTCUSDT")[0]
    assert trade.stop == 100.0


def test_time_stop_exits_after_max_hold_minutes() -> None:
    cfg = Settings(_env_file=None, data_dir="data/test_strategy_v2", max_hold_minutes=1)
    db = Database(cfg)
    db.reset_all()
    trader = PaperTrader(cfg, db)
    opened_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.open_trade("ETHUSDT", entry=100.0, stop=99.0, take_profit=101.5, size=1.0, side="long", opened_at=opened_at)
    scheduler = DecisionScheduler(cfg, StateStore(), database=db, paper_trader=trader)
    scheduler._close_time_stop_trades("ETHUSDT", price=100.2, now=datetime.now(timezone.utc))
    closed = [t for t in db.fetch_trades() if t.closed_at is not None]
    assert closed
    assert closed[0].result == "time_stop_close"


def test_no_trade_score_below_min_reason() -> None:
    cfg = Settings(_env_file=None, min_signal_score=100, candle_interval="5m")
    candles = [_candle(100, 101, 99.8, 100.4, 120) for _ in range(90)]
    request = DecisionRequest(
        tradingview=TradingViewPayload(
            symbol="BTCUSDT",
            direction_hint=Direction.long,
            entry_low=100.0,
            entry_high=100.5,
            sl_hint=99.0,
            setup_type=SetupType.break_retest,
            tf_entry="5m",
        ),
        market=MarketSnapshot(funding_rate=0.0, oi_change_24h=0.0, leverage_ratio=1.0, trend_strength=0.8),
        bias=BiasSignal(direction=Direction.long, confidence=1.0),
        candles=candles,
        interval="5m",
        timestamp=datetime.now(timezone.utc),
    )
    plan = decide(request, StateStore(), cfg)
    assert plan.status.value == "NO_TRADE"
    assert "score_below_min" in plan.rationale
