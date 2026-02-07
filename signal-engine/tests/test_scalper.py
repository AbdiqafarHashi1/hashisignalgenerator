from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.models import (
    BiasSignal,
    Candle,
    DecisionRequest,
    Direction,
    MarketSnapshot,
    SetupType,
    TradingViewPayload,
)
from app.state import StateStore
from app.strategy import scalper as scalper_engine
from app.strategy.decision import decide


def test_engulfing_detection_bullish_and_bearish() -> None:
    cfg = Settings(_env_file=None)
    ema = 100.0
    bullish = [
        Candle(open=101, high=102, low=99.8, close=100.2, volume=100),
        Candle(open=100.4, high=100.9, low=99.7, close=99.9, volume=90),
        Candle(open=99.8, high=101.2, low=99.6, close=101.0, volume=150),
    ]
    bearish = [
        Candle(open=99.5, high=101.5, low=98.9, close=100.6, volume=110),
        Candle(open=100.4, high=101.1, low=100.1, close=100.9, volume=95),
        Candle(open=101.2, high=101.3, low=99.0, close=99.4, volume=160),
    ]
    assert scalper_engine.engulfing_trigger(bullish, Direction.long, ema, cfg)
    assert scalper_engine.engulfing_trigger(bearish, Direction.short, ema, cfg)


def test_trend_filter_blocks_opposite_direction() -> None:
    cfg = Settings(adx_threshold=0, candle_interval="5m", min_signal_score=0, _env_file=None)
    candles = [Candle(open=100, high=101, low=99, close=100, volume=100) for _ in range(48)]
    candles.extend(
        [
            Candle(open=104.5, high=105.0, low=104.2, close=104.8, volume=120),
            Candle(open=105.5, high=105.7, low=103.6, close=104.0, volume=160),
        ]
    )
    trend, ema = scalper_engine.trend_direction(candles, cfg.ema_length)
    assert trend == Direction.long
    assert ema is not None
    assert scalper_engine.engulfing_trigger(candles, Direction.short, ema, cfg)

    request = DecisionRequest(
        tradingview=TradingViewPayload(
            symbol="BTCUSDT",
            direction_hint=Direction.short,
            entry_low=103.0,
            entry_high=105.0,
            sl_hint=106.0,
            setup_type=SetupType.break_retest,
        ),
        market=MarketSnapshot(
            funding_rate=0.01,
            oi_change_24h=0.0,
            leverage_ratio=1.0,
            trend_strength=0.7,
        ),
        bias=BiasSignal(direction=Direction.long, confidence=0.7),
        timestamp=datetime.now(timezone.utc),
        interval="5m",
        candles=candles,
    )
    plan = decide(request, StateStore(), cfg)
    assert plan.status.value == "RISK_OFF"
    assert "no_trigger" in plan.rationale


def test_risk_gate_blocks_with_reason() -> None:
    cfg = Settings(_env_file=None)
    candles = [Candle(open=100, high=101, low=99, close=100, volume=100) for _ in range(60)]
    request = DecisionRequest(
        tradingview=TradingViewPayload(
            symbol="BTCUSDT",
            direction_hint=Direction.long,
            entry_low=99.5,
            entry_high=100.5,
            sl_hint=99.0,
            setup_type=SetupType.break_retest,
        ),
        market=MarketSnapshot(
            funding_rate=0.01,
            oi_change_24h=0.0,
            leverage_ratio=1.0,
            trend_strength=0.7,
        ),
        bias=BiasSignal(direction=Direction.long, confidence=0.7),
        timestamp=datetime.now(timezone.utc),
        interval="5m",
        candles=candles,
    )
    state = StateStore()
    daily_state = state.get_daily_state("BTCUSDT")
    daily_state.pnl_usd = -(cfg.account_size * cfg.max_daily_loss_pct) - 1

    plan = decide(request, state, cfg)
    assert plan.status.value == "RISK_OFF"
    assert "daily_loss_limit" in plan.rationale
