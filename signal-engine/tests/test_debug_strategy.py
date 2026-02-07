from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.models import BiasSignal, Candle, DecisionRequest, Direction, MarketSnapshot, SetupType, TradingViewPayload, Status
from app.state import StateStore
from app.strategy.decision import decide


def _bullish_engulfing_candles() -> list[Candle]:
    return [
        Candle(open=100.0, high=101.0, low=99.0, close=100.0, volume=100),
        Candle(open=100.0, high=102.0, low=99.5, close=101.0, volume=110),
        Candle(open=101.0, high=102.0, low=100.0, close=100.5, volume=120),
        Candle(open=100.5, high=101.0, low=99.5, close=100.0, volume=130),
        Candle(open=99.8, high=102.0, low=99.7, close=101.5, volume=140),
    ]


@pytest.mark.parametrize("strategy", ["scalper", "baseline"])
def test_debug_loosen_engulfing_enters(strategy: str) -> None:
    cfg = Settings(
        debug_loosen=True,
        strategy=strategy,
        ema_length=3,
        adx_period=2,
        adx_threshold=0.0,
        atr_period=2,
        atr_sma_period=2,
        ema_pullback_pct=0.1,
        volume_confirm_enabled=False,
        _env_file=None,
    )
    state = StateStore()
    candles = _bullish_engulfing_candles()
    request = DecisionRequest(
        tradingview=TradingViewPayload(
            symbol="BTCUSDT",
            direction_hint=Direction.long,
            entry_low=100.0,
            entry_high=101.0,
            sl_hint=98.0,
            setup_type=SetupType.break_retest,
            tf_entry=cfg.candle_interval,
        ),
        market=MarketSnapshot(
            funding_rate=0.0,
            oi_change_24h=0.0,
            leverage_ratio=1.0,
            trend_strength=0.2,
        ),
        bias=BiasSignal(direction=Direction.long, confidence=0.9),
        timestamp=datetime.now(timezone.utc),
        interval=cfg.candle_interval,
        candles=candles,
    )
    plan = decide(request, state, cfg)
    assert plan.status == Status.TRADE
    assert plan.direction == Direction.long
