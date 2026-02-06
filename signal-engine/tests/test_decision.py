from datetime import datetime, timezone

from app.config import Settings
from app.models import BiasSignal, Candle, DecisionRequest, Direction, MarketSnapshot, SetupType, TradingViewPayload
from app.state import StateStore
from app.strategy.decision import decide


def test_decision_risk_off_on_extreme_market() -> None:
    cfg = Settings()
    state = StateStore()
    candles = [
        Candle(open=100, high=101, low=99, close=100, volume=100),
        Candle(open=100, high=101, low=99, close=100, volume=100),
    ]
    request = DecisionRequest(
        tradingview=TradingViewPayload(
            symbol="BTCUSDT",
            direction_hint=Direction.long,
            entry_low=100.0,
            entry_high=101.0,
            sl_hint=98.0,
            setup_type=SetupType.break_retest,
        ),
        market=MarketSnapshot(
            funding_rate=0.05,
            oi_change_24h=0.01,
            leverage_ratio=1.0,
            trend_strength=0.2,
        ),
        bias=BiasSignal(direction=Direction.long, confidence=0.9),
        timestamp=datetime.now(timezone.utc),
        interval="5m",
        candles=candles,
    )
    plan = decide(request, state, cfg)
    assert plan.status.value == "RISK_OFF"


def test_decision_no_trade_no_momentum() -> None:
    cfg = Settings(MODE="prop_cfd", adx_threshold=80)
    state = StateStore()
    candles = [
        Candle(open=100, high=101, low=99, close=100, volume=100)
        for _ in range(55)
    ]
    candles.extend(
        [
            Candle(open=101, high=103, low=100, close=102, volume=120),
            Candle(open=102, high=104, low=101, close=103, volume=130),
            Candle(open=103, high=105, low=102, close=104, volume=140),
            Candle(open=104, high=106, low=103, close=105, volume=150),
            Candle(open=105, high=107, low=104, close=106, volume=160),
        ]
    )
    request = DecisionRequest(
        tradingview=TradingViewPayload(
            symbol="ETHUSDT",
            direction_hint=Direction.long,
            entry_low=100.0,
            entry_high=101.0,
            sl_hint=99.5,
            setup_type=SetupType.break_retest,
        ),
        market=MarketSnapshot(
            funding_rate=0.02,
            oi_change_24h=0.2,
            leverage_ratio=2.4,
            trend_strength=0.2,
        ),
        bias=BiasSignal(direction=Direction.long, confidence=0.55),
        timestamp=datetime.now(timezone.utc),
        interval="5m",
        candles=candles,
    )
    plan = decide(request, state, cfg)
    assert plan.status.value == "RISK_OFF"
    assert "no_momentum" in plan.rationale
