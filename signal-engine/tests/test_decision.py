from datetime import datetime, timezone

from app.config import Settings
from app.models import BiasSignal, DecisionRequest, Direction, MarketSnapshot, SetupType, TradingViewPayload
from app.state import StateStore
from app.strategy.decision import decide


def test_decision_risk_off_on_extreme_market() -> None:
    cfg = Settings()
    state = StateStore()
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
    )
    plan = decide(request, state, cfg)
    assert plan.status.value == "RISK_OFF"


def test_decision_no_trade_low_score() -> None:
    cfg = Settings(MODE="prop_cfd")
    state = StateStore()
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
    )
    plan = decide(request, state, cfg)
    assert plan.status.value in {"NO_TRADE", "RISK_OFF"}
