from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.models import BiasSignal, Candle, DecisionRequest, Direction, MarketSnapshot, SetupType, TradingViewPayload
from app.state import StateStore
from app.strategy.decision import decide, evaluate_warmup_status


def _candles(count: int) -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for _ in range(count):
        candles.append(Candle(open=price, high=price + 1, low=price - 1, close=price + 0.2, volume=100.0))
        price += 0.1
    return candles


def _request(candles: list[Candle]) -> DecisionRequest:
    return DecisionRequest(
        tradingview=TradingViewPayload(
            symbol="BTCUSDT",
            direction_hint=Direction.long,
            entry_low=100.0,
            entry_high=101.0,
            sl_hint=99.0,
            setup_type=SetupType.break_retest,
        ),
        market=MarketSnapshot(funding_rate=0.0, oi_change_24h=0.0, leverage_ratio=1.0, trend_strength=0.3),
        bias=BiasSignal(direction=Direction.long, confidence=0.6),
        timestamp=datetime.now(timezone.utc),
        interval="5m",
        candles=candles,
    )


def test_warmup_ready_after_min_bars_without_htf() -> None:
    cfg = Settings(
        _env_file=None,
        HTF_BIAS_ENABLED=False,
        SCALP_REGIME_ENABLED=False,
        STRATEGY_BIAS_EMA_SLOW=80,
        WARMUP_MIN_BARS_5M=250,
    )
    status = evaluate_warmup_status(_candles(250), cfg)
    assert status.ready is True
    assert status.bars_5m_have >= status.bars_5m_need


def test_warmup_requires_htf_when_enabled() -> None:
    cfg = Settings(
        _env_file=None,
        HTF_BIAS_ENABLED=True,
        HTF_INTERVAL="1h",
        WARMUP_MIN_BARS_5M=250,
        WARMUP_MIN_BARS_1H=220,
    )
    not_ready = evaluate_warmup_status(_candles((220 * 12) - 1), cfg)
    assert not_ready.ready is False
    assert "htf_ema" in not_ready.missing_components

    ready = evaluate_warmup_status(_candles(220 * 12), cfg)
    assert ready.ready is True
    assert ready.bars_htf_have >= ready.bars_htf_need


def test_decision_does_not_return_insufficient_once_warmup_satisfied() -> None:
    cfg = Settings(
        _env_file=None,
        HTF_BIAS_ENABLED=False,
        SCALP_REGIME_ENABLED=False,
        STRATEGY_BIAS_EMA_SLOW=80,
        WARMUP_MIN_BARS_5M=250,
    )
    plan = decide(_request(_candles(260)), StateStore(), cfg)
    assert "insufficient_candles" not in plan.rationale
