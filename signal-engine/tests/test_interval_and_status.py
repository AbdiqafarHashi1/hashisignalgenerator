from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.models import BiasSignal, Candle, DecisionRequest, Direction, MarketSnapshot, SetupType, TradingViewPayload
from app.state import StateStore
from app.strategy import decision as decision_module
from app.strategy.decision import decide
from app.utils.intervals import interval_to_ms


def test_interval_to_ms_common_values() -> None:
    assert interval_to_ms("1m") == 60_000
    assert interval_to_ms("5m") == 300_000
    assert interval_to_ms("15m") == 900_000


def test_no_valid_setup_maps_to_no_trade(monkeypatch) -> None:
    cfg = Settings(candle_interval="5m", min_signal_score=0, _env_file=None)
    state = StateStore()
    candles = [
        Candle(open=100 + i, high=101 + i, low=99 + i, close=100 + i, volume=100)
        for i in range(60)
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
            funding_rate=0.001,
            oi_change_24h=0.0,
            leverage_ratio=1.0,
            trend_strength=1.0,
        ),
        bias=BiasSignal(direction=Direction.long, confidence=0.9),
        timestamp=datetime.now(timezone.utc),
        interval="5m",
        candles=candles,
    )

    monkeypatch.setattr(decision_module.scalper_engine, "trend_direction", lambda *args, **kwargs: (Direction.long, 100.0))
    monkeypatch.setattr(decision_module.scalper_engine, "pullback_continuation_trigger", lambda *args, **kwargs: False)
    monkeypatch.setattr(decision_module.scalper_engine, "breakout_expansion_trigger", lambda *args, **kwargs: False)

    plan = decide(request, state, cfg)

    assert plan.status.value == "NO_TRADE"
    assert "no_valid_setup" in plan.rationale
