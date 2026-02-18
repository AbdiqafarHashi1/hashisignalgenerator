from __future__ import annotations

from dataclasses import dataclass

from ..models import Candle, Direction
from .features import FeatureSnapshot
from .regime import RegimeSnapshot


@dataclass(frozen=True)
class SignalDecision:
    should_trade: bool
    direction: Direction
    setup_type: str | None
    entry: float | None
    score: int
    skip_reason: str | None


def _momentum_ok(features: FeatureSnapshot, direction: Direction) -> bool:
    if direction == Direction.long:
        return (features.rsi > 50 and features.rsi >= features.rsi_prev) or (
            features.macd_hist > 0 and features.macd_hist >= features.macd_hist_prev
        )
    if direction == Direction.short:
        return (features.rsi < 50 and features.rsi <= features.rsi_prev) or (
            features.macd_hist < 0 and features.macd_hist <= features.macd_hist_prev
        )
    return False


def generate_entry_signal(candles: list[Candle], features: FeatureSnapshot, regime: RegimeSnapshot) -> SignalDecision:
    if not regime.trend_allowed:
        return SignalDecision(False, regime.bias, None, None, 0, regime.skip_reason or "regime_blocked")
    if len(candles) < 3:
        return SignalDecision(False, regime.bias, None, None, 0, "insufficient_candles")
    cur = candles[-1]
    prev = candles[-2]

    if not _momentum_ok(features, regime.bias):
        return SignalDecision(False, regime.bias, None, None, 0, "no_momentum")

    setup_type: str | None = None
    if regime.bias == Direction.long:
        breakout = cur.close > features.swing_high
        pullback = prev.low <= features.ema20 and cur.close > features.ema20 and cur.close > prev.close
        if breakout:
            setup_type = "breakout"
        elif pullback:
            setup_type = "pullback"
    elif regime.bias == Direction.short:
        breakout = cur.close < features.swing_low
        pullback = prev.high >= features.ema20 and cur.close < features.ema20 and cur.close < prev.close
        if breakout:
            setup_type = "breakout"
        elif pullback:
            setup_type = "pullback"

    if not setup_type:
        return SignalDecision(False, regime.bias, None, None, 0, "no_valid_setup")

    score = 50
    score += 20 if setup_type == "breakout" else 15
    score += 10 if abs(features.ema50_slope) > 0.00025 else 0
    score += 10 if features.adx >= 25 else 0
    return SignalDecision(True, regime.bias, setup_type, cur.close, min(100, score), None)
