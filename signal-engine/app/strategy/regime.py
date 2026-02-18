from __future__ import annotations

from dataclasses import dataclass

from ..models import Direction
from .features import BiasSnapshot, FeatureSnapshot


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: str
    trend_allowed: bool
    volatility_bucket: str
    bias: Direction
    skip_reason: str | None


def classify_regime(
    features: FeatureSnapshot,
    bias: BiasSnapshot,
    adx_threshold: float,
    min_atr_pct: float,
    max_atr_pct: float,
    slope_threshold: float,
) -> RegimeSnapshot:
    if bias.direction == Direction.none:
        return RegimeSnapshot("RANGE", False, "normal", Direction.none, "bias_unclear")
    if features.atr_pct < min_atr_pct:
        return RegimeSnapshot("LOW_VOL", False, "low", bias.direction, "atr_too_low")
    if features.atr_pct > max_atr_pct:
        return RegimeSnapshot("HIGH_VOL", False, "high", bias.direction, "atr_too_high")

    trend_alignment = (
        (bias.direction == Direction.long and features.ema20 > features.ema50)
        or (bias.direction == Direction.short and features.ema20 < features.ema50)
    )
    trend_energy = features.adx >= adx_threshold or (
        abs(features.ema50_slope) >= slope_threshold and features.atr_pct >= min_atr_pct * 1.15
    )
    if trend_alignment and trend_energy:
        return RegimeSnapshot("TREND", True, "normal", bias.direction, None)
    return RegimeSnapshot("RANGE", False, "normal", bias.direction, "range_regime")
