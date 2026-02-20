from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from signal_engine.features import FeatureSet


class Regime(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    DEAD = "DEAD"


class Bias(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class RegimeDecision:
    regime: Regime
    bias: Bias
    confidence: float
    why: list[str]


def classify_regime(
    features: FeatureSet,
    *,
    adx_threshold: float = 25.0,
    min_atr_pct: float = 0.0015,
    max_atr_pct: float = 0.012,
    slope_threshold: float = 0.0002,
) -> RegimeDecision:
    why: list[str] = []
    htf_bias = Bias.NEUTRAL
    if features.htf.ema50 > features.htf.ema200 and features.htf.slope > 0:
        htf_bias = Bias.LONG
    elif features.htf.ema50 < features.htf.ema200 and features.htf.slope < 0:
        htf_bias = Bias.SHORT

    atr_ok = min_atr_pct <= features.atr_pct <= max_atr_pct
    trend_aligned = (htf_bias == Bias.LONG and features.ema_fast > features.ema_slow) or (htf_bias == Bias.SHORT and features.ema_fast < features.ema_slow)

    if features.atr_pct < min_atr_pct:
        why.append("atr_below_min")
        regime = Regime.DEAD
    elif features.adx >= adx_threshold and atr_ok and trend_aligned and abs(features.ema_slope) >= slope_threshold:
        why.extend(["adx_strong", "ema_aligned", "slope_ok"])
        regime = Regime.TREND
    elif features.adx < adx_threshold and features.atr_pct >= min_atr_pct:
        why.append("adx_sub_trend")
        regime = Regime.RANGE
    else:
        why.append("chop_or_extreme")
        regime = Regime.DEAD

    adx_strength = max(0.0, min(1.0, features.adx / max(adx_threshold, 1.0)))
    slope_strength = max(0.0, min(1.0, abs(features.ema_slope) / max(slope_threshold, 1e-6)))
    atr_centered = 1.0 - min(1.0, abs(features.atr_percentile - 0.5) * 2)
    confidence = max(0.0, min(1.0, (adx_strength * 0.45) + (slope_strength * 0.35) + (atr_centered * 0.2)))

    return RegimeDecision(regime=regime, bias=htf_bias, confidence=confidence, why=why)
