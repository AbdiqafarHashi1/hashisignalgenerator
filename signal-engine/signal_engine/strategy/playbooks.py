from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from signal_engine.features import FeatureSet
from signal_engine.regime import Bias


class Playbook(str, Enum):
    TREND_PULLBACK = "TREND_PULLBACK"
    RANGE_REVERSION = "RANGE_REVERSION"


@dataclass(frozen=True)
class EntrySignal:
    playbook: Playbook
    side: str
    entry: float
    stop: float
    target_r: float
    reasons: list[str]


def trend_pullback_signal(
    close: float,
    high: float,
    low: float,
    tr: float,
    features: FeatureSet,
    bias: Bias,
    *,
    expansion_mult: float = 1.2,
    no_chase_atr_mult: float = 1.4,
    parabolic_atr_mult: float = 2.5,
) -> EntrySignal | None:
    if bias == Bias.NEUTRAL:
        return None
    side = "long" if bias == Bias.LONG else "short"
    in_zone = min(features.ema_fast, features.ema_slow) <= close <= max(features.ema_fast, features.ema_slow)
    rsi_reset = (features.rsi_prev < 50 <= features.rsi) if side == "long" else (features.rsi_prev > 50 >= features.rsi)
    expansion = tr >= (expansion_mult * features.atr)
    spike = tr > (parabolic_atr_mult * features.atr)
    if spike:
        return None
    dist_from_ema = abs(close - features.ema_fast)
    if dist_from_ema > (no_chase_atr_mult * features.atr):
        return None

    if side == "long":
        pivot = max((p.price for p in features.pivots_high), default=high)
        bos = close > pivot
        stop = min((p.price for p in features.pivots_low), default=low)
    else:
        pivot = min((p.price for p in features.pivots_low), default=low)
        bos = close < pivot
        stop = max((p.price for p in features.pivots_high), default=high)

    if not (in_zone and rsi_reset and bos and expansion):
        return None
    return EntrySignal(playbook=Playbook.TREND_PULLBACK, side=side, entry=close, stop=stop, target_r=2.2, reasons=["pullback_zone", "rsi_reset", "bos", "expansion_confirmed"])


def range_reversion_signal(close: float, features: FeatureSet, *, confidence: float, range_conf_min: float = 0.55) -> EntrySignal | None:
    if confidence < range_conf_min:
        return None
    hi = max((p.price for p in features.pivots_high[-5:]), default=close)
    lo = min((p.price for p in features.pivots_low[-5:]), default=close)
    if close <= lo and features.rsi <= 35:
        return EntrySignal(playbook=Playbook.RANGE_REVERSION, side="long", entry=close, stop=lo * 0.998, target_r=1.4, reasons=["range_low_touch", "rsi_extreme"])
    if close >= hi and features.rsi >= 65:
        return EntrySignal(playbook=Playbook.RANGE_REVERSION, side="short", entry=close, stop=hi * 1.002, target_r=1.4, reasons=["range_high_touch", "rsi_extreme"])
    return None
