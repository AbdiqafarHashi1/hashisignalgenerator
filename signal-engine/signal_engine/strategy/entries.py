from __future__ import annotations

from dataclasses import dataclass

from signal_engine.features import FeatureSet
from signal_engine.regime import Regime, RegimeDecision
from signal_engine.strategy.playbooks import EntrySignal, Playbook, range_reversion_signal, trend_pullback_signal


@dataclass(frozen=True)
class EntryDecision:
    signal: EntrySignal | None
    block_reasons: list[str]


def evaluate_entries(
    *,
    regime: RegimeDecision,
    features: FeatureSet,
    close: float,
    high: float,
    low: float,
    equity_state: str,
) -> EntryDecision:
    if regime.regime == Regime.DEAD:
        return EntryDecision(signal=None, block_reasons=["regime_dead"])

    true_range = max(high - low, abs(high - close), abs(low - close))
    if regime.regime == Regime.TREND:
        signal = trend_pullback_signal(close, high, low, true_range, features, regime.bias)
        return EntryDecision(signal=signal, block_reasons=[] if signal else ["trend_pullback_not_ready"])

    if regime.regime == Regime.RANGE:
        if equity_state == "DEFENSIVE":
            return EntryDecision(signal=None, block_reasons=["range_disabled_defensive"])
        signal = range_reversion_signal(close, features, confidence=regime.confidence)
        return EntryDecision(signal=signal, block_reasons=[] if signal else ["range_reversion_not_ready"])

    return EntryDecision(signal=None, block_reasons=["unknown_regime"])
