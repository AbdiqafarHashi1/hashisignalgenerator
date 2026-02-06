from __future__ import annotations

from ..config import Settings
from ..models import BiasSignal, MarketSnapshot, Posture, ScoreBreakdown, TradingViewPayload


def _clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def score_signal(
    market: MarketSnapshot,
    bias: BiasSignal,
    payload: TradingViewPayload,
    posture: Posture,
    cfg: Settings,
) -> ScoreBreakdown:
    """
    Score 0..100:
      - Regime:   0..30
      - Bias:     0..30
      - Structure:0..40
      - Bonus:    0..10

    Coach fix:
      - Structure entry-width scoring is now % based (not raw price units),
        so BTC/ETH scoring behaves consistently.
    """
    # --- Regime ---
    regime_quality = 0.0
    if abs(market.funding_rate) <= cfg.funding_elevated_abs:
        regime_quality += 12
    else:
        regime_quality += 6

    if market.leverage_ratio <= cfg.leverage_elevated:
        regime_quality += 10
    else:
        regime_quality += 5

    if abs(market.oi_change_24h) < cfg.oi_spike_pct:
        regime_quality += 8
    else:
        regime_quality += 4

    # --- Bias ---
    bias_quality = bias.confidence * 30.0

    # --- Structure ---
    structure_quality = 0.0
    if payload.setup_type.value == "sweep_reclaim":
        structure_quality += 26
    else:
        structure_quality += 22

    entry_low = float(payload.entry_low)
    entry_high = float(payload.entry_high)
    entry_mid = (entry_low + entry_high) / 2.0 if (entry_low + entry_high) > 0 else 0.0

    # width as % of price
    if entry_mid > 0:
        entry_width_pct = (entry_high - entry_low) / entry_mid
    else:
        entry_width_pct = 1.0

    # tighter entry zones score higher (5m)
    if entry_width_pct <= 0.0015:        # 0.15%
        structure_quality += 14
    elif entry_width_pct <= 0.0030:      # 0.30%
        structure_quality += 10
    elif entry_width_pct <= 0.0060:      # 0.60%
        structure_quality += 6
    else:
        structure_quality += 2

    # --- Bonus ---
    bonus = 0
    if payload.setup_type.value == "sweep_reclaim":
        bonus += 3
    if posture == Posture.OPPORTUNISTIC:
        bonus += 4

    regime_score = _clamp_int(regime_quality, 0, 30)
    bias_score = _clamp_int(bias_quality, 0, 30)
    structure_score = _clamp_int(structure_quality, 0, 40)
    bonus_score = _clamp_int(bonus, 0, 10)

    total = _clamp_int(regime_score + bias_score + structure_score + bonus_score, 0, 100)
    return ScoreBreakdown(
        regime=regime_score,
        bias=bias_score,
        structure=structure_score,
        bonus=bonus_score,
        total=total,
    )
