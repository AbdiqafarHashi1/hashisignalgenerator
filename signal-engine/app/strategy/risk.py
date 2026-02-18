from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..models import Direction, Posture, RiskResult


@dataclass(frozen=True)
class StopPlan:
    stop_price: float
    stop_distance: float
    stop_type: str


@dataclass(frozen=True)
class TargetPlan:
    target_price: float
    partial_r: float
    target_r: float


@dataclass(frozen=True)
class RiskProposal:
    risk_pct: float
    size_usd: float


def build_stop_plan(
    entry: float,
    direction: Direction,
    swing_low: float,
    swing_high: float,
    atr: float,
    settings: Settings,
) -> StopPlan:
    atr_distance = atr * settings.strategy_atr_stop_mult
    if direction == Direction.long:
        swing_stop = swing_low * (1 - settings.strategy_swing_stop_buffer_bps / 10_000)
        atr_stop = entry - atr_distance
        stop = min(swing_stop, atr_stop)
        dist = entry - stop
    else:
        swing_stop = swing_high * (1 + settings.strategy_swing_stop_buffer_bps / 10_000)
        atr_stop = entry + atr_distance
        stop = max(swing_stop, atr_stop)
        dist = stop - entry

    min_distance = entry * settings.strategy_min_stop_pct
    max_distance = entry * settings.strategy_max_stop_pct
    clamped = min(max(dist, min_distance), max_distance)
    if direction == Direction.long:
        stop = entry - clamped
    else:
        stop = entry + clamped

    stop_type = "blended"
    if abs(stop - swing_stop) < 1e-9:
        stop_type = "swing"
    elif abs(stop - atr_stop) < 1e-9:
        stop_type = "atr"
    return StopPlan(stop, clamped, stop_type)


def build_target_plan(entry: float, direction: Direction, stop_distance: float, settings: Settings) -> TargetPlan:
    target_r = settings.strategy_target_r
    partial_r = settings.strategy_partial_r
    move = stop_distance * target_r
    target = entry + move if direction == Direction.long else entry - move
    return TargetPlan(target, partial_r=partial_r, target_r=target_r)


def suggest_position_size(account_balance: float, entry: float, stop_distance: float, settings: Settings) -> RiskProposal:
    risk_pct = min(max(settings.prop_risk_base_pct, settings.prop_risk_min_pct), settings.prop_risk_max_pct)
    risk_usd = max(0.0, account_balance * risk_pct)
    if stop_distance <= 0 or entry <= 0:
        return RiskProposal(risk_pct=risk_pct, size_usd=0.0)
    units = risk_usd / stop_distance
    return RiskProposal(risk_pct=risk_pct, size_usd=max(0.0, units * entry))


# Backward-compatible API for tests/legacy imports.
def choose_risk_pct(posture: Posture, signal_score: int, settings: Settings) -> float:
    """Compatibility wrapper for legacy risk selection API.

    Canonical sizing logic now lives in suggest_position_size/prop governor flow.
    This helper preserves the old import surface without altering strategy behavior.
    """
    base = float(settings.base_risk_pct or 0.0)
    max_risk = float(settings.max_risk_pct or base)
    if posture == Posture.RISK_OFF:
        return 0.0
    if posture == Posture.OPPORTUNISTIC and signal_score >= 80:
        return max_risk
    return base


# Backward-compatible API for tests/legacy imports.
def position_size(entry: float, stop_loss: float, risk_pct: float, settings: Settings) -> RiskResult:
    stop_distance = abs(entry - stop_loss)
    stop_distance_pct = (stop_distance / entry) if entry > 0 else 0.0
    risk_usd = max(0.0, float(settings.account_size or 0.0) * float(risk_pct))
    if entry <= 0 or stop_distance <= 0:
        position_size_usd = 0.0
    else:
        units = risk_usd / stop_distance
        raw_notional = max(0.0, units * entry)
        max_notional = max(0.0, float(settings.account_size or 0.0) * settings.max_notional_account_multiplier)
        position_size_usd = min(raw_notional, max_notional)

    return RiskResult(
        risk_pct_used=float(risk_pct),
        stop_distance_pct=stop_distance_pct,
        position_size_usd=position_size_usd,
    )
