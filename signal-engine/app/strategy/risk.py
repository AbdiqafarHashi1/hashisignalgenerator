from __future__ import annotations

from ..config import Settings
from ..models import Posture, RiskResult


def choose_risk_pct(posture: Posture, score: int, cfg: Settings) -> float:
    """
    Base risk always applies.
    Risk bump only when posture is OPPORTUNISTIC and score is high.
    """
    risk_pct = cfg.base_risk_pct
    if posture == Posture.OPPORTUNISTIC:
        if cfg.MODE == "prop_cfd" and score >= 80:
            risk_pct = cfg.max_risk_pct
        if cfg.MODE == "personal_crypto" and score >= 75:
            risk_pct = cfg.max_risk_pct
    return min(risk_pct, cfg.max_risk_pct)


def position_size(entry_price: float, stop_loss: float, risk_pct: float, cfg: Settings) -> RiskResult:
    """
    Test-compatible signature.

    Coach upgrade (non-breaking):
      - optional min stop distance gate (only active if cfg.min_stop_distance_pct exists and > 0)
      - still returns RiskResult (position may be 0 when rejected)
    """
    stop_distance = abs(entry_price - stop_loss)
    if entry_price <= 0:
        stop_distance_pct = 0.0
    else:
        stop_distance_pct = stop_distance / entry_price

    if stop_distance_pct <= 0:
        return RiskResult(risk_pct_used=risk_pct, stop_distance_pct=0.0, position_size_usd=0.0)

    # Optional safety: reject ultra-tight stops only if configured
    min_stop = float(getattr(cfg, "min_stop_distance_pct", 0.0) or 0.0)
    if min_stop > 0 and stop_distance_pct < min_stop:
        return RiskResult(risk_pct_used=risk_pct, stop_distance_pct=stop_distance_pct, position_size_usd=0.0)

    position_usd = (cfg.account_size * risk_pct) / stop_distance_pct
    max_notional = cfg.account_size * 3.0
    position_usd = min(position_usd, max_notional)

    return RiskResult(
        risk_pct_used=risk_pct,
        stop_distance_pct=stop_distance_pct,
        position_size_usd=position_usd,
    )


# Backward-compat helper for tests that import this name directly
def position_size_usd(account_size: float, risk_pct: float, entry_price: float, stop_loss: float) -> float:
    stop_distance = abs(entry_price - stop_loss)
    if entry_price <= 0 or stop_distance <= 0:
        return 0.0
    stop_distance_pct = stop_distance / entry_price
    if stop_distance_pct <= 0:
        return 0.0
    position = (account_size * risk_pct) / stop_distance_pct
    return min(position, account_size * 3.0)
