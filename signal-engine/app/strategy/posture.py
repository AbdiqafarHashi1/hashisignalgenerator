from __future__ import annotations

from datetime import datetime, timezone

from ..config import Settings
from ..models import MarketSnapshot, Posture, PostureSnapshot


def compute_posture(symbol: str, market: MarketSnapshot, cfg: Settings, now: datetime | None = None) -> PostureSnapshot:
    if now is None:
        now = datetime.now(timezone.utc)
    if cfg.is_blackout(now):
        return PostureSnapshot(
            symbol=symbol,
            date=now.date().isoformat(),
            posture=Posture.RISK_OFF,
            reason="news_blackout",
            computed_at=now,
        )

    funding_extreme = abs(market.funding_rate) >= cfg.funding_extreme_abs
    leverage_extreme = market.leverage_ratio >= cfg.leverage_extreme

    if funding_extreme or leverage_extreme:
        return PostureSnapshot(
            symbol=symbol,
            date=now.date().isoformat(),
            posture=Posture.RISK_OFF,
            reason="extreme_funding_or_leverage",
            computed_at=now,
        )

    clean_funding = abs(market.funding_rate) <= cfg.funding_elevated_abs
    clean_leverage = market.leverage_ratio <= cfg.leverage_elevated
    clean_oi = abs(market.oi_change_24h) < cfg.oi_spike_pct

    if clean_funding and clean_leverage and clean_oi and market.trend_strength >= cfg.trend_strength_min:
        posture = Posture.OPPORTUNISTIC
        reason = "clean_conditions_expanding_trend"
    else:
        posture = Posture.NORMAL
        reason = "standard_conditions"

    return PostureSnapshot(
        symbol=symbol,
        date=now.date().isoformat(),
        posture=posture,
        reason=reason,
        computed_at=now,
    )
