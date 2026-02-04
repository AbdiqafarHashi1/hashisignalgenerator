from __future__ import annotations

from ..config import Settings
from ..models import MarketSnapshot


def regime_allows(market: MarketSnapshot, cfg: Settings) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if abs(market.funding_rate) >= cfg.funding_extreme_abs:
        reasons.append("funding_extreme")
    if market.leverage_ratio >= cfg.leverage_elevated:
        reasons.append("leverage_elevated")
    if abs(market.oi_change_24h) >= cfg.oi_spike_pct and abs(market.funding_rate) >= cfg.funding_elevated_abs:
        reasons.append("oi_spike_with_non_neutral_funding")
    return len(reasons) == 0, reasons
