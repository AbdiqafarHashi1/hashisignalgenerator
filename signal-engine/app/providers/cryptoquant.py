from __future__ import annotations

from .models import MarketSnapshot


def parse_market_snapshot(payload: dict) -> MarketSnapshot:
    return MarketSnapshot(**payload)
