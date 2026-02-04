from __future__ import annotations

from .models import TradingViewPayload


def parse_tradingview(payload: dict) -> TradingViewPayload:
    return TradingViewPayload(**payload)
