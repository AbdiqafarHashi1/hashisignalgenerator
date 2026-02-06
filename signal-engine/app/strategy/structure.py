from __future__ import annotations

from ..models import Direction, TradingViewPayload


def validate_structure(payload: TradingViewPayload) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if payload.entry_low >= payload.entry_high:
        reasons.append("invalid_entry_zone")
    if payload.direction_hint == Direction.long and payload.sl_hint >= payload.entry_low:
        reasons.append("stop_not_below_entry_for_long")
    if payload.direction_hint == Direction.short and payload.sl_hint <= payload.entry_high:
        reasons.append("stop_not_above_entry_for_short")
    if payload.tf_bias.lower() != "4h":
        reasons.append("bias_timeframe_mismatch")
    if payload.tf_entry.lower() != "5m":
        reasons.append("entry_timeframe_mismatch")
    return len(reasons) == 0, reasons
