from __future__ import annotations

from datetime import datetime, timezone


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def trading_day_key(now: datetime) -> str:
    """Canonical trading-day key used across replay/live accounting and governor logic."""
    return _as_aware_utc(now).date().isoformat()


def trading_day_start(now: datetime) -> datetime:
    ts = _as_aware_utc(now)
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)
