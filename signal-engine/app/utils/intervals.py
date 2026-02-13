from __future__ import annotations


def interval_to_ms(interval: str) -> int:
    raw = (interval or "").strip()
    if not raw:
        return 300_000
    normalized = raw.lower()

    fixed = {
        "d": 86_400_000,
        "1d": 86_400_000,
        "w": 604_800_000,
        "1w": 604_800_000,
    }
    if normalized in fixed:
        return fixed[normalized]

    if normalized.isdigit():
        return int(normalized) * 60_000

    unit = normalized[-1]
    value = normalized[:-1]
    if value.isdigit():
        size = int(value)
        if unit == "m":
            return size * 60_000
        if unit == "h":
            return size * 3_600_000

    return 300_000

