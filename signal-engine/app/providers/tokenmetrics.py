from __future__ import annotations

from .models import BiasSignal, Direction


def parse_bias_signal(payload: dict) -> BiasSignal:
    direction = payload.get("direction", "none")
    if direction not in {"long", "short", "none"}:
        direction = "none"
    payload = {**payload, "direction": direction}
    return BiasSignal(**payload)
