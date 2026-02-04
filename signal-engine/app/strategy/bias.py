from __future__ import annotations

from ..config import Settings
from ..models import BiasSignal, Posture, Direction


def bias_allows(bias: BiasSignal, posture: Posture, cfg: Settings) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if bias.direction == Direction.none:
        reasons.append("no_direction")
    if bias.confidence < 0.55:
        reasons.append("low_confidence")
    if posture == Posture.OPPORTUNISTIC and bias.confidence < 0.75:
        reasons.append("opportunistic_requires_high_confidence")
    return len(reasons) == 0, reasons
