from __future__ import annotations

from ..models import Posture, SetupType


def select_rr(
    posture: Posture,
    setup_type: SetupType,
    score: int | None = None,
    cfg: object | None = None,
) -> float | None:
    """
    Backward-compatible:
      - Tests and older code can still call: select_rr(posture, setup_type)
    Enhanced:
      - If score + cfg are provided, OPPORTUNISTIC sweep_reclaim can upgrade to 4.0
        when score is elite (mode-dependent).
    """
    if posture == Posture.RISK_OFF:
        return None

    if posture == Posture.NORMAL:
        if setup_type == SetupType.break_retest:
            return 2.0
        return 2.5

    # OPPORTUNISTIC posture
    if setup_type == SetupType.break_retest:
        return 3.0

    # sweep_reclaim
    # Default remains 3.5 (test-safe)
    rr = 3.5

    # If we have score + cfg, allow elite upgrade to 4.0
    if score is not None and cfg is not None:
        mode = getattr(cfg, "MODE", None)
        elite = 85 if mode == "prop_cfd" else 80
        if score >= elite:
            rr = 4.0

    return rr
