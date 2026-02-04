from app.models import Posture, SetupType
from app.strategy.rr import select_rr


def test_rr_selection() -> None:
    assert select_rr(Posture.NORMAL, SetupType.break_retest) == 2.0
    assert select_rr(Posture.NORMAL, SetupType.sweep_reclaim) == 2.5
    assert select_rr(Posture.OPPORTUNISTIC, SetupType.break_retest) == 3.0
    assert select_rr(Posture.OPPORTUNISTIC, SetupType.sweep_reclaim) == 3.5
