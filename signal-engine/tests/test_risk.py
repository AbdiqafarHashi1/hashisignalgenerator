from app.config import Settings
from app.models import Posture
from app.strategy.risk import choose_risk_pct, position_size


def test_risk_bump_and_clamp() -> None:
    cfg = Settings(MODE="prop_cfd")
    risk_pct = choose_risk_pct(Posture.OPPORTUNISTIC, 85, cfg)
    assert risk_pct == cfg.max_risk_pct

    risk = position_size(100.0, 99.0, risk_pct, cfg)
    assert risk.position_size_usd <= cfg.account_size * 3.0
