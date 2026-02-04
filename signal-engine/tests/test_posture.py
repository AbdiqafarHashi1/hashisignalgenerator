from datetime import datetime, timezone

from app.config import Settings
from app.models import MarketSnapshot, Posture
from app.strategy.posture import compute_posture


def test_posture_risk_off_extreme_funding() -> None:
    cfg = Settings()
    market = MarketSnapshot(
        funding_rate=0.05,
        oi_change_24h=0.01,
        leverage_ratio=1.5,
        trend_strength=0.5,
    )
    snapshot = compute_posture("BTCUSDT", market, cfg, now=datetime.now(timezone.utc))
    assert snapshot.posture == Posture.RISK_OFF


def test_posture_opportunistic_clean_trend() -> None:
    cfg = Settings()
    market = MarketSnapshot(
        funding_rate=0.0,
        oi_change_24h=0.01,
        leverage_ratio=1.2,
        trend_strength=0.8,
    )
    snapshot = compute_posture("BTCUSDT", market, cfg, now=datetime.now(timezone.utc))
    assert snapshot.posture == Posture.OPPORTUNISTIC
