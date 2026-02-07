from app.config import Settings
from app.models import BiasSignal, MarketSnapshot, Posture, TradingViewPayload, SetupType, Direction
from app.strategy.scoring import score_signal


def test_scoring_meets_prop_threshold() -> None:
    cfg = Settings(MODE="prop_cfd", _env_file=None)
    market = MarketSnapshot(
        funding_rate=0.0,
        oi_change_24h=0.01,
        leverage_ratio=1.2,
        trend_strength=0.7,
    )
    bias = BiasSignal(direction=Direction.long, confidence=0.9)
    tv = TradingViewPayload(
        symbol="BTCUSDT",
        direction_hint=Direction.long,
        entry_low=100.0,
        entry_high=101.0,
        sl_hint=98.0,
        setup_type=SetupType.sweep_reclaim,
    )
    score = score_signal(market, bias, tv, Posture.OPPORTUNISTIC, cfg)
    assert score.total >= cfg.min_signal_score
