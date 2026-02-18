from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.models import BiasSignal, Candle, DecisionRequest, Direction, MarketSnapshot, SetupType, TradingViewPayload
from app.services.scheduler import DecisionScheduler
from app.services.database import Database
from app.state import StateStore
from app.strategy.decision import choose_effective_blocker, decide
from app.strategy.features import compute_bias, compute_features
from app.strategy.regime import classify_regime
from app.strategy.risk import build_stop_plan, build_target_plan


def _req(candles: list[Candle], direction: Direction = Direction.long) -> DecisionRequest:
    return DecisionRequest(
        tradingview=TradingViewPayload(
            symbol="ETHUSDT",
            direction_hint=direction,
            entry_low=candles[-1].close * 0.999,
            entry_high=candles[-1].close * 1.001,
            sl_hint=candles[-1].close * 0.995,
            setup_type=SetupType.break_retest,
            tf_entry="5m",
        ),
        market=MarketSnapshot(funding_rate=0.0, oi_change_24h=0.0, leverage_ratio=1.0, trend_strength=0.8),
        bias=BiasSignal(direction=direction, confidence=0.9),
        candles=candles,
        interval="5m",
        timestamp=datetime.now(timezone.utc),
    )


def _trend_candles(n: int = 260) -> list[Candle]:
    out: list[Candle] = []
    price = 1800.0
    for i in range(n):
        drift = 0.9
        noise = (i % 7 - 3) * 0.08
        o = price
        c = price + drift + noise
        h = max(o, c) + 3.2
        l = min(o, c) - 2.8
        out.append(Candle(open=o, high=h, low=l, close=c, volume=100 + i))
        price = c
    last = out[-1]
    out[-1] = Candle(open=last.open, high=last.high + 3.5, low=last.low, close=last.close + 3.0, volume=last.volume)
    return out


def _range_candles(n: int = 260) -> list[Candle]:
    out: list[Candle] = []
    base = 1800.0
    for i in range(n):
        swing = 3.2 if i % 2 == 0 else -3.2
        o = base + swing
        c = base - swing * 0.9
        h = max(o, c) + 0.6
        l = min(o, c) - 0.6
        out.append(Candle(open=o, high=h, low=l, close=c, volume=120))
    return out


def test_regime_detection_trend_vs_range() -> None:
    cfg = Settings(_env_file=None)
    trend = _trend_candles()
    feat_trend = compute_features(trend, cfg.strategy_swing_lookback)
    assert feat_trend is not None
    bias_trend = compute_bias(trend, cfg.strategy_bias_ema_fast, cfg.strategy_bias_ema_slow)
    regime_trend = classify_regime(feat_trend, bias_trend, cfg.strategy_adx_threshold, cfg.strategy_min_atr_pct, cfg.strategy_max_atr_pct, cfg.strategy_trend_slope_threshold)
    assert regime_trend.regime == "TREND"

    rng = _range_candles()
    feat_rng = compute_features(rng, cfg.strategy_swing_lookback)
    assert feat_rng is not None
    bias_rng = compute_bias(rng, cfg.strategy_bias_ema_fast, cfg.strategy_bias_ema_slow)
    regime_rng = classify_regime(feat_rng, bias_rng, cfg.strategy_adx_threshold, cfg.strategy_min_atr_pct, cfg.strategy_max_atr_pct, cfg.strategy_trend_slope_threshold)
    assert regime_rng.regime in {"RANGE", "LOW_VOL"}


def test_stop_target_calculation() -> None:
    cfg = Settings(_env_file=None)
    stop = build_stop_plan(2000.0, Direction.long, swing_low=1988.0, swing_high=2012.0, atr=8.0, settings=cfg)
    target = build_target_plan(2000.0, Direction.long, stop.stop_distance, cfg)
    assert stop.stop_price < 2000.0
    assert stop.stop_distance > 0
    assert target.target_price > 2000.0
    assert target.target_r == cfg.strategy_target_r


def test_blocker_precedence_single_effective() -> None:
    effective, blockers = choose_effective_blocker(terminal="kill", governor="daily_lock", risk="risk_gate", strategy="no_setup")
    assert effective is not None
    assert effective.layer == "terminal"
    assert effective.code == "kill"
    assert len(blockers) == 4


def test_trades_in_trend_and_skips_in_range() -> None:
    cfg = Settings(_env_file=None)
    trend_plan = decide(_req(_trend_candles()), StateStore(), cfg)
    range_plan = decide(_req(_range_candles()), StateStore(), cfg)
    assert trend_plan.status.value == "TRADE"
    assert range_plan.status.value != "TRADE"
    assert any(r in {"range_regime", "bias_unclear", "no_valid_setup", "no_momentum"} for r in range_plan.rationale)


def test_day_rollover_clears_governor_lock() -> None:
    cfg = Settings(_env_file=None, data_dir="data/test_strategy_pipeline")
    db = Database(cfg)
    db.reset_all()
    db.set_runtime_state("prop.governor", value_text=db.dumps_json({
        "day_key": "2026-01-01",
        "daily_net_r": -1.2,
        "daily_losses": 2,
        "daily_trades": 2,
        "consecutive_losses": 2,
        "locked_until_ts": "2026-01-01T12:00:00+00:00",
    }))
    scheduler = DecisionScheduler(cfg, StateStore(), database=db)
    rolled = scheduler._roll_governor_day(datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc))
    assert rolled is not None
    assert rolled["day_key"] == "2026-01-02"
    assert rolled["daily_losses"] == 0
    assert rolled["daily_trades"] == 0
    assert rolled["locked_until_ts"] is None


def test_risk_rules_enforced_by_governor() -> None:
    cfg = Settings(_env_file=None, data_dir="data/test_strategy_pipeline_governor")
    db = Database(cfg)
    db.reset_all()
    db.set_runtime_state("prop.governor", value_text=db.dumps_json({
        "day_key": "2026-01-02",
        "daily_net_r": 0.0,
        "daily_losses": 0,
        "daily_trades": cfg.prop_max_trades_per_day,
        "consecutive_losses": 0,
        "locked_until_ts": None,
    }))
    scheduler = DecisionScheduler(cfg, StateStore(), database=db)
    blockers = scheduler._governor_blockers(datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc))
    assert any(b.code == "prop_max_trades_per_day" for b in blockers)
