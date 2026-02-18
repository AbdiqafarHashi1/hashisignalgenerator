from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import Settings
from ..models import DecisionRequest, Direction, Posture, Status, TradePlan
from ..state import StateStore
from . import manager
from .features import compute_bias, compute_features
from .regime import classify_regime
from .risk import build_stop_plan, build_target_plan, suggest_position_size
from . import scalper as scalper_engine
from .signals import generate_entry_signal


@dataclass(frozen=True)
class PipelineBlocker:
    layer: str
    code: str


def choose_effective_blocker(*, terminal: str | None = None, governor: str | None = None, risk: str | None = None, strategy: str | None = None) -> tuple[PipelineBlocker | None, list[PipelineBlocker]]:
    blockers = [
        PipelineBlocker("terminal", terminal) if terminal else None,
        PipelineBlocker("governor", governor) if governor else None,
        PipelineBlocker("risk", risk) if risk else None,
        PipelineBlocker("strategy", strategy) if strategy else None,
    ]
    filtered = [b for b in blockers if b is not None]
    return (filtered[0] if filtered else None), filtered


def decide(request: DecisionRequest, state: StateStore, cfg: Settings) -> TradePlan:
    now = request.timestamp or datetime.now(timezone.utc)
    symbol = request.tradingview.symbol
    candles = request.candles or []

    allowed, risk_status, risk_reasons = state.check_limits(symbol, cfg, now)
    if not allowed:
        return _skip_plan(request, Posture.RISK_OFF, risk_status, *risk_reasons)

    if abs(request.market.funding_rate) >= cfg.funding_extreme_abs:
        return _skip_plan(request, Posture.RISK_OFF, Status.RISK_OFF, "funding_extreme")
    if len(candles) < 5:
        return _skip_plan(request, Posture.NORMAL, Status.RISK_OFF, "insufficient_candles")

    trend_direction, trend_ema = scalper_engine.trend_direction(candles, cfg.ema_length)
    has_pullback = False
    has_breakout = False
    if trend_direction is not None and request.tradingview.direction_hint != trend_direction:
        return _skip_plan(request, Posture.RISK_OFF, Status.RISK_OFF, "no_trigger")
    if trend_direction is not None and trend_ema is not None:
        has_pullback = scalper_engine.pullback_continuation_trigger(candles, request.tradingview.direction_hint, trend_ema, cfg)
        has_breakout = scalper_engine.breakout_expansion_trigger(candles, request.tradingview.direction_hint, cfg)
        if cfg.debug_loosen and (has_pullback or has_breakout):
            entry = candles[-1].close
            direction = request.tradingview.direction_hint
            stop = request.tradingview.sl_hint
            tp = entry * (1 + cfg.take_profit_pct) if direction == Direction.long else entry * (1 - cfg.take_profit_pct)
            return TradePlan(
                status=Status.TRADE,
                direction=direction,
                entry_zone=(request.tradingview.entry_low, request.tradingview.entry_high),
                stop_loss=stop,
                take_profit=tp,
                risk_pct_used=float(cfg.base_risk_pct or 0.0),
                position_size_usd=float(cfg.account_size or 0.0) * float(cfg.base_risk_pct or 0.0),
                signal_score=int(max(0.0, min(1.0, request.market.trend_strength)) * 100),
                posture=Posture.NORMAL,
                rationale=["debug_loosen", "setup_pullback" if has_pullback else "setup_breakout"],
                raw_input_snapshot={"debug": True},
            )

    signal_floor_score = int(max(0.0, min(1.0, request.market.trend_strength)) * 100)
    if not cfg.debug_loosen and cfg.min_signal_score is not None and signal_floor_score < cfg.min_signal_score:
        return _skip_plan(request, Posture.NORMAL, Status.NO_TRADE, "score_below_min")

    features = compute_features(candles, swing_lookback=cfg.strategy_swing_lookback)
    if features is None:
        return _skip_plan(request, Posture.NORMAL, Status.RISK_OFF, "feature_calc_failed")
    bias = compute_bias(candles, ema_fast=cfg.strategy_bias_ema_fast, ema_slow=cfg.strategy_bias_ema_slow)
    effective_adx_threshold = max(cfg.strategy_adx_threshold, float(getattr(cfg, "adx_threshold", cfg.strategy_adx_threshold)))
    regime = classify_regime(
        features,
        bias,
        adx_threshold=effective_adx_threshold,
        min_atr_pct=cfg.strategy_min_atr_pct,
        max_atr_pct=cfg.strategy_max_atr_pct,
        slope_threshold=cfg.strategy_trend_slope_threshold,
    )
    if not cfg.debug_loosen and features.adx < effective_adx_threshold:
        return _skip_plan(
            request,
            Posture.RISK_OFF,
            Status.RISK_OFF,
            "no_momentum",
            details=_decision_snapshot(now, symbol, features, bias, regime.regime, type("S", (), {"setup_type": None, "score": 0, "skip_reason": "no_momentum"})(), None, None),
        )

    if len(candles) <= 120 and trend_direction is not None and trend_ema is not None and not has_pullback and not has_breakout:
        return _skip_plan(request, Posture.NORMAL, Status.NO_TRADE, "no_valid_setup")

    signal = generate_entry_signal(candles, features, regime)
    if not signal.should_trade or signal.entry is None:
        reason = signal.skip_reason or regime.skip_reason or "no_signal"
        return _skip_plan(
            request,
            Posture.RISK_OFF if reason == "no_momentum" else Posture.NORMAL,
            Status.RISK_OFF if reason == "no_momentum" else Status.NO_TRADE,
            reason,
            details=_decision_snapshot(now, symbol, features, bias, regime.regime, signal, None, None),
        )

    stop_plan = build_stop_plan(
        signal.entry,
        signal.direction,
        swing_low=features.swing_low,
        swing_high=features.swing_high,
        atr=features.atr,
        settings=cfg,
    )
    target_plan = build_target_plan(signal.entry, signal.direction, stop_plan.stop_distance, cfg)
    sizing = suggest_position_size(cfg.account_size, signal.entry, stop_plan.stop_distance, cfg)
    if sizing.size_usd <= 0:
        return _skip_plan(
            request,
            Posture.RISK_OFF,
            Status.RISK_OFF,
            "risk_rejected_or_zero_position",
            details=_decision_snapshot(now, symbol, features, bias, regime.regime, signal, stop_plan, target_plan),
        )

    mgmt = manager.default_management_plan()
    rationale = ["qualified_trade", f"setup_{signal.setup_type}"]
    return TradePlan(
        status=Status.TRADE,
        direction=signal.direction,
        entry_zone=(signal.entry * (1 - cfg.regime_entry_buffer_pct), signal.entry * (1 + cfg.regime_entry_buffer_pct)),
        stop_loss=stop_plan.stop_price,
        take_profit=target_plan.target_price,
        risk_pct_used=sizing.risk_pct,
        position_size_usd=sizing.size_usd,
        signal_score=signal.score,
        posture=Posture.NORMAL,
        rationale=rationale,
        raw_input_snapshot=_decision_snapshot(now, symbol, features, bias, regime.regime, signal, stop_plan, target_plan)
        | {
            "management": {
                "be_enabled": mgmt.break_even_enabled,
                "be_trigger_r": mgmt.break_even_r,
                "be_min_seconds_open": mgmt.break_even_min_seconds,
                "partial_enabled": mgmt.partial_enabled,
                "partial_r": mgmt.partial_r,
                "partial_close_pct": mgmt.partial_close_pct,
                "trail_enabled": mgmt.trail_enabled,
                "trail_start_r": mgmt.trail_start_r,
                "trail_atr_mult": mgmt.trail_atr_mult,
            }
        },
    )


def _decision_snapshot(now: datetime, symbol: str, features, bias, regime: str, signal, stop_plan, target_plan) -> dict[str, object]:
    return {
        "timestamp": now.isoformat(),
        "symbol": symbol,
        "regime": regime,
        "bias_1h": bias.direction.value,
        "bias_strength": bias.strength,
        "atr_pct": features.atr_pct,
        "adx": features.adx,
        "ema20_slope": features.ema20_slope,
        "ema50_slope": features.ema50_slope,
        "momentum": {
            "rsi": features.rsi,
            "rsi_prev": features.rsi_prev,
            "macd_hist": features.macd_hist,
            "macd_hist_prev": features.macd_hist_prev,
        },
        "setup_type": signal.setup_type,
        "entry_strength": signal.score,
        "stop_type": (stop_plan.stop_type if stop_plan else None),
        "stop_distance": (stop_plan.stop_distance if stop_plan else None),
        "r_target": (target_plan.target_r if target_plan else None),
        "skip_reason": signal.skip_reason,
    }


def _skip_plan(
    request: DecisionRequest,
    posture: Posture,
    status: Status,
    reason: str,
    details: dict[str, object] | None = None,
) -> TradePlan:
    return TradePlan(
        status=status,
        direction=request.tradingview.direction_hint if request.tradingview.direction_hint != Direction.none else Direction.long,
        entry_zone=None,
        stop_loss=None,
        take_profit=None,
        risk_pct_used=None,
        position_size_usd=None,
        signal_score=None,
        posture=posture,
        rationale=[reason],
        raw_input_snapshot=details or request.model_dump(),
    )
