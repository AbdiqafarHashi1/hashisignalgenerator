from __future__ import annotations

from datetime import datetime, timezone

from ..config import Settings
from ..models import DecisionRequest, Direction, Posture, Status, TradePlan
from ..state import StateStore
from . import posture as posture_engine
from . import risk as risk_engine
from . import scalper as scalper_engine
from . import scoring as scoring_engine


def decide(request: DecisionRequest, state: StateStore, cfg: Settings) -> TradePlan:
    """
    Test-compatible signature: decide(request, state, cfg)
    """
    if cfg.strategy == "baseline":
        return _decide_baseline(request, state, cfg)
    return _decide_scalper(request, state, cfg)


def _decide_baseline(request: DecisionRequest, state: StateStore, cfg: Settings) -> TradePlan:
    return _decide_scalper(request, state, cfg)


def _decide_scalper(request: DecisionRequest, state: StateStore, cfg: Settings) -> TradePlan:
    now = request.timestamp or datetime.now(timezone.utc)
    symbol = request.tradingview.symbol
    rationale: list[str] = []

    # 1) Posture (cached per symbol/day)
    cached_posture = state.get_posture(symbol)
    if cached_posture is None:
        posture_snapshot = posture_engine.compute_posture(symbol, request.market, cfg, now=now)
        state.set_posture(posture_snapshot)
    else:
        posture_snapshot = cached_posture

    if posture_snapshot.posture == Posture.RISK_OFF:
        rationale.append(posture_snapshot.reason)
        return TradePlan(
            status=Status.RISK_OFF,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=None,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    # 2) State limits gate
    limits_ok, limit_status, limit_reasons = state.check_limits(symbol, cfg, now)
    if not limits_ok:
        rationale.extend(limit_reasons)
        return TradePlan(
            status=limit_status,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=None,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    score_breakdown = scoring_engine.score_signal(
        request.market,
        request.bias,
        request.tradingview,
        posture_snapshot.posture,
        cfg,
    )
    signal_score = score_breakdown.total
    if cfg.min_signal_score is not None and signal_score < cfg.min_signal_score:
        rationale.append("score_below_min")
        return TradePlan(
            status=Status.NO_TRADE,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=signal_score,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    candles = request.candles or []
    interval = (request.interval or request.tradingview.tf_entry or "").lower()
    if interval != cfg.candle_interval.lower():
        rationale.append("entry_timeframe_mismatch")
        return TradePlan(
            status=Status.RISK_OFF,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=signal_score,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )
    min_candles = max(cfg.ema_length, cfg.atr_period + cfg.atr_sma_period + 2)
    if len(candles) < min_candles:
        if not cfg.debug_loosen:
            rationale.append("no_candles")
            return TradePlan(
                status=Status.RISK_OFF,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=signal_score,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
            )

    if not scalper_engine.momentum_ok(candles, cfg):
        rationale.append("no_momentum")
        return TradePlan(
            status=Status.RISK_OFF,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=signal_score,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    trend_direction, ema = scalper_engine.trend_direction(candles, cfg.ema_length)
    if trend_direction is None or ema is None:
        rationale.append("no_trend")
        return TradePlan(
            status=Status.RISK_OFF,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=signal_score,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    if (
        cfg.scalp_trend_filter_enabled
        and request.tradingview.direction_hint in {Direction.long, Direction.short}
        and request.tradingview.direction_hint != trend_direction
    ):
        rationale.append("no_trigger")
        return TradePlan(
            status=Status.RISK_OFF,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=signal_score,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    adx_value = scalper_engine._compute_adx(candles, cfg.adx_period)
    if adx_value is None or adx_value < cfg.adx_threshold:
        rationale.append("adx_too_low")
        return TradePlan(
            status=Status.NO_TRADE,
            direction=trend_direction,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=signal_score,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    regime_signal = scalper_engine.generate_regime_signal(candles, cfg)
    if regime_signal is None:
        setup_name: str | None = None
        if scalper_engine.pullback_continuation_trigger(candles, trend_direction, ema, cfg):
            setup_name = "pullback_continuation"
            rationale.append("setup_pullback_continuation")
        elif cfg.disable_breakout_chase:
            rationale.append("breakout_chase_disabled")
            return TradePlan(
                status=Status.NO_TRADE,
                direction=trend_direction,
                entry_zone=None,
                stop_loss=None,
                take_profit=None,
                risk_pct_used=None,
                position_size_usd=None,
                signal_score=signal_score,
                posture=posture_snapshot.posture,
                rationale=rationale,
                raw_input_snapshot=request.model_dump(),
            )
        elif scalper_engine.breakout_expansion_trigger(candles, trend_direction, cfg):
            setup_name = "breakout_expansion"
            rationale.append("setup_breakout_expansion")
        else:
            rationale.append("no_valid_setup")
            return TradePlan(
                status=Status.NO_TRADE,
                direction=trend_direction,
                entry_zone=None,
                stop_loss=None,
                take_profit=None,
                risk_pct_used=None,
                position_size_usd=None,
                signal_score=signal_score,
                posture=posture_snapshot.posture,
                rationale=rationale,
                raw_input_snapshot=request.model_dump(),
            )
        levels = scalper_engine.build_trade_levels(candles, trend_direction, cfg, setup_name)
    else:
        setup_name = f"regime_{regime_signal.regime.lower()}"
        trend_direction = regime_signal.direction
        signal_score = regime_signal.signal_score
        rationale.extend(regime_signal.rationale)
        levels = scalper_engine.TradeLevels(
            entry=regime_signal.entry,
            stop_loss=regime_signal.stop_loss,
            take_profit=regime_signal.take_profit,
            entry_zone=(
                regime_signal.entry * (1 - cfg.regime_entry_buffer_pct),
                regime_signal.entry * (1 + cfg.regime_entry_buffer_pct),
            ),
        )

    expected_profit_after_costs = scalper_engine.expected_pnl_after_costs(levels, trend_direction, cfg)
    if expected_profit_after_costs <= 0:
        rationale.append("cost_model_reject")
        return TradePlan(
            status=Status.NO_TRADE,
            direction=trend_direction,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=signal_score,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    risk_pct = risk_engine.choose_risk_pct(posture_snapshot.posture, 0, cfg)
    risk_result = risk_engine.position_size(levels.entry, levels.stop_loss, risk_pct, cfg)

    # Optional: if min stop gate is enabled and position becomes 0, treat as NO_TRADE
    if risk_result.position_size_usd <= 0:
        rationale.append("risk_rejected_or_zero_position")
        return TradePlan(
            status=Status.RISK_OFF,
            direction=trend_direction,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=None,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    rationale.append("maker_first" if setup_name == "pullback_continuation" else "taker_allowed")
    rationale.append("qualified_trade")
    return TradePlan(
        status=Status.TRADE,
        direction=trend_direction,
        entry_zone=levels.entry_zone,
        stop_loss=levels.stop_loss,
        take_profit=levels.take_profit,
        risk_pct_used=risk_result.risk_pct_used,
        position_size_usd=risk_result.position_size_usd,
        signal_score=signal_score,
        posture=posture_snapshot.posture,
        rationale=rationale,
        raw_input_snapshot=request.model_dump(),
    )
