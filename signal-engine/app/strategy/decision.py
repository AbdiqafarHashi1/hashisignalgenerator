from __future__ import annotations

from datetime import datetime, timezone

from ..config import Settings
from ..models import DecisionRequest, Direction, Posture, Status, TradePlan
from ..state import StateStore
from . import posture as posture_engine
from . import risk as risk_engine
from . import scalper as scalper_engine


def decide(request: DecisionRequest, state: StateStore, cfg: Settings) -> TradePlan:
    """
    Test-compatible signature: decide(request, state, cfg)
    """
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
            signal_score=None,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )
    min_candles = max(cfg.ema_length, cfg.adx_period + 1, cfg.atr_period + cfg.atr_sma_period)
    if len(candles) < min_candles:
        rationale.append("no_candles")
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
            signal_score=None,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot=request.model_dump(),
        )

    if not scalper_engine.momentum_ok(candles, cfg):
        rationale.append("no_momentum")
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

    if not scalper_engine.engulfing_trigger(candles, trend_direction, ema, cfg):
        rationale.append("no_trigger")
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

    levels = scalper_engine.build_trade_levels(candles, trend_direction, cfg)

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

    rationale.append("qualified_trade")
    return TradePlan(
        status=Status.TRADE,
        direction=trend_direction,
        entry_zone=levels.entry_zone,
        stop_loss=levels.stop_loss,
        take_profit=levels.take_profit,
        risk_pct_used=risk_result.risk_pct_used,
        position_size_usd=risk_result.position_size_usd,
        signal_score=None,
        posture=posture_snapshot.posture,
        rationale=rationale,
        raw_input_snapshot=request.model_dump(),
    )
