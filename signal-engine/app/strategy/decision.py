from __future__ import annotations

from datetime import datetime, timezone

from ..config import Settings
from ..models import DecisionRequest, Direction, Posture, Status, TradePlan
from ..state import StateStore
from . import bias as bias_gate
from . import posture as posture_engine
from . import regime as regime_gate
from . import risk as risk_engine
from . import rr as rr_engine
from . import scoring as scoring_engine
from . import structure as structure_gate


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

    # 2) Regime gate
    regime_ok, regime_reasons = regime_gate.regime_allows(request.market, cfg)
    if not regime_ok:
        rationale.extend(regime_reasons)
        return TradePlan(
            status=Status.NO_TRADE,
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

    # 3) Bias gate
    bias_ok, bias_reasons = bias_gate.bias_allows(request.bias, posture_snapshot.posture, cfg)
    if not bias_ok:
        rationale.extend(bias_reasons)
        return TradePlan(
            status=Status.NO_TRADE,
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

    # 4) Structure gate
    structure_ok, structure_reasons = structure_gate.validate_structure(request.tradingview)
    if not structure_ok:
        rationale.extend(structure_reasons)
        return TradePlan(
            status=Status.NO_TRADE,
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

    # 5) State limits gate
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

    # 6) Scoring gate
    score = scoring_engine.score_signal(
        request.market,
        request.bias,
        request.tradingview,
        posture_snapshot.posture,
        cfg,
    )
    if score.total < cfg.min_signal_score:
        rationale.append("score_below_threshold")
        return TradePlan(
            status=Status.NO_TRADE,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=score.total,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot={"request": request.model_dump(), "score": score.model_dump()},
        )

    # 7) Dynamic R:R (supports 2-arg call, enhanced with score+cfg)
    rr = rr_engine.select_rr(posture_snapshot.posture, request.tradingview.setup_type, score.total, cfg)
    if rr is None:
        rationale.append("risk_off_rr")
        return TradePlan(
            status=Status.RISK_OFF,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=score.total,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot={"request": request.model_dump(), "score": score.model_dump()},
        )

    entry_mid = (request.tradingview.entry_low + request.tradingview.entry_high) / 2.0
    if request.tradingview.direction_hint == Direction.long:
        take_profit = entry_mid + (entry_mid - request.tradingview.sl_hint) * rr
    else:
        take_profit = entry_mid - (request.tradingview.sl_hint - entry_mid) * rr

    # 8) Risk sizing
    risk_pct = risk_engine.choose_risk_pct(posture_snapshot.posture, score.total, cfg)
    risk_result = risk_engine.position_size(entry_mid, request.tradingview.sl_hint, risk_pct, cfg)

    # Optional: if min stop gate is enabled and position becomes 0, treat as NO_TRADE
    if risk_result.position_size_usd <= 0:
        rationale.append("risk_rejected_or_zero_position")
        return TradePlan(
            status=Status.NO_TRADE,
            direction=request.tradingview.direction_hint,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=score.total,
            posture=posture_snapshot.posture,
            rationale=rationale,
            raw_input_snapshot={"request": request.model_dump(), "score": score.model_dump()},
        )

    rationale.append("qualified_trade")
    return TradePlan(
        status=Status.TRADE,
        direction=request.tradingview.direction_hint,
        entry_zone=(request.tradingview.entry_low, request.tradingview.entry_high),
        stop_loss=request.tradingview.sl_hint,
        take_profit=take_profit,
        risk_pct_used=risk_result.risk_pct_used,
        position_size_usd=risk_result.position_size_usd,
        signal_score=score.total,
        posture=posture_snapshot.posture,
        rationale=rationale,
        raw_input_snapshot={"request": request.model_dump(), "score": score.model_dump()},
    )
