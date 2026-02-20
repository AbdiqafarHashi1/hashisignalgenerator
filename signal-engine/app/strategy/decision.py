from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


from ..config import Settings
from ..models import DecisionRequest, Direction, Posture, Status, TradePlan
from ..state import StateStore
from signal_engine.features import compute_features
from signal_engine.regime import Bias, Regime, classify_regime
from signal_engine.strategy.entries import evaluate_entries


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


def _equity_state(state: StateStore, symbol: str, cfg: Settings, now: datetime) -> tuple[str, float, str]:
    daily = state.get_daily_state(symbol, now)
    _, global_dd = state.global_drawdown(float(cfg.account_size or 0.0))
    if global_dd >= 0.04 or daily.consecutive_losses >= 2:
        return "DEFENSIVE", 0.6, "dd_or_loss_streak"
    if daily.pnl_usd > (float(cfg.account_size or 0.0) * 0.01) and global_dd < 0.02:
        return "HOT", 1.2, "recent_performance_strong"
    return "NEUTRAL", 1.0, "balanced"


def decide(request: DecisionRequest, state: StateStore, cfg: Settings) -> TradePlan:
    now = request.timestamp or datetime.now(timezone.utc)
    symbol = request.tradingview.symbol
    candles = request.candles or []

    allowed, risk_status, risk_reasons = state.check_limits(symbol, cfg, now)
    if not allowed:
        return _skip_plan(request, Posture.RISK_OFF, risk_status, *risk_reasons)
    if len(candles) < 220:
        return _skip_plan(request, Posture.NORMAL, Status.NO_TRADE, "insufficient_candles")

    rows = [c.model_dump() for c in candles]
    feats = compute_features(rows, ema_fast=cfg.strategy_bias_ema_fast, ema_slow=cfg.strategy_bias_ema_slow)
    regime = classify_regime(
        feats,
        adx_threshold=cfg.strategy_adx_threshold,
        min_atr_pct=cfg.strategy_min_atr_pct,
        max_atr_pct=cfg.strategy_max_atr_pct,
        slope_threshold=cfg.strategy_trend_slope_threshold,
    )
    eq_state, risk_mult, eq_reason = _equity_state(state, symbol, cfg, now)
    candle = candles[-1]
    entry = evaluate_entries(regime=regime, features=feats, close=candle.close, high=candle.high, low=candle.low, equity_state=eq_state)
    if entry.signal is None:
        reason = entry.block_reasons[0] if entry.block_reasons else "no_entry"
        return _skip_plan(
            request,
            Posture.RISK_OFF if regime.regime == Regime.DEAD else Posture.NORMAL,
            Status.NO_TRADE,
            reason,
            {
                "regime": regime.regime.value,
                "bias": regime.bias.value,
                "confidence": regime.confidence,
                "regime_why": regime.why,
                "entry_block_reasons": entry.block_reasons,
                "equity_state": eq_state,
                "equity_state_reason": eq_reason,
            },
        )

    sig = entry.signal
    stop = sig.stop
    stop_pct = abs(sig.entry - stop) / sig.entry if sig.entry > 0 else 0.0
    if stop_pct < cfg.strategy_min_stop_pct:
        stop = sig.entry * (1 - cfg.strategy_min_stop_pct) if sig.side == "long" else sig.entry * (1 + cfg.strategy_min_stop_pct)
    elif stop_pct > cfg.strategy_max_stop_pct:
        stop = sig.entry * (1 - cfg.strategy_max_stop_pct) if sig.side == "long" else sig.entry * (1 + cfg.strategy_max_stop_pct)

    risk = abs(sig.entry - stop)
    target_r = 3.0 if regime.confidence >= 0.75 and sig.playbook.value == "TREND_PULLBACK" else sig.target_r
    tp = sig.entry + (risk * target_r) if sig.side == "long" else sig.entry - (risk * target_r)
    direction = Direction.long if sig.side == "long" else Direction.short
    risk_pct = min(float(cfg.max_risk_pct or cfg.base_risk_pct or 0.0), float((cfg.base_risk_pct or 0.0) * risk_mult))
    size = float(cfg.account_size or 0.0) * risk_pct

    return TradePlan(
        status=Status.TRADE,
        direction=direction,
        entry_zone=(sig.entry, sig.entry),
        stop_loss=stop,
        take_profit=tp,
        risk_pct_used=risk_pct,
        position_size_usd=size,
        signal_score=int(100 * regime.confidence),
        posture=Posture.NORMAL,
        rationale=[sig.playbook.value, *sig.reasons],
        raw_input_snapshot={
            "regime": regime.regime.value,
            "bias": regime.bias.value,
            "confidence": regime.confidence,
            "regime_why": regime.why,
            "entry_reasons": sig.reasons,
            "equity_state": eq_state,
            "equity_state_reason": eq_reason,
            "risk_multiplier": risk_mult,
        },
    )


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
