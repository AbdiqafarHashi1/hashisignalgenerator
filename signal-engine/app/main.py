from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services.notifier import format_trade_message, send_telegram_message
from .config import get_settings
from .models import BiasSignal, DecisionRequest, Direction, MarketSnapshot, Posture, SetupType, Status, TradeOutcome, TradePlan
from .state import StateStore
from .services.database import Database
from .services.paper_trader import PaperTrader
from .services.stats import compute_stats
from .storage.store import log_event
from .strategy.decision import decide
from .services.scheduler import DecisionScheduler

app = FastAPI(title="signal-engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
state = StateStore()
state.set_symbols(settings.symbols)
database = Database(settings)
paper_trader = PaperTrader(settings, database)

running = False
last_heartbeat_ts: datetime | None = None
last_action: str | None = None
last_correlation_id: str | None = None


def _record_heartbeat() -> None:
    global last_heartbeat_ts
    last_heartbeat_ts = datetime.now(timezone.utc)


def _record_action(action_type: str) -> None:
    global last_action
    last_action = action_type


def _record_correlation_id(correlation_id: str) -> None:
    global last_correlation_id
    last_correlation_id = correlation_id


scheduler = DecisionScheduler(settings, state, database, paper_trader, heartbeat_cb=_record_heartbeat)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/test/telegram")
async def test_telegram() -> dict:
    ok = await send_telegram_message("✅ Telegram test from Swagger", settings)
    return {"status": "sent" if ok else "failed"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/run")
async def run_once(force: bool = Query(False)) -> dict:
    _record_action("run")
    results = await scheduler.run_once(force=force)
    return {"status": "ok", "results": results}


@app.get("/state")
async def latest_state() -> dict:
    symbols = state.get_symbols()
    decisions = {}
    for symbol in symbols:
        daily_state = state.get_daily_state(symbol)
        decisions[symbol] = daily_state.latest_decision
    return {"decisions": decisions}


@app.get("/start")
async def start_scheduler() -> dict[str, str]:
    started = await scheduler.start()
    global running
    running = scheduler.running
    _record_action("start")
    return {"status": "started" if started else "already_running"}


@app.get("/stop")
async def stop_scheduler() -> dict[str, str]:
    stopped = await scheduler.stop()
    global running
    running = scheduler.running
    _record_action("stop")
    return {"status": "stopped" if stopped else "already_stopped"}


@app.get("/stats")
async def stats() -> dict:
    summary = compute_stats(database.fetch_trades())
    return summary.__dict__


@app.get("/trades")
async def trades() -> dict:
    return {"trades": [trade.__dict__ for trade in database.fetch_trades()]}


@app.get("/equity")
async def equity() -> dict:
    summary = compute_stats(database.fetch_trades())
    return {"equity_curve": summary.equity_curve}


@app.get("/positions")
async def positions() -> dict:
    return {"positions": [trade.__dict__ for trade in database.fetch_open_trades()]}


@app.get("/symbols")
async def symbols() -> dict:
    return {"symbols": state.get_symbols()}


@app.post("/symbols")
async def update_symbols(payload: dict) -> dict:
    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
        raise HTTPException(status_code=400, detail="invalid_symbols")
    state.set_symbols(symbols)
    return {"symbols": state.get_symbols()}


@app.get("/paper/reset")
async def reset_paper() -> dict:
    database.reset_trades()
    return {"status": "reset"}


@app.get("/heartbeat")
async def heartbeat():
    return {
        "status": "alive",
        "ts_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/engine/status")
async def engine_status() -> dict[str, Any]:
    global running
    running = scheduler.running
    return {
        "status": "RUNNING" if running else "STOPPED",
        "mode": settings.MODE,
        "last_heartbeat_ts": last_heartbeat_ts.isoformat() if last_heartbeat_ts else None,
        "last_action": last_action,
    }

@app.post("/webhook/tradingview")
async def tradingview_webhook(request: DecisionRequest) -> dict:
    correlation_id = str(uuid4())
    _record_correlation_id(correlation_id)
    log_event(settings, "webhook", request.model_dump(), correlation_id)

    plan = decide(request, state, settings)
    if plan.status == Status.TRADE:
        state.record_trade(request.tradingview.symbol)
    state.set_latest_decision(request.tradingview.symbol, plan)

    log_event(settings, "decision", plan.model_dump(), correlation_id)
    return {"correlation_id": correlation_id, "plan": plan}


class DebugForceSignalRequest(BaseModel):
    symbol: str | None = Field(default=None, min_length=1)
    direction: Direction | None = None
    strategy: Literal["scalper", "baseline"] | None = None
    bypass_soft_gates: bool = False


@app.get("/decision/latest")
async def decision_latest(symbol: str = Query(..., min_length=1)) -> dict:
    daily_state = state.get_daily_state(symbol)
    if daily_state.latest_decision is None:
        return {"symbol": symbol, "decision": None}
    return {"symbol": symbol, "decision": daily_state.latest_decision}


@app.post("/trade_outcome")
async def trade_outcome(outcome: TradeOutcome) -> dict:
    state.record_outcome(outcome.symbol, outcome.pnl_usd, outcome.win, outcome.timestamp)
    return {"status": "recorded"}


@app.get("/state/today")
async def state_today(symbol: str = Query(..., min_length=1)) -> dict:
    daily_state = state.get_daily_state(symbol)
    return {"symbol": symbol, "state": daily_state}


@app.get("/debug/runtime")
async def debug_runtime() -> dict:
    now = datetime.now(timezone.utc)
    symbols = state.get_symbols() or list(settings.symbols)
    storage_health = _storage_health_check(settings.data_dir)
    symbol_data = []
    for symbol in symbols:
        snapshot = scheduler.last_snapshot(symbol)
        latest_candle_ts = snapshot.candle.close_time.isoformat() if snapshot else None
        last_tick_ts = scheduler.last_symbol_tick_time(symbol)
        last_processed_ms = state.get_last_processed_close_time_ms(symbol)
        last_processed_iso = None
        if last_processed_ms is not None:
            last_processed_iso = datetime.fromtimestamp(last_processed_ms / 1000, tz=timezone.utc).isoformat()
        last_decision_ts = state.get_last_decision_ts(symbol)
        latest_decision = state.get_daily_state(symbol).latest_decision
        symbol_data.append(
            {
                "symbol": symbol,
                "last_candle_ts": latest_candle_ts,
                "last_tick_time": last_tick_ts.isoformat() if last_tick_ts else None,
                "candles_fetched_count": scheduler.last_fetch_count(symbol),
                "last_processed_candle_ts": last_processed_iso,
                "last_decision_ts": last_decision_ts.isoformat() if last_decision_ts else None,
                "active_risk_gates": _active_risk_gates(symbol, now),
                "last_decision_status": latest_decision.status if latest_decision else None,
                "gate_reasons_top3": _top_gate_reasons(latest_decision),
            }
        )
    return {
        "settings": {
            **settings.resolved_settings(),
            "telegram_bot_token": _mask_secret(settings.telegram_bot_token),
            "telegram_enabled": settings.telegram_enabled,
        },
        "scheduler": {
            "tick_interval_seconds": scheduler.tick_interval,
            "last_tick_time": scheduler.last_tick_time().isoformat() if scheduler.last_tick_time() else None,
        },
        "symbols": symbol_data,
        "storage": {
            "data_dir": settings.data_dir,
            "last_decision_file": _decision_log_path(settings.data_dir),
            "last_decision_key": last_correlation_id,
            "health_check": storage_health,
        },
    }


@app.post("/debug/force_signal")
async def debug_force_signal(payload: DebugForceSignalRequest) -> dict:
    if not settings.debug_loosen:
        raise HTTPException(status_code=400, detail="debug_loosen_required")
    symbols = state.get_symbols() or list(settings.symbols)
    symbol = payload.symbol or (symbols[0] if symbols else None)
    if symbol is None:
        raise HTTPException(status_code=400, detail="no_symbol_available")
    direction = payload.direction or Direction.long
    snapshot = scheduler.last_snapshot(symbol)
    if snapshot is None:
        raise HTTPException(status_code=400, detail="no_recent_market_data")
    effective_settings = settings.model_copy(
        update={"strategy": payload.strategy or settings.strategy}
    )
    plan = _build_force_plan(
        payload=payload,
        symbol=symbol,
        direction=direction,
        snapshot=snapshot,
        effective_settings=effective_settings,
    )
    state.set_latest_decision(symbol, plan)
    correlation_id = str(uuid4())
    _record_correlation_id(correlation_id)
    log_event(settings, "decision", plan.model_dump(), correlation_id)
    if plan.status == Status.TRADE and settings.telegram_enabled:
        message = format_trade_message(symbol, plan)
        await send_telegram_message(message, settings)
    return {"symbol": symbol, "decision": plan}


def _mask_secret(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}****{value[-4:]}"


def _decision_log_path(data_dir: str) -> str:
    date_key = datetime.now(timezone.utc).date().isoformat()
    return str(Path(data_dir) / "logs" / f"{date_key}.jsonl")


def _storage_health_check(data_dir: str) -> dict[str, str]:
    path = Path(data_dir) / ".healthcheck"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = datetime.now(timezone.utc).isoformat()
        path.write_text(content, encoding="utf-8")
        read_back = path.read_text(encoding="utf-8")
        return {"status": "ok" if read_back == content else "mismatch", "path": str(path)}
    except OSError as exc:
        return {"status": "error", "path": str(path), "error": str(exc)}


def _active_risk_gates(symbol: str, now: datetime) -> list[dict[str, str]]:
    active: list[dict[str, str]] = []
    daily_state = state.get_daily_state(symbol)
    account_loss_limit = settings.account_size * settings.max_daily_loss_pct
    if settings.is_blackout(now):
        active.append({"gate": "blackout", "reason": "news_blackout"})
    if not settings.debug_disable_hard_risk_gates and daily_state.pnl_usd <= -account_loss_limit:
        active.append({"gate": "daily_loss", "reason": "daily_loss_limit"})
    if not settings.debug_disable_hard_risk_gates and settings.max_losses_per_day:
        if daily_state.losses >= settings.max_losses_per_day:
            active.append({"gate": "max_losses", "reason": "max_losses_per_day"})
    if daily_state.trades >= settings.max_trades_per_day:
        active.append({"gate": "max_trades", "reason": "max_trades"})
    if not settings.debug_loosen and daily_state.last_loss_ts is not None:
        minutes_since = (now - daily_state.last_loss_ts).total_seconds() / 60.0
        if minutes_since < settings.cooldown_minutes_after_loss:
            active.append({"gate": "cooldown", "reason": "cooldown"})
    return active


def _top_gate_reasons(plan: TradePlan | None) -> list[str]:
    if plan is None:
        return []
    if plan.status not in {Status.RISK_OFF, Status.NO_TRADE}:
        return []
    return list(plan.rationale)[:3]


def _build_force_plan(
    payload: DebugForceSignalRequest,
    symbol: str,
    direction: Direction,
    snapshot,
    effective_settings,
) -> TradePlan:
    now = datetime.now(timezone.utc)
    hard_gate_reasons = _hard_risk_gate_reasons(symbol, now)
    if hard_gate_reasons:
        return TradePlan(
            status=Status.RISK_OFF,
            direction=direction,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=None,
            posture=Posture.RISK_OFF,
            rationale=hard_gate_reasons,
            raw_input_snapshot={"symbol": symbol, "direction": direction.value},
        )
    request = _build_force_request(symbol, direction, snapshot)
    plan = decide(request, state, effective_settings)
    if payload.bypass_soft_gates and plan.status in {Status.RISK_OFF, Status.NO_TRADE}:
        if not _contains_hard_gate(plan.rationale):
            return _forced_trade_plan(symbol, direction, snapshot, plan, effective_settings)
    return plan


def _build_force_request(symbol: str, direction: Direction, snapshot) -> DecisionRequest:
    candle = snapshot.candle
    entry_low = min(candle.open, candle.close)
    entry_high = max(candle.open, candle.close)
    sl_hint = candle.low if direction == Direction.long else candle.high
    tradingview_payload = {
        "symbol": symbol,
        "direction_hint": direction,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "sl_hint": sl_hint,
        "setup_type": SetupType.break_retest,
        "tf_entry": snapshot.interval,
    }
    market = MarketSnapshot(
        funding_rate=0.01,
        oi_change_24h=0.0,
        leverage_ratio=1.0,
        trend_strength=0.5,
    )
    bias = BiasSignal(direction=direction, confidence=0.6)
    return DecisionRequest(
        tradingview=tradingview_payload,
        market=market,
        bias=bias,
        timestamp=datetime.now(timezone.utc),
        interval=snapshot.interval,
        candles=[
            {
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume,
            }
            for item in snapshot.candles
        ],
    )


def _forced_trade_plan(
    symbol: str,
    direction: Direction,
    snapshot,
    prior_plan: TradePlan,
    effective_settings,
) -> TradePlan:
    entry = snapshot.candle.close
    stop_loss = entry * (0.99 if direction == Direction.long else 1.01)
    take_profit = entry * (1.01 if direction == Direction.long else 0.99)
    return TradePlan(
        status=Status.TRADE,
        direction=direction,
        entry_zone=(entry, entry),
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_pct_used=effective_settings.base_risk_pct,
        position_size_usd=None,
        signal_score=prior_plan.signal_score,
        posture=prior_plan.posture,
        rationale=list(prior_plan.rationale) + ["debug_force_signal", "bypass_soft_gates"],
        raw_input_snapshot={
            "symbol": symbol,
            "direction": direction.value,
            "strategy": effective_settings.strategy,
        },
    )


def _hard_risk_gate_reasons(symbol: str, now: datetime) -> list[str]:
    daily_state = state.get_daily_state(symbol)
    account_loss_limit = settings.account_size * settings.max_daily_loss_pct
    reasons: list[str] = []
    for start_t, end_t in settings.blackout_windows():
        if start_t <= now.time() <= end_t:
            reasons.append("news_blackout")
            break
    if daily_state.pnl_usd <= -account_loss_limit:
        reasons.append("daily_loss_limit")
    if settings.max_losses_per_day and daily_state.losses >= settings.max_losses_per_day:
        reasons.append("max_losses_per_day")
    if daily_state.trades >= settings.max_trades_per_day:
        reasons.append("max_trades")
    if daily_state.last_loss_ts is not None:
        minutes_since = (now - daily_state.last_loss_ts).total_seconds() / 60.0
        if minutes_since < settings.cooldown_minutes_after_loss:
            reasons.append("cooldown")
    return reasons


def _contains_hard_gate(rationale: list[str]) -> bool:
    hard_gates = {
        "news_blackout",
        "daily_loss_limit",
        "max_losses_per_day",
        "max_trades",
        "cooldown",
    }
    return any(reason in hard_gates for reason in rationale)
