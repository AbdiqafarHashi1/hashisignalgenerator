from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services.notifier import format_trade_message, send_telegram_message
from .config import Settings, get_settings
from .models import BiasSignal, DecisionRequest, Direction, MarketSnapshot, Posture, SetupType, Status, TradeOutcome, TradePlan
from .state import StateStore
from .services.database import Database
from .services.paper_trader import PaperTrader
from .services.stats import compute_stats
from .storage.store import log_event
from .strategy.decision import decide
from .services.scheduler import DecisionScheduler

logger = logging.getLogger(__name__)

settings = None
state = None
database = None
paper_trader = None
scheduler = None

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


def _initialize_engine(app: FastAPI) -> None:
    global settings, state, database, paper_trader, scheduler
    global running, last_heartbeat_ts, last_action, last_correlation_id

    settings = get_settings()
    state = StateStore()
    state.set_symbols(settings.symbols)
    database = Database(settings)
    paper_trader = PaperTrader(settings, database)
    scheduler = DecisionScheduler(
        settings,
        state,
        database,
        paper_trader,
        interval_seconds=settings.tick_interval_seconds,
        heartbeat_cb=_record_heartbeat,
    )
    running = False
    last_heartbeat_ts = None
    last_action = None
    last_correlation_id = None

    app.state.settings = settings
    app.state.state_store = state
    app.state.database = database
    app.state.paper_trader = paper_trader
    app.state.scheduler = scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    _initialize_engine(app)
    if settings is not None:
        logger.info(
            "engine_ready symbols=%s mode=%s engine_mode=%s",
            ",".join(settings.symbols),
            settings.MODE,
            settings.engine_mode,
        )
        logger.info(
            "engine_config tick_interval_seconds=%s force_trade_mode=%s smoke_test_force_trade=%s "
            "force_trade_every_seconds=%s force_trade_cooldown_seconds=%s force_trade_auto_close_seconds=%s "
            "force_trade_random_direction=%s",
            settings.tick_interval_seconds,
            settings.force_trade_mode,
            settings.smoke_test_force_trade,
            settings.force_trade_every_seconds,
            settings.force_trade_cooldown_seconds,
            settings.force_trade_auto_close_seconds,
            settings.force_trade_random_direction,
        )
    try:
        yield
    finally:
        if scheduler is not None and scheduler.running:
            await scheduler.stop()
            logger.info("engine_stopped reason=shutdown")


app = FastAPI(title="signal-engine", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _require_scheduler() -> DecisionScheduler:
    if scheduler is None:
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return scheduler


def _require_state() -> StateStore:
    if state is None:
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return state


def _require_settings() -> Settings:
    if settings is None:
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return settings


def _require_database() -> Database:
    if database is None:
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return database


def _require_paper_trader() -> PaperTrader:
    if paper_trader is None:
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return paper_trader


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
    scheduler = _require_scheduler()
    results = await scheduler.run_once(force=force)
    return {"status": "ok", "results": results}


@app.get("/state")
async def latest_state() -> dict:
    state_store = _require_state()
    symbols = state_store.get_symbols()
    decisions = {}
    for symbol in symbols:
        daily_state = state_store.get_daily_state(symbol)
        decisions[symbol] = daily_state.latest_decision
    return {"decisions": decisions}


@app.get("/start")
async def start_scheduler() -> dict[str, str]:
    scheduler = _require_scheduler()
    started = await scheduler.start()
    global running
    running = scheduler.running
    _record_action("start")
    logger.info("engine_start status=%s", "started" if started else "already_running")
    return {"status": "started" if started else "already_running"}


@app.get("/engine/start")
async def engine_start() -> dict[str, str]:
    return await start_scheduler()


@app.get("/stop")
async def stop_scheduler() -> dict[str, str]:
    scheduler = _require_scheduler()
    stopped = await scheduler.stop()
    global running
    running = scheduler.running
    _record_action("stop")
    logger.info("engine_stop status=%s", "stopped" if stopped else "already_stopped")
    return {"status": "stopped" if stopped else "already_stopped"}


@app.get("/engine/stop")
async def engine_stop() -> dict[str, str]:
    return await stop_scheduler()


@app.get("/stats")
async def stats() -> dict:
    summary = compute_stats(_require_database().fetch_trades())
    return summary.__dict__


@app.get("/trades")
async def trades() -> dict:
    return {"trades": [trade.__dict__ for trade in _require_database().fetch_trades()]}


@app.get("/equity")
async def equity() -> dict:
    summary = compute_stats(_require_database().fetch_trades())
    return {"equity_curve": summary.equity_curve}


@app.get("/positions")
async def positions() -> dict:
    return {"positions": [trade.__dict__ for trade in _require_database().fetch_open_trades()]}


@app.get("/symbols")
async def symbols() -> dict:
    return {"symbols": _require_state().get_symbols()}


@app.post("/symbols")
async def update_symbols(payload: dict) -> dict:
    state_store = _require_state()
    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
        raise HTTPException(status_code=400, detail="invalid_symbols")
    state_store.set_symbols(symbols)
    return {"symbols": state_store.get_symbols()}


@app.get("/paper/reset")
async def reset_paper() -> dict:
    _require_database().reset_trades()
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
    scheduler = _require_scheduler()
    running = scheduler.running
    return {
        "status": "RUNNING" if running else "STOPPED",
        "mode": _require_settings().MODE,
        "last_heartbeat_ts": last_heartbeat_ts.isoformat() if last_heartbeat_ts else None,
        "last_action": last_action,
        "last_tick_time": scheduler.last_tick_time().isoformat() if scheduler.last_tick_time() else None,
    }

@app.post("/webhook/tradingview")
async def tradingview_webhook(request: DecisionRequest) -> dict:
    correlation_id = str(uuid4())
    _record_correlation_id(correlation_id)
    settings = _require_settings()
    state_store = _require_state()
    log_event(settings, "webhook", request.model_dump(), correlation_id)

    plan = decide(request, state_store, settings)
    if plan.status == Status.TRADE:
        state_store.record_trade(request.tradingview.symbol)
    state_store.set_latest_decision(request.tradingview.symbol, plan)

    log_event(settings, "decision", plan.model_dump(), correlation_id)
    return {"correlation_id": correlation_id, "plan": plan}


class DebugForceSignalRequest(BaseModel):
    symbol: str | None = Field(default=None, min_length=1)
    direction: Direction | None = None
    strategy: Literal["scalper", "baseline"] | None = None
    bypass_soft_gates: bool = False


class DebugSmokeCycleRequest(BaseModel):
    symbol: str | None = Field(default=None, min_length=1)
    direction: Direction | None = None
    hold_seconds: int = Field(2, ge=0)
    entry_price: float = Field(100.0, gt=0)


@app.get("/decision/latest")
async def decision_latest(symbol: str = Query(..., min_length=1)) -> dict:
    state_store = _require_state()
    scheduler = _require_scheduler()
    daily_state = state_store.get_daily_state(symbol)
    if daily_state.latest_decision is None:
        if scheduler.last_tick_time() is None:
            return {"symbol": symbol, "decision": None}
        return {"symbol": symbol, "decision": _fallback_decision(symbol)}
    return {"symbol": symbol, "decision": daily_state.latest_decision}


@app.get("/engine/run_once")
async def engine_run_once(force: bool = Query(True)) -> dict:
    _record_action("run_once")
    scheduler = _require_scheduler()
    logger.info("engine_run_once force=%s", force)
    results = await scheduler.run_once(force=force)
    return {"status": "ok", "results": results}


@app.post("/trade_outcome")
async def trade_outcome(outcome: TradeOutcome) -> dict:
    _require_state().record_outcome(outcome.symbol, outcome.pnl_usd, outcome.win, outcome.timestamp)
    return {"status": "recorded"}


@app.get("/state/today")
async def state_today(symbol: str = Query(..., min_length=1)) -> dict:
    daily_state = _require_state().get_daily_state(symbol)
    return {"symbol": symbol, "state": daily_state}


@app.get("/debug/runtime")
async def debug_runtime() -> dict:
    now = datetime.now(timezone.utc)
    settings = _require_settings()
    state_store = _require_state()
    scheduler = _require_scheduler()
    symbols = state_store.get_symbols() or list(settings.symbols)
    storage_health = _storage_health_check(settings.data_dir)
    symbol_data = []
    for symbol in symbols:
        snapshot = scheduler.last_snapshot(symbol)
        latest_candle_ts = snapshot.candle.close_time.isoformat() if snapshot else None
        last_tick_ts = scheduler.last_symbol_tick_time(symbol)
        last_processed_ms = state_store.get_last_processed_close_time_ms(symbol)
        last_processed_iso = None
        if last_processed_ms is not None:
            last_processed_iso = datetime.fromtimestamp(last_processed_ms / 1000, tz=timezone.utc).isoformat()
        last_decision_ts = state_store.get_last_decision_ts(symbol)
        latest_decision = state_store.get_daily_state(symbol).latest_decision
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
    settings = _require_settings()
    state_store = _require_state()
    scheduler = _require_scheduler()
    force_reasons: list[str] = []
    if settings.smoke_test_force_trade:
        force_reasons.append("smoke_test_force_trade")
    if payload.bypass_soft_gates:
        force_reasons.append("bypass_soft_gates")
    force_trade = bool(force_reasons)
    if not settings.debug_loosen and not force_trade:
        raise HTTPException(status_code=400, detail="debug_loosen_required")
    symbols = state_store.get_symbols() or list(settings.symbols)
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
        force_trade=force_trade,
        force_reasons=force_reasons,
    )
    state_store.set_latest_decision(symbol, plan)
    correlation_id = str(uuid4())
    _record_correlation_id(correlation_id)
    log_event(settings, "decision", plan.model_dump(), correlation_id)
    if plan.status == Status.TRADE and settings.telegram_enabled:
        message = format_trade_message(symbol, plan)
        await send_telegram_message(message, settings)
    executed, trade_id = _execute_trade_plan(symbol, plan)
    return {
        "correlation_id": correlation_id,
        "symbol": symbol,
        "plan": plan,
        "executed": executed,
        "trade_id": trade_id,
    }


@app.post("/debug/smoke/run_full_cycle")
async def debug_smoke_run_full_cycle(payload: DebugSmokeCycleRequest) -> dict:
    settings = _require_settings()
    state_store = _require_state()
    database = _require_database()
    trader = _require_paper_trader()
    symbols = state_store.get_symbols() or list(settings.symbols)
    symbol = payload.symbol or (symbols[0] if symbols else None)
    if symbol is None:
        raise HTTPException(status_code=400, detail="no_symbol_available")
    direction = payload.direction or Direction.long
    entry = payload.entry_price
    entry_low = entry * (1 - 0.0002)
    entry_high = entry * (1 + 0.0002)
    stop_loss = entry * (0.999 if direction == Direction.long else 1.001)
    take_profit = entry * (1.001 if direction == Direction.long else 0.999)
    plan = TradePlan(
        status=Status.TRADE,
        direction=direction,
        entry_zone=(entry_low, entry_high),
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_pct_used=0.001,
        position_size_usd=10.0,
        signal_score=None,
        posture=Posture.NORMAL,
        rationale=["debug_smoke_cycle"],
        raw_input_snapshot={"symbol": symbol, "direction": direction.value},
    )
    correlation_id = str(uuid4())
    _record_correlation_id(correlation_id)
    log_event(settings, "decision", plan.model_dump(), correlation_id)
    state_store.set_latest_decision(symbol, plan)
    before_stats = compute_stats(database.fetch_trades())
    trade_id = trader.maybe_open_trade(symbol, plan, allow_multiple=True)
    opened_at = datetime.now(timezone.utc)
    if trade_id is None:
        return {
            "status": "failed",
            "reason": "trade_not_opened",
            "symbol": symbol,
            "correlation_id": correlation_id,
        }
    state_store.record_trade(symbol)
    await asyncio.sleep(payload.hold_seconds)
    close_price = entry * (1.001 if direction == Direction.long else 0.999)
    close_results = trader.force_close_trades(symbol, close_price, reason="debug_smoke_cycle")
    closed_at = datetime.now(timezone.utc)
    pnl_usd = next((result.pnl_usd for result in close_results if result.trade_id == trade_id), 0.0)
    win = pnl_usd > 0
    state_store.record_outcome(symbol, pnl_usd, win, closed_at)
    after_stats = compute_stats(database.fetch_trades())
    return {
        "status": "ok",
        "symbol": symbol,
        "correlation_id": correlation_id,
        "trade_id": trade_id,
        "open_ts": opened_at.isoformat(),
        "close_ts": closed_at.isoformat(),
        "pnl_usd": pnl_usd,
        "equity_delta": after_stats.total_pnl - before_stats.total_pnl,
        "assertions": {
            "opened": trade_id is not None,
            "closed": any(result.trade_id == trade_id for result in close_results),
            "pnl_recorded": pnl_usd != 0.0,
        },
    }


@app.post("/debug/storage/reset")
async def debug_storage_reset() -> dict:
    settings = _require_settings()
    _require_database().reset_all()
    _require_state().reset()
    logs_dir = Path(settings.data_dir) / "logs"
    if logs_dir.exists():
        for path in logs_dir.glob("*.jsonl"):
            path.unlink()
    return {"status": "reset"}


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
    settings = _require_settings()
    daily_state = _require_state().get_daily_state(symbol)
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
    force_trade: bool,
    force_reasons: list[str],
) -> TradePlan:
    now = datetime.now(timezone.utc)
    if force_trade:
        return _forced_trade_plan(symbol, direction, snapshot, effective_settings, force_reasons)
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
            return _forced_trade_plan(
                symbol,
                direction,
                snapshot,
                effective_settings,
                list(plan.rationale) + ["bypass_soft_gates"],
            )
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
    effective_settings,
    extra_rationale: list[str] | None = None,
) -> TradePlan:
    entry = snapshot.candle.close
    entry_low = entry * (1 - 0.0002)
    entry_high = entry * (1 + 0.0002)
    stop_loss = entry * (0.999 if direction == Direction.long else 1.001)
    take_profit = entry * (1.001 if direction == Direction.long else 0.999)
    return TradePlan(
        status=Status.TRADE,
        direction=direction,
        entry_zone=(entry_low, entry_high),
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_pct_used=0.001,
        position_size_usd=10.0,
        signal_score=None,
        posture=Posture.NORMAL,
        rationale=["debug_force_signal", *(extra_rationale or [])],
        raw_input_snapshot={
            "symbol": symbol,
            "direction": direction.value,
            "strategy": effective_settings.strategy,
        },
    )


def _execute_trade_plan(symbol: str, plan: TradePlan) -> tuple[bool, int | None]:
    if plan.status != Status.TRADE:
        return False, None
    settings = _require_settings()
    if settings.engine_mode not in {"paper", "live"} or paper_trader is None:
        return False, None
    trade_id = paper_trader.maybe_open_trade(symbol, plan)
    if trade_id is None:
        return False, None
    _require_state().record_trade(symbol)
    return True, trade_id


def _hard_risk_gate_reasons(symbol: str, now: datetime) -> list[str]:
    settings = _require_settings()
    daily_state = _require_state().get_daily_state(symbol)
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


def _fallback_decision(symbol: str) -> TradePlan:
    return TradePlan(
        status=Status.RISK_OFF,
        direction=Direction.none,
        entry_zone=None,
        stop_loss=None,
        take_profit=None,
        risk_pct_used=None,
        position_size_usd=None,
        signal_score=None,
        posture=Posture.RISK_OFF,
        rationale=["no_decision_yet"],
        raw_input_snapshot={"symbol": symbol},
    )


def _contains_hard_gate(rationale: list[str]) -> bool:
    hard_gates = {
        "news_blackout",
        "daily_loss_limit",
        "max_losses_per_day",
        "max_trades",
        "cooldown",
    }
    return any(reason in hard_gates for reason in rationale)
