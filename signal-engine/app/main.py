from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .services.notifier import format_trade_message, send_telegram_message
from .config import Settings, get_settings
from .models import BiasSignal, DecisionRequest, Direction, EngineState, MarketSnapshot, Posture, SetupType, Status, TradeOutcome, TradePlan
from .state import StateStore
from .services.database import Database
from .services.paper_trader import PaperTrader
from .services.stats import compute_stats
from .services.performance import PerformanceStore, build_performance_snapshot
from .storage.store import log_event
from .strategy.decision import decide
from .services.scheduler import DecisionScheduler
from .providers.bybit import BybitClient
def _as_datetime_utc(value: Any) -> Optional[datetime]:
    """
    Normalize DB timestamps to timezone-aware UTC datetimes.
    Accepts:
      - datetime
      - ISO string
      - None
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    return None
class AccountSummary(BaseModel):
    starting_balance_usd: float
    balance_usd: float
    equity_usd: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    total_pnl_usd: float
    pnl_pct: float | None
    open_positions: int
    trades_today: int
    wins_today: int
    losses_today: int
    win_rate_today: float
    profit_factor: float
    expectancy: float
    max_drawdown_pct: float
    realized_pnl_today_usd: float
    daily_pct: float
    last_trade_ts: str | None
    engine_status: str
    pnl_curve: list[float] = Field(default_factory=list)
    last_updated_ts: str
    equity_curve: list[float] = Field(default_factory=list)


logger = logging.getLogger(__name__)

settings = None
state = None
database = None
paper_trader = None
scheduler = None
performance_store = None

running = False
last_heartbeat_ts: datetime | None = None
last_action: str | None = None
last_correlation_id: str | None = None
last_status_reason_logged: str | None = None
state_subscribers: set[asyncio.Queue[str]] = set()
latest_engine_state: EngineState | None = None


def _record_heartbeat() -> None:
    global last_heartbeat_ts
    last_heartbeat_ts = datetime.now(timezone.utc)


def _record_action(action_type: str) -> None:
    global last_action
    last_action = action_type


def _record_correlation_id(correlation_id: str) -> None:
    global last_correlation_id
    last_correlation_id = correlation_id


def _funding_blackout_active(now: datetime, cfg: Settings) -> bool:
    minutes = int(now.timestamp() // 60)
    interval = max(1, cfg.funding_interval_minutes)
    minute_in_window = minutes % interval
    block_start = max(0, interval - cfg.funding_block_before_minutes)
    return minute_in_window >= block_start or minute_in_window <= 1


def _trade_to_dict(trade: Any) -> dict[str, Any]:
    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "entry": trade.entry,
        "exit": trade.exit,
        "stop": trade.stop,
        "take_profit": trade.take_profit,
        "size": trade.size,
        "pnl_usd": trade.pnl_usd,
        "side": trade.side,
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
        "result": trade.result,
        "trade_mode": trade.trade_mode,
    }


def _build_engine_state_snapshot() -> EngineState:
    scheduler = _require_scheduler()
    cfg = _require_settings()
    db = _require_database()
    state_store = _require_state()
    trader = _require_paper_trader()
    now = datetime.now(timezone.utc)

    trades = db.fetch_trades()
    summary = compute_stats(trades)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    wins = 0
    losses = 0
    trades_today = 0
    realized_pnl_today = 0.0
    for trade in trades:
        closed_at_dt = _as_datetime_utc(getattr(trade, "closed_at", None))
        if closed_at_dt is None or closed_at_dt < start_of_day:
            continue
        if getattr(trade, "trade_mode", "paper") == "test":
            continue
        pnl = float(getattr(trade, "pnl_usd", None) or 0.0)
        realized_pnl_today += pnl
        trades_today += 1
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    unrealized = trader.total_unrealized_pnl_usd()
    balance = cfg.account_size + summary.total_pnl
    equity = balance + unrealized
    max_dd_today_pct = (summary.max_drawdown / cfg.account_size) * 100 if cfg.account_size else 0.0

    symbols = state_store.get_symbols()
    risk_symbol = symbols[0] if symbols else (cfg.symbols[0] if cfg.symbols else "BTCUSDT")
    risk = state_store.risk_snapshot(risk_symbol, cfg, now)

    last_tick = scheduler.last_tick_time()
    last_tick_age_seconds = (now - last_tick).total_seconds() if last_tick else None

    current_mode = cfg.current_mode
    decision_meta = state_store.get_decision_meta(risk_symbol)

    return EngineState(
        timestamp=now,
        last_tick_age_seconds=last_tick_age_seconds,
        running=scheduler.running,
        balance=balance,
        equity=equity,
        unrealized_pnl_usd=unrealized,
        realized_pnl_today_usd=realized_pnl_today,
        trades_today=trades_today,
        wins=wins,
        losses=losses,
        win_rate=(wins / trades_today) if trades_today else 0.0,
        profit_factor=summary.profit_factor,
        max_dd_today_pct=max_dd_today_pct,
        daily_loss_remaining_usd=float(risk.get("daily_loss_remaining_usd", 0.0)),
        daily_loss_pct=float(cfg.max_daily_loss_pct or 0.0),
        open_positions=[_trade_to_dict(trade) for trade in db.fetch_open_trades()],
        recent_trades=[_trade_to_dict(trade) for trade in trades[:50]],
        cooldown_active=bool(risk.get("cooldown_active", False)),
        funding_blackout=_funding_blackout_active(now, cfg),
        swings_enabled=cfg.sweet8_enabled,
        current_mode=str(current_mode),
        consecutive_losses=int(risk.get("consecutive_losses", 0) or 0),
        last_decision=decision_meta.get("decision"),
        last_skip_reason=decision_meta.get("skip_reason"),
        final_entry_gate=decision_meta.get("final_entry_gate"),
        regime_label=decision_meta.get("regime_label"),
        allowed_side=decision_meta.get("allowed_side"),
        atr_pct=decision_meta.get("atr_pct"),
        ema_fast=decision_meta.get("ema_fast"),
        ema_slow=decision_meta.get("ema_slow"),
        ema_trend=decision_meta.get("ema_trend"),
    )


def _publish_state_snapshot() -> None:
    global latest_engine_state
    snapshot = _build_engine_state_snapshot()
    latest_engine_state = snapshot
    logger.info(
        "state_tick_summary equity=%.2f unrealized=%.2f open_positions=%s",
        snapshot.equity,
        snapshot.unrealized_pnl_usd,
        len(snapshot.open_positions),
    )
    payload = f"event: state\ndata: {json.dumps(snapshot.model_dump(mode='json'))}\n\n"
    stale: list[asyncio.Queue[str]] = []
    for queue in state_subscribers:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            stale.append(queue)
    for queue in stale:
        state_subscribers.discard(queue)


def _initialize_engine(app: FastAPI) -> None:
    global settings, state, database, paper_trader, scheduler, performance_store
    global running, last_heartbeat_ts, last_action, last_correlation_id, last_status_reason_logged

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
    scheduler.add_tick_listener(_publish_state_snapshot)
    performance_store = PerformanceStore(settings.data_dir)
    running = False
    last_heartbeat_ts = None
    last_action = None
    last_correlation_id = None
    last_status_reason_logged = None

    app.state.settings = settings
    app.state.state_store = state
    app.state.database = database
    app.state.paper_trader = paper_trader
    app.state.scheduler = scheduler
    app.state.performance_store = performance_store


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


def _require_performance_store() -> PerformanceStore:
    if performance_store is None:
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return performance_store


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


@app.get("/state", response_model=EngineState)
async def latest_state() -> EngineState:
    global latest_engine_state
    snapshot = _build_engine_state_snapshot()
    latest_engine_state = snapshot
    return snapshot


@app.get("/events/state")
async def stream_state_events(request: Request) -> StreamingResponse:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    state_subscribers.add(queue)
    logger.info("state_stream_subscribe subscribers=%s", len(state_subscribers))

    if latest_engine_state is not None:
        queue.put_nowait(f"event: state\ndata: {json.dumps(latest_engine_state.model_dump(mode='json'))}\n\n")
    else:
        queue.put_nowait(f"event: state\ndata: {json.dumps(_build_engine_state_snapshot().model_dump(mode='json'))}\n\n")

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield message
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            state_subscribers.discard(queue)
            logger.info("state_stream_unsubscribe subscribers=%s", len(state_subscribers))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/start")
async def start_scheduler() -> dict[str, str]:
    scheduler = _require_scheduler()
    started = await scheduler.start()
    global running
    running = scheduler.running
    _record_action("start")
    status = "started" if started else "already_running"
    if not started and scheduler.stop_reason:
        status = "failed"
        logger.error("engine_start status=failed reason=%s", scheduler.stop_reason)
    else:
        logger.info("engine_start status=%s", status)
    _publish_state_snapshot()
    return {"status": status}


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
    _publish_state_snapshot()
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


@app.get("/metrics/performance")
async def performance_metrics() -> dict[str, Any]:
    settings = _require_settings()
    database = _require_database()
    state_store = _require_state()
    snapshot = build_performance_snapshot(
        database.fetch_trades(),
        account_size=float(settings.account_size or 0.0),
        skip_reason_counts=state_store.skip_reason_counts(),
    )
    store = _require_performance_store()
    store.save(snapshot)
    return snapshot.__dict__


@app.get("/account/summary", response_model=AccountSummary)
async def account_summary() -> AccountSummary:
    settings = _require_settings()
    state_store = _require_state()
    database = _require_database()
    summary = compute_stats(database.fetch_trades())
    open_positions = database.fetch_open_trades()

    starting_balance = settings.account_size or 0.0
    realized_pnl = summary.total_pnl
    unrealized_pnl = _require_paper_trader().total_unrealized_pnl_usd()
    balance = starting_balance + realized_pnl
    equity = balance + unrealized_pnl
    total_pnl = realized_pnl + unrealized_pnl
    pnl_pct = None
    if starting_balance > 0:
        pnl_pct = ((equity - starting_balance) / starting_balance) * 100
    elif equity != 0:
        pnl_pct = (total_pnl / abs(equity)) * 100

    # --- TODAY METRICS from DB (correct, not state_store) ---
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    trades_today = 0
    wins_today = 0
    losses_today = 0

    all_trades = database.fetch_trades()
    last_trade_ts: str | None = None
    realized_pnl_today = 0.0

    for t in all_trades:
        closed_at_dt = _as_datetime_utc(getattr(t, "closed_at", None))
        if closed_at_dt is None:
            continue
        if closed_at_dt < start_of_day:
            continue

        if getattr(t, "trade_mode", "paper") == "test":
            continue
        # 🚫 ignore forced / debug closes
        result = (getattr(t, "result", None) or "").lower()
        if "force" in result or "debug" in result:
            continue
        pnl = float(getattr(t, "pnl_usd", None) or 0.0)
        realized_pnl_today += pnl

        trades_today += 1
        if pnl > 0:
            wins_today += 1
        elif pnl < 0:
            losses_today += 1

    for t in all_trades:
        closed_at_dt = _as_datetime_utc(getattr(t, "closed_at", None))
        if closed_at_dt is None or getattr(t, "trade_mode", "paper") == "test":
            continue
        last_trade_ts = closed_at_dt.isoformat()
        break

    win_rate_today = (wins_today / trades_today) if trades_today else 0.0
    max_drawdown_pct = (summary.max_drawdown / starting_balance) * 100 if starting_balance else 0.0
    equity_curve = [starting_balance + point for point in summary.equity_curve]

    return AccountSummary(
        starting_balance_usd=starting_balance,
        balance_usd=balance,
        equity_usd=equity,
        realized_pnl_usd=realized_pnl,
        unrealized_pnl_usd=unrealized_pnl,
        total_pnl_usd=total_pnl,
        pnl_pct=pnl_pct,
        open_positions=len(open_positions),
        trades_today=trades_today,
        wins_today=wins_today,
        losses_today=losses_today,
        win_rate_today=win_rate_today,
        profit_factor=summary.profit_factor,
        expectancy=summary.expectancy,
        max_drawdown_pct=max_drawdown_pct,
        realized_pnl_today_usd=realized_pnl_today,
        daily_pct=((realized_pnl_today / starting_balance) * 100) if starting_balance else 0.0,
        last_trade_ts=last_trade_ts,
        engine_status="RUNNING" if _require_scheduler().running else "STOPPED",
        pnl_curve=summary.equity_curve,
        last_updated_ts=datetime.now(timezone.utc).isoformat(),
        equity_curve=equity_curve,
    )




@app.get("/risk/summary")
async def risk_summary(symbol: str = Query(None)) -> dict[str, Any]:
    settings = _require_settings()
    state_store = _require_state()
    now = datetime.now(timezone.utc)
    use_symbol = symbol or (settings.symbols[0] if settings.symbols else "BTCUSDT")
    snapshot = state_store.risk_snapshot(use_symbol, settings, now)
    minutes = int(now.timestamp() // 60)
    interval = max(1, settings.funding_interval_minutes)
    minute_in_window = minutes % interval
    block_start = max(0, interval - settings.funding_block_before_minutes)
    snapshot["funding_blackout_active"] = minute_in_window >= block_start or minute_in_window <= 1
    snapshot["symbol"] = use_symbol
    snapshot["sweet8_enabled"] = settings.sweet8_enabled
    snapshot["sweet8_current_mode"] = state_store.get_last_notified_key("__sweet8_current_mode__") or settings.sweet8_current_mode
    snapshot["open_positions"] = len(_require_database().fetch_open_trades())
    snapshot["blocked_premature_exits"] = settings.sweet8_blocked_close_total
    snapshot["daily_loss_pct"] = float(settings.max_daily_loss_pct or 0.0)
    snapshot["risk_per_trade_pct"] = float(settings.base_risk_pct or 0.0)
    return snapshot
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
    normalized = [item.strip().upper() for item in symbols if item and item.strip()]
    if not normalized:
        raise HTTPException(status_code=400, detail="invalid_symbols")
    state_store.set_symbols(normalized)
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
    global running, last_status_reason_logged
    scheduler = _require_scheduler()
    running = scheduler.running
    stop_reason = scheduler.stop_reason
    if not running and stop_reason and stop_reason != last_status_reason_logged:
        logger.warning("engine_state_change status=STOPPED reason=%s", stop_reason)
        last_status_reason_logged = stop_reason
    if running:
        last_status_reason_logged = None
    uptime_seconds = 0
    if running and scheduler.started_ts is not None:
        uptime_seconds = max(0, int(time.time() - scheduler.started_ts))
    return {
        "running": running,
        "status": "RUNNING" if running else "STOPPED",
        "stop_reason": stop_reason,
        "mode": _require_settings().MODE,
        "last_heartbeat_ts": last_heartbeat_ts.isoformat() if last_heartbeat_ts else None,
        "last_action": last_action,
        "last_tick_time": scheduler.last_tick_time().isoformat() if scheduler.last_tick_time() else None,
        "last_tick_ts": scheduler.last_tick_ts,
        "uptime_seconds": uptime_seconds,
        "sweet8_enabled": _require_settings().sweet8_enabled,
        "sweet8_current_mode": _require_state().get_last_notified_key("__sweet8_current_mode__") or _require_settings().sweet8_current_mode,
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
    state_store = _require_state()
    db = _require_database()

    # Keep latest_decision from state store (UI uses it)
    daily_state = state_store.get_daily_state(symbol)
    latest_decision = getattr(daily_state, "latest_decision", None)

    # Compute today metrics from DB trades (source of truth)
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    trades = db.fetch_trades()

    # Only trades for this symbol, CLOSED today (so pnl is final)
    closed_today = []
    for t in trades:
        if getattr(t, "symbol", None) != symbol:
            continue

        closed_at_dt = _as_datetime_utc(getattr(t, "closed_at", None))
        if closed_at_dt is None:
            continue

        if closed_at_dt >= start_of_day:
            closed_today.append(t)

    pnl_list = [(getattr(t, "pnl_usd", None) or 0.0) for t in closed_today]
    trades_count = len(closed_today)

    losses_count = sum(1 for p in pnl_list if p < 0)
    wins_count = sum(1 for p in pnl_list if p > 0)
    breakeven_count = sum(1 for p in pnl_list if p == 0)

    pnl_today = float(sum(pnl_list))

    def _closed_key(tr):
        return _as_datetime_utc(getattr(tr, "closed_at", None)) or datetime.min.replace(tzinfo=timezone.utc)

    # consecutive_losses (from most recent closed trade going backwards)
    consecutive_losses = 0
    for t in sorted(closed_today, key=_closed_key, reverse=True):
        p = (getattr(t, "pnl_usd", None) or 0.0)
        if p < 0:
            consecutive_losses += 1
        else:
            break

    # last_loss_ts
    last_loss_ts = None
    for t in sorted(closed_today, key=_closed_key, reverse=True):
        p = (getattr(t, "pnl_usd", None) or 0.0)
        if p < 0:
            last_loss_ts = _as_datetime_utc(getattr(t, "closed_at", None))
            break

    state = {
        "date": start_of_day.date().isoformat(),
        "trades": trades_count,
        "wins": wins_count,
        "losses": losses_count,
        "breakeven": breakeven_count,
        "consecutive_losses": consecutive_losses,
        "pnl_usd": pnl_today,
        "last_loss_ts": last_loss_ts.isoformat() if last_loss_ts else None,
        "latest_decision": latest_decision,
    }

    return {"symbol": symbol, "state": state}

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


@app.get("/debug/kline")
async def debug_kline(interval: str = Query("5m")) -> dict[str, Any]:
    settings = _require_settings()
    client = BybitClient(
        api_key=settings.bybit_api_key,
        api_secret=settings.bybit_api_secret,
        rest_base=settings.bybit_rest_base,
    )
    symbols = ["BTCUSDT", "ETHUSDT"]
    output: list[dict[str, Any]] = []
    for symbol in symbols:
        debug_rows = await client.fetch_kline_debug_rows(symbol=symbol, interval=interval, limit=3)
        logger.info("bybit_kline_debug symbol=%s rows=%s", symbol, debug_rows["rows"])
        output.append(debug_rows)
    return {"interval": interval, "symbols": output}


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
        "decision": plan,
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
    if force_trade:
        return _forced_trade_plan(symbol, direction, snapshot, effective_settings, force_reasons)
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
        status=Status.NO_TRADE,
        direction=Direction.none,
        entry_zone=None,
        stop_loss=None,
        take_profit=None,
        risk_pct_used=None,
        position_size_usd=None,
        signal_score=None,
        posture=Posture.NORMAL,
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
