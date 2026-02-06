from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .services.notifier import send_telegram_message
from .config import get_settings
from .models import DecisionRequest, TradeOutcome, Status
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
last_action: dict[str, Any] = {
    "type": "boot",
    "ts": datetime.now(timezone.utc),
    "detail": "engine_initialized",
}


def _record_heartbeat() -> None:
    global last_heartbeat_ts
    last_heartbeat_ts = datetime.now(timezone.utc)


def _record_action(action_type: str, detail: str | None = None) -> None:
    global last_action
    payload: dict[str, Any] = {
        "type": action_type,
        "ts": datetime.now(timezone.utc),
    }
    if detail:
        payload["detail"] = detail
    last_action = payload


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
    _record_action("run_once")
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
    _record_action("start", "scheduler_started" if started else "already_running")
    return {"status": "started" if started else "already_running"}


@app.get("/stop")
async def stop_scheduler() -> dict[str, str]:
    stopped = await scheduler.stop()
    global running
    running = scheduler.running
    _record_action("stop", "scheduler_stopped" if stopped else "already_stopped")
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
        "running": running,
        "mode": settings.engine_mode,
        "symbols": settings.symbols,
        "last_heartbeat_ts": last_heartbeat_ts.isoformat() if last_heartbeat_ts else None,
        "last_action": {
            **last_action,
            "ts": last_action.get("ts").isoformat() if last_action.get("ts") else None,
        },
    }

@app.post("/webhook/tradingview")
async def tradingview_webhook(request: DecisionRequest) -> dict:
    correlation_id = str(uuid4())
    log_event(settings, "webhook", request.model_dump(), correlation_id)

    plan = decide(request, state, settings)
    if plan.status == Status.TRADE:
        state.record_trade(request.tradingview.symbol)
    state.set_latest_decision(request.tradingview.symbol, plan)

    log_event(settings, "decision", plan.model_dump(), correlation_id)
    return {"correlation_id": correlation_id, "plan": plan}


@app.get("/decision/latest")
async def decision_latest(symbol: str = Query(..., min_length=1)) -> dict:
    daily_state = state.get_daily_state(symbol)
    if daily_state.latest_decision is None:
        raise HTTPException(status_code=404, detail="no_decision")
    return {"symbol": symbol, "decision": daily_state.latest_decision}


@app.post("/trade_outcome")
async def trade_outcome(outcome: TradeOutcome) -> dict:
    state.record_outcome(outcome.symbol, outcome.pnl_usd, outcome.win, outcome.timestamp)
    return {"status": "recorded"}


@app.get("/state/today")
async def state_today(symbol: str = Query(..., min_length=1)) -> dict:
    daily_state = state.get_daily_state(symbol)
    return {"symbol": symbol, "state": daily_state}
