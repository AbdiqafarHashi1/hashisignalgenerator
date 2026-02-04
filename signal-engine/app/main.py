from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query

from .config import get_settings
from .models import DecisionRequest, TradeOutcome, Status
from .state import StateStore
from .storage.store import log_event
from .strategy.decision import decide

app = FastAPI(title="signal-engine", version="1.0.0")
settings = get_settings()
state = StateStore()


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
