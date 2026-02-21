from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
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
from .services.dashboard_metrics import build_dashboard_metrics
from .services.challenge import ChallengeService
from .services.prop_governor import PropRiskGovernor
from .strategy.decision import decide
from .services.scheduler import DecisionScheduler
from .providers.bybit import BybitClient, fetch_symbol_klines, replay_reset, replay_reset_all_state, replay_status, replay_validate_dataset
from .providers.replay import ReplayDatasetError
from .utils.clock import RealClock, ReplayClock
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


class DashboardOverview(BaseModel):
    account: dict[str, Any]
    risk: dict[str, Any]
    activity: dict[str, Any]
    symbols: dict[str, dict[str, Any]]
    recent_trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    skip_reasons: dict[str, Any]


logger = logging.getLogger(__name__)

settings = None
state = None
database = None
paper_trader = None
scheduler = None
performance_store = None
challenge_service = None
prop_governor = None
engine_reset_lock = asyncio.Lock()

running = False
last_heartbeat_ts: datetime | None = None
last_action: str | None = None
last_correlation_id: str | None = None
last_status_reason_logged: str | None = None
replay_last_error: str | None = None
state_subscribers: set[asyncio.Queue[str]] = set()
latest_engine_state: EngineState | None = None
runtime_clock: RealClock | ReplayClock | None = None


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


def _mark_price_for_trade(trader: PaperTrader, trade: Any) -> float:
    return float(trader._last_mark_prices.get(trade.symbol, trade.entry))


def _trade_fee(trade: Any, cfg: Settings) -> float:
    if getattr(trade, "fees", None) is not None:
        return float(trade.fees or 0.0)
    if trade.exit is None:
        return 0.0
    return ((trade.entry * trade.size) + (trade.exit * trade.size)) * (float(cfg.fee_rate_bps or 0.0) / 10000)


def _trade_to_dict(trade: Any, trader: PaperTrader | None = None, cfg: Settings | None = None) -> dict[str, Any]:
    mark = _mark_price_for_trade(trader, trade) if trader is not None else None
    unrealized = None
    unrealized_r = None
    if trade.closed_at is None and mark is not None:
        side_sign = 1.0 if trade.side == "long" else -1.0
        unrealized = (mark - trade.entry) * trade.size * side_sign
        risk = abs(trade.entry - trade.stop)
        unrealized_r = ((mark - trade.entry) * side_sign) / risk if risk > 0 else None

    fee_value = _trade_fee(trade, cfg) if cfg is not None else (float(getattr(trade, "fees", 0.0) or 0.0) if getattr(trade, "fees", None) is not None else None)

    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "entry": trade.entry,
        "exit": trade.exit,
        "stop": trade.stop,
        "take_profit": trade.take_profit,
        "size": trade.size,
        "pnl_usd": trade.pnl_usd,
        "fees": fee_value,
        "side": trade.side,
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
        "result": trade.result,
        "trade_mode": trade.trade_mode,
        "mark_price": mark,
        "unrealized_pnl": unrealized,
        "unrealized_r": unrealized_r,
    }


def cfg_db_path(cfg: Settings) -> str:
    return cfg.database_url or f"sqlite:///{Path(cfg.data_dir) / 'trades.db'}"


def _scan_csv_bounds(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    first_ts = None
    last_ts = None
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = row.get("close_time") or row.get("timestamp")
            if not ts:
                continue
            first_ts = first_ts or ts
            last_ts = ts
    return {"path": str(path), "exists": True, "size": path.stat().st_size, "first_ts": first_ts, "last_ts": last_ts}


def _get_runtime_float(db: Database, key: str, default: float) -> float:
    row = db.get_runtime_state(key)
    if row is None or row.value_number is None:
        return default
    return float(row.value_number)


def _get_runtime_text(db: Database, key: str, default: str) -> str:
    row = db.get_runtime_state(key)
    if row is None or row.value_text is None:
        return default
    return str(row.value_text)


def _build_engine_state_snapshot() -> EngineState:
    scheduler = _require_scheduler()
    cfg = _require_settings()
    db = _require_database()
    state_store = _require_state()
    trader = _require_paper_trader()
    now = _runtime_now()

    acct = build_dashboard_metrics(cfg, db, trader, state_store, now)
    challenge = None
    challenge_ready = False
    challenge_error: str | None = "engine_not_ready"
    maybe_challenge_service = _get_challenge_service_optional()
    if maybe_challenge_service is not None:
        try:
            challenge = maybe_challenge_service.update(
                equity=float(acct["equity_now"]),
                daily_start_equity=float(acct["day_start_equity"]),
                now=now,
                traded_today=bool(acct["trades_today"]),
            )
            challenge_ready = True
            challenge_error = None
        except Exception as exc:
            challenge_error = f"{type(exc).__name__}: {exc}"
            logger.warning("challenge_state_publish_failed error=%s", challenge_error)
    trades = acct["trades"]
    summary = compute_stats(trades)

    wins = int(acct["wins_today"])
    losses = int(acct["losses_today"])
    trades_today = int(acct["trades_today"])
    realized_pnl_today = float(acct["pnl_realized_today"])

    unrealized = float(acct["pnl_unrealized"])
    balance = float(acct["balance"])
    equity = float(acct["equity_now"])
    max_dd_today_pct = float(acct["daily_dd_pct"]) * 100
    db.record_equity(equity=equity, realized_pnl_today=realized_pnl_today, drawdown_pct=max_dd_today_pct)

    symbols = state_store.get_symbols()
    risk_symbol = symbols[0] if symbols else (cfg.symbols[0] if cfg.symbols else "BTCUSDT")
    risk = state_store.risk_snapshot(risk_symbol, cfg, now)

    last_tick = scheduler.last_tick_time()
    last_tick_age_seconds = (now - last_tick).total_seconds() if last_tick else None

    current_mode = cfg.current_mode
    decision_meta = state_store.get_decision_meta(risk_symbol)

    return EngineState(
        timestamp=now,
        server_ts=datetime.now(timezone.utc),
        candle_ts=now,
        replay_cursor_ts=now,
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
        blocker_code=decision_meta.get("blocker_code"),
        blocker_detail=decision_meta.get("blocker_detail") or "No active blockers.",
        blocker_layer=decision_meta.get("blocker_layer") if decision_meta.get("blocker_layer") in {"terminal", "governor", "risk", "strategy", "none"} else "none",
        blocker_until_ts=_as_datetime_utc(decision_meta.get("blocker_until_ts")),
        blockers=list(decision_meta.get("blockers") or []),
        regime_label=decision_meta.get("regime_label"),
        allowed_side=decision_meta.get("allowed_side"),
        atr_pct=decision_meta.get("atr_pct"),
        ema_fast=decision_meta.get("ema_fast"),
        ema_slow=decision_meta.get("ema_slow"),
        ema_trend=decision_meta.get("ema_trend"),
        starting_equity=float(acct["equity_start"]),
        realized_pnl=float(acct["pnl_realized_total"]),
        unrealized_pnl=float(acct["pnl_unrealized"]),
        fees_total=float(acct["fees_total"]),
        fees_today=float(acct["fees_today"]),
        equity_reconcile_delta=float(acct["equity_reconcile_delta"]),
        daily_start_equity=float(acct["day_start_equity"]),
        daily_peak_equity=float(acct["day_start_equity"]),
        global_peak_equity=float(acct["equity_high_watermark"]),
        daily_dd_pct=float(acct["daily_dd_pct"]),
        global_dd_pct=float(acct["global_dd_pct"]),
        trades_today_by_symbol=dict(acct["trades_today_by_symbol"]),
        realized_pnl_by_symbol=dict(acct["realized_pnl_by_symbol"]),
        fees_by_symbol=dict(acct["fees_by_symbol"]),
        challenge=challenge.__dict__ if challenge is not None else None,
        challenge_ready=challenge_ready,
        challenge_error=challenge_error,
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
    global settings, state, database, paper_trader, scheduler, performance_store, challenge_service, prop_governor, runtime_clock
    global running, last_heartbeat_ts, last_action, last_correlation_id, last_status_reason_logged, replay_last_error

    settings = get_settings()
    runtime_clock = ReplayClock() if settings.run_mode == "replay" else RealClock()
    state = StateStore()
    state.set_symbols(settings.symbols)
    database = Database(settings)
    database.init_schema()
    challenge_service = ChallengeService(settings, database)
    challenge_service.load(now=runtime_clock.now_dt(), start_equity=float(settings.account_size or 0.0))
    prop_governor = PropRiskGovernor(settings, database)
    prop_governor.load(now=runtime_clock.now_dt())
    paper_trader = PaperTrader(settings, database)
    scheduler = DecisionScheduler(
        settings,
        state,
        database,
        paper_trader,
        interval_seconds=settings.tick_interval_seconds,
        heartbeat_cb=_record_heartbeat,
        clock=runtime_clock,
    )
    state.set_clock(_runtime_now)
    paper_trader.set_clock(scheduler.engine_now)
    scheduler.add_tick_listener(_publish_state_snapshot)
    performance_store = PerformanceStore(settings.data_dir)
    running = False
    last_heartbeat_ts = None
    last_action = None
    last_correlation_id = None
    last_status_reason_logged = None
    replay_last_error = None

    if os.getenv("DEBUG_RUNTIME_DIAG", "false").strip().lower() in {"1", "true", "yes", "on"}:
        env_file = str(Path.cwd() / ".env") if (Path.cwd() / ".env").exists() else None
        logger.info("runtime_diag env_file=%s", env_file)
        logger.info("runtime_diag settings=%s", settings.resolved_settings())
        logger.info("runtime_diag replay_path=%s", Path(settings.market_data_replay_path).resolve())
        logger.info("runtime_diag db_path=%s", cfg_db_path(settings))

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

@app.on_event("startup")
async def ensure_database_schema() -> None:
    if database is not None:
        database.init_schema()


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




def _require_challenge_service() -> ChallengeService:
    if challenge_service is None:
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return challenge_service


def _get_challenge_service_optional() -> ChallengeService | None:
    return challenge_service


def _require_prop_governor() -> PropRiskGovernor:
    if prop_governor is None:
        raise HTTPException(status_code=503, detail="engine_not_ready")
    return prop_governor


def _runtime_now() -> datetime:
    if scheduler is not None:
        cfg = _require_settings()
        if cfg.run_mode == "replay" or cfg.market_data_provider == "replay":
            return scheduler.engine_now()
    if runtime_clock is not None:
        return runtime_clock.now_dt()
    return datetime.now(timezone.utc)

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
async def health() -> dict[str, Any]:
    cfg_errors: list[str] = []
    try:
        _require_settings()
    except Exception as exc:
        cfg_errors.append(str(exc))
    cfg = _require_settings()
    sch = _require_scheduler()
    return {
        "ok": True,
        "status": "ok",
        "mode": cfg.run_mode,
        "provider": cfg.market_data_provider,
        "provider_type": cfg.market_data_provider,
        "replay_running": sch.replay_active,
        "version": "1",
        "config_errors": cfg_errors,
        "config_summary": {
            "run_mode": cfg.run_mode,
            "symbols": cfg.symbols,
            "candle_interval": cfg.candle_interval,
            "replay_path": cfg.market_data_replay_path,
        },
    }




@app.get("/settings")
async def settings_payload() -> dict[str, Any]:
    cfg = _require_settings()
    symbol = cfg.symbols[0] if cfg.symbols else "ETHUSDT"
    interval = cfg.candle_interval or "5m"
    data_range = replay_validate_dataset(
        cfg.market_data_replay_path,
        symbol,
        interval,
        start_ts=cfg.replay_start_ts,
        end_ts=cfg.replay_end_ts,
    ) if cfg.run_mode == "replay" else None
    return {
        "effective": cfg.resolved_settings(),
        "data_range": data_range,
    }


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
    global replay_last_error
    scheduler = _require_scheduler()
    cfg = _require_settings()
    if cfg.run_mode == "replay":
        try:
            replay_validate_dataset(cfg.market_data_replay_path, cfg.symbols[0], cfg.candle_interval or "3m", start_ts=cfg.replay_start_ts, end_ts=cfg.replay_end_ts)
            replay_last_error = None
        except ReplayDatasetError as exc:
            replay_last_error = str(exc)
            logger.error("engine_start status=failed reason=%s", replay_last_error)
            return {"status": "failed"}
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


@app.post("/engine/replay/start")
@app.post("/replay/start")
async def engine_replay_start() -> dict[str, str]:
    return await start_scheduler()


@app.post("/engine/replay/stop")
@app.post("/replay/stop")
async def engine_replay_stop() -> dict[str, str]:
    return await stop_scheduler()


@app.post("/engine/replay/reset")
async def engine_replay_reset(clear_storage: bool = False) -> dict[str, str]:
    cfg = _require_settings()
    db = _require_database()
    state_store = _require_state()
    async with engine_reset_lock:
        for symbol in cfg.symbols:
            replay_reset(cfg.market_data_replay_path, symbol, cfg.candle_interval or "5m")
        if clear_storage:
            db.reset_all()
            state_store.reset()
        _require_prop_governor().reset(_runtime_now())
    return {"status": "ok"}




@app.get("/challenge/status")
async def challenge_status() -> dict[str, Any]:
    now = _runtime_now()
    return _require_challenge_service().status_payload(now=now)


@app.post("/challenge/reset")
async def challenge_reset() -> dict[str, Any]:
    cfg = _require_settings()
    db = _require_database()
    state_store = _require_state()
    sch = _require_scheduler()
    start = time.perf_counter()
    async with engine_reset_lock:
        await sch.stop()
        db.reset_all()
        state_store.reset()
        if cfg.run_mode == "replay":
            for symbol in cfg.symbols:
                replay_reset(cfg.market_data_replay_path, symbol, cfg.candle_interval or "3m")
        now = _runtime_now()
        _require_prop_governor().reset(now)
        challenge = _require_challenge_service().reset(now, float(cfg.account_size or 0.0))
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {"status": "reset", "elapsed_ms": elapsed_ms, "challenge": challenge.__dict__}


@app.get("/replay/progress")
async def replay_progress() -> dict[str, Any]:
    cfg = _require_settings()
    symbol = cfg.symbols[0] if cfg.symbols else "ETHUSDT"
    interval = cfg.candle_interval or "5m"
    status = replay_status(cfg.market_data_replay_path, symbol, interval, replay_resume=cfg.replay_resume)
    trades_closed = len([t for t in _require_database().fetch_trades() if t.closed_at is not None])
    snapshot = _require_state().get_decision_meta(symbol)
    bars_processed = status.get("bars_processed", status.get("bar_index", 0) + 1)
    total_bars = status.get("total_bars", status.get("row_count", 0))
    warmup_meta = snapshot.get("warmup_status") or {
        "ready": False,
        "bars_5m_have": int(snapshot.get("candles_loaded_5m_count") or 0),
        "bars_5m_need": int(cfg.warmup_min_bars_5m),
        "missing_components": ["ema_slow"],
    }
    if cfg.htf_bias_enabled and "bars_htf_need" not in warmup_meta:
        warmup_meta["bars_htf_have"] = int(snapshot.get("candles_loaded_htf_count") or 0)
        warmup_meta["bars_htf_need"] = int(cfg.warmup_min_bars_1h)
    return {
        "replay_current_ts": status.get("current_ts"),
        "replay_start_ts": status.get("first_ts"),
        "replay_end_ts": status.get("last_ts"),
        "current_ts": status.get("current_ts"),
        "bars_processed": bars_processed,
        "total_bars": total_bars,
        "trades_closed": trades_closed,
        "speed": float(cfg.market_data_replay_speed or 1.0),
        "candles_loaded_5m_count": snapshot.get("candles_loaded_5m_count", bars_processed),
        "candles_loaded_htf_count": snapshot.get("candles_loaded_htf_count", 0),
        "warmup_status": warmup_meta,
        "state": snapshot.get("equity_state"),
        "regime": snapshot.get("regime_label") or snapshot.get("regime"),
        "bias": snapshot.get("bias"),
        "confidence": snapshot.get("confidence"),
        "entry_reasons": snapshot.get("entry_reasons") or [],
        "entry_block_reasons": snapshot.get("entry_block_reasons") or [],
    }

@app.get("/replay/status")
async def replay_dataset_status() -> dict[str, Any]:
    cfg = _require_settings()
    symbol = cfg.symbols[0] if cfg.symbols else "ETHUSDT"
    interval = cfg.candle_interval or "3m"
    status = replay_status(cfg.market_data_replay_path, symbol, interval, replay_resume=cfg.replay_resume)
    return status

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
    start_of_day = _runtime_now().replace(hour=0, minute=0, second=0, microsecond=0)

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
        last_updated_ts=_runtime_now().isoformat(),
        equity_curve=equity_curve,
    )




@app.get("/risk/summary")
async def risk_summary(symbol: str = Query(None)) -> dict[str, Any]:
    settings = _require_settings()
    state_store = _require_state()
    now = _runtime_now()
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
    configured_risk_usd = float(settings.risk_per_trade_usd or 0.0)
    account_size = float(settings.account_size or 0.0)
    snapshot["risk_per_trade_pct"] = (
        (configured_risk_usd / account_size) if configured_risk_usd > 0 and account_size > 0 else float(settings.base_risk_pct or 0.0)
    )
    return snapshot


@app.post("/engine/kill-switch")
async def set_kill_switch(enabled: bool = Query(...)) -> dict[str, Any]:
    cfg = _require_settings()
    cfg.manual_kill_switch = enabled
    return {"status": "ok", "manual_kill_switch": cfg.manual_kill_switch}


@app.get("/dashboard/overview", response_model=DashboardOverview)
async def dashboard_overview() -> DashboardOverview:
    cfg = _require_settings()
    state_store = _require_state()
    db = _require_database()
    trader = _require_paper_trader()
    scheduler = _require_scheduler()
    now = _runtime_now()
    acct = build_dashboard_metrics(cfg, db, trader, state_store, now)
    trades = acct["trades"]
    perf = build_performance_snapshot(trades, account_size=float(cfg.account_size or 0.0), skip_reason_counts=state_store.skip_reason_counts())

    starting_equity = float(acct["equity_start"])
    realized = float(acct["pnl_realized_total"])
    unrealized = float(acct["pnl_unrealized"])
    equity = float(acct["equity_now"])
    fees_total = float(acct["fees_total"])
    fees_today = float(acct["fees_today"])
    trades_today = int(acct["trades_today"])

    daily_dd_pct = float(acct["daily_dd_pct"])
    global_dd_pct = float(acct["global_dd_pct"])
    daily_dd_usd = float(acct["daily_dd_abs"])
    global_dd_usd = float(acct["global_dd_abs"])

    active_reasons = _active_risk_gates(cfg.symbols[0] if cfg.symbols else "BTCUSDT", now)

    open_trades = db.fetch_open_trades()
    open_positions = [_trade_to_dict(t, trader, cfg) for t in open_trades]
    open_orders: list[dict[str, Any]] = []
    executions = [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "price": t.exit,
            "qty": t.size,
            "fee": _trade_fee(t, cfg),
            "status": t.result,
            "time": t.closed_at,
        }
        for t in trades[:200]
        if t.closed_at is not None and t.exit is not None
    ]

    symbol_data: dict[str, dict[str, Any]] = {}
    for symbol in cfg.symbols:
        meta = state_store.get_decision_meta(symbol)
        symbol_data[symbol] = {
            "regime": "TRENDING" if str(meta.get("regime_label", "")).lower() in {"bull", "bear", "trend"} else "RANGING",
            "last_decision": meta.get("decision"),
            "last_skip_reason": meta.get("skip_reason"),
            "blocker_code": meta.get("blocker_code"),
            "atr_pct": meta.get("atr_pct"),
            "trend_strength": meta.get("trend_strength"),
            "signal_score": meta.get("signal_score"),
            "provider": meta.get("provider"),
            "last_candle_age_seconds": meta.get("last_candle_age_seconds"),
            "market_data_status": meta.get("market_data_status", "OK"),
            "open_position": next((p for p in [_trade_to_dict(t, trader, cfg) for t in db.fetch_open_trades(symbol)]), None),
        }

    skip_by_symbol: dict[str, dict[str, int]] = {symbol: {} for symbol in cfg.symbols}
    for symbol in cfg.symbols:
        reason = state_store.get_decision_meta(symbol).get("skip_reason")
        if reason:
            skip_by_symbol[symbol][str(reason)] = skip_by_symbol[symbol].get(str(reason), 0) + 1

    raw_events = db.fetch_events(limit=200)
    event_tape = [
        {
            "time": item.get("timestamp"),
            "type": item.get("event_type"),
            "correlation_id": item.get("correlation_id"),
            "payload": item.get("payload", {}),
        }
        for item in raw_events
    ]

    return DashboardOverview(
        account={
            "live_equity": equity,
            "starting_equity": starting_equity,
            "balance": float(acct["balance"]),
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "realized_pnl_today": float(acct["pnl_realized_today"]),
            "fees_total": fees_total,
            "fees_today": fees_today,
            "metrics_version": int(acct["metrics_version"]),
            "equity_reconcile_delta": float(acct["equity_reconcile_delta"]),
            "daily_start_equity": float(acct["day_start_equity"]),
            "global_peak_equity": float(acct["equity_high_watermark"]),
            "daily_drawdown_usd": daily_dd_usd,
            "daily_drawdown_pct": daily_dd_pct,
            "global_drawdown_usd": global_dd_usd,
            "global_drawdown_pct": global_dd_pct,
            "max_global_dd_abs": float(acct["max_global_dd_abs"]),
            "max_global_dd_pct": float(acct["max_global_dd_pct"]),
            "trades_today_by_symbol": dict(acct["trades_today_by_symbol"]),
            "realized_pnl_by_symbol": dict(acct["realized_pnl_by_symbol"]),
            "fees_by_symbol": dict(acct["fees_by_symbol"]),
            "open_positions_detail": open_positions,
            "open_orders": open_orders,
            "executions": executions,
            "event_tape": event_tape,
            "status": "PAUSED" if active_reasons else "ACTIVE",
            "pause_reasons": [item.get("reason") for item in active_reasons],
            "engine_running": scheduler.running,
            "last_tick_time": scheduler.last_tick_time().isoformat() if scheduler.last_tick_time() else None,
            "last_tick_age_seconds": (now - scheduler.last_tick_time()).total_seconds() if scheduler.last_tick_time() else None,
            "tick_interval_seconds": scheduler.tick_interval,
        },
        risk={
            "risk_pct_per_trade": float(cfg.base_risk_pct or 0.0),
            "risk_per_trade_usd": float(cfg.risk_per_trade_usd or 0.0),
            "daily_loss_limit_pct": float(cfg.max_daily_loss_pct or 0.0),
            "global_dd_limit_pct": float(cfg.global_drawdown_limit_pct or 0.0),
            "max_trades_per_day": int(cfg.max_trades_per_day or 0),
            "trades_today": trades_today,
            "consecutive_losses": max((int(ds.consecutive_losses) for ds in [state_store.get_daily_state(s) for s in cfg.symbols]), default=0),
            "cooldown_remaining_seconds": max((int((state_store.risk_snapshot(s, cfg, now).get("cooldown_remaining_minutes", 0) or 0) * 60) for s in cfg.symbols), default=0),
        },
        activity={
            "trades_today": trades_today,
            "win_rate_today": perf.win_rate,
            "profit_factor": perf.profit_factor,
            "expectancy": perf.expectancy_r,
            "avg_win": perf.avg_win,
            "avg_loss": perf.avg_loss,
            "fees_today": fees_today,
            "fees_rolling": fees_total,
            "slippage_bps": float(cfg.slippage_bps or 0.0),
            "market_data_errors": state_store.market_data_error_counts(),
        },
        symbols=symbol_data,
        recent_trades=[_trade_to_dict(t, trader, cfg) for t in trades[:100]],
        equity_curve=[{"index": idx, "equity": starting_equity + val} for idx, val in enumerate(perf.equity_curve)],
        skip_reasons={"global": state_store.skip_reason_counts(), "by_symbol": skip_by_symbol},
    )


@app.get("/dashboard/metrics")
async def dashboard_metrics() -> dict[str, Any]:
    cfg = _require_settings()
    state_store = _require_state()
    db = _require_database()
    trader = _require_paper_trader()
    return build_dashboard_metrics(cfg, db, trader, state_store, _runtime_now())
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
    _require_database().log_event("webhook", request.model_dump(), correlation_id)

    plan = decide(request, state_store, settings)
    if plan.status == Status.TRADE:
        state_store.record_trade(request.tradingview.symbol)
    state_store.set_latest_decision(request.tradingview.symbol, plan)

    _require_database().log_event("decision", plan.model_dump(), correlation_id)
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
    start_of_day = _runtime_now().replace(hour=0, minute=0, second=0, microsecond=0)

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



@app.get("/debug/config")
def debug_config() -> dict[str, Any]:
    cfg = _require_settings()
    effective = cfg.resolved_settings()
    for secret_key in ("bybit_api_key", "bybit_api_secret", "telegram_bot_token"):
        if secret_key in effective:
            effective[secret_key] = "***"
    return {
        "effective": effective,
        "sources": cfg.config_sources(),
        "env_keys": cfg.env_alias_map(),
    }


@app.get("/debug/db")
def debug_db() -> dict[str, Any]:
    db = _require_database()
    cfg = _require_settings()
    database_url = cfg.database_url or f"sqlite:///{Path(cfg.data_dir) / 'trades.db'}"
    trades = db.fetch_trades()
    return {
        "database_url": database_url,
        "database_type": "sqlite" if database_url.startswith("sqlite") else "postgresql",
        "connected": True,
        "trade_count": len(trades),
    }
@app.get("/debug/runtime")
async def debug_runtime() -> dict[str, Any]:
    cfg = _require_settings()
    db = _require_database()
    st = _require_state()
    sch = _require_scheduler()
    trader = _require_paper_trader()
    symbols = st.get_symbols() or list(cfg.symbols)

    aliases = cfg.env_alias_map()
    sources = cfg.config_sources()
    resolved = cfg.resolved_settings()
    env_keys = [
        "MODE", "ENGINE_MODE", "RUN_MODE", "STRATEGY_PROFILE", "PROFILE", "SETTINGS_ENABLE_LEGACY", "SYMBOLS",
        "MARKET_DATA_PROVIDER", "MARKET_DATA_REPLAY_PATH", "CANDLE_INTERVAL", "CANDLE_HISTORY_LIMIT", "TICK_INTERVAL_SECONDS", "MARKET_DATA_REPLAY_SPEED",
        "REPLAY_START_TS", "REPLAY_END_TS", "REPLAY_RESUME", "REPLAY_MAX_TRADES", "REPLAY_MAX_BARS", "REPLAY_SEED", "REPLAY_HISTORY_LIMIT",
        "ACCOUNT_SIZE", "PROP_ENABLED", "PROP_GOVERNOR_ENABLED",
        "PROP_RISK_BASE_PCT", "PROP_RISK_MIN_PCT", "PROP_RISK_MAX_PCT",
        "RISK_PER_TRADE_USD", "BASE_RISK_PCT", "DATABASE_URL", "DATA_DIR",
    ]
    env_resolution: dict[str, Any] = {}
    for env_key in env_keys:
        field_name = next((k for k, v in aliases.items() if v == env_key), env_key.lower())
        source = "legacy_alias" if env_key != aliases.get(field_name, env_key) else ("env" if env_key in os.environ else ("computed" if sources.get(field_name) == "profile_default" else "default"))
        env_resolution[env_key] = {
            "raw_env_value": os.environ.get(env_key),
            "parsed_value": getattr(cfg, field_name, resolved.get(field_name)),
            "source": source,
        }

    replay_root = Path(cfg.market_data_replay_path).resolve()
    csv_files = []
    for interval in ("5m", "1h"):
        for symbol in symbols:
            csv_files.append(_scan_csv_bounds(replay_root / symbol / f"{interval}.csv"))

    data_root = Path("/app/data") if Path("/app/data").exists() else Path(cfg.data_dir).resolve()
    replay_resume_files = [
        str(path) for path in data_root.rglob("*")
        if path.is_file() and ("runtime_state" in path.name or "resume" in path.name or "cursor" in path.name or path.name.endswith(".state"))
    ]

    interval = cfg.candle_interval or "5m"
    replay_truth: dict[str, Any] = {}
    if symbols:
        try:
            replay_truth = replay_status(cfg.market_data_replay_path, symbols[0], interval, replay_resume=cfg.replay_resume)
        except ReplayDatasetError as exc:
            replay_truth = {"error": str(exc)}
    replay_provider_module = "app.providers.replay.ReplayProvider"
    start_resolution = replay_truth.get("start_resolution", {}) if isinstance(replay_truth, dict) else {}
    resume_file = replay_truth.get("resume_state_file") if isinstance(replay_truth, dict) else None
    resolved_source = start_resolution.get("source") or "csv_first_ts_fallback"
    resolved_effective_start_ts = start_resolution.get("effective_replay_start_ts") or replay_truth.get("current_ts") or replay_truth.get("first_ts")
    resume_in_use = bool(cfg.replay_resume and resolved_source == "resume_enabled")

    db_url = cfg_db_path(cfg)
    sqlite_path = None
    if db_url.startswith("sqlite:///"):
        sqlite_path = str(Path(db_url.replace("sqlite:///", "")).resolve())
    db_files = [
        {"path": str(path), "size": path.stat().st_size}
        for path in data_root.rglob("*.db") if path.is_file()
    ]
    runtime_rows = db.list_runtime_state()
    gov_row = db.get_runtime_state("prop.governor")
    gov_payload = None
    if gov_row and gov_row.value_text:
        try:
            gov_payload = json.loads(gov_row.value_text)
        except json.JSONDecodeError:
            gov_payload = {"raw": gov_row.value_text}

    symbol_gate = symbols[0] if symbols else "BTCUSDT"
    return {
        "settings": {
            "profile": resolved.get("profile", resolved.get("PROFILE")),
            "min_signal_score": cfg.min_signal_score,
            "run_mode": cfg.run_mode,
        },
        "scheduler": {
            "running": sch.running,
            "last_tick_time": sch.last_tick_time().isoformat() if sch.last_tick_time() else None,
            "tick_interval_seconds": sch.tick_interval,
        },
        "symbols": [
            {
                "symbol": symbol,
                "candles_fetched_count": len((sch.last_snapshot(symbol).candles if sch.last_snapshot(symbol) else []) or []),
            }
            for symbol in symbols
        ],
        "environment_settings_resolution": env_resolution,
        "effective_settings_snapshot": {"canonical": resolved, "aliases": aliases},
        "replay_state": {
            "replay_provider_class": replay_provider_module,
            "replay_path_resolved": str(replay_root),
            "csv_files_found": csv_files,
            "replay_resume_files_found": replay_resume_files,
            "replay_resume_files_in_use": {
                "enabled": cfg.replay_resume,
                "files": [resume_file] if resume_in_use and resume_file else [],
                "selection_logic": start_resolution.get("selection_logic", "REPLAY_START_TS >= first candle > csv_first_ts_fallback"),
            },
            "trade_start_ts": start_resolution.get("trade_start_ts") or replay_truth.get("first_ts"),
            "history_preload_start_ts": start_resolution.get("history_preload_start_ts") or replay_truth.get("first_ts"),
            "warmup_ready_at_ts": start_resolution.get("warmup_ready_at_ts"),
            "warmup_missing_bars": start_resolution.get("warmup_missing_bars", 0),
            "effective_replay_start_ts": resolved_effective_start_ts,
            "why_start_ts_overridden": start_resolution.get("why_start_ts_overridden", resolved_source),
            "replay_pointer_now": {
                "engine_time": _runtime_now().isoformat(),
                "cursor_index": replay_truth.get("bar_index"),
                "last_processed_timestamp": replay_truth.get("current_ts"),
            },
            "truth_precedence": {
                "replay_resume": "resume_enabled > start_ts_used > csv_first_ts_fallback" if cfg.replay_resume else "start_ts_used > csv_first_ts_fallback (resume disabled)",
                "trade_start_ts": resolved_source,
                "warmup": "history_preload_start_ts is used for indicators only; trade_start_ts remains user anchor",
            },
        },
        "database_truth": {
            "database_url_resolved": db_url,
            "sqlite_file_path": sqlite_path,
            "db_files_found": db_files,
            "active_runtime_state_keys": [row.key for row in runtime_rows],
            "prop_governor_state_row": gov_payload,
        },
        "risk_sizing_truth": {
            symbol: ({
                **(trader.last_sizing_decision(symbol) or {}),
                "position_size_usd_cap_source_key": "TradePlan.position_size_usd",
            }) for symbol in symbols
        },
        "strategy_gating_blockers": {
            symbol: st.gate_events(symbol, limit=50) for symbol in symbols
        },
        "dashboard_vs_engine": {
            "dashboard_base_risk_pct": cfg.base_risk_pct,
            "engine_last_risk_pct": (trader.last_sizing_decision(symbol_gate) or {}).get("engine_risk_pct"),
        },
    }


class DebugResetRequest(BaseModel):
    reset_replay_state: bool = False
    reset_governor_state: bool = False
    reset_trades_db: bool = False
    reset_performance: bool = False
    dry_run: bool = False


@app.post("/debug/reset")
async def debug_reset(payload: DebugResetRequest) -> dict[str, Any]:
    cfg = _require_settings()
    db = _require_database()
    data_root = Path("/app/data") if Path("/app/data").exists() else Path(cfg.data_dir).resolve()
    replay_root = Path(cfg.market_data_replay_path).resolve()

    replay_targets = sorted({
        data_root / "replay" / "replay_runtime_state.json",
        replay_root / "replay_runtime_state.json",
        *[
            path for path in data_root.rglob("*")
            if path.is_file() and (path.name.endswith(".state") or "cursor" in path.name or "runtime_state" in path.name or "resume" in path.name)
        ],
    }, key=lambda p: str(p))
    deleted: dict[str, Any] = {"files": [], "db_rows": {}, "in_memory": {}}

    if payload.reset_replay_state:
        deleted["in_memory"]["replay"] = replay_reset_all_state()
        for path in replay_targets:
            existed = path.exists()
            deleted["files"].append({"path": str(path.resolve()), "existed": existed})
            if not payload.dry_run:
                path.unlink(missing_ok=True)

    if payload.reset_performance:
        perf_targets = [Path(cfg.data_dir) / "performance.json", data_root / "performance.json"]
        for perf in perf_targets:
            existed = perf.exists()
            deleted["files"].append({"path": str(perf.resolve()), "existed": existed})
            if not payload.dry_run:
                perf.unlink(missing_ok=True)

    if payload.reset_governor_state:
        keys = [row.key for row in db.list_runtime_state() if row.key.startswith("prop.governor") or row.key.startswith("prop.")]
        deleted["db_rows"]["governor_keys"] = keys
        if not payload.dry_run:
            db.delete_runtime_state_keys(keys)
            _require_prop_governor().reset(_runtime_now())

    if payload.reset_trades_db:
        db_url = cfg_db_path(cfg)
        if db_url.startswith("sqlite:///"):
            sqlite_path = Path(db_url.replace("sqlite:///", "")).resolve()
            deleted["files"].append({"path": str(sqlite_path), "existed": sqlite_path.exists()})
            if not payload.dry_run and sqlite_path.exists():
                sqlite_path.unlink()
                db.init_schema()
        else:
            deleted["db_rows"]["trades_reset"] = True
            if not payload.dry_run:
                db.reset_trades()

    return {"dry_run": payload.dry_run, "deleted": deleted, "replay_csvs_deleted": []}


@app.get("/debug/kline")
async def debug_kline(interval: str = Query("5m")) -> dict[str, Any]:
    settings = _require_settings()
    symbols = ["BTCUSDT", "ETHUSDT"]
    output: list[dict[str, Any]] = []
    scheduler = _require_scheduler()
    for symbol in symbols:
        snapshot = None
        provider = "none"
        error: str | None = None
        try:
            snapshot = await fetch_symbol_klines(
                symbol=symbol,
                interval=interval,
                limit=3,
                rest_base=settings.bybit_rest_base,
                provider=settings.market_data_provider,
                fallback_provider=settings.market_data_fallbacks,
                failover_threshold=settings.market_data_failover_threshold,
                backoff_base_ms=settings.market_data_backoff_base_ms,
                backoff_max_ms=settings.market_data_backoff_max_ms,
                replay_path=settings.market_data_replay_path,
                replay_speed=settings.market_data_replay_speed,
                replay_start_ts=settings.replay_start_ts,
                replay_end_ts=settings.replay_end_ts,
                replay_history_limit=getattr(settings, "replay_history_limit", None),
                replay_resume=settings.replay_resume,
            )
        except Exception as exc:
            error = type(exc).__name__
            cached = scheduler.last_snapshot(symbol)
            if cached is not None:
                snapshot = cached
                provider = f"cached:{cached.provider_name}"
        if snapshot is None:
            output.append({"symbol": symbol, "provider": provider, "rows": [], "error": error})
            continue
        rows = [
            {
                "open_time": candle.open_time.isoformat(),
                "close_time": candle.close_time.isoformat(),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in snapshot.candles[-3:]
        ]
        output.append({"symbol": symbol, "provider": provider if provider != "none" else getattr(snapshot, "provider_name", "bybit"), "rows": rows, "error": error})
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
    _require_database().log_event("decision", plan.model_dump(), correlation_id)
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
    _require_database().log_event("decision", plan.model_dump(), correlation_id)
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


@app.post("/storage/reset")
@app.post("/debug/storage/reset")
async def debug_storage_reset() -> dict:
    payload = await challenge_reset()
    settings = _require_settings()
    logs_dir = Path(settings.data_dir) / "logs"
    if logs_dir.exists():
        for path in logs_dir.glob("*.jsonl"):
            path.unlink()
    return payload


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

    now = _runtime_now()
    closed_today = 0
    for trade in _require_database().fetch_trades():
        closed_at = _as_datetime_utc(getattr(trade, "closed_at", None))
        if closed_at is not None and closed_at.date() == now.date() and str(getattr(trade, "symbol", "")).upper() == symbol.upper():
            closed_today += 1
    risk_ok, risk_reason = _require_state().risk_check(symbol, settings, now, trades_today_closed=closed_today)
    if not risk_ok:
        _require_state().record_skip_reason(risk_reason)
        return False, None

    trade_id = paper_trader.maybe_open_trade(symbol, plan)
    if trade_id is None:
        return False, None
    _require_state().record_trade(symbol)
    return True, trade_id


def _hard_risk_gate_reasons(symbol: str, now: datetime) -> list[str]:
    settings = _require_settings()
    state_store = _require_state()
    reasons: list[str] = []
    for start_t, end_t in settings.blackout_windows():
        if start_t <= now.time() <= end_t:
            reasons.append("news_blackout")
            break

    closed_today = 0
    for trade in _require_database().fetch_trades():
        closed_at = _as_datetime_utc(getattr(trade, "closed_at", None))
        if closed_at is not None and closed_at.date() == now.date() and str(getattr(trade, "symbol", "")).upper() == symbol.upper():
            closed_today += 1

    risk_ok, risk_reason = state_store.risk_check(symbol, settings, now, trades_today_closed=closed_today)
    if not risk_ok and risk_reason:
        reasons.append(risk_reason)

    daily_state = state_store.get_daily_state(symbol)
    if settings.max_losses_per_day and daily_state.losses >= settings.max_losses_per_day:
        reasons.append("max_losses_per_day")
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
        "daily_loss_limit_hit",
        "global_dd_limit_hit",
        "max_losses_per_day",
        "max_trades_exceeded",
        "cooldown",
        "max_consecutive_losses",
    }
    return any(reason in hard_gates for reason in rationale)
