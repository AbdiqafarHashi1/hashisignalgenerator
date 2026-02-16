from __future__ import annotations

import asyncio
import logging
import random
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone

from ..config import Settings
from ..models import (
    BiasSignal,
    DecisionRequest,
    Direction,
    MarketSnapshot,
    Posture,
    SetupType,
    Status,
    TradePlan,
)
from ..state import StateStore
from ..strategy.decision import decide
from ..providers.bybit import BybitKlineSnapshot, fetch_symbol_klines
from ..utils.intervals import interval_to_ms
from .notifier import format_trade_message, send_telegram_message

logger = logging.getLogger(__name__)

# ---- JSON SAFE HELPERS ----
import json
from datetime import date
from decimal import Decimal
from enum import Enum

def _jsonable(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)

def _to_json_dict(x):
    return json.loads(json.dumps(x, default=_jsonable))
# ---------------------------

class DecisionScheduler:
    def __init__(
        self,
        settings: Settings,
        state: StateStore,
        database=None,
        paper_trader=None,
        interval_seconds: int = 60,
        heartbeat_cb: Callable[[], None] | None = None,
    ) -> None:
        self._settings = settings
        self._state = state
        self._database = database
        self._paper_trader = paper_trader
        self._interval = interval_seconds
        if settings.smoke_test_force_trade or settings.force_trade_mode:
            self._interval = min(self._interval, settings.force_trade_every_seconds)
            if interval_seconds > settings.force_trade_every_seconds:
                logger.warning(
                    "force_trade_interval_adjusted tick_interval=%s force_every=%s",
                    interval_seconds,
                    settings.force_trade_every_seconds,
                )
        self._heartbeat_cb = heartbeat_cb
        self._last_tick_time: datetime | None = None
        self.last_tick_ts: float | None = None
        self.started_ts: float | None = None
        self._last_snapshots: dict[str, BybitKlineSnapshot] = {}
        self._last_fetch_counts: dict[str, int] = {}
        self._last_symbol_tick_time: dict[str, datetime] = {}
        self._last_force_trade_ts: dict[str, datetime] = {}
        self._last_exit_eval_close_ms: dict[str, int] = {}
        self._last_skip_telegram_ts: dict[str, datetime] = {}
        self._next_fetch_after_ms: dict[str, int] = {}
        self._tick_listeners: list[Callable[[], None]] = []

        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._stop_reason: str | None = None
        self._consecutive_failures = 0
        self._stopping_requested = False

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def tick_interval(self) -> int:
        return self._interval

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def last_tick_time(self) -> datetime | None:
        return self._last_tick_time

    def last_snapshot(self, symbol: str) -> BybitKlineSnapshot | None:
        return self._last_snapshots.get(symbol)

    def last_fetch_count(self, symbol: str) -> int | None:
        return self._last_fetch_counts.get(symbol)

    def last_symbol_tick_time(self, symbol: str) -> datetime | None:
        return self._last_symbol_tick_time.get(symbol)

    def add_tick_listener(self, listener: Callable[[], None]) -> None:
        self._tick_listeners.append(listener)

    def _notify_tick_listeners(self) -> None:
        for listener in self._tick_listeners:
            try:
                listener()
            except Exception:
                logger.exception("scheduler_tick_listener_error")

    async def start(self) -> bool:
        async with self._lock:
            if self.running:
                logger.info("scheduler_start skipped=already_running")
                return False
            self._stop_event = asyncio.Event()
            self._stopping_requested = False
            self._stop_reason = None
            self._consecutive_failures = 0
            self.started_ts = time.time()
            self._task = asyncio.create_task(self._run_loop(), name="decision_scheduler")
            self._task.add_done_callback(self._on_task_done)
        await asyncio.sleep(0)
        if self._task is None or self._task.done():
            error = self._task.exception() if self._task is not None else RuntimeError("scheduler_task_not_created")
            reason = f"{type(error).__name__}: {error}" if error is not None else "scheduler_task_exited_early"
            self._stop_reason = reason
            logger.error("scheduler_start status=failed reason=%s", reason)
            return False
        async with self._lock:
            logger.info("scheduler_start status=started interval=%s", self._interval)
            return True

    async def stop(self) -> bool:
        async with self._lock:
            if not self.running:
                logger.info("scheduler_stop skipped=already_stopped")
                return False
            self._stopping_requested = True
            self._stop_reason = "stopped_by_user"
            self._stop_event.set()
            task = self._task
        if task is not None:
            await task
        self._task = None
        logger.info("scheduler_stop status=stopped")
        return True

    def _closed_trades_today_by_symbol(self, now: datetime) -> dict[str, int]:
        if self._database is None:
            return {}
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        counts: dict[str, int] = {}
        for trade in self._database.fetch_trades():
            closed_at = getattr(trade, "closed_at", None)
            if closed_at is None:
                continue
            closed_dt = closed_at if isinstance(closed_at, datetime) else datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
            if closed_dt.tzinfo is None:
                closed_dt = closed_dt.replace(tzinfo=timezone.utc)
            if closed_dt < start_of_day:
                continue
            symbol = str(getattr(trade, "symbol", "") or "")
            if not symbol:
                continue
            counts[symbol] = counts.get(symbol, 0) + 1
        return counts

    def _risk_gate_reason(self, symbol: str, now: datetime, closed_trades_today: dict[str, int]) -> str | None:
        closed_count = closed_trades_today.get(symbol)
        allowed, reason = self._state.risk_check(symbol, self._settings, now, trades_today_closed=closed_count)
        return None if allowed else reason

    async def run_once(self, force: bool = False) -> list[dict[str, object]]:
        self.last_tick_ts = time.time()
        if self._heartbeat_cb is not None:
            self._heartbeat_cb()
        symbols = list(self._settings.symbols)
        self._last_tick_time = datetime.now(timezone.utc)
        force_mode = force or self._settings.smoke_test_force_trade or self._settings.force_trade_mode
        dedupe_bypass = force
        logger.info("scheduler_tick_start symbols=%s force=%s", ",".join(symbols), force_mode)
        if force_mode:
            logger.info("FORCE MODE ACTIVE")
            logger.info(
                "force_tick ts=%s force=%s symbols=%s",
                self._last_tick_time.isoformat(),
                force_mode,
                ",".join(symbols),
            )
        results: list[dict[str, object]] = []
        if not self._settings.market_data_enabled:
            logger.warning("scheduler_tick_skipped reason=market_data_disabled")
            for symbol in symbols:
                results.append(
                    {
                        "symbol": symbol,
                        "plan": None,
                        "reason": "market_data_disabled",
                        "candles_fetched": 0,
                        "latest_candle_ts": None,
                        "decision_status": None,
                        "persisted": False,
                        "dedupe_key": None,
                        "telegram_sent": False,
                        "trade_opened": False,
                        "trade_id": None,
                        "decision": "skip",
                        "skip_reason": "market_data_disabled",
                    }
                )
            for _ in symbols:
                self._state.record_skip_reason("market_data_disabled")
            self._notify_tick_listeners()
            return results
        closed_trades_today = self._closed_trades_today_by_symbol(self._last_tick_time)
        for symbol in symbols:
            tick_ts = datetime.now(timezone.utc)
            self._last_symbol_tick_time[symbol] = tick_ts
            gate_reason = self._risk_gate_reason(symbol, tick_ts, closed_trades_today)
            if gate_reason:
                self._state.set_decision_meta(symbol, {"decision": "skip", "skip_reason": gate_reason, "final_entry_gate": gate_reason})
                self._state.record_skip_reason(gate_reason)
                results.append(
                    {
                        "symbol": symbol,
                        "plan": None,
                        "reason": gate_reason,
                        "candles_fetched": 0,
                        "latest_candle_ts": None,
                        "decision_status": None,
                        "persisted": False,
                        "dedupe_key": None,
                        "telegram_sent": False,
                        "trade_opened": False,
                        "trade_id": None,
                        "decision": "skip",
                        "skip_reason": gate_reason,
                    }
                )
                continue
            try:
                snapshot = self._last_snapshots.get(symbol)
                now_ms = int(tick_ts.timestamp() * 1000)
                next_fetch_after_ms = self._next_fetch_after_ms.get(symbol, 0)
                should_refresh = snapshot is None or force_mode or now_ms >= next_fetch_after_ms
                if should_refresh:
                    snapshot = await fetch_symbol_klines(
                        symbol=symbol,
                        interval=self._settings.candle_interval,
                        limit=self._settings.candle_history_limit,
                        rest_base=self._settings.bybit_rest_base,
                        provider=self._settings.market_data_provider,
                        fallback_provider=self._settings.market_data_fallbacks,
                        failover_threshold=self._settings.market_data_failover_threshold,
                        backoff_base_ms=self._settings.market_data_backoff_base_ms,
                        backoff_max_ms=self._settings.market_data_backoff_max_ms,
                        replay_path=self._settings.market_data_replay_path,
                        replay_speed=self._settings.market_data_replay_speed,
                    )
                if snapshot is None:
                    logger.info("candle_fetch symbol=%s status=not_ready reason=candle_open", symbol)
                    self._last_fetch_counts[symbol] = 0
                    results.append(
                        {
                            "symbol": symbol,
                            "plan": None,
                            "reason": "candle_open",
                            "candles_fetched": 0,
                            "latest_candle_ts": None,
                            "decision_status": None,
                            "persisted": False,
                            "dedupe_key": None,
                            "telegram_sent": False,
                            "trade_opened": False,
                            "trade_id": None,
                            "decision": "skip",
                            "skip_reason": "candle_open",
                        }
                    )
                    self._state.set_decision_meta(symbol, {"decision": "skip", "skip_reason": "candle_open"})
                    self._state.record_skip_reason("candle_open")
                    continue
                last_candle_age_seconds = max(0.0, (now_ms - snapshot.kline_close_time_ms) / 1000.0)
                stale_gate_enabled = snapshot.kline_close_time_ms >= 1_600_000_000_000
                stale_threshold_seconds = (interval_to_ms(snapshot.interval) / 1000.0) + float(self._settings.market_data_allow_stale or 0)
                if stale_gate_enabled and last_candle_age_seconds > stale_threshold_seconds:
                    stale_reason = "MARKET_DATA_STALE"
                    self._state.record_market_data_error(stale_reason)
                    self._state.set_decision_meta(
                        symbol,
                        {
                            "decision": "skip",
                            "skip_reason": stale_reason,
                            "final_entry_gate": stale_reason,
                            "provider": getattr(snapshot, "provider_name", "bybit"),
                            "last_candle_age_seconds": last_candle_age_seconds,
                            "market_data_status": "STALE",
                        },
                    )
                    self._state.record_skip_reason(stale_reason)
                    results.append(
                        {
                            "symbol": symbol,
                            "plan": None,
                            "reason": stale_reason,
                            "candles_fetched": len(snapshot.candles),
                            "latest_candle_ts": snapshot.candle.close_time.isoformat(),
                            "decision_status": None,
                            "persisted": False,
                            "dedupe_key": None,
                            "telegram_sent": False,
                            "trade_opened": False,
                            "trade_id": None,
                            "decision": "skip",
                            "skip_reason": stale_reason,
                        }
                    )
                    continue
                self._state.set_decision_meta(symbol, {
                    **self._state.get_decision_meta(symbol),
                    "provider": getattr(snapshot, "provider_name", "bybit"),
                    "last_candle_age_seconds": last_candle_age_seconds,
                    "market_data_status": "OK",
                })
                self._last_snapshots[symbol] = snapshot
                self._last_fetch_counts[symbol] = len(snapshot.candles)
                next_refresh = snapshot.kline_close_time_ms if not snapshot.kline_is_closed else snapshot.kline_close_time_ms + interval_to_ms(snapshot.interval)
                self._next_fetch_after_ms[symbol] = next_refresh
                if self._settings.force_trade_mode or self._settings.smoke_test_force_trade:
                    self._auto_close_forced_trades(symbol, snapshot, tick_ts)
                if self._paper_trader is not None:
                    self._paper_trader.update_mark_price(symbol, snapshot.candle.close)
                    last_eval_close = self._last_exit_eval_close_ms.get(symbol)
                    if last_eval_close != snapshot.kline_close_time_ms:
                        self._paper_trader.evaluate_open_trades(
                            symbol,
                            snapshot.candle.close,
                            candle_high=snapshot.candle.high,
                            candle_low=snapshot.candle.low,
                        )
                        self._last_exit_eval_close_ms[symbol] = snapshot.kline_close_time_ms
                    self._paper_trader.adjust_stop_dynamic(symbol, trigger_r=self._settings.risk_reduction_trigger_r, target_r=self._settings.risk_reduction_target_r)
                    self._close_time_stop_trades(symbol, snapshot.candle.close, tick_ts)
                active_mode = self.detect_regime(snapshot)
                if self._settings.sweet8_enabled:
                    self._settings.sweet8_current_mode = active_mode
                    self._state.set_last_notified_key("__sweet8_current_mode__", active_mode)
                logger.info(
                    "candle_fetch symbol=%s candles=%s latest=%s closed=%s",
                    snapshot.symbol,
                    len(snapshot.candles),
                    snapshot.candle.close_time.isoformat(),
                    snapshot.kline_is_closed,
                )
                logger.info(
                    "candle_selected symbol=%s start=%s end=%s now=%s closed=%s",
                    snapshot.symbol,
                    datetime.fromtimestamp(snapshot.kline_open_time_ms / 1000, tz=timezone.utc).isoformat(),
                    datetime.fromtimestamp(snapshot.kline_close_time_ms / 1000, tz=timezone.utc).isoformat(),
                    tick_ts.isoformat(),
                    snapshot.kline_is_closed,
                )
            except Exception as exc:
                logger.exception("scheduler_symbol_error symbol=%s error=%s", symbol, exc)
                error_reason = f"symbol_error:{type(exc).__name__}"
                blocked_tokens = {"HTTP_401", "HTTP_403", "HTTP_418", "HTTP_429", "BYBIT_RATE_LIMIT_10006", "MARKET_DATA_BLOCKED", "ProxyError", "403 Forbidden"}
                if any(token in str(exc) for token in blocked_tokens) or type(exc).__name__ in {"ProxyError", "HTTPStatusError", "MarketDataBlockedError", "BybitRateLimitError"}:
                    error_reason = "MARKET_DATA_BLOCKED"
                self._state.record_market_data_error(error_reason)
                results.append(
                    {
                        "symbol": symbol,
                        "plan": None,
                        "reason": error_reason,
                        "candles_fetched": 0,
                        "latest_candle_ts": None,
                        "decision_status": None,
                        "persisted": False,
                        "dedupe_key": None,
                        "telegram_sent": False,
                        "trade_opened": False,
                        "trade_id": None,
                        "decision": "skip",
                        "skip_reason": error_reason,
                    }
                )
                symbol_error_reason = error_reason
                self._state.set_decision_meta(
                    symbol,
                    {
                        "decision": "skip",
                        "skip_reason": symbol_error_reason,
                        "final_entry_gate": symbol_error_reason,
                        "market_data_status": "BLOCKED",
                    },
                )
                self._state.record_skip_reason(error_reason)
                continue

            funding_state = _funding_blackout_state(tick_ts, self._settings)
            if self._paper_trader is not None and self._settings.funding_blackout_force_close:
                util_pct = self._paper_trader.margin_utilization_pct()
                unrealized = self._paper_trader.symbol_unrealized_pnl_usd(symbol)
                should_force_close = (
                    funding_state["close_positions"]
                    and (
                        util_pct >= self._settings.funding_blackout_max_util_pct
                        or unrealized <= -abs(self._settings.funding_blackout_max_loss_usd)
                    )
                )
                if should_force_close:
                    self._paper_trader.force_close_trades(symbol, snapshot.candle.close, reason="funding_blackout_close")
                    logger.warning("funding_blackout_forced_close symbol=%s", symbol)

            plan: TradePlan | None = None
            reason: str | None = None
            persisted = False
            decision_status: str | None = None
            dedupe_key: str | None = None
            telegram_sent = False
            trade_opened = False
            trade_id: str | None = None
            decision = "skip"
            skip_reason: str | None = None
            scalp_meta: dict[str, object] = {}
            latest_closed_ms = snapshot.kline_close_time_ms
            if not snapshot.kline_is_closed and snapshot.candles:
                latest_closed_ms = int(snapshot.candles[-1].close_time.timestamp() * 1000)
            last_processed = self._state.get_last_processed_close_time_ms(snapshot.symbol)
            should_process = bool(force_mode or dedupe_bypass or (latest_closed_ms and latest_closed_ms != last_processed))

            if not snapshot.kline_is_closed and not force_mode:
                reason = "candle_open"
                skip_reason = "candle_open"
                should_process = False
            elif funding_state["block_new_entries"]:
                reason = "funding_blackout_entries_blocked"
                skip_reason = "funding_blackout_entries_blocked"
                should_process = False
            elif not should_process:
                reason = "candle_already_processed" if latest_closed_ms == last_processed else "candle_open"
                skip_reason = reason
            else:
                forced_trade = False
                if (self._settings.force_trade_mode or self._settings.smoke_test_force_trade) and self._force_trade_due(snapshot.symbol, tick_ts):
                    plan = _build_forced_trade_plan(snapshot, self._settings)
                    self._last_force_trade_ts[snapshot.symbol] = tick_ts
                    forced_trade = True
                else:
                    request = _build_decision_request(snapshot)
                    settings_for_decision = self._settings
                    if self._settings.sweet8_enabled:
                        settings_for_decision = self._settings.model_copy(
                            update={"strategy": "scalper" if active_mode == "scalper" else "baseline"}
                        )
                    plan = decide(request, self._state, settings_for_decision)
                if plan.status == Status.TRADE:
                    plan, mode_skip_reason, scalp_meta = _apply_mode_overrides(plan, snapshot, self._settings)
                    if plan.status != Status.TRADE:
                        reason = mode_skip_reason or "setup_not_confirmed"
                        skip_reason = mode_skip_reason or "setup_not_confirmed"
                if (
                    plan.status == Status.NO_TRADE
                    and self._paper_trader is not None
                    and (plan.signal_score is not None and plan.signal_score < self._settings.exit_score_min)
                ):
                    self._paper_trader.force_close_trades(snapshot.symbol, snapshot.candle.close, reason="weakness_exit")
                decision_status = plan.status.value
                self._state.set_latest_decision(snapshot.symbol, plan)
                persisted = True
                if forced_trade:
                    logger.info(
                        "force_trade_decision symbol=%s status=%s rationale=%s",
                        snapshot.symbol,
                        plan.status.value,
                        ",".join(plan.rationale),
                    )
                else:
                    logger.info(
                        "decision_computed symbol=%s status=%s rationale=%s",
                        snapshot.symbol,
                        plan.status.value,
                        ",".join(plan.rationale),
                    )
                if self._database is not None:
                    self._database.log_event(
                        "signal_generated",
                        {"symbol": snapshot.symbol, "status": plan.status.value, "rationale": list(plan.rationale)},
                        f"tick:{int(tick_ts.timestamp())}:{snapshot.symbol}",
                    )
                    entry = None
                    if plan.entry_zone is not None:
                        entry = sum(plan.entry_zone) / 2.0
                    self._database.add_signal(
                        timestamp=datetime.now(timezone.utc),
                        symbol=snapshot.symbol,
                        score=plan.signal_score,
                        status=plan.status.value,
                        rationale=",".join(plan.rationale),
                        entry=entry,
                        stop=plan.stop_loss,
                        take_profit=plan.take_profit,
                        decision=("enter_long" if plan.status == Status.TRADE and plan.direction == Direction.long else "enter_short" if plan.status == Status.TRADE else "skip"),
                        skip_reason=_plan_skip_reason(plan) if plan.status != Status.TRADE else None,
                        regime=active_mode,
                        scores={"signal_score": plan.signal_score},
                        inputs_snapshot=_to_json_dict(plan.raw_input_snapshot),
                    )
                self._state.set_last_processed_close_time_ms(snapshot.symbol, latest_closed_ms)
                if self._database is not None:
                    self._database.set_runtime_state(
                        key=f"last_processed_candle:{snapshot.symbol}",
                        value_number=float(latest_closed_ms),
                        symbol=snapshot.symbol,
                    )
                if plan.status == Status.TRADE:
                    decision = "enter_long" if plan.direction == Direction.long else "enter_short"
                    if forced_trade:
                        dedupe_key = _force_trade_key(snapshot.symbol, plan, tick_ts)
                    else:
                        dedupe_key = _trade_key(snapshot, plan)
                    last_trade_key = self._state.get_last_trade_key(snapshot.symbol)
                    if dedupe_key != last_trade_key:
                        message = format_trade_message(snapshot.symbol, plan, snapshot)
                        telegram_sent = await send_telegram_message(message, self._settings)
                        if telegram_sent:
                            logger.info(
                                "telegram_sent symbol=%s dedupe_key=%s",
                                snapshot.symbol,
                                dedupe_key,
                            )
                        if self._settings.engine_mode in {"paper", "live"} and self._paper_trader is not None:
                            allow_multiple = (
                                (self._settings.force_trade_mode or self._settings.smoke_test_force_trade)
                                and self._settings.force_trade_auto_close_seconds == 0
                                and self._settings.current_mode != "SCALP"
                            )
                            self._database.log_event(
                                "order_sent",
                                {"symbol": snapshot.symbol, "side": ("long" if plan.direction == Direction.long else "short")},
                                f"tick:{int(tick_ts.timestamp())}:{snapshot.symbol}",
                            ) if self._database is not None else None
                            trade_id = self._paper_trader.maybe_open_trade(
                                snapshot.symbol,
                                plan,
                                allow_multiple=allow_multiple,
                                snapshot=snapshot,
                                regime=active_mode,
                            )
                            if trade_id is not None:
                                self._database.log_event(
                                    "order_accepted",
                                    {"symbol": snapshot.symbol, "trade_id": trade_id},
                                    f"tick:{int(tick_ts.timestamp())}:{snapshot.symbol}",
                                ) if self._database is not None else None
                                trade_opened = True
                                self._state.record_trade(snapshot.symbol)
                                logger.info(
                                    "paper_trade_created id=%s symbol=%s dedupe_key=%s",
                                    trade_id,
                                    snapshot.symbol,
                                    dedupe_key,
                                )
                        self._state.set_last_trade_key(snapshot.symbol, dedupe_key)
                    else:
                        logger.info(
                            "trade_deduped symbol=%s dedupe_key=%s",
                            snapshot.symbol,
                            dedupe_key,
                        )
                if plan is not None and decision_status is None:
                    decision_status = plan.status.value
                if plan is not None and plan.status != Status.TRADE and skip_reason is None:
                    skip_reason = _plan_skip_reason(plan)
            outcome = "waiting"
            reasons: list[str] = []
            if plan is not None:
                if plan.status == Status.TRADE:
                    outcome = "ENTER"
                elif plan.status == Status.NO_TRADE:
                    outcome = "waiting"
                else:
                    outcome = plan.status.value
                reasons = list(plan.rationale)
            elif reason is not None:
                reasons = [reason]
            logger.info(
                "scheduler_tick symbol=%s candles=%s latest=%s outcome=%s reasons=%s persisted=%s",
                symbol,
                len(snapshot.candles),
                snapshot.candle.close_time.isoformat(),
                outcome,
                ",".join(reasons),
                persisted,
            )
            debug_reason = skip_reason if skip_reason is not None else ("ready" if should_process else "candle_open")
            self._state.set_decision_meta(
                snapshot.symbol,
                {
                    **self._state.get_decision_meta(snapshot.symbol),
                    "latest_candle_ts": snapshot.kline_close_time_ms,
                    "last_processed_candle_ts": last_processed,
                    "should_process": should_process,
                    "should_process_reason": debug_reason,
                },
            )
            if self._database is not None:
                self._database.set_runtime_state(key=f"latest_candle:{snapshot.symbol}", value_number=float(snapshot.kline_close_time_ms), symbol=snapshot.symbol)

            if skip_reason is None and decision == "skip":
                skip_reason = reason
            final_entry_gate = skip_reason if decision == "skip" else None
            decision_meta = {
                **self._state.get_decision_meta(symbol),
                "decision": decision,
                "skip_reason": skip_reason,
                "final_entry_gate": final_entry_gate,
                "regime_label": scalp_meta.get("regime_label"),
                "allowed_side": scalp_meta.get("allowed_side"),
                "atr_pct": scalp_meta.get("atr_pct"),
                "ema_fast": scalp_meta.get("ema_fast"),
                "ema_slow": scalp_meta.get("ema_slow"),
                "ema_trend": scalp_meta.get("ema_trend"),
                "signal_score": (plan.signal_score if plan is not None else None),
                "trend_strength": (
                    plan.raw_input_snapshot.get("market", {}).get("trend_strength")
                    if plan is not None and isinstance(plan.raw_input_snapshot, dict)
                    else None
                ),
                "provider": getattr(snapshot, "provider_name", "bybit"),
                "last_candle_age_seconds": max(0.0, (now_ms - snapshot.kline_close_time_ms) / 1000.0),
                "market_data_status": "OK",
            }
            self._state.set_decision_meta(symbol, decision_meta)
            if decision == "skip":
                self._state.record_skip_reason(skip_reason)
            if self._settings.telegram_debug_skips and decision == "skip":
                await self._maybe_send_skip_debug(symbol, skip_reason)
            logger.info(
                "tick symbol=%s ts=%s decision=%s skip_reason=%s",
                symbol,
                tick_ts.isoformat(),
                decision,
                skip_reason,
            )
            results.append(
                {
                    "symbol": symbol,
                    "plan": plan,
                    "reason": reason,
                    "candles_fetched": len(snapshot.candles),
                    "latest_candle_ts": snapshot.candle.close_time.isoformat(),
                    "decision_status": decision_status,
                    "persisted": persisted,
                    "dedupe_key": dedupe_key,
                    "telegram_sent": telegram_sent,
                    "trade_opened": trade_opened,
                    "trade_id": trade_id,
                    "decision": decision,
                    "skip_reason": skip_reason,
                    "final_entry_gate": final_entry_gate,
                    "regime_label": decision_meta.get("regime_label"),
                    "allowed_side": decision_meta.get("allowed_side"),
                    "atr_pct": decision_meta.get("atr_pct"),
                }
            )
        self._notify_tick_listeners()
        return results


    async def _maybe_send_skip_debug(self, symbol: str, skip_reason: str | None) -> None:
        if not skip_reason:
            return
        now = datetime.now(timezone.utc)
        last_sent = self._last_skip_telegram_ts.get(symbol)
        if last_sent is not None and (now - last_sent).total_seconds() < 600:
            return
        message = f"⏭️ Skip\nSymbol: {symbol}\nReason: {skip_reason}"
        sent = await send_telegram_message(message, self._settings)
        if sent:
            self._last_skip_telegram_ts[symbol] = now

    def _force_trade_due(self, symbol: str, now: datetime) -> bool:
        last_forced = self._last_force_trade_ts.get(symbol)
        if last_forced is None:
            return True
        elapsed = (now - last_forced).total_seconds()
        return elapsed >= max(self._settings.force_trade_every_seconds, self._settings.force_trade_cooldown_seconds)

    def _auto_close_forced_trades(
        self,
        symbol: str,
        snapshot: BybitKlineSnapshot,
        now: datetime,
    ) -> None:
        if self._settings.sweet8_enabled:
            return
        if self._settings.force_trade_auto_close_seconds <= 0:
            return
        if self._database is None or self._paper_trader is None:
            return
        open_trades = self._database.fetch_open_trades(symbol)
        if not open_trades:
            return
        for trade in open_trades:
            opened_at = datetime.fromisoformat(trade.opened_at)
            elapsed = (now - opened_at).total_seconds()
            if elapsed >= self._settings.force_trade_auto_close_seconds:
                self._paper_trader.force_close_trades(symbol, snapshot.candle.close, reason="force_trade_auto_close")
                logger.info(
                    "force_trade_auto_close symbol=%s elapsed=%.2fs",
                    symbol,
                    elapsed,
                )

    def _close_time_stop_trades(self, symbol: str, price: float, now: datetime) -> None:
        if self._settings.sweet8_enabled:
            return
        if self._database is None or self._paper_trader is None:
            return
        hold_minutes = self._settings.scalp_max_hold_minutes if self._settings.current_mode == "SCALP" else self._settings.max_hold_minutes
        if hold_minutes <= 0:
            return
        for trade in self._database.fetch_open_trades(symbol):
            opened_at = datetime.fromisoformat(trade.opened_at)
            elapsed_minutes = (now - opened_at).total_seconds() / 60.0
            if elapsed_minutes >= hold_minutes:
                self._paper_trader.force_close_trades(symbol, price, reason="time_stop_close")

    def detect_regime(self, snapshot: BybitKlineSnapshot) -> str:
        if self._settings.sweet8_mode != "auto":
            return self._settings.sweet8_mode
        adx = _compute_adx(snapshot, period=max(2, self._settings.adx_period))
        atr_values = _compute_atr_values(snapshot, period=max(2, self._settings.atr_period))
        if adx is None or len(atr_values) < 2:
            return "swing"
        atr = atr_values[-1]
        atr_avg = sum(atr_values[:-1]) / max(1, len(atr_values) - 1)
        vol_ratio = (atr / atr_avg) if atr_avg > 0 else 0.0
        if adx >= float(self._settings.sweet8_regime_adx_threshold) and vol_ratio >= self._settings.sweet8_regime_vol_threshold:
            return "scalper"
        return "swing"

    async def _run_loop(self) -> None:
        logger.info("scheduler_loop_start")
        while not self._stop_event.is_set():
            logger.info("scheduler_loop_tick_start")
            try:
                await self.run_once()
                self._consecutive_failures = 0
                logger.info("scheduler_loop_tick_end status=ok")
            except Exception as exc:
                self._consecutive_failures += 1
                backoff_seconds = self._compute_backoff_seconds(self._consecutive_failures)
                self._stop_reason = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "scheduler_loop_tick_end status=error failures=%s backoff=%ss reason=%s",
                    self._consecutive_failures,
                    backoff_seconds,
                    self._stop_reason,
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff_seconds)
                except asyncio.TimeoutError:
                    continue
                break
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue
        logger.info("scheduler_loop_end stop_requested=%s reason=%s", self._stopping_requested, self._stop_reason)

    def _compute_backoff_seconds(self, failures: int) -> int:
        schedule = [1, 2, 5, 10, 20, 30]
        return schedule[min(max(failures - 1, 0), len(schedule) - 1)]

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            if self._stop_reason is None:
                self._stop_reason = "scheduler_task_cancelled"
            logger.warning("scheduler_task_done status=cancelled reason=%s", self._stop_reason)
            self._task = None
            return

        error = task.exception()
        if error is None:
            logger.info("scheduler_task_done status=completed stop_requested=%s", self._stopping_requested)
            if self._stopping_requested:
                self._task = None
                return
            self._stop_reason = self._stop_reason or "scheduler_task_completed_unexpectedly"
            logger.warning("scheduler_task_done status=unexpected_completion reason=%s", self._stop_reason)
        else:
            formatted_tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            self._stop_reason = f"{type(error).__name__}: {error}"
            logger.error(
                "scheduler_task_done status=crashed reason=%s traceback=%s",
                self._stop_reason,
                formatted_tb,
            )

        if self._stopping_requested:
            self._task = None
            return

        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run_loop(), name="decision_scheduler_restart")
            self._task.add_done_callback(self._on_task_done)
            logger.warning("scheduler_task_restart reason=%s", self._stop_reason)
        except RuntimeError:
            self._task = None
            logger.exception("scheduler_task_restart_failed reason=no_running_loop")


def _build_forced_trade_plan(snapshot: BybitKlineSnapshot, settings: Settings) -> TradePlan:
    entry = snapshot.candle.close
    entry_low = entry * (1 - 0.0002)
    entry_high = entry * (1 + 0.0002)
    direction = _pick_direction(settings, snapshot)
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
        rationale=["force_trade_mode", "debug_force_trade"],
        raw_input_snapshot={
            "symbol": snapshot.symbol,
            "direction": direction.value,
            "strategy": settings.strategy,
        },
    )


def _pick_direction(settings: Settings, snapshot: BybitKlineSnapshot) -> Direction:
    if settings.force_trade_random_direction:
        return random.choice([Direction.long, Direction.short])
    return Direction.long if snapshot.candle.close >= snapshot.candle.open else Direction.short


def _trend_strength(candle: BybitKlineSnapshot) -> float:
    price_move = abs(candle.candle.close - candle.candle.open)
    if candle.candle.open == 0:
        return 0.0
    strength = (price_move / candle.candle.open) * 10
    return min(1.0, strength)


def _build_decision_request(snapshot: BybitKlineSnapshot) -> DecisionRequest:
    candle = snapshot.candle
    direction = Direction.long if candle.close >= candle.open else Direction.short
    entry_low = min(candle.open, candle.close)
    entry_high = max(candle.open, candle.close)
    sl_hint = candle.low if direction == Direction.long else candle.high
    bias = BiasSignal(direction=direction, confidence=0.6)
    market = MarketSnapshot(
        funding_rate=0.01,
        oi_change_24h=0.0,
        leverage_ratio=1.0,
        trend_strength=_trend_strength(snapshot),
    )
    tradingview_payload = {
        "symbol": snapshot.symbol,
        "direction_hint": direction,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "sl_hint": sl_hint,
        "setup_type": SetupType.break_retest,
        "tf_entry": snapshot.interval,
        "tf_bias": "1h",
    }
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



def _apply_mode_overrides(
    plan: TradePlan,
    snapshot: BybitKlineSnapshot,
    settings: Settings,
) -> tuple[TradePlan, str | None, dict[str, object]]:
    if settings.current_mode != "SCALP":
        return plan, None, {}
    score = plan.signal_score or 0
    if score < settings.scalp_min_score:
        return (
            plan.model_copy(update={"status": Status.NO_TRADE, "rationale": [*plan.rationale, "scalp_score_below_min"]}),
            "setup_not_confirmed",
            {},
        )

    regime = classify_scalp_regime(snapshot, settings)
    regime_label = regime["regime_label"]
    allowed_side = regime["allowed_side"]
    scalp_meta = {
        "regime_label": regime_label,
        "allowed_side": allowed_side,
        "atr_pct": regime["atr_pct"],
        "ema_fast": regime["ema_fast"],
        "ema_slow": regime["ema_slow"],
        "ema_trend": regime["ema_trend"],
    }

    if settings.scalp_regime_enabled:
        if regime_label == "dead":
            return plan.model_copy(update={"status": Status.NO_TRADE}), "atr_too_low", scalp_meta
        if regime_label == "too_hot":
            return plan.model_copy(update={"status": Status.NO_TRADE}), "atr_too_high", scalp_meta
        if regime_label == "chop":
            return plan.model_copy(update={"status": Status.NO_TRADE}), "regime_chop", scalp_meta
        if allowed_side is None:
            return plan.model_copy(update={"status": Status.NO_TRADE}), "trend_mismatch", scalp_meta
        if (plan.direction == Direction.long and allowed_side != "long") or (
            plan.direction == Direction.short and allowed_side != "short"
        ):
            return plan.model_copy(update={"status": Status.NO_TRADE}), "trend_mismatch", scalp_meta

    if settings.scalp_trend_filter_enabled:
        trend_bias = _derive_trend_bias(snapshot, settings=settings)
        if trend_bias is not None and plan.direction != trend_bias:
            return plan.model_copy(update={"status": Status.NO_TRADE}), "trend_mismatch", scalp_meta

    setup_confirmed = is_scalp_setup_confirmed(snapshot, settings, plan.direction)
    if not setup_confirmed:
        _, setup_reason = scalp_setup_gate_reason(snapshot, settings, plan.direction)
        return plan.model_copy(update={"status": Status.NO_TRADE}), setup_reason, scalp_meta

    entry = snapshot.candle.close
    if plan.direction == Direction.long:
        stop_loss = entry * (1 - settings.scalp_sl_pct)
        take_profit = entry * (1 + settings.scalp_tp_pct)
    else:
        stop_loss = entry * (1 + settings.scalp_sl_pct)
        take_profit = entry * (1 - settings.scalp_tp_pct)

    updated = plan.model_copy(
        update={
            "entry_zone": (entry, entry),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rationale": [*plan.rationale, "mode_scalp"],
        }
    )
    return updated, None, scalp_meta


def classify_scalp_regime(snapshot: BybitKlineSnapshot, settings: Settings) -> dict[str, object]:
    closes = [candle.close for candle in snapshot.candles if candle.close > 0]
    highs = [candle.high for candle in snapshot.candles]
    lows = [candle.low for candle in snapshot.candles]
    if not closes:
        return {"regime_label": "chop", "allowed_side": None, "atr_pct": 0.0, "ema_fast": 0.0, "ema_slow": 0.0, "ema_trend": 0.0}

    close = closes[-1]
    ema_fast = _ema(closes, settings.scalp_ema_fast)
    ema_slow = _ema(closes, settings.scalp_ema_slow)
    ema_trend = _ema(closes, settings.scalp_ema_trend if len(closes) >= settings.scalp_ema_trend else settings.scalp_ema_slow)
    atr = _atr(highs, lows, closes, settings.scalp_atr_period)
    atr_pct = (atr / close) if close > 0 else 0.0

    if atr_pct < settings.scalp_atr_pct_min:
        regime_label = "dead"
        allowed_side = None
    elif atr_pct > settings.scalp_atr_pct_max:
        regime_label = "too_hot"
        allowed_side = None
    else:
        slope = _ema_slope(closes, settings.scalp_ema_slow)
        if close > ema_trend and ema_fast > ema_slow and slope >= settings.scalp_trend_slope_min:
            regime_label = "bull"
            allowed_side = "long"
        elif close < ema_trend and ema_fast < ema_slow and slope <= -settings.scalp_trend_slope_min:
            regime_label = "bear"
            allowed_side = "short"
        else:
            regime_label = "chop"
            allowed_side = None

    return {
        "regime_label": regime_label,
        "allowed_side": allowed_side,
        "atr_pct": atr_pct,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "ema_trend": ema_trend,
    }


def is_scalp_setup_confirmed(snapshot: BybitKlineSnapshot, settings: Settings, direction: Direction) -> bool:
    confirmed, _ = scalp_setup_gate_reason(snapshot, settings, direction)
    return confirmed


def scalp_setup_gate_reason(
    snapshot: BybitKlineSnapshot,
    settings: Settings,
    direction: Direction,
) -> tuple[bool, str]:
    if direction not in {Direction.long, Direction.short}:
        return False, "setup_invalid_direction"
    if settings.scalp_setup_mode in {"pullback_engulfing", "either"} and _confirm_pullback_engulfing(snapshot, settings, direction):
        return True, ""
    if settings.scalp_setup_mode in {"breakout_retest", "either"} and _confirm_breakout_retest(snapshot, settings, direction):
        return True, ""
    return False, f"setup_not_confirmed:{settings.scalp_setup_mode}"


def _confirm_pullback_engulfing(snapshot: BybitKlineSnapshot, settings: Settings, direction: Direction) -> bool:
    candles = snapshot.candles
    if len(candles) < settings.setup_min_candles:
        return False
    closes = [c.close for c in candles]
    ema_pull = _ema(closes, settings.scalp_pullback_ema)
    current = candles[-1]
    prev = candles[-2]
    dist = abs(current.close - ema_pull) / current.close if current.close > 0 else 1.0
    if dist > settings.scalp_pullback_max_dist_pct:
        return False

    current_body = abs(current.close - current.open)
    min_body = current.close * settings.scalp_engulfing_min_body_pct

    if direction == Direction.long:
        engulfing = current.close > current.open and prev.close < prev.open and current.open <= prev.close and current.close >= prev.open
        strong_close = current.close > prev.high and current_body >= min_body
    else:
        engulfing = current.close < current.open and prev.close > prev.open and current.open >= prev.close and current.close <= prev.open
        strong_close = current.close < prev.low and current_body >= min_body
    if not (engulfing or strong_close):
        return False

    if settings.scalp_rsi_confirm:
        rsi = _rsi(closes, settings.scalp_rsi_period)
        if rsi is None:
            return False
        if direction == Direction.long and rsi < settings.scalp_rsi_long_min:
            return False
        if direction == Direction.short and rsi > settings.scalp_rsi_short_max:
            return False
    return True


def _confirm_breakout_retest(snapshot: BybitKlineSnapshot, settings: Settings, direction: Direction) -> bool:
    candles = snapshot.candles
    lookback = max(settings.setup_min_candles, settings.scalp_breakout_lookback)
    if len(candles) < lookback + 2:
        return False
    window = candles[-(lookback + settings.scalp_retest_max_bars + 1):]
    breakout_level = max(c.high for c in window[:lookback]) if direction == Direction.long else min(c.low for c in window[:lookback])
    for i in range(lookback, len(window)):
        candle = window[i]
        if direction == Direction.long and candle.close > breakout_level:
            retest_window = window[i + 1 : i + 1 + settings.scalp_retest_max_bars]
            return any(item.low <= breakout_level <= item.close for item in retest_window)
        if direction == Direction.short and candle.close < breakout_level:
            retest_window = window[i + 1 : i + 1 + settings.scalp_retest_max_bars]
            return any(item.high >= breakout_level >= item.close for item in retest_window)
    return False


def _derive_trend_bias(snapshot: BybitKlineSnapshot, lookback: int | None = None, settings: Settings | None = None) -> Direction | None:
    if lookback is None:
        lookback = settings.trend_bias_lookback if settings is not None else 20
    closes = [candle.close for candle in snapshot.candles[-lookback:] if candle.close > 0]
    if len(closes) < (settings.trend_min_candles if settings is not None else 4):
        return None
    midpoint = len(closes) // 2
    first_avg = sum(closes[:midpoint]) / midpoint
    second_avg = sum(closes[midpoint:]) / (len(closes) - midpoint)
    if second_avg > first_avg:
        return Direction.long
    if second_avg < first_avg:
        return Direction.short
    return None


def _ema(values: list[float], period: int) -> float:
    period = max(2, min(period, len(values)))
    alpha = 2 / (period + 1)
    ema = values[0]
    for value in values[1:]:
        ema = (value * alpha) + (ema * (1 - alpha))
    return ema


def _ema_slope(values: list[float], period: int) -> float:
    if len(values) < period + 2:
        return 0.0
    ema_now = _ema(values, period)
    ema_prev = _ema(values[:-1], period)
    return ((ema_now - ema_prev) / ema_prev) if ema_prev else 0.0


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return 0.0
    trs: list[float] = []
    for idx in range(1, len(closes)):
        trs.append(max(highs[idx] - lows[idx], abs(highs[idx] - closes[idx - 1]), abs(lows[idx] - closes[idx - 1])))
    return sum(trs[-period:]) / period if len(trs) >= period else 0.0


def _rsi(closes: list[float], period: int) -> float | None:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1 + rs))


def _plan_skip_reason(plan: TradePlan) -> str:
    rationale = set(plan.rationale)
    if "cooldown" in rationale:
        return "cooldown"
    if "max_losses" in rationale:
        return "max_consecutive_losses"
    if "daily_loss_limit" in rationale:
        return "daily_dd_limit"
    if "max_losses_per_day" in rationale:
        return "max_losses_per_day"
    if "max_trades" in rationale:
        return "max_trades"
    for soft_reason in ("no_valid_setup", "setup_not_confirmed", "atr_too_low", "no_trend", "no_candles"):
        if soft_reason in rationale:
            return soft_reason
    if plan.rationale:
        return f"setup_not_confirmed:{plan.rationale[0]}"
    return "setup_not_confirmed"


def _trade_key(snapshot: BybitKlineSnapshot, plan: TradePlan) -> str:
    return (
        f"{snapshot.symbol}:{snapshot.kline_close_time_ms}:"
        f"{plan.direction.value}"
    )


def _funding_blackout_state(now: datetime, settings: Settings) -> dict[str, bool]:
    minutes = int(now.timestamp() // 60)
    interval = max(1, settings.funding_interval_minutes)
    minute_in_window = minutes % interval
    block_start = max(0, interval - settings.funding_block_before_minutes)
    close_start = max(0, interval - settings.funding_close_before_minutes)
    block_new_entries = minute_in_window >= block_start or minute_in_window <= settings.funding_guard_tail_minute
    close_positions = minute_in_window >= close_start
    return {"block_new_entries": block_new_entries, "close_positions": close_positions}


def _force_trade_key(symbol: str, plan: TradePlan, now: datetime) -> str:
    return f"{symbol}:{int(now.timestamp() * 1000)}:{plan.direction.value}"


def _compute_atr_values(snapshot: BybitKlineSnapshot, period: int) -> list[float]:
    candles = snapshot.candles
    if period <= 0 or len(candles) < period + 1:
        return []
    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        current = candles[i]
        prev_close = candles[i - 1].close
        true_ranges.append(max(current.high - current.low, abs(current.high - prev_close), abs(current.low - prev_close)))
    if len(true_ranges) < period:
        return []
    atr_values: list[float] = []
    atr = sum(true_ranges[:period]) / period
    atr_values.append(atr)
    for tr in true_ranges[period:]:
        atr = ((atr * (period - 1)) + tr) / period
        atr_values.append(atr)
    return atr_values


def _compute_adx(snapshot: BybitKlineSnapshot, period: int) -> float | None:
    candles = snapshot.candles
    if period <= 1 or len(candles) < (period * 2) + 1:
        return None
    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, len(candles)):
        cur = candles[i]
        prev = candles[i - 1]
        up_move = cur.high - prev.high
        down_move = prev.low - cur.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    if len(trs) < period:
        return None
    smoothed_tr = sum(trs[:period])
    smoothed_plus_dm = sum(plus_dm[:period])
    smoothed_minus_dm = sum(minus_dm[:period])
    dx_values: list[float] = []
    for i in range(period, len(trs)):
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + trs[i]
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]
        if smoothed_tr <= 0:
            continue
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        denom = plus_di + minus_di
        if denom <= 0:
            continue
        dx_values.append(100 * (abs(plus_di - minus_di) / denom))
    if len(dx_values) < period:
        return None
    return sum(dx_values[-period:]) / period
