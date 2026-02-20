from __future__ import annotations

import asyncio
import logging
import random
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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
from ..strategy.decision import decide, evaluate_warmup_status, required_warmup_bars_5m
from ..providers.bybit import BybitKlineSnapshot, fetch_symbol_klines
from ..utils.intervals import interval_to_ms
from ..utils.clock import Clock, ReplayClock
from ..utils.trading_day import trading_day_key, trading_day_start
from .notifier import format_trade_message, send_telegram_message

logger = logging.getLogger(__name__)


@dataclass
class Blocker:
    code: str
    layer: str
    detail: str
    until_ts: str | None = None

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
        clock: Clock | None = None,
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
        self._clock = clock
        self._last_tick_time: datetime | None = None
        self._last_heartbeat_monotonic = time.monotonic()
        self._stall_recoveries = 0
        self.last_tick_ts: float | None = None
        self.started_ts: float | None = None
        self._last_snapshots: dict[str, BybitKlineSnapshot] = {}
        self._last_fetch_counts: dict[str, int] = {}
        self._last_symbol_tick_time: dict[str, datetime] = {}
        self._last_force_trade_ts: dict[str, datetime] = {}
        self._last_exit_eval_close_ms: dict[str, int] = {}
        self._last_skip_telegram_ts: dict[str, datetime] = {}
        self._last_blocker_log: dict[str, tuple[str, datetime]] = {}
        self._blocker_log_interval_seconds = 30.0
        self._next_fetch_after_ms: dict[str, int] = {}
        self._tick_listeners: list[Callable[[], None]] = []
        self._listener_error_log_window_seconds = 10.0
        self._last_listener_error_message: str | None = None
        self._last_listener_error_traceback_ts: float = 0.0

        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._stop_reason: str | None = None
        self._consecutive_failures = 0
        self._stopping_requested = False
        self._replay_active = False
        self._replay_bars_processed = 0
        self._engine_clock: datetime | None = None
        if settings.run_mode == "replay" and settings.replay_seed is not None:
            random.seed(settings.replay_seed)

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

    def engine_now(self) -> datetime:
        if self._settings.run_mode == "replay" and self._engine_clock is not None:
            return self._engine_clock
        if self._clock is not None:
            return self._clock.now_dt()
        return datetime.now(timezone.utc)

    @property
    def replay_active(self) -> bool:
        return self._replay_active

    @property
    def replay_bars_processed(self) -> int:
        return self._replay_bars_processed

    def add_tick_listener(self, listener: Callable[[], None]) -> None:
        self._tick_listeners.append(listener)

    def _notify_tick_listeners(self) -> None:
        for listener in self._tick_listeners:
            try:
                listener()
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                now_monotonic = time.monotonic()
                should_log_traceback = (
                    message != self._last_listener_error_message
                    or (now_monotonic - self._last_listener_error_traceback_ts) >= self._listener_error_log_window_seconds
                )
                if should_log_traceback:
                    self._last_listener_error_message = message
                    self._last_listener_error_traceback_ts = now_monotonic
                    logger.exception("scheduler_tick_listener_error error=%s", message)
                else:
                    logger.warning("scheduler_tick_listener_error_suppressed error=%s", message)

    async def start(self) -> bool:
        async with self._lock:
            if self.running:
                logger.info("scheduler_start skipped=already_running")
                return False
            self._stop_event = asyncio.Event()
            self._stopping_requested = False
            self._stop_reason = None
            self._consecutive_failures = 0
            self._replay_active = self._settings.run_mode == "replay"
            self._replay_bars_processed = 0
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
        self._replay_active = False
        logger.info("scheduler_stop status=stopped")
        return True

    def _closed_trades_today_by_symbol(self, now: datetime) -> dict[str, int]:
        if self._database is None:
            return {}
        start_of_day = trading_day_start(now)
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


    def _roll_governor_day(self, now: datetime) -> dict[str, object] | None:
        if self._database is None:
            return None
        governor_row = self._database.get_runtime_state("prop.governor")
        if governor_row is None or not governor_row.value_text:
            return None
        try:
            gov = json.loads(governor_row.value_text)
        except json.JSONDecodeError:
            return None
        day_key = trading_day_key(now)
        if gov.get("day_key") == day_key:
            return gov
        gov["day_key"] = day_key
        gov["daily_net_r"] = 0.0
        gov["daily_losses"] = 0
        gov["daily_trades"] = 0
        if self._settings.prop_reset_consec_losses_on_day_rollover:
            gov["consecutive_losses"] = 0
        gov["locked_until_ts"] = None
        self._database.set_runtime_state("prop.governor", value_text=self._database.dumps_json(gov))
        return gov

    def _parse_iso_datetime(self, value: object) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _challenge_blockers(self) -> list[Blocker]:
        if not self._settings.prop_enabled or self._database is None:
            return []
        row = self._database.get_runtime_state("challenge.state")
        if row is None or not row.value_text:
            return []
        try:
            challenge = json.loads(row.value_text)
        except json.JSONDecodeError:
            return [Blocker(code="challenge_state_unavailable", layer="terminal", detail="Challenge state is unreadable; trading disabled for safety.")]
        status = str(challenge.get("status", "")).upper()
        if status in {"PASSED", "FAILED"}:
            return [Blocker(code=f"challenge_{status.lower()}", layer="terminal", detail=f"Challenge status is {status}.")]
        return []

    def _governor_blockers(self, now: datetime) -> list[Blocker]:
        blockers: list[Blocker] = []
        gov = self._roll_governor_day(now)
        if not gov:
            return blockers
        try:
            daily_losses = int(gov.get("daily_losses", 0) or 0)
            daily_trades = int(gov.get("daily_trades", 0) or 0)
            consec_losses = int(gov.get("consecutive_losses", 0) or 0)
            daily_net_r = float(gov.get("daily_net_r", 0.0) or 0.0)
        except (TypeError, ValueError):
            return [Blocker(code="prop_governor_state_invalid", layer="governor", detail="Governor state is invalid; unable to safely evaluate limits.")]

        lock_start = self._parse_iso_datetime(gov.get("locked_until_ts"))
        if lock_start is not None:
            until_dt = lock_start
            if self._settings.prop_time_cooldown_minutes > 0:
                until_dt = lock_start + timedelta(minutes=int(self._settings.prop_time_cooldown_minutes))
            if until_dt > now:
                blockers.append(Blocker(code="prop_time_cooldown", layer="governor", detail="Prop governor cooldown is active.", until_ts=until_dt.isoformat()))

        if daily_trades >= self._settings.prop_max_trades_per_day:
            blockers.append(Blocker(code="prop_max_trades_per_day", layer="governor", detail="Prop max trades/day reached."))
        if consec_losses >= self._settings.prop_max_consec_losses:
            blockers.append(Blocker(code="prop_max_consecutive_losses", layer="governor", detail="Prop max consecutive losses reached."))
        if daily_losses >= self._settings.prop_daily_stop_after_losses:
            blockers.append(Blocker(code="prop_daily_stop_after_losses", layer="governor", detail="Prop daily stop after losses reached."))
        if daily_net_r >= self._settings.prop_daily_stop_after_net_r:
            blockers.append(Blocker(code="prop_daily_stop_after_net_r", layer="governor", detail="Prop daily stop after net R reached."))
        return blockers

    def _risk_blockers(self, symbol: str, now: datetime, closed_trades_today: dict[str, int], funding_blocked: bool = False) -> list[Blocker]:
        blockers: list[Blocker] = []
        if self._settings.is_blackout(now):
            blockers.append(Blocker(code="news_blackout", layer="risk", detail="News blackout window active."))
        if funding_blocked:
            blockers.append(Blocker(code="funding_blackout_entries_blocked", layer="risk", detail="Funding blackout blocks new entries."))
        closed_count = closed_trades_today.get(symbol)
        allowed, reason = self._state.risk_check(symbol, self._settings, now, trades_today_closed=closed_count)
        if not allowed:
            blockers.append(Blocker(code=str(reason or "risk_gate_blocked"), layer="risk", detail=f"Global risk gate blocked entries: {reason or 'unknown'}"))
        return blockers

    def _strategy_blockers(self, plan: TradePlan | None, skip_reason: str | None) -> list[Blocker]:
        if skip_reason:
            return [Blocker(code=skip_reason, layer="strategy", detail=f"Strategy or execution filter blocked entry: {skip_reason}.")]
        if plan is not None and plan.status != Status.TRADE:
            code = _plan_skip_reason(plan)
            return [Blocker(code=code, layer="strategy", detail=f"Strategy rejected setup: {code}.")]
        return []

    def _compute_effective_blockers(
        self,
        symbol: str,
        now: datetime,
        closed_trades_today: dict[str, int],
        plan: TradePlan | None,
        skip_reason: str | None,
        funding_blocked: bool = False,
    ) -> tuple[Blocker | None, list[Blocker]]:
        blockers = [
            *self._challenge_blockers(),
            *self._governor_blockers(now),
            *self._risk_blockers(symbol, now, closed_trades_today, funding_blocked=funding_blocked),
            *self._strategy_blockers(plan, skip_reason),
        ]
        if blockers:
            return blockers[0], blockers
        return None, []

    def _log_skip_blocker(self, symbol: str, blocker: Blocker | None, now: datetime) -> None:
        if blocker is None:
            return
        signature = f"{blocker.layer}:{blocker.code}:{blocker.until_ts or ''}"
        previous = self._last_blocker_log.get(symbol)
        should_log = previous is None or previous[0] != signature or (now - previous[1]).total_seconds() >= self._blocker_log_interval_seconds
        if not should_log:
            return
        logger.info(
            "trade_skip_blocker symbol=%s layer=%s code=%s detail=%s until_ts=%s",
            symbol,
            blocker.layer,
            blocker.code,
            blocker.detail,
            blocker.until_ts,
        )
        self._last_blocker_log[symbol] = (signature, now)

    async def run_once(self, force: bool = False) -> list[dict[str, object]]:
        self.last_tick_ts = time.time()
        self._last_heartbeat_monotonic = time.monotonic()
        if self._heartbeat_cb is not None:
            self._heartbeat_cb()
        symbols = list(self._settings.symbols)
        self._last_tick_time = self.engine_now()
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
        closed_trades_today = self._closed_trades_today_by_symbol(self.engine_now())
        for symbol in symbols:
            try:
                snapshot = self._last_snapshots.get(symbol)
                tick_ts = self.engine_now()
                now_ms = int(tick_ts.timestamp() * 1000)
                next_fetch_after_ms = self._next_fetch_after_ms.get(symbol, 0)
                should_refresh = (
                    self._settings.run_mode == "replay"
                    or self._settings.market_data_provider == "replay"
                    or snapshot is None
                    or force_mode
                    or now_ms >= next_fetch_after_ms
                )
                if self._settings.run_mode == "replay" or self._settings.market_data_provider == "replay":
                    logger.info(
                        "replay_scheduler_fetch_gate symbol=%s now_ms=%s next_fetch_after_ms=%s should_refresh=%s",
                        symbol,
                        now_ms,
                        next_fetch_after_ms,
                        should_refresh,
                    )
                if should_refresh:
                    required_history = required_warmup_bars_5m(self._settings)
                    fetch_limit = max(int(self._settings.candle_history_limit), int(required_history))
                    snapshot = await fetch_symbol_klines(
                        symbol=symbol,
                        interval=self._settings.candle_interval,
                        limit=fetch_limit,
                        rest_base=self._settings.bybit_rest_base,
                        provider=self._settings.market_data_provider,
                        fallback_provider=self._settings.market_data_fallbacks,
                        failover_threshold=self._settings.market_data_failover_threshold,
                        backoff_base_ms=self._settings.market_data_backoff_base_ms,
                        backoff_max_ms=self._settings.market_data_backoff_max_ms,
                        replay_path=self._settings.market_data_replay_path,
                        replay_speed=self._settings.market_data_replay_speed,
                        replay_start_ts=self._settings.replay_start_ts,
                        replay_end_ts=self._settings.replay_end_ts,
                    )
                if self._settings.run_mode == "replay" or self._settings.market_data_provider == "replay":
                    tick_ts = snapshot.candle.close_time
                    self._engine_clock = tick_ts
                    if isinstance(self._clock, ReplayClock):
                        self._clock.set_ts(int(tick_ts.timestamp()))
                    now_ms = snapshot.kline_close_time_ms
                self._last_symbol_tick_time[symbol] = tick_ts
                if snapshot is None:
                    self._last_fetch_counts[symbol] = 0
                else:
                    self._last_fetch_counts[symbol] = len(snapshot.candles)
                effective_blocker, blockers = self._compute_effective_blockers(
                    symbol=symbol,
                    now=tick_ts,
                    closed_trades_today=closed_trades_today,
                    plan=None,
                    skip_reason=None,
                    funding_blocked=False,
                )
                if effective_blocker is not None and effective_blocker.layer in {"terminal", "governor", "risk"}:
                    gate_reason = effective_blocker.code
                    self._state.set_decision_meta(
                        symbol,
                        {
                            "decision": "skip",
                            "skip_reason": gate_reason,
                            "final_entry_gate": gate_reason,
                            "blocker_code": effective_blocker.code,
                            "blocker_detail": effective_blocker.detail,
                            "blocker_layer": effective_blocker.layer,
                            "blocker_until_ts": effective_blocker.until_ts,
                            "blockers": [blocker.__dict__ for blocker in blockers],
                        },
                    )
                    self._state.record_skip_reason(gate_reason)
                    self._log_skip_blocker(symbol, effective_blocker, tick_ts)
                    results.append(
                        {
                            "symbol": symbol,
                            "plan": None,
                            "reason": gate_reason,
                            "candles_fetched": len(snapshot.candles) if snapshot else 0,
                            "latest_candle_ts": snapshot.candle.close_time.isoformat() if snapshot else None,
                            "decision_status": None,
                            "persisted": False,
                            "dedupe_key": None,
                            "telegram_sent": False,
                            "trade_opened": False,
                            "trade_id": None,
                            "decision": "skip",
                            "skip_reason": gate_reason,
                            "final_entry_gate": gate_reason,
                            "blocker_code": effective_blocker.code,
                            "blocker_detail": effective_blocker.detail,
                            "blocker_layer": effective_blocker.layer,
                            "blocker_until_ts": effective_blocker.until_ts,
                            "blockers": [blocker.__dict__ for blocker in blockers],
                        }
                    )
                    continue
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
                    and util_pct >= self._settings.funding_blackout_max_util_pct
                    and unrealized <= -abs(self._settings.funding_blackout_max_loss_usd)
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

            if not snapshot.kline_is_closed and self._settings.require_candle_close_confirm:
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
                if self._settings.run_mode == "replay" or self._settings.market_data_provider == "replay":
                    logger.info(
                        "replay_candle_skipped symbol=%s reason=%s latest_closed_ms=%s last_processed_ms=%s",
                        snapshot.symbol,
                        reason,
                        latest_closed_ms,
                        last_processed,
                    )
            else:
                forced_trade = False
                if (self._settings.force_trade_mode or self._settings.smoke_test_force_trade) and self._force_trade_due(snapshot.symbol, tick_ts):
                    plan = _build_forced_trade_plan(snapshot, self._settings)
                    self._last_force_trade_ts[snapshot.symbol] = tick_ts
                    forced_trade = True
                else:
                    request = _build_decision_request(snapshot, timestamp=tick_ts)
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
                    elif self._database is not None:
                        direction_block = _direction_limit_reason(plan.direction, self._database.fetch_open_trades(), self._settings)
                        if direction_block is not None:
                            plan = plan.model_copy(update={"status": Status.NO_TRADE, "rationale": [*plan.rationale, direction_block]})
                            reason = direction_block
                            skip_reason = direction_block
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
                        timestamp=tick_ts,
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
                if self._settings.run_mode == "replay" or self._settings.market_data_provider == "replay":
                    logger.info(
                        "replay_candle_processed symbol=%s latest_closed_ms=%s last_processed_ms=%s",
                        snapshot.symbol,
                        latest_closed_ms,
                        last_processed,
                    )
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
            warmup = evaluate_warmup_status(snapshot.candles, self._settings)
            warmup_meta: dict[str, object] = {
                "ready": warmup.ready,
                "bars_5m_have": warmup.bars_5m_have,
                "bars_5m_need": warmup.bars_5m_need,
                "missing_components": warmup.missing_components,
            }
            if self._settings.htf_bias_enabled:
                warmup_meta["bars_htf_have"] = warmup.bars_htf_have
                warmup_meta["bars_htf_need"] = warmup.bars_htf_need
            self._state.set_decision_meta(
                snapshot.symbol,
                {
                    **self._state.get_decision_meta(snapshot.symbol),
                    "latest_candle_ts": snapshot.kline_close_time_ms,
                    "last_processed_candle_ts": last_processed,
                    "should_process": should_process,
                    "should_process_reason": debug_reason,
                    "warmup_status": warmup_meta,
                    "candles_loaded_5m_count": len(snapshot.candles),
                    "candles_loaded_htf_count": warmup.bars_htf_have if self._settings.htf_bias_enabled else 0,
                },
            )
            if self._database is not None:
                self._database.set_runtime_state(key=f"latest_candle:{snapshot.symbol}", value_number=float(snapshot.kline_close_time_ms), symbol=snapshot.symbol)

            if skip_reason is None and decision == "skip":
                skip_reason = reason
            effective_blocker, blockers = self._compute_effective_blockers(
                symbol=symbol,
                now=tick_ts,
                closed_trades_today=closed_trades_today,
                plan=plan,
                skip_reason=skip_reason if decision == "skip" else None,
                funding_blocked=bool(funding_state["block_new_entries"]),
            )
            final_entry_gate = effective_blocker.code if decision == "skip" and effective_blocker is not None else (skip_reason if decision == "skip" else None)
            decision_meta = {
                **self._state.get_decision_meta(symbol),
                "decision": decision,
                "skip_reason": skip_reason,
                "final_entry_gate": final_entry_gate,
                "blocker_code": (effective_blocker.code if effective_blocker is not None else None),
                "blocker_detail": (effective_blocker.detail if effective_blocker is not None else ("ELIGIBLE" if decision != "skip" else "No blocker detail available.")),
                "blocker_layer": (effective_blocker.layer if effective_blocker is not None else "none"),
                "blocker_until_ts": (effective_blocker.until_ts if effective_blocker is not None else None),
                "blockers": [blocker.__dict__ for blocker in blockers],
                "regime_label": scalp_meta.get("regime_label"),
                "allowed_side": scalp_meta.get("allowed_side"),
                "atr_pct": scalp_meta.get("atr_pct"),
                "ema_fast": scalp_meta.get("ema_fast"),
                "ema_slow": scalp_meta.get("ema_slow"),
                "ema_trend": scalp_meta.get("ema_trend"),
                "htf_bias_reject": scalp_meta.get("htf_bias_reject"),
                "trigger_body_ratio_reject": scalp_meta.get("trigger_body_ratio_reject"),
                "trigger_close_location_reject": scalp_meta.get("trigger_close_location_reject"),
                "signal_score": (plan.signal_score if plan is not None else None),
                "trend_strength": (
                    plan.raw_input_snapshot.get("market", {}).get("trend_strength")
                    if plan is not None and isinstance(plan.raw_input_snapshot, dict)
                    else None
                ),
                "regime": (plan.raw_input_snapshot.get("regime") if plan is not None and isinstance(plan.raw_input_snapshot, dict) else None),
                "bias": (plan.raw_input_snapshot.get("bias") if plan is not None and isinstance(plan.raw_input_snapshot, dict) else None),
                "confidence": (plan.raw_input_snapshot.get("confidence") if plan is not None and isinstance(plan.raw_input_snapshot, dict) else None),
                "entry_reasons": (plan.raw_input_snapshot.get("entry_reasons") if plan is not None and isinstance(plan.raw_input_snapshot, dict) else None),
                "entry_block_reasons": (plan.raw_input_snapshot.get("entry_block_reasons") if plan is not None and isinstance(plan.raw_input_snapshot, dict) else None),
                "equity_state": (plan.raw_input_snapshot.get("equity_state") if plan is not None and isinstance(plan.raw_input_snapshot, dict) else None),
                "provider": getattr(snapshot, "provider_name", "bybit"),
                "last_candle_age_seconds": max(0.0, (now_ms - snapshot.kline_close_time_ms) / 1000.0),
                "market_data_status": "OK",
            }
            self._state.set_decision_meta(symbol, decision_meta)
            if decision == "skip":
                self._state.record_skip_reason(skip_reason)
                self._log_skip_blocker(symbol, effective_blocker, tick_ts)
            if self._settings.telegram_debug_skips and decision == "skip":
                await self._maybe_send_skip_debug(symbol, skip_reason)
            logger.info(
                "tick symbol=%s ts=%s decision=%s skip_reason=%s blocker_layer=%s blocker_code=%s htf_bias_reject=%s trigger_body_ratio_reject=%s trigger_close_location_reject=%s",
                symbol,
                tick_ts.isoformat(),
                decision,
                skip_reason,
                decision_meta.get("blocker_layer"),
                decision_meta.get("blocker_code"),
                decision_meta.get("htf_bias_reject"),
                decision_meta.get("trigger_body_ratio_reject"),
                decision_meta.get("trigger_close_location_reject"),
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
                    "blocker_code": decision_meta.get("blocker_code"),
                    "blocker_detail": decision_meta.get("blocker_detail"),
                    "blocker_layer": decision_meta.get("blocker_layer"),
                    "blocker_until_ts": decision_meta.get("blocker_until_ts"),
                    "blockers": decision_meta.get("blockers", []),
                    "regime_label": decision_meta.get("regime_label"),
                    "allowed_side": decision_meta.get("allowed_side"),
                    "atr_pct": decision_meta.get("atr_pct"),
                    "htf_bias_reject": decision_meta.get("htf_bias_reject"),
                    "trigger_body_ratio_reject": decision_meta.get("trigger_body_ratio_reject"),
                    "trigger_close_location_reject": decision_meta.get("trigger_close_location_reject"),
                }
            )
        self._notify_tick_listeners()
        if self._settings.run_mode == "replay":
            self._replay_bars_processed += 1
        return results


    async def _maybe_send_skip_debug(self, symbol: str, skip_reason: str | None) -> None:
        if not skip_reason:
            return
        now = self.engine_now()
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
        replay_speed = max(0.1, float(self._settings.market_data_replay_speed or 1.0))
        replay_pause_seconds = 1.0 / replay_speed
        progress_log_every = max(50, int(replay_speed * 25))
        while not self._stop_event.is_set():
            try:
                heartbeat_age = time.monotonic() - self._last_heartbeat_monotonic
                if heartbeat_age > max(5.0, self._interval * 3):
                    self._stall_recoveries += 1
                    logger.error("scheduler_watchdog_stall_detected age=%.2fs recoveries=%s", heartbeat_age, self._stall_recoveries)
                await self.run_once()
                if self._settings.run_mode == "replay":
                    closed = len([trade for trade in self._database.fetch_trades() if getattr(trade, "closed_at", None)]) if self._database is not None else 0
                    if closed >= max(1, self._settings.replay_max_trades):
                        self._stop_reason = "replay_max_trades_reached"
                        self._stop_event.set()
                        break
                    if self._replay_bars_processed >= max(1, self._settings.replay_max_bars):
                        self._stop_reason = "replay_max_bars_reached"
                        self._stop_event.set()
                        break
                    if self._replay_bars_processed % progress_log_every == 0:
                        logger.info(
                            "replay_progress bars_processed=%s replay_max_bars=%s replay_speed=%s",
                            self._replay_bars_processed,
                            self._settings.replay_max_bars,
                            replay_speed,
                        )
                self._consecutive_failures = 0
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
            if self._settings.run_mode == "replay":
                try:
                    if replay_pause_seconds > 0:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=replay_pause_seconds)
                    else:
                        await asyncio.sleep(0)
                except asyncio.TimeoutError:
                    pass
                continue
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
        if self._settings.run_mode == "replay" and self._stop_reason in {"replay_max_trades_reached", "replay_max_bars_reached"}:
            self._task = None
            self._replay_active = False
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


def _build_decision_request(snapshot: BybitKlineSnapshot, timestamp: datetime | None = None) -> DecisionRequest:
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
        timestamp=timestamp or candle.close_time,
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
        "htf_bias_reject": False,
        "trigger_body_ratio_reject": False,
        "trigger_close_location_reject": False,
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

    if settings.htf_bias_enabled:
        if not _passes_htf_bias(snapshot, settings, plan.direction):
            scalp_meta["htf_bias_reject"] = True
            return plan.model_copy(update={"status": Status.NO_TRADE}), "htf_bias_reject", scalp_meta

    setup_confirmed = is_scalp_setup_confirmed(snapshot, settings, plan.direction)
    if not setup_confirmed:
        _, setup_reason = scalp_setup_gate_reason(snapshot, settings, plan.direction)
        return plan.model_copy(update={"status": Status.NO_TRADE}), setup_reason, scalp_meta

    trigger_ok, trigger_reason = _trigger_quality_gate(snapshot, settings, plan.direction)
    if not trigger_ok:
        if trigger_reason == "trigger_body_ratio_reject":
            scalp_meta["trigger_body_ratio_reject"] = True
        if trigger_reason == "trigger_close_location_reject":
            scalp_meta["trigger_close_location_reject"] = True
        return plan.model_copy(update={"status": Status.NO_TRADE}), trigger_reason, scalp_meta

    entry = snapshot.candle.close
    closes = [candle.close for candle in snapshot.candles if candle.close > 0]
    highs = [candle.high for candle in snapshot.candles]
    lows = [candle.low for candle in snapshot.candles]
    atr_value = _atr(highs, lows, closes, settings.scalp_atr_period)
    atr_stop_distance = atr_value * settings.sl_atr_mult if atr_value > 0 else float("inf")
    pct_stop_distance = entry * settings.scalp_sl_pct
    risk_distance = min(pct_stop_distance, atr_stop_distance)

    atr_tp_distance = atr_value * settings.tp_atr_mult if atr_value > 0 else float("inf")
    pct_tp_distance = entry * settings.scalp_tp_pct
    capped_tp_distance = min(pct_tp_distance, atr_tp_distance)
    target_distance = min(2.0 * risk_distance, capped_tp_distance)

    if plan.direction == Direction.long:
        stop_loss = entry - risk_distance
        take_profit = entry + target_distance
    else:
        stop_loss = entry + risk_distance
        take_profit = entry - target_distance

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
    pullback_ok, pullback_reason = _confirm_pullback_engulfing(snapshot, settings, direction)
    if settings.scalp_setup_mode in {"pullback_engulfing", "either"} and pullback_ok:
        return True, ""
    if settings.scalp_setup_mode == "pullback_engulfing" and pullback_reason:
        return False, pullback_reason

    if settings.disable_breakout_chase and settings.scalp_setup_mode in {"breakout_retest", "either"}:
        return False, "breakout_chase_disabled"

    breakout_ok, breakout_reason = _confirm_breakout_retest(snapshot, settings, direction)
    if settings.scalp_setup_mode in {"breakout_retest", "either"} and breakout_ok:
        return True, ""
    if settings.scalp_setup_mode == "breakout_retest" and breakout_reason:
        return False, breakout_reason

    return False, f"setup_not_confirmed:{settings.scalp_setup_mode}"


def _passes_htf_bias(snapshot: BybitKlineSnapshot, settings: Settings, direction: Direction) -> bool:
    htf_closes = _aggregate_closes_for_interval(snapshot, settings.htf_interval)
    needed = max(settings.htf_ema_fast, settings.htf_ema_slow) + 2
    if len(htf_closes) < needed:
        return False

    ema_fast = _ema(htf_closes, settings.htf_ema_fast)
    ema_slow = _ema(htf_closes, settings.htf_ema_slow)
    if direction == Direction.long and ema_fast <= ema_slow:
        return False
    if direction == Direction.short and ema_fast >= ema_slow:
        return False

    if not settings.htf_bias_require_slope:
        return True

    slope = _ema_slope(htf_closes, settings.htf_ema_slow)
    if direction == Direction.long:
        return slope > 0
    return slope < 0


def _aggregate_closes_for_interval(snapshot: BybitKlineSnapshot, interval: str) -> list[float]:
    target_ms = interval_to_ms(interval)
    source_ms = interval_to_ms(snapshot.interval)
    if target_ms <= source_ms or source_ms <= 0:
        return [candle.close for candle in snapshot.candles if candle.close > 0]

    merged: list[float] = []
    candles = [candle for candle in snapshot.candles if candle.close > 0]
    if not candles:
        return merged

    current_bucket = (int(candles[0].close_time.timestamp() * 1000) // target_ms) * target_ms
    last_close = candles[0].close
    for candle in candles:
        close_ms = int(candle.close_time.timestamp() * 1000)
        bucket = (close_ms // target_ms) * target_ms
        if bucket != current_bucket:
            merged.append(last_close)
            current_bucket = bucket
        last_close = candle.close
    merged.append(last_close)
    return merged


def _trigger_quality_gate(snapshot: BybitKlineSnapshot, settings: Settings, direction: Direction) -> tuple[bool, str | None]:
    current = snapshot.candle
    candle_range = max(current.high - current.low, 1e-12)
    body_ratio = abs(current.close - current.open) / candle_range
    if settings.trigger_body_ratio_min > 0 and body_ratio < settings.trigger_body_ratio_min:
        return False, "trigger_body_ratio_reject"

    if settings.trigger_close_location_min > 0:
        if direction == Direction.long:
            close_loc = (current.close - current.low) / candle_range
        else:
            close_loc = (current.high - current.close) / candle_range
        if close_loc < settings.trigger_close_location_min:
            return False, "trigger_close_location_reject"

    return True, None


def _confirm_pullback_engulfing(snapshot: BybitKlineSnapshot, settings: Settings, direction: Direction) -> tuple[bool, str]:
    candles = snapshot.candles
    if len(candles) < settings.setup_min_candles:
        return False, "setup_not_confirmed:pullback_engulfing"
    closes = [c.close for c in candles]
    ema_pull = _ema(closes, settings.scalp_pullback_ema)
    current = candles[-1]
    prev = candles[-2]
    dist = abs(current.close - ema_pull) / current.close if current.close > 0 else 1.0
    if dist > settings.scalp_pullback_max_dist_pct:
        return False, "setup_not_confirmed:pullback_engulfing"
    min_dist = max(0.0, float(settings.scalp_pullback_min_dist_pct or 0.0))
    dist_from_ema = abs(current.close - ema_pull) / ema_pull if ema_pull > 0 else 1.0
    if dist_from_ema < min_dist:
        return False, "pullback_too_shallow"

    adx = _compute_adx(snapshot, period=max(2, settings.adx_period))
    if adx is None or adx < settings.adx_threshold:
        return False, "adx_too_low"

    current_body = abs(current.close - current.open)
    min_body = current.close * settings.scalp_engulfing_min_body_pct

    if direction == Direction.long:
        engulfing = current.close > current.open and prev.close < prev.open and current.open <= prev.close and current.close >= prev.open
        strong_close = current.close > prev.high and current_body >= min_body
    else:
        engulfing = current.close < current.open and prev.close > prev.open and current.open >= prev.close and current.close <= prev.open
        strong_close = current.close < prev.low and current_body >= min_body
    if not (engulfing or strong_close):
        return False, "setup_not_confirmed:pullback_engulfing"

    if settings.scalp_rsi_confirm:
        rsi = _rsi(closes, settings.scalp_rsi_period)
        if rsi is None:
            return False, "setup_not_confirmed:pullback_engulfing"
        if direction == Direction.long and rsi < settings.scalp_rsi_long_min:
            return False, "setup_not_confirmed:pullback_engulfing"
        if direction == Direction.short and rsi > settings.scalp_rsi_short_max:
            return False, "setup_not_confirmed:pullback_engulfing"
        if direction == Direction.long and rsi > settings.scalp_rsi_long_max:
            return False, "rsi_exhausted_long"
        if direction == Direction.short and rsi < settings.scalp_rsi_short_min:
            return False, "rsi_exhausted_short"
    return True, ""


def _confirm_breakout_retest(snapshot: BybitKlineSnapshot, settings: Settings, direction: Direction) -> tuple[bool, str]:
    candles = snapshot.candles
    lookback = max(settings.setup_min_candles, settings.scalp_breakout_lookback)
    if len(candles) < lookback + 2:
        return False, "setup_not_confirmed:breakout_retest"
    adx = _compute_adx(snapshot, period=max(2, settings.adx_period))
    if adx is None or adx < settings.adx_threshold:
        return False, "adx_too_low"
    window = candles[-(lookback + settings.scalp_retest_max_bars + 1):]
    breakout_level = max(c.high for c in window[:lookback]) if direction == Direction.long else min(c.low for c in window[:lookback])
    for i in range(lookback, len(window)):
        candle = window[i]
        if direction == Direction.long and candle.close > breakout_level:
            retest_window = window[i + 1 : i + 1 + settings.scalp_retest_max_bars]
            return any(item.low <= breakout_level <= item.close for item in retest_window), "setup_not_confirmed:breakout_retest"
        if direction == Direction.short and candle.close < breakout_level:
            retest_window = window[i + 1 : i + 1 + settings.scalp_retest_max_bars]
            return any(item.high >= breakout_level >= item.close for item in retest_window), "setup_not_confirmed:breakout_retest"
    return False, "setup_not_confirmed:breakout_retest"


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




def _direction_limit_reason(direction: Direction, open_trades: list[object], settings: Settings) -> str | None:
    limit = int(settings.max_open_positions_per_direction or 0)
    if limit <= 0:
        return None
    open_longs = sum(1 for trade in open_trades if getattr(trade, "side", "") == "long")
    open_shorts = sum(1 for trade in open_trades if getattr(trade, "side", "") == "short")
    if direction == Direction.long and open_longs >= limit:
        return "dir_limit_long"
    if direction == Direction.short and open_shorts >= limit:
        return "dir_limit_short"
    return None
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
