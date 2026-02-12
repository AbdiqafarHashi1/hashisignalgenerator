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
from .notifier import format_trade_message, send_telegram_message

logger = logging.getLogger(__name__)


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

    async def run_once(self, force: bool = False) -> list[dict[str, object]]:
        self.last_tick_ts = time.time()
        if self._heartbeat_cb is not None:
            self._heartbeat_cb()
        symbols = list(self._settings.symbols)
        self._last_tick_time = datetime.now(timezone.utc)
        force_mode = force or self._settings.smoke_test_force_trade or self._settings.force_trade_mode
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
                    }
                )
            return results
        for symbol in symbols:
            tick_ts = datetime.now(timezone.utc)
            self._last_symbol_tick_time[symbol] = tick_ts
            try:
                snapshot = await fetch_symbol_klines(
                    symbol=symbol,
                    interval=self._settings.candle_interval,
                    limit=self._settings.candle_history_limit,
                    rest_base=self._settings.bybit_rest_base,
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
                        }
                    )
                    continue
                self._last_snapshots[symbol] = snapshot
                self._last_fetch_counts[symbol] = len(snapshot.candles)
                if self._settings.force_trade_mode or self._settings.smoke_test_force_trade:
                    self._auto_close_forced_trades(symbol, snapshot, tick_ts)
                if self._paper_trader is not None:
                    self._paper_trader.evaluate_open_trades(symbol, snapshot.candle.close)
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
                results.append(
                    {
                        "symbol": symbol,
                        "plan": None,
                        "reason": f"symbol_error:{type(exc).__name__}",
                        "candles_fetched": 0,
                        "latest_candle_ts": None,
                        "decision_status": None,
                        "persisted": False,
                        "dedupe_key": None,
                        "telegram_sent": False,
                        "trade_opened": False,
                        "trade_id": None,
                    }
                )
                continue

            funding_state = _funding_blackout_state(tick_ts, self._settings)
            if funding_state["close_positions"] and self._paper_trader is not None:
                self._paper_trader.force_close_trades(symbol, snapshot.candle.close, reason="funding_blackout_close")

            plan: TradePlan | None = None
            reason: str | None = None
            persisted = False
            decision_status: str | None = None
            dedupe_key: str | None = None
            telegram_sent = False
            trade_opened = False
            trade_id: str | None = None
            if not snapshot.kline_is_closed:
                reason = "candle_open"
            elif funding_state["block_new_entries"]:
                reason = "funding_blackout_active"
            else:
                last_processed = self._state.get_last_processed_close_time_ms(snapshot.symbol)
                if (
                    not force_mode
                    and last_processed == snapshot.kline_close_time_ms
                ):
                    reason = "candle_already_processed"
                else:
                    forced_trade = False
                    if force_mode and self._force_trade_due(snapshot.symbol, tick_ts):
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
                        )
                    if snapshot.kline_is_closed:
                        self._state.set_last_processed_close_time_ms(
                            snapshot.symbol,
                            snapshot.kline_close_time_ms,
                        )
                    if plan.status == Status.TRADE:
                        if forced_trade:
                            dedupe_key = _force_trade_key(snapshot.symbol, plan, tick_ts)
                        else:
                            dedupe_key = _trade_key(snapshot, plan)
                        last_trade_key = self._state.get_last_trade_key(snapshot.symbol)
                        if dedupe_key != last_trade_key:
                            message = format_trade_message(snapshot.symbol, plan)
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
                                )
                                trade_id = self._paper_trader.maybe_open_trade(
                                    snapshot.symbol,
                                    plan,
                                    allow_multiple=allow_multiple,
                                    snapshot=snapshot,
                                    regime=active_mode,
                                )
                                if trade_id is not None:
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
            logger.info(
                "tick symbol=%s ts=%s decision=%s",
                symbol,
                tick_ts.isoformat(),
                plan.status.value if plan is not None else (reason or "none"),
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
                }
            )
        return results

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
        if self._settings.time_stop_minutes <= 0:
            return
        for trade in self._database.fetch_open_trades(symbol):
            opened_at = datetime.fromisoformat(trade.opened_at)
            elapsed_minutes = (now - opened_at).total_seconds() / 60.0
            if elapsed_minutes >= self._settings.time_stop_minutes:
                self._paper_trader.force_close_trades(symbol, price, reason="time_stop")

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
    block_new_entries = minute_in_window >= block_start or minute_in_window <= 1
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
