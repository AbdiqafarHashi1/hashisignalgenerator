from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from ..config import Settings
from ..models import (
    BiasSignal,
    DecisionRequest,
    Direction,
    MarketSnapshot,
    SetupType,
    Status,
    TradePlan,
)
from ..state import StateStore
from ..strategy.decision import decide
from ..providers.binance import BinanceKlineSnapshot, fetch_symbol_klines
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
        self._heartbeat_cb = heartbeat_cb
        self._last_tick_time: datetime | None = None
        self._last_snapshots: dict[str, BinanceKlineSnapshot] = {}
        self._last_fetch_counts: dict[str, int] = {}
        self._last_symbol_tick_time: dict[str, datetime] = {}

        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def tick_interval(self) -> int:
        return self._interval

    def last_tick_time(self) -> datetime | None:
        return self._last_tick_time

    def last_snapshot(self, symbol: str) -> BinanceKlineSnapshot | None:
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
            self._task = asyncio.create_task(self._run_loop())
            logger.info("scheduler_start status=started interval=%s", self._interval)
            return True

    async def stop(self) -> bool:
        async with self._lock:
            if not self.running:
                logger.info("scheduler_stop skipped=already_stopped")
                return False
            self._stop_event.set()
            task = self._task
        if task is not None:
            await task
        logger.info("scheduler_stop status=stopped")
        return True

    async def run_once(self, force: bool = False) -> list[dict[str, object]]:
        if self._heartbeat_cb is not None:
            self._heartbeat_cb()
        symbols = list(self._settings.symbols)
        self._last_tick_time = datetime.now(timezone.utc)
        logger.info("scheduler_tick_start symbols=%s force=%s", ",".join(symbols), force)
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
            snapshot = await fetch_symbol_klines(
                symbol=symbol,
                interval=self._settings.candle_interval,
                limit=self._settings.candle_history_limit,
            )
            self._last_snapshots[symbol] = snapshot
            self._last_fetch_counts[symbol] = len(snapshot.candles)
            logger.info(
                "candle_fetch symbol=%s candles=%s latest=%s closed=%s",
                snapshot.symbol,
                len(snapshot.candles),
                snapshot.candle.close_time.isoformat(),
                snapshot.kline_is_closed,
            )

            plan: TradePlan | None = None
            reason: str | None = None
            persisted = False
            decision_status: str | None = None
            dedupe_key: str | None = None
            telegram_sent = False
            trade_opened = False
            trade_id: str | None = None
            if not force and not snapshot.kline_is_closed:
                reason = "candle_open"
            else:
                last_processed = self._state.get_last_processed_close_time_ms(snapshot.symbol)
                if not force and last_processed == snapshot.kline_close_time_ms:
                    reason = "candle_already_processed"
                else:
                    request = _build_decision_request(snapshot)
                    plan = decide(request, self._state, self._settings)
                    decision_status = plan.status.value
                    self._state.set_latest_decision(snapshot.symbol, plan)
                    persisted = True
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
                                trade_id = self._paper_trader.maybe_open_trade(snapshot.symbol, plan)
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

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue


def _trend_strength(candle: BinanceKlineSnapshot) -> float:
    price_move = abs(candle.candle.close - candle.candle.open)
    if candle.candle.open == 0:
        return 0.0
    strength = (price_move / candle.candle.open) * 10
    return min(1.0, strength)


def _build_decision_request(snapshot: BinanceKlineSnapshot) -> DecisionRequest:
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


def _trade_key(snapshot: BinanceKlineSnapshot, plan: TradePlan) -> str:
    return (
        f"{snapshot.symbol}:{snapshot.kline_close_time_ms}:"
        f"{plan.direction.value}"
    )
