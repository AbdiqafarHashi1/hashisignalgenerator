from __future__ import annotations

import asyncio
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
from ..providers.binance import BinanceKlineSnapshot, fetch_btcusdt_klines
from .notifier import format_trade_message, send_telegram_message


class DecisionScheduler:
    def __init__(self, settings: Settings, state: StateStore, interval_seconds: int = 60) -> None:
        self._settings = settings
        self._state = state
        self._interval = interval_seconds
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> bool:
        async with self._lock:
            if self.running:
                return False
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(self._run_loop())
            return True

    async def stop(self) -> bool:
        async with self._lock:
            if not self.running:
                return False
            self._stop_event.set()
            task = self._task
        if task is not None:
            await task
        return True

    async def run_once(self, force: bool = False) -> tuple[TradePlan | None, str | None]:
        snapshot = await fetch_btcusdt_klines()
        if not force and not snapshot.kline_is_closed:
            return None, "candle_open"
        last_processed = self._state.get_last_processed_close_time_ms()
        if not force and last_processed == snapshot.kline_close_time_ms:
            return None, "candle_already_processed"
        request = _build_decision_request(snapshot)
        plan = decide(request, self._state, self._settings)
        self._state.set_latest_decision(snapshot.symbol, plan)
        if snapshot.kline_is_closed:
            self._state.set_last_processed_close_time_ms(snapshot.kline_close_time_ms)
        if plan.status == Status.TRADE:
            dedupe_key = _notification_key(snapshot, plan)
            last_notified = self._state.get_last_notified_key()
            if dedupe_key != last_notified:
                message = format_trade_message(snapshot.symbol, plan)
                await send_telegram_message(message, self._settings)
                self._state.set_last_notified_key(dedupe_key)
        return plan, None

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
    }
    return DecisionRequest(
        tradingview=tradingview_payload,
        market=market,
        bias=bias,
        timestamp=datetime.now(timezone.utc),
    )


def _notification_key(snapshot: BinanceKlineSnapshot, plan: TradePlan) -> str:
    entry = (
        "none"
        if plan.entry_zone is None
        else f"{plan.entry_zone[0]:.2f}-{plan.entry_zone[1]:.2f}"
    )
    stop = "none" if plan.stop_loss is None else f"{plan.stop_loss:.2f}"
    take_profit = "none" if plan.take_profit is None else f"{plan.take_profit:.2f}"
    score = "none" if plan.signal_score is None else str(plan.signal_score)
    return (
        f"{snapshot.symbol}:{snapshot.kline_close_time_ms}:"
        f"{plan.status.value}:{plan.direction.value}:{entry}:{stop}:{take_profit}:{score}"
    )
