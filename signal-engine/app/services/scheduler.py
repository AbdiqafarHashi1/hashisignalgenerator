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
from ..providers.binance import BinanceKlineSnapshot, fetch_klines
from .database import Database
from .notifier import (
    format_heartbeat_message,
    format_trade_close_message,
    format_trade_message,
    format_trade_open_message,
    handle_status_commands,
    send_telegram_message,
)
from .paper_trader import PaperTrader
from .stats import compute_stats


class DecisionScheduler:
    def __init__(
        self,
        settings: Settings,
        state: StateStore,
        database: Database,
        paper_trader: PaperTrader,
        interval_seconds: int = 60,
    ) -> None:
        self._settings = settings
        self._state = state
        self._database = database
        self._paper_trader = paper_trader
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

    async def run_once(
        self,
        force: bool = False,
        symbols: list[str] | None = None,
    ) -> dict[str, dict[str, str | TradePlan | None]]:
        results: dict[str, dict[str, str | TradePlan | None]] = {}
        targets = symbols or self._state.get_symbols()
        for symbol in targets:
            try:
                result = await self._process_symbol(symbol, force=force)
            except Exception as exc:  # noqa: BLE001
                results[symbol] = {"status": "error", "reason": str(exc), "plan": None}
                continue
            results[symbol] = result
        return results

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            await self.run_once()
            await self._handle_heartbeat()
            await self._handle_status_commands()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue

    async def _process_symbol(self, symbol: str, force: bool = False) -> dict[str, str | TradePlan | None]:
        snapshot = await fetch_klines(symbol)
        if not force and not snapshot.kline_is_closed:
            return {"status": "skipped", "reason": "candle_open", "plan": None}
        last_processed = self._state.get_last_processed_close_time_ms(symbol)
        if not force and last_processed == snapshot.kline_close_time_ms:
            return {"status": "skipped", "reason": "candle_already_processed", "plan": None}
        request = _build_decision_request(snapshot)
        plan = decide(request, self._state, self._settings)
        self._state.set_latest_decision(symbol, plan)
        if snapshot.kline_is_closed:
            self._state.set_last_processed_close_time_ms(symbol, snapshot.kline_close_time_ms)
        self._database.add_signal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            score=plan.signal_score,
            status=plan.status.value,
            rationale=",".join(plan.rationale),
            entry=None if plan.entry_zone is None else sum(plan.entry_zone) / 2.0,
            stop=plan.stop_loss,
            take_profit=plan.take_profit,
        )
        await self._maybe_notify_signal(symbol, snapshot, plan)
        await self._maybe_paper_trade(symbol, snapshot, plan)
        return {"status": "ok", "reason": None, "plan": plan}

    async def _maybe_notify_signal(self, symbol: str, snapshot: BinanceKlineSnapshot, plan: TradePlan) -> None:
        if plan.status != Status.TRADE:
            return
        dedupe_key = _notification_key(snapshot, plan)
        last_notified = self._state.get_last_notified_key(symbol)
        if dedupe_key == last_notified:
            return
        message = format_trade_message(symbol, plan)
        sent = await send_telegram_message(message, self._settings)
        if sent:
            self._state.set_last_notified_key(symbol, dedupe_key)

    async def _maybe_paper_trade(self, symbol: str, snapshot: BinanceKlineSnapshot, plan: TradePlan) -> None:
        if self._settings.engine_mode != "paper":
            return
        results = self._paper_trader.evaluate_open_trades(symbol, snapshot.price)
        for result in results:
            close_message = format_trade_close_message(
                result.symbol,
                result.exit_price,
                result.pnl_usd,
                result.pnl_r,
                result.result,
            )
            self._state.record_outcome(result.symbol, result.pnl_usd, result.result == "win", datetime.now(timezone.utc))
            await send_telegram_message(close_message, self._settings)
        if plan.status != Status.TRADE:
            return
        trade_id = self._paper_trader.maybe_open_trade(symbol, plan)
        if trade_id is None:
            return
        self._state.record_trade(symbol)
        entry = sum(plan.entry_zone) / 2.0 if plan.entry_zone else snapshot.price
        open_message = format_trade_open_message(
            symbol,
            entry,
            plan.stop_loss or entry,
            plan.take_profit or entry,
            plan.position_size_usd or 0.0,
        )
        await send_telegram_message(open_message, self._settings)

    async def _handle_heartbeat(self) -> None:
        if not self._settings.telegram_enabled:
            return
        last_sent = self._state.get_last_heartbeat_ts()
        now = datetime.now(timezone.utc)
        if last_sent and (now - last_sent).total_seconds() < self._settings.heartbeat_minutes * 60:
            return
        symbols = self._state.get_symbols()
        trades_today = sum(self._state.get_daily_state(symbol).trades for symbol in symbols)
        pnl_today = sum(self._state.get_daily_state(symbol).pnl_usd for symbol in symbols)
        last_candle_time = None
        if symbols:
            last_processed = self._state.get_last_processed_close_time_ms(symbols[0])
            if last_processed:
                last_candle_time = datetime.fromtimestamp(last_processed / 1000, tz=timezone.utc)
        message = format_heartbeat_message(last_candle_time, trades_today, pnl_today)
        sent = await send_telegram_message(message, self._settings)
        if sent:
            self._state.set_last_heartbeat_ts(now)

    async def _handle_status_commands(self) -> None:
        stats = compute_stats(self._database.fetch_trades())
        await handle_status_commands(
            settings=self._settings,
            state=self._state,
            symbols=self._state.get_symbols(),
            stats=stats,
            engine_running=self.running,
        )

    async def trigger_heartbeat(self) -> None:
        await self._handle_heartbeat()


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
        f"{plan.direction.value}:{entry}:{stop}:{take_profit}:{score}"
    )
