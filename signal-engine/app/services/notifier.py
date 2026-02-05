from __future__ import annotations

from datetime import datetime

import httpx

from ..config import Settings
from ..models import TradePlan
from ..state import StateStore
from .stats import StatsSummary


def format_trade_message(symbol: str, plan: TradePlan) -> str:
    entry = "-"
    if plan.entry_zone is not None:
        entry = f"{plan.entry_zone[0]:.2f} - {plan.entry_zone[1]:.2f}"
    stop = "-" if plan.stop_loss is None else f"{plan.stop_loss:.2f}"
    take_profit = "-" if plan.take_profit is None else f"{plan.take_profit:.2f}"
    score = "-" if plan.signal_score is None else str(plan.signal_score)
    return (
        "📈 Trade Signal\n"
        f"Symbol: {symbol}\n"
        f"Status: {plan.status.value}\n"
        f"Direction: {plan.direction.value}\n"
        f"Entry: {entry}\n"
        f"Stop: {stop}\n"
        f"Take Profit: {take_profit}\n"
        f"Score: {score}"
    )


def format_trade_open_message(symbol: str, entry: float, stop: float, take_profit: float, size: float) -> str:
    return (
        "🟢 Paper Trade Opened\n"
        f"Symbol: {symbol}\n"
        f"Entry: {entry:.2f}\n"
        f"Stop: {stop:.2f}\n"
        f"Take Profit: {take_profit:.2f}\n"
        f"Size: {size:.2f}"
    )


def format_trade_close_message(symbol: str, exit_price: float, pnl_usd: float, pnl_r: float, result: str) -> str:
    return (
        "🔴 Paper Trade Closed\n"
        f"Symbol: {symbol}\n"
        f"Exit: {exit_price:.2f}\n"
        f"PnL: {pnl_usd:.2f} ({pnl_r:.2f}R)\n"
        f"Result: {result}"
    )


def format_status_message(
    settings: Settings,
    state: StateStore,
    stats: StatsSummary,
    symbols: list[str],
    engine_running: bool,
) -> str:
    decisions = []
    for symbol in symbols:
        daily_state = state.get_daily_state(symbol)
        latest = daily_state.latest_decision
        if latest is None:
            decisions.append(f"{symbol}: none")
        else:
            decisions.append(f"{symbol}: {latest.status.value} ({latest.signal_score or '-'})")
    return (
        "🧠 Engine Status\n"
        f"Running: {'yes' if engine_running else 'no'}\n"
        f"Symbols: {', '.join(symbols)}\n"
        f"Mode: {settings.engine_mode}\n"
        f"Trades today: {sum(state.get_daily_state(symbol).trades for symbol in symbols)}\n"
        f"Total trades: {stats.total_trades}\n"
        f"Win rate: {stats.win_rate:.2%}\n"
        f"PnL today: {sum(state.get_daily_state(symbol).pnl_usd for symbol in symbols):.2f}\n"
        f"Total PnL: {stats.total_pnl:.2f}\n"
        f"Last decisions: {' | '.join(decisions)}"
    )


def format_heartbeat_message(
    last_candle_time: datetime | None,
    trades_today: int,
    pnl_today: float,
) -> str:
    last_candle = last_candle_time.isoformat() if last_candle_time else "unknown"
    return (
        "✅ Engine alive\n"
        f"Last candle: {last_candle}\n"
        f"Trades today: {trades_today}\n"
        f"PnL today: {pnl_today:.2f}"
    )


async def send_telegram_message(message: str, settings: Settings) -> bool:
    if not settings.telegram_enabled:
        return False
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {"chat_id": settings.telegram_chat_id, "text": message}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

    return True


async def fetch_updates(settings: Settings, offset: int | None) -> list[dict]:
    if not settings.telegram_bot_token:
        return []
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
    params = {"timeout": 1}
    if offset is not None:
        params["offset"] = offset
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    return payload.get("result", [])


async def handle_status_commands(
    settings: Settings,
    state: StateStore,
    symbols: list[str],
    stats: StatsSummary,
    engine_running: bool,
) -> None:
    if not settings.telegram_enabled:
        return
    offset = state.get_last_telegram_update_id()
    updates = await fetch_updates(settings, None if offset is None else offset + 1)
    last_update_id: int | None = None
    for update in updates:
        update_id = update.get("update_id")
        message = update.get("message") or {}
        text = (message.get("text") or "").strip()
        if text == "/status":
            status_message = format_status_message(settings, state, stats, symbols, engine_running)
            sent = await send_telegram_message(status_message, settings)
            if sent and isinstance(update_id, int):
                last_update_id = update_id
        if isinstance(update_id, int):
            last_update_id = update_id
    if last_update_id is not None:
        state.set_last_telegram_update_id(last_update_id)
