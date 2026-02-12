from __future__ import annotations

import httpx

from ..config import Settings
from ..models import TradePlan
from ..providers.bybit import BybitKlineSnapshot


def format_trade_message(symbol: str, plan: TradePlan, snapshot: BybitKlineSnapshot | None = None) -> str:
    entry = "-"
    if plan.entry_zone is not None:
        entry = f"{plan.entry_zone[0]:.2f} - {plan.entry_zone[1]:.2f}"
    stop = "-" if plan.stop_loss is None else f"{plan.stop_loss:.2f}"
    take_profit = "-" if plan.take_profit is None else f"{plan.take_profit:.2f}"
    score = "-" if plan.signal_score is None else str(plan.signal_score)
    if snapshot is not None:
        last_price = getattr(getattr(snapshot, "candle", None), "close", None)
        provider_name = getattr(snapshot, "provider_name", "unknown")
        provider_category = getattr(snapshot, "provider_category", "unknown")
        provider_symbol = getattr(snapshot, "symbol", symbol)
        last_price_text = "-" if last_price is None else f"{float(last_price):.2f}"
        return (
            "📈 Trade Signal\n"
            f"Symbol: {symbol}\n"
            f"Status: {plan.status.value}\n"
            f"Direction: {plan.direction.value}\n"
            f"Entry: {entry}\n"
            f"Stop: {stop}\n"
            f"Take Profit: {take_profit}\n"
            f"Score: {score}\n"
            f"Last price: {last_price_text}\n"
            f"Data source: {provider_name} {provider_category} {provider_symbol}"
        )
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
