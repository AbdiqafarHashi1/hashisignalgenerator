from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class AccountingSnapshot:
    equity_start: float
    equity_now: float
    day_start_equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    daily_dd_pct: float
    global_dd_pct: float
    hwm: float
    trading_days_count: int
    progress_to_target_pct: float
    profit_target_reached: bool


def _as_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def compute_accounting_snapshot(
    *,
    equity_start: float,
    realized_pnl: float,
    unrealized_pnl: float,
    fees: float,
    day_start_equity: float,
    hwm: float,
    trade_close_dates: list[str | datetime],
    profit_target_pct: float,
) -> AccountingSnapshot:
    equity_now = equity_start + realized_pnl + unrealized_pnl - fees
    daily_dd_pct = ((equity_now - day_start_equity) / day_start_equity) if day_start_equity else 0.0
    active_hwm = max(hwm, equity_now)
    global_dd_pct = ((equity_now - active_hwm) / active_hwm) if active_hwm else 0.0
    trading_days = len({_as_utc(v).date().isoformat() for v in trade_close_dates})
    progress = ((equity_now - equity_start) / equity_start) if equity_start else 0.0
    return AccountingSnapshot(
        equity_start=equity_start,
        equity_now=equity_now,
        day_start_equity=day_start_equity,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        fees=fees,
        daily_dd_pct=daily_dd_pct,
        global_dd_pct=global_dd_pct,
        hwm=active_hwm,
        trading_days_count=trading_days,
        progress_to_target_pct=progress,
        profit_target_reached=progress >= profit_target_pct,
    )
