from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from signal_engine.accounting import AccountingSnapshot


@dataclass(frozen=True)
class GovernorBlock:
    code: str
    message: str
    remaining_trades: int
    remaining_daily_loss_room: float
    remaining_global_dd_room: float
    cooldown_remaining: int


def _as_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def evaluate_prop_block(
    *,
    now: datetime,
    day_key: str,
    stored_day_key: str | None,
    daily_trades: int,
    consecutive_losses: int,
    last_loss_at: datetime | None,
    accounting: AccountingSnapshot,
    max_daily_loss_pct: float,
    max_global_dd_pct: float,
    max_trades_per_day: int,
    max_consecutive_losses: int,
    cooldown_minutes: int,
) -> GovernorBlock | None:
    if stored_day_key and stored_day_key != day_key:
        daily_trades = 0
        consecutive_losses = 0
        last_loss_at = None

    remaining_trades = max(0, max_trades_per_day - daily_trades)
    daily_room = max(0.0, max_daily_loss_pct + accounting.daily_dd_pct)
    global_room = max(0.0, max_global_dd_pct + accounting.global_dd_pct)
    cooldown_remaining = 0
    if last_loss_at is not None:
        elapsed = now - _as_utc(last_loss_at)
        cooldown_remaining = max(0, int((timedelta(minutes=cooldown_minutes) - elapsed).total_seconds() // 60))

    if accounting.daily_dd_pct <= -max_daily_loss_pct:
        return GovernorBlock("daily_loss_limit", "Daily loss breached", remaining_trades, daily_room, global_room, cooldown_remaining)
    if accounting.global_dd_pct <= -max_global_dd_pct:
        return GovernorBlock("global_dd_limit", "Global drawdown breached", remaining_trades, daily_room, global_room, cooldown_remaining)
    if daily_trades >= max_trades_per_day:
        return GovernorBlock("max_trades_per_day", "Max trades per day reached", remaining_trades, daily_room, global_room, cooldown_remaining)
    if consecutive_losses >= max_consecutive_losses:
        return GovernorBlock("max_consecutive_losses", "Consecutive loss cap reached", remaining_trades, daily_room, global_room, cooldown_remaining)
    if cooldown_remaining > 0:
        return GovernorBlock("loss_cooldown", "Loss cooldown active", remaining_trades, daily_room, global_room, cooldown_remaining)
    return None
