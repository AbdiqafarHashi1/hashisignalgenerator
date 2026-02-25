from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from ..config import Settings
from ..state import StateStore
from ..utils.trading_day import trading_day_key, trading_day_start
from .database import Database, TradeRecord
from .paper_trader import PaperTrader


@dataclass(frozen=True)
class MetricsSnapshot:
    equity_start: float
    equity_now: float
    realized_gross: float
    realized_fees: float
    realized_net: float
    unrealized_net: float
    fees_today: float
    fees_total: float
    daily_start_equity: float
    peak_equity: float
    daily_dd_pct: float
    global_dd_pct: float
    profit_target_amt: float
    profit_target_progress_pct: float
    trades_today: int
    consecutive_losses: int
    cooldown_remaining: int
    current_time: str
    replay_cursor_time: str | None
    bars_processed: int
    trades_processed: int


def _trade_fee(trade: TradeRecord, cfg: Settings) -> float:
    if getattr(trade, "fees", None) is not None:
        return float(trade.fees or 0.0)
    if trade.exit is None:
        return 0.0
    qty = abs(float(trade.size or 0.0))
    return ((float(trade.entry or 0.0) * qty) + (float(trade.exit or 0.0) * qty)) * (float(cfg.fee_rate_bps or 0.0) / 10000.0)


def compute_metrics(
    cfg: Settings,
    db: Database,
    trader: PaperTrader,
    state_store: StateStore,
    now: datetime,
    replay_status: dict[str, Any] | None = None,
    *,
    persist_runtime_state: bool = True,
) -> dict[str, Any]:
    trades = db.fetch_trades()
    day_start = trading_day_start(now)
    day_key = trading_day_key(now)
    challenge_start_row = db.get_runtime_state("accounting.challenge_start_ts")
    challenge_start_ts = (
        datetime.fromisoformat(str(challenge_start_row.value_text).replace("Z", "+00:00"))
        if challenge_start_row and challenge_start_row.value_text
        else None
    )

    equity_start = float(cfg.account_size or 0.0)
    realized_net = 0.0
    fees_total = 0.0
    unrealized = 0.0
    fees_today = 0.0
    trades_today = 0
    wins_today = 0
    losses_today = 0
    realized_today = 0.0
    scoped_trades: list[TradeRecord] = []

    for trade in trades:
        if trade.closed_at:
            closed_at = datetime.fromisoformat(str(trade.closed_at).replace("Z", "+00:00"))
            if challenge_start_ts is not None and closed_at < challenge_start_ts:
                continue
            scoped_trades.append(trade)
            pnl_net = float(trade.pnl_usd or 0.0)
            fee = _trade_fee(trade, cfg)
            realized_net += pnl_net
            fees_total += fee
            if closed_at >= day_start:
                fees_today += fee
                trades_today += 1
                realized_today += pnl_net
                if pnl_net > 0:
                    wins_today += 1
                elif pnl_net < 0:
                    losses_today += 1
        else:
            opened_at = datetime.fromisoformat(str(trade.opened_at).replace("Z", "+00:00"))
            if challenge_start_ts is not None and opened_at < challenge_start_ts:
                continue
            scoped_trades.append(trade)
            mark = float(trader._last_mark_prices.get(trade.symbol, trade.entry))
            side_sign = 1.0 if trade.side == "long" else -1.0
            unrealized += (mark - trade.entry) * trade.size * side_sign

    realized_gross = realized_net + fees_total
    equity_now = equity_start + realized_net + unrealized

    prev_day_key = db.get_runtime_state("accounting.day_key")
    prev_day_text = prev_day_key.value_text if prev_day_key and prev_day_key.value_text else None
    if persist_runtime_state and prev_day_text != day_key:
        db.set_runtime_state("accounting.day_key", value_text=day_key)
        db.set_runtime_state("accounting.day_start_equity", value_number=equity_now)

    day_start_row = db.get_runtime_state("accounting.day_start_equity")
    daily_start_equity = float(day_start_row.value_number) if day_start_row and day_start_row.value_number is not None else equity_now

    peak_row = db.get_runtime_state("accounting.equity_high_watermark")
    prev_peak = float(peak_row.value_number) if peak_row and peak_row.value_number is not None else equity_now
    peak_equity = max(prev_peak, equity_now)
    if persist_runtime_state:
        db.set_runtime_state("accounting.equity_high_watermark", value_number=peak_equity)

    daily_dd_pct = (max(0.0, daily_start_equity - equity_now) / daily_start_equity) if daily_start_equity > 0 else 0.0
    global_dd_pct = (max(0.0, equity_start - equity_now) / equity_start) if equity_start > 0 else 0.0

    target_pct = float(cfg.prop_profit_target_pct or cfg.daily_profit_target_pct or 0.0)
    profit_target_amt = equity_start * target_pct
    realized_progress = max(0.0, equity_now - equity_start)
    progress_pct = (realized_progress / profit_target_amt * 100.0) if profit_target_amt > 0 else 0.0

    consecutive_losses = max((int(state_store.get_daily_state(s).consecutive_losses) for s in cfg.symbols), default=0)
    cooldown_remaining = max((int((state_store.risk_snapshot(s, cfg, now).get("cooldown_remaining_minutes", 0) or 0) * 60) for s in cfg.symbols), default=0)

    state_store.set_global_equity(equity_now)

    replay_status = replay_status or {}
    snapshot = MetricsSnapshot(
        equity_start=equity_start,
        equity_now=equity_now,
        realized_gross=realized_gross,
        realized_fees=fees_total,
        realized_net=realized_net,
        unrealized_net=unrealized,
        fees_today=fees_today,
        fees_total=fees_total,
        daily_start_equity=daily_start_equity,
        peak_equity=peak_equity,
        daily_dd_pct=daily_dd_pct,
        global_dd_pct=global_dd_pct,
        profit_target_amt=profit_target_amt,
        profit_target_progress_pct=progress_pct,
        trades_today=trades_today,
        consecutive_losses=consecutive_losses,
        cooldown_remaining=cooldown_remaining,
        current_time=now.isoformat(),
        replay_cursor_time=replay_status.get("current_ts"),
        bars_processed=int(replay_status.get("bars_processed", 0) or 0),
        trades_processed=len([t for t in scoped_trades if t.closed_at]),
    )
    data = asdict(snapshot)
    data.update(
        {
            "trades": scoped_trades,
            "trades_all": trades,
            "wins_today": wins_today,
            "losses_today": losses_today,
            "pnl_realized_today": realized_today,
            "challenge_start_ts": challenge_start_ts.isoformat() if challenge_start_ts else None,
        }
    )
    return data
