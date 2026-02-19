from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import Settings
from ..state import StateStore
from ..utils.trading_day import trading_day_key, trading_day_start
from .database import Database, TradeRecord
from .paper_trader import PaperTrader

EPSILON = 1e-9


def _trade_fee(trade: TradeRecord, cfg: Settings) -> float:
    if getattr(trade, "fees", None) is not None:
        return float(trade.fees or 0.0)
    if trade.exit is None:
        return 0.0
    return ((trade.entry * trade.size) + (trade.exit * trade.size)) * (float(cfg.fee_rate_bps or 0.0) / 10000)


def build_dashboard_metrics(cfg: Settings, db: Database, trader: PaperTrader, state_store: StateStore, now: datetime) -> dict[str, Any]:
    """Canonical dashboard/accounting metrics.

    Root cause fixed here: old logic mixed net/gross realized pnl and used daily *peak* for daily DD,
    which inflated daily DD after intraday gains. This model uses one reconciliation identity and
    day-start/high-watermark anchors consistently.
    """
    trades = db.fetch_trades()
    day_start = trading_day_start(now)
    day_key = trading_day_key(now)

    equity_start = float(cfg.account_size or 0.0)
    pnl_realized_net_total = 0.0
    fees_total = 0.0
    pnl_unrealized = 0.0

    pnl_realized_today = 0.0
    fees_today = 0.0
    trades_today = 0
    wins_today = 0
    losses_today = 0

    trades_today_by_symbol: dict[str, int] = {}
    realized_pnl_by_symbol: dict[str, float] = {}
    fees_by_symbol: dict[str, float] = {}

    for trade in trades:
        symbol = str(getattr(trade, "symbol", ""))
        if trade.closed_at is not None:
            pnl_net = float(getattr(trade, "pnl_usd", 0.0) or 0.0)
            fee = _trade_fee(trade, cfg)
            pnl_realized_net_total += pnl_net
            fees_total += fee
            realized_pnl_by_symbol[symbol] = realized_pnl_by_symbol.get(symbol, 0.0) + pnl_net
            fees_by_symbol[symbol] = fees_by_symbol.get(symbol, 0.0) + fee
            closed_at = datetime.fromisoformat(str(trade.closed_at).replace("Z", "+00:00"))
            if closed_at >= day_start:
                trades_today += 1
                trades_today_by_symbol[symbol] = trades_today_by_symbol.get(symbol, 0) + 1
                pnl_realized_today += pnl_net
                fees_today += fee
                if pnl_net > 0:
                    wins_today += 1
                elif pnl_net < 0:
                    losses_today += 1
        else:
            mark = float(trader._last_mark_prices.get(trade.symbol, trade.entry))
            side_sign = 1.0 if trade.side == "long" else -1.0
            pnl_unrealized += (mark - trade.entry) * trade.size * side_sign

    # Canonical formula identity uses gross realized = net + fees.
    pnl_realized_total = pnl_realized_net_total + fees_total
    equity_now = equity_start + pnl_realized_total + pnl_unrealized - fees_total
    balance = equity_start + pnl_realized_total - fees_total
    equity_reconcile_delta = equity_now - (equity_start + pnl_realized_total + pnl_unrealized - fees_total)

    prev_day_key_row = db.get_runtime_state("accounting.day_key")
    prev_day_key = prev_day_key_row.value_text if prev_day_key_row and prev_day_key_row.value_text else None
    if prev_day_key != day_key:
        db.set_runtime_state("accounting.day_key", value_text=day_key)
        db.set_runtime_state("accounting.day_start_equity", value_number=equity_now)

    day_start_row = db.get_runtime_state("accounting.day_start_equity")
    day_start_equity = float(day_start_row.value_number) if day_start_row and day_start_row.value_number is not None else equity_now

    day_dd_abs = max(0.0, day_start_equity - equity_now)
    day_dd_pct = (day_dd_abs / day_start_equity) if day_start_equity > 0 else 0.0

    peak_row = db.get_runtime_state("accounting.equity_high_watermark")
    prev_peak = float(peak_row.value_number) if peak_row and peak_row.value_number is not None else equity_now
    equity_high_watermark = max(prev_peak, equity_now)
    db.set_runtime_state("accounting.equity_high_watermark", value_number=equity_high_watermark)

    global_dd_abs = max(0.0, equity_high_watermark - equity_now)
    global_dd_pct = (global_dd_abs / equity_high_watermark) if equity_high_watermark > 0 else 0.0

    max_abs_row = db.get_runtime_state("accounting.max_global_dd_abs")
    max_global_dd_abs_prev = float(max_abs_row.value_number) if max_abs_row and max_abs_row.value_number is not None else 0.0
    max_global_dd_abs = max(max_global_dd_abs_prev, global_dd_abs)
    db.set_runtime_state("accounting.max_global_dd_abs", value_number=max_global_dd_abs)

    max_pct_row = db.get_runtime_state("accounting.max_global_dd_pct")
    max_global_dd_pct_prev = float(max_pct_row.value_number) if max_pct_row and max_pct_row.value_number is not None else 0.0
    max_global_dd_pct = max(max_global_dd_pct_prev, global_dd_pct)
    db.set_runtime_state("accounting.max_global_dd_pct", value_number=max_global_dd_pct)

    state_store.set_global_equity(equity_now)

    return {
        "metrics_version": 1,
        "trades": trades,
        "equity_start": equity_start,
        "balance": balance,
        "pnl_realized_total": pnl_realized_total,
        "pnl_realized_net_total": pnl_realized_net_total,
        "pnl_unrealized": pnl_unrealized,
        "fees_total": fees_total,
        "equity_now": equity_now,
        "equity_reconcile_delta": 0.0 if abs(equity_reconcile_delta) < EPSILON else equity_reconcile_delta,
        "day_start_equity": day_start_equity,
        "pnl_realized_today": pnl_realized_today,
        "fees_today": fees_today,
        "trades_today": trades_today,
        "wins_today": wins_today,
        "losses_today": losses_today,
        "daily_dd_abs": day_dd_abs,
        "daily_dd_pct": day_dd_pct,
        "equity_high_watermark": equity_high_watermark,
        "global_dd_abs": global_dd_abs,
        "global_dd_pct": global_dd_pct,
        "max_global_dd_abs": max_global_dd_abs,
        "max_global_dd_pct": max_global_dd_pct,
        "trades_today_by_symbol": trades_today_by_symbol,
        "realized_pnl_by_symbol": realized_pnl_by_symbol,
        "fees_by_symbol": fees_by_symbol,
    }
