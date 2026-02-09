from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from .database import TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class StatsSummary:
    total_trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    equity_curve: list[float]


def _pnl_value(trade: TradeRecord) -> float:
    """
    Returns the realized PnL for a trade.
    Current implementation uses pnl_usd if present; otherwise 0.0.

    NOTE: If you later want compute-on-the-fly PnL when pnl_usd is None,
    this is the function to extend (needs TradeRecord entry/exit/side/size).
    """
    return float(trade.pnl_usd) if getattr(trade, "pnl_usd", None) is not None else 0.0


def compute_stats(trades: list[TradeRecord], *, debug: bool = False) -> StatsSummary:
    # Closed trades only
    closed = [t for t in trades if getattr(t, "closed_at", None)]

    pnls = [_pnl_value(t) for t in closed]
    total_trades = len(closed)

    wins = sum(1 for p in pnls if p > 0.0)
    losses = sum(1 for p in pnls if p < 0.0)
    breakeven = sum(1 for p in pnls if p == 0.0)

    total_pnl = sum(pnls)
    avg_pnl = (total_pnl / total_trades) if total_trades else 0.0
    win_rate = (wins / total_trades) if total_trades else 0.0

    # Average win/loss
    sum_wins = sum(p for p in pnls if p > 0.0)
    sum_losses = sum(p for p in pnls if p < 0.0)  # negative number

    avg_win = (sum_wins / wins) if wins else 0.0
    avg_loss = (sum_losses / losses) if losses else 0.0  # negative

    # Expectancy in USD per trade
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)

    # Profit factor = gross profit / gross loss (absolute)
    # If no losing trades, set PF to 0.0 (or you can choose float("inf"))
    profit_factor = (sum_wins / abs(sum_losses)) if sum_losses < 0.0 else 0.0

    # Equity curve (running pnl) + max drawdown
    equity_curve: list[float] = []
    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        running += pnl
        equity_curve.append(running)
        if running > peak:
            peak = running
        drawdown = peak - running
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    if debug:
        neg_samples = [
            (getattr(t, "id", None), _pnl_value(t), getattr(t, "result", None))
            for t in closed
            if _pnl_value(t) < 0.0
        ][:5]
        logger.info(
            "STATS_DEBUG closed=%d wins=%d losses=%d breakeven=%d total_pnl=%.6f sample_neg=%s",
            total_trades,
            wins,
            losses,
            breakeven,
            total_pnl,
            neg_samples,
        )

    return StatsSummary(
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        expectancy=expectancy,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        equity_curve=equity_curve,
    )