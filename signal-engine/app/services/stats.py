from __future__ import annotations

from dataclasses import dataclass

from .database import TradeRecord


@dataclass
class StatsSummary:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    equity_curve: list[float]


def compute_stats(trades: list[TradeRecord]) -> StatsSummary:
    closed = [trade for trade in trades if trade.closed_at]
    pnls = [trade.pnl_usd or 0.0 for trade in closed]
    total_trades = len(closed)
    wins = len([trade for trade in closed if (trade.pnl_usd or 0.0) > 0])
    losses = len([trade for trade in closed if (trade.pnl_usd or 0.0) <= 0])
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / total_trades if total_trades else 0.0
    win_rate = wins / total_trades if total_trades else 0.0
    avg_win = (
        sum(trade.pnl_usd or 0.0 for trade in closed if (trade.pnl_usd or 0.0) > 0) / wins
        if wins
        else 0.0
    )
    avg_loss = (
        sum(trade.pnl_usd or 0.0 for trade in closed if (trade.pnl_usd or 0.0) <= 0) / losses
        if losses
        else 0.0
    )
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    profit_factor = (
        (sum(trade.pnl_usd or 0.0 for trade in closed if (trade.pnl_usd or 0.0) > 0))
        / abs(sum(trade.pnl_usd or 0.0 for trade in closed if (trade.pnl_usd or 0.0) < 0))
        if any((trade.pnl_usd or 0.0) < 0 for trade in closed)
        else 0.0
    )
    equity_curve = []
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
    return StatsSummary(
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        expectancy=expectancy,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        equity_curve=equity_curve,
    )
