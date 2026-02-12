from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .database import TradeRecord


@dataclass
class TradePerformance:
    trade_id: int
    symbol: str
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float
    hold_seconds: float
    reason: str


@dataclass
class PerformanceSnapshot:
    generated_at: str
    trades: list[dict[str, Any]]
    trades_today: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy_r: float
    profit_factor: float
    max_drawdown_pct: float
    consecutive_wins: int
    consecutive_losses: int
    avg_hold_time: float
    sharpe_like: float
    skip_reason_counts: dict[str, int]
    equity_curve: list[float]
    drawdown_curve_pct: list[float]
    win_loss_distribution: dict[str, int]


def _to_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    txt = value.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _normalize_reason(result: str | None) -> str:
    value = (result or "manual").lower()
    if "tp" in value:
        return "tp"
    if "sl" in value:
        return "sl"
    if "time" in value or "timeout" in value:
        return "timeout"
    return "manual"


def _trade_performance(trade: TradeRecord) -> TradePerformance | None:
    if trade.closed_at is None or trade.exit is None:
        return None
    entry_time = _to_utc(trade.opened_at)
    exit_time = _to_utc(trade.closed_at)
    if entry_time is None or exit_time is None:
        return None
    pnl_usd = float(trade.pnl_usd or 0.0)
    signed_move = (trade.exit - trade.entry) if trade.side == "long" else (trade.entry - trade.exit)
    pnl_pct = (signed_move / trade.entry) * 100 if trade.entry else 0.0
    stop_distance = abs(trade.entry - trade.stop)
    r_multiple = (signed_move / stop_distance) if stop_distance else 0.0
    return TradePerformance(
        trade_id=int(trade.id),
        symbol=trade.symbol,
        entry_time=entry_time.isoformat(),
        exit_time=exit_time.isoformat(),
        side=trade.side,
        entry_price=float(trade.entry),
        exit_price=float(trade.exit),
        pnl_usd=pnl_usd,
        pnl_pct=pnl_pct,
        r_multiple=float(trade.pnl_r if trade.pnl_r is not None else r_multiple),
        hold_seconds=max(0.0, (exit_time - entry_time).total_seconds()),
        reason=_normalize_reason(trade.result),
    )


def _max_consecutive(values: list[bool], target: bool) -> int:
    best = 0
    current = 0
    for value in values:
        if value == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def build_performance_snapshot(
    trades: list[TradeRecord],
    *,
    account_size: float,
    skip_reason_counts: dict[str, int] | None = None,
) -> PerformanceSnapshot:
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    filtered = [t for t in trades if t.closed_at and getattr(t, "trade_mode", "paper") != "test"]
    ordered = sorted(filtered, key=lambda t: t.closed_at or "")

    trade_rows = [perf for perf in (_trade_performance(t) for t in ordered) if perf is not None]

    pnls = [trade.pnl_usd for trade in trade_rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    r_values = [trade.r_multiple for trade in trade_rows]

    trades_today = sum(1 for trade in trade_rows if _to_utc(trade.exit_time) and _to_utc(trade.exit_time) >= start_of_day)
    total = len(trade_rows)
    win_rate = (len(wins) / total) if total else 0.0
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    expectancy_r = mean(r_values) if r_values else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else 0.0

    equity_curve: list[float] = []
    drawdown_curve_pct: list[float] = []
    running = 0.0
    peak = 0.0
    for pnl in pnls:
        running += pnl
        equity_curve.append(running)
        peak = max(peak, running)
        drawdown = peak - running
        drawdown_curve_pct.append((drawdown / account_size) * 100 if account_size else 0.0)
    max_drawdown_pct = max(drawdown_curve_pct) if drawdown_curve_pct else 0.0

    outcomes = [p > 0 for p in pnls if p != 0]
    consecutive_wins = _max_consecutive(outcomes, True)
    consecutive_losses = _max_consecutive(outcomes, False)
    avg_hold_time = mean([trade.hold_seconds for trade in trade_rows]) if trade_rows else 0.0

    sharpe_like = 0.0
    if pnls:
        std_dev = pstdev(pnls)
        if std_dev > 0:
            sharpe_like = mean(pnls) / std_dev

    distribution = {
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": sum(1 for p in pnls if p == 0),
    }

    return PerformanceSnapshot(
        generated_at=now.isoformat(),
        trades=[asdict(trade) for trade in trade_rows],
        trades_today=trades_today,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        consecutive_wins=consecutive_wins,
        consecutive_losses=consecutive_losses,
        avg_hold_time=avg_hold_time,
        sharpe_like=sharpe_like,
        skip_reason_counts=dict(skip_reason_counts or {}),
        equity_curve=equity_curve,
        drawdown_curve_pct=drawdown_curve_pct,
        win_loss_distribution=distribution,
    )


class PerformanceStore:
    def __init__(self, data_dir: str) -> None:
        self._path = Path(data_dir) / "performance.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: PerformanceSnapshot) -> None:
        payload = asdict(snapshot)
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load(self) -> dict[str, Any] | None:
        if not self._path.exists():
            return None
        with self._path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
