from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import Settings
from ..models import Direction, TradePlan
from .database import Database


@dataclass
class PaperTradeResult:
    trade_id: int
    symbol: str
    exit_price: float
    pnl_usd: float
    pnl_r: float
    result: str


class PaperTrader:
    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._db = database

    def maybe_open_trade(self, symbol: str, plan: TradePlan, allow_multiple: bool = False) -> int | None:
        if not allow_multiple and self._db.fetch_open_trades(symbol):
            return None
        if plan.entry_zone is None or plan.stop_loss is None or plan.take_profit is None:
            return None
        entry = sum(plan.entry_zone) / 2.0
        size = plan.position_size_usd or self._settings.account_size * (plan.risk_pct_used or 0)
        side = "long" if plan.direction == Direction.long else "short"
        return self._db.open_trade(
            symbol=symbol,
            entry=entry,
            stop=plan.stop_loss,
            take_profit=plan.take_profit,
            size=size,
            side=side,
            opened_at=datetime.now(timezone.utc),
            trade_mode=self._settings.engine_mode if self._settings.engine_mode in {"paper", "live"} else "paper",
        )

    def force_close_trades(self, symbol: str, price: float, reason: str = "force_close") -> list[PaperTradeResult]:
        results: list[PaperTradeResult] = []
        for trade in self._db.fetch_open_trades(symbol):
            pnl_usd, pnl_r = self._calculate_pnl(trade.side, trade.entry, price, trade.stop, trade.size)
            self._db.close_trade(
                trade_id=trade.id,
                exit_price=price,
                pnl_usd=pnl_usd,
                pnl_r=pnl_r,
                closed_at=datetime.now(timezone.utc),
                result=reason,
            )
            results.append(PaperTradeResult(trade.id, trade.symbol, price, pnl_usd, pnl_r, reason))
        return results

    def evaluate_open_trades(self, symbol: str, price: float) -> list[PaperTradeResult]:
        results: list[PaperTradeResult] = []
        for trade in self._db.fetch_open_trades(symbol):
            hit_result = self._check_exit(trade.side, trade.entry, trade.stop, trade.take_profit, price)
            if hit_result is None:
                continue
            exit_price, result = hit_result
            pnl_usd, pnl_r = self._calculate_pnl(trade.side, trade.entry, exit_price, trade.stop, trade.size)
            self._db.close_trade(
                trade_id=trade.id,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_r=pnl_r,
                closed_at=datetime.now(timezone.utc),
                result=result,
            )
            results.append(PaperTradeResult(trade.id, trade.symbol, exit_price, pnl_usd, pnl_r, result))
        return results

    def _check_exit(
        self,
        side: str,
        entry: float,
        stop: float,
        take_profit: float,
        price: float,
    ) -> tuple[float, str] | None:
        if side == "long":
            if price <= stop:
                return stop, "loss"
            if price >= take_profit:
                return take_profit, "win"
        else:
            if price >= stop:
                return stop, "loss"
            if price <= take_profit:
                return take_profit, "win"
        return None

    def _calculate_pnl(
        self,
        side: str,
        entry: float,
        exit_price: float,
        stop: float,
        size: float,
    ) -> tuple[float, float]:
        if side == "long":
            pnl_pct = (exit_price - entry) / entry
            risk_per_unit = (entry - stop)
            pnl_r = (exit_price - entry) / risk_per_unit if risk_per_unit else 0.0
        else:
            pnl_pct = (entry - exit_price) / entry
            risk_per_unit = (stop - entry)
            pnl_r = (entry - exit_price) / risk_per_unit if risk_per_unit else 0.0
        return pnl_pct * size, pnl_r
