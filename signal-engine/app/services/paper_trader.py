from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from ..config import Settings
from ..models import Direction, TradePlan
from ..providers.bybit import BybitKlineSnapshot
from .database import Database


logger = logging.getLogger(__name__)


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

    def maybe_open_trade(
        self,
        symbol: str,
        plan: TradePlan,
        allow_multiple: bool = False,
        snapshot: BybitKlineSnapshot | None = None,
        regime: str | None = None,
    ) -> int | None:
        if self._settings.sweet8_enabled:
            open_total = len(self._db.fetch_open_trades())
            open_symbol = len(self._db.fetch_open_trades(symbol))
            if open_total >= self._settings.sweet8_max_open_positions_total:
                return None
            if open_symbol >= self._settings.sweet8_max_open_positions_per_symbol:
                return None
            if snapshot is None:
                return None
            score = plan.signal_score or 0
            active_regime = regime or self._settings.sweet8_current_mode
            atr = _compute_atr(snapshot, period=14)
            if atr is None or atr <= 0:
                return None
            entry = snapshot.candle.close
            if active_regime == "scalper":
                sl_distance = atr * self._settings.sweet8_scalp_atr_sl_mult
                tp_distance = atr * self._settings.sweet8_scalp_atr_tp_mult
                min_score = self._settings.sweet8_scalp_min_score
            else:
                sl_distance = atr * self._settings.sweet8_swing_atr_sl_mult
                tp_distance = atr * self._settings.sweet8_swing_atr_tp_mult
                min_score = self._settings.sweet8_swing_min_score
            if score < min_score:
                return None
            risk_pct = min(self._settings.sweet8_base_risk_pct, self._settings.sweet8_max_risk_pct)
            size = (self._settings.account_size * risk_pct / sl_distance) if sl_distance > 0 else 0.0
            if size <= 0:
                return None
            if plan.direction == Direction.long:
                stop = entry - sl_distance
                take_profit = entry + tp_distance
                side = "long"
            else:
                stop = entry + sl_distance
                take_profit = entry - tp_distance
                side = "short"
            return self._db.open_trade(
                symbol=symbol,
                entry=entry,
                stop=stop,
                take_profit=take_profit,
                size=size,
                side=side,
                opened_at=datetime.now(timezone.utc),
                trade_mode=self._settings.engine_mode if self._settings.engine_mode in {"paper", "live"} else "paper",
            )
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
        if self._settings.sweet8_enabled and reason in {"time_stop", "force_trade_auto_close"}:
            self._settings.sweet8_blocked_close_total += 1
            if reason == "time_stop":
                self._settings.sweet8_blocked_close_time_stop += 1
            if reason == "force_trade_auto_close":
                self._settings.sweet8_blocked_close_force += 1
            logger.info("sweet8_blocked_premature_close symbol=%s reason=%s", symbol, reason)
            return []
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


def _compute_atr(snapshot: BybitKlineSnapshot, period: int = 14) -> float | None:
    candles = snapshot.candles
    if len(candles) < period + 1:
        return None
    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        current = candles[i]
        prev_close = candles[i - 1].close
        true_ranges.append(max(current.high - current.low, abs(current.high - prev_close), abs(current.low - prev_close)))
    if len(true_ranges) < period:
        return None
    return sum(true_ranges[-period:]) / period
