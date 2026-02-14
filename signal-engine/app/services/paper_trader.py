from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
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
        self._last_mark_prices: dict[str, float] = {}

    def update_mark_price(self, symbol: str, price: float) -> None:
        self._last_mark_prices[symbol] = price

    def total_unrealized_pnl_usd(self) -> float:
        total = 0.0
        for trade in self._db.fetch_open_trades():
            mark_price = self._last_mark_prices.get(trade.symbol, trade.entry)
            total += self._unrealized_pnl(trade.side, trade.entry, mark_price, trade.size)
        return total

    def total_margin_used_usd(self) -> float:
        leverage = self._effective_leverage()
        if leverage <= 0:
            return 0.0
        return sum((trade.entry * trade.size) / leverage for trade in self._db.fetch_open_trades())

    def margin_utilization_pct(self) -> float:
        account_size = float(self._settings.account_size or 0.0)
        if account_size <= 0:
            return 0.0
        return (self.total_margin_used_usd() / account_size) * 100.0

    def symbol_unrealized_pnl_usd(self, symbol: str) -> float:
        total = 0.0
        for trade in self._db.fetch_open_trades(symbol):
            mark_price = self._last_mark_prices.get(trade.symbol, trade.entry)
            total += self._unrealized_pnl(trade.side, trade.entry, mark_price, trade.size)
        return total

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
            atr = _compute_atr(snapshot, period=self._settings.atr_period)
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
                take_profit = self._cap_take_profit(entry, entry + tp_distance, "long")
                side = "long"
            else:
                stop = entry + sl_distance
                take_profit = self._cap_take_profit(entry, entry - tp_distance, "short")
                side = "short"
            entry_with_costs = self._apply_entry_price(side, entry)
            take_profit = self._cap_take_profit(entry_with_costs, take_profit, side)
            if not self._can_open_trade(entry_with_costs, size):
                logger.info("paper_trade_rejected symbol=%s reason=insufficient_margin", symbol)
                return None
            return self._db.open_trade(
                symbol=symbol,
                entry=entry_with_costs,
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
        stop = plan.stop_loss
        if stop is None:
            return None
        side = "long" if plan.direction == Direction.long else "short"
        entry_with_costs = self._apply_entry_price(side, entry)
        risk_usd = self._settings.account_size * float(plan.risk_pct_used or self._settings.base_risk_pct or 0.0)
        stop_distance = abs(entry_with_costs - stop)
        size = (risk_usd / stop_distance) if stop_distance > 0 else 0.0
        if plan.position_size_usd:
            size = min(size, float(plan.position_size_usd) / entry_with_costs)
        if size <= 0:
            return None
        take_profit = self._cap_take_profit(entry_with_costs, plan.take_profit, side)
        if not self._is_reentry_allowed(symbol, side):
            logger.info("paper_trade_rejected symbol=%s reason=reentry_cooldown", symbol)
            return None
        if not self._can_open_trade(entry_with_costs, size):
            logger.info("paper_trade_rejected symbol=%s reason=insufficient_margin", symbol)
            return None
        return self._db.open_trade(
            symbol=symbol,
            entry=entry_with_costs,
            stop=stop,
            take_profit=take_profit,
            size=size,
            side=side,
            opened_at=datetime.now(timezone.utc),
            trade_mode=self._settings.engine_mode if self._settings.engine_mode in {"paper", "live"} else "paper",
        )

    def force_close_trades(self, symbol: str, price: float, reason: str = "force_close") -> list[PaperTradeResult]:
        if self._settings.sweet8_enabled and reason in {"time_stop_close", "force_trade_auto_close"}:
            self._settings.sweet8_blocked_close_total += 1
            if reason == "time_stop_close":
                self._settings.sweet8_blocked_close_time_stop += 1
            if reason == "force_trade_auto_close":
                self._settings.sweet8_blocked_close_force += 1
            logger.info("sweet8_blocked_premature_close symbol=%s reason=%s", symbol, reason)
            return []
        results: list[PaperTradeResult] = []
        for trade in self._db.fetch_open_trades(symbol):
            exit_price = self._apply_exit_price(trade.side, price)
            pnl_usd, pnl_r = self._calculate_pnl(trade.side, trade.entry, exit_price, trade.stop, trade.size)
            self._db.close_trade(
                trade_id=trade.id,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_r=pnl_r,
                closed_at=datetime.now(timezone.utc),
                result=reason,
            )
            results.append(PaperTradeResult(trade.id, trade.symbol, exit_price, pnl_usd, pnl_r, reason))
        return results

    def evaluate_open_trades(
        self,
        symbol: str,
        price: float,
        candle_high: float | None = None,
        candle_low: float | None = None,
    ) -> list[PaperTradeResult]:
        results: list[PaperTradeResult] = []
        self.update_mark_price(symbol, price)
        for trade in self._db.fetch_open_trades(symbol):
            hit_result = self._check_exit(
                trade.side,
                trade.entry,
                trade.stop,
                trade.take_profit,
                price,
                candle_high=candle_high,
                candle_low=candle_low,
            )
            if hit_result is None:
                continue
            exit_price, result = hit_result
            exit_with_costs = self._apply_exit_price(trade.side, exit_price)
            pnl_usd, pnl_r = self._calculate_pnl(trade.side, trade.entry, exit_with_costs, trade.stop, trade.size)
            self._db.close_trade(
                trade_id=trade.id,
                exit_price=exit_with_costs,
                pnl_usd=pnl_usd,
                pnl_r=pnl_r,
                closed_at=datetime.now(timezone.utc),
                result=result,
            )
            results.append(PaperTradeResult(trade.id, trade.symbol, exit_with_costs, pnl_usd, pnl_r, result))
        return results


    def move_stop_to_breakeven(self, symbol: str, trigger_r: float | None = None) -> int:
        trigger_r = self._settings.move_to_breakeven_trigger_r if trigger_r is None else trigger_r
        moved = 0
        for trade in self._db.fetch_open_trades(symbol):
            risk = abs(trade.entry - trade.stop)
            if risk <= 0:
                continue
            mark = self._last_mark_prices.get(symbol, trade.entry)
            progress = ((mark - trade.entry) / risk) if trade.side == "long" else ((trade.entry - mark) / risk)
            if progress < trigger_r:
                continue
            breakeven = trade.entry
            if trade.side == "long" and trade.stop < breakeven:
                self._db.update_trade_stop(trade.id, breakeven)
                moved += 1
            elif trade.side == "short" and trade.stop > breakeven:
                self._db.update_trade_stop(trade.id, breakeven)
                moved += 1
        return moved

    def _effective_leverage(self) -> float:
        return max(1.0, float(getattr(self._settings, "leverage_elevated", 1.0)))

    def _available_margin(self) -> float:
        closed_pnl = sum(float(trade.pnl_usd or 0.0) for trade in self._db.fetch_trades() if trade.closed_at)
        equity = self._settings.account_size + closed_pnl + self.total_unrealized_pnl_usd()
        return max(0.0, equity - self.total_margin_used_usd())

    def _can_open_trade(self, entry_price: float, qty_base: float) -> bool:
        leverage = self._effective_leverage()
        notional = entry_price * qty_base
        margin_required = notional / leverage if leverage > 0 else notional
        return margin_required <= self._available_margin()

    def _apply_entry_price(self, side: str, price: float) -> float:
        spread = self._settings.spread_bps / 10_000
        slippage = self._settings.slippage_bps / 10_000
        impact = spread + slippage
        if side == "long":
            return price * (1 + impact)
        return price * (1 - impact)

    def _apply_exit_price(self, side: str, price: float) -> float:
        spread = self._settings.spread_bps / 10_000
        slippage = self._settings.slippage_bps / 10_000
        impact = spread + slippage
        if side == "long":
            return price * (1 - impact)
        return price * (1 + impact)

    def _fees_usd(self, entry: float, exit_price: float, qty_base: float) -> float:
        fee_rate = self._settings.fee_rate_bps / 10_000
        return (entry * qty_base + exit_price * qty_base) * fee_rate

    def _unrealized_pnl(self, side: str, entry: float, mark_price: float, qty_base: float) -> float:
        side_sign = 1.0 if side == "long" else -1.0
        return (mark_price - entry) * qty_base * side_sign

    def _check_exit(
        self,
        side: str,
        entry: float,
        stop: float,
        take_profit: float,
        price: float,
        candle_high: float | None = None,
        candle_low: float | None = None,
    ) -> tuple[float, str] | None:
        high = price if candle_high is None else candle_high
        low = price if candle_low is None else candle_low
        if math.isnan(high) or math.isnan(low) or high < low:
            return None
        if side == "long":
            tp_hit = high >= take_profit
            sl_hit = low <= stop
            if tp_hit and sl_hit:
                return stop, "sl_close"
            if sl_hit:
                return stop, "sl_close"
            if tp_hit:
                return take_profit, "tp_close"
        else:
            tp_hit = low <= take_profit
            sl_hit = high >= stop
            if tp_hit and sl_hit:
                return stop, "sl_close"
            if sl_hit:
                return stop, "sl_close"
            if tp_hit:
                return take_profit, "tp_close"
        return None

    def _is_reentry_allowed(self, symbol: str, side: str) -> bool:
        # Global override: explicit zero disables re-entry cooldown checks in all modes.
        if max(0, self._settings.reentry_cooldown_minutes) == 0:
            return True
        cooldown = (
            max(0, self._settings.scalp_reentry_cooldown_minutes)
            if self._settings.current_mode == "SCALP"
            else max(0, self._settings.reentry_cooldown_minutes)
        )
        if cooldown <= 0:
            return True
        for trade in self._db.fetch_trades():
            if trade.symbol != symbol or trade.closed_at is None:
                continue
            if trade.result != "tp_close" or trade.side != side:
                continue
            closed_at = datetime.fromisoformat(trade.closed_at)
            elapsed_minutes = (datetime.now(timezone.utc) - closed_at).total_seconds() / 60.0
            return elapsed_minutes >= cooldown
        return True

    def _cap_take_profit(self, entry: float, take_profit: float, side: str) -> float:
        if side == "long":
            return min(take_profit, entry * (1 + self._settings.tp_cap_long_pct))
        return max(take_profit, entry * (1 - self._settings.tp_cap_short_pct))

    def _calculate_pnl(
        self,
        side: str,
        entry: float,
        exit_price: float,
        stop: float,
        size: float,
    ) -> tuple[float, float]:
        side_sign = 1.0 if side == "long" else -1.0
        gross_pnl = (exit_price - entry) * size * side_sign
        fees = self._fees_usd(entry, exit_price, size)
        pnl_usd = gross_pnl - fees
        risk_per_unit = abs(entry - stop)
        pnl_r = ((exit_price - entry) * side_sign) / risk_per_unit if risk_per_unit else 0.0
        return pnl_usd, pnl_r


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
