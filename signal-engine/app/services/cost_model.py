from __future__ import annotations

from dataclasses import dataclass


class CostModel:
    def apply_entry_price(self, side: str, price: float) -> float:
        raise NotImplementedError

    def apply_exit_price(self, side: str, price: float) -> float:
        raise NotImplementedError

    def fees(self, *, symbol: str, entry: float, exit_price: float, qty: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class CryptoCostModel(CostModel):
    fee_rate_bps: float
    spread_bps: float
    slippage_bps: float

    def _impact(self) -> float:
        return (self.spread_bps + self.slippage_bps) / 10000.0

    def apply_entry_price(self, side: str, price: float) -> float:
        impact = self._impact()
        return price * (1 + impact) if side == "long" else price * (1 - impact)

    def apply_exit_price(self, side: str, price: float) -> float:
        impact = self._impact()
        return price * (1 - impact) if side == "long" else price * (1 + impact)

    def fees(self, *, symbol: str, entry: float, exit_price: float, qty: float) -> float:
        rate = self.fee_rate_bps / 10000.0
        return (entry * qty + exit_price * qty) * rate


@dataclass(frozen=True)
class ForexCostModel(CostModel):
    spread_bps: float
    commission_bps: float

    def _pip_size(self, symbol: str) -> float:
        pair = symbol.upper().replace("_", "")
        return 0.01 if pair.endswith("JPY") else 0.0001

    def _spread_price(self, symbol: str, price: float) -> float:
        pips = (self.spread_bps / 10000.0) * price
        pip = self._pip_size(symbol)
        return round(pips / pip) * pip

    def apply_entry_price(self, side: str, price: float) -> float:
        # Use half-spread on entry and half on exit.
        return price

    def apply_exit_price(self, side: str, price: float) -> float:
        return price

    def fees(self, *, symbol: str, entry: float, exit_price: float, qty: float) -> float:
        spread_cost = self._spread_price(symbol, (entry + exit_price) / 2.0) * qty
        commission_rate = self.commission_bps / 10000.0
        commission_cost = (entry * qty + exit_price * qty) * commission_rate
        return spread_cost + commission_cost
