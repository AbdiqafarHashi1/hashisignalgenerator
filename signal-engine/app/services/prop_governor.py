from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime

from ..config import Settings
from ..utils.trading_day import trading_day_key
from .database import Database
from signal_engine.accounting import compute_accounting_snapshot
from signal_engine.governor.prop import GovernorBlock, evaluate_prop_block

STATE_KEY = "prop.governor"


@dataclass
class GovernorState:
    risk_pct: float
    consecutive_losses: int = 0
    trades_since_loss: int = 999
    daily_net_r: float = 0.0
    daily_losses: int = 0
    daily_trades: int = 0
    locked_until_ts: str | None = None
    day_key: str | None = None


class PropRiskGovernor:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._s = settings
        self._db = db

    def load(self, now: datetime) -> GovernorState:
        row = self._db.get_runtime_state(STATE_KEY)
        if row and row.value_text:
            data = self._db.loads_json(row.value_text)
            return GovernorState(**data)
        st = GovernorState(risk_pct=self._s.prop_risk_base_pct, day_key=trading_day_key(now))
        self.save(st)
        return st

    def save(self, state: GovernorState) -> None:
        self._db.set_runtime_state(STATE_KEY, value_text=self._db.dumps_json(asdict(state)))

    def reset(self, now: datetime) -> GovernorState:
        state = GovernorState(risk_pct=self._s.prop_risk_base_pct, day_key=trading_day_key(now))
        self.save(state)
        return state

    def block_reason(self, now: datetime) -> GovernorBlock | None:
        st = self.load(now)
        trades = self._db.fetch_trades()
        realized = sum(float(t.pnl_usd or 0.0) for t in trades if t.closed_at)
        fees = sum(float(t.fees or 0.0) for t in trades if t.closed_at)
        unrealized = 0.0
        acct = compute_accounting_snapshot(
            equity_start=float(self._s.account_size or 0.0),
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            fees=fees,
            day_start_equity=float(self._s.account_size or 0.0),
            hwm=float(self._s.account_size or 0.0),
            trade_close_dates=[t.closed_at for t in trades if t.closed_at],
            profit_target_pct=float(self._s.prop_profit_target_pct or 0.0),
        )
        return evaluate_prop_block(
            now=now,
            day_key=trading_day_key(now),
            stored_day_key=st.day_key,
            daily_trades=st.daily_trades,
            consecutive_losses=st.consecutive_losses,
            last_loss_at=datetime.fromisoformat(st.locked_until_ts) if st.locked_until_ts else None,
            accounting=acct,
            max_daily_loss_pct=float(self._s.prop_max_daily_loss_pct or 0.0),
            max_global_dd_pct=float(self._s.prop_max_global_dd_pct or 0.0),
            max_trades_per_day=int(self._s.prop_max_trades_per_day or 0),
            max_consecutive_losses=int(self._s.prop_max_consec_losses or 0),
            cooldown_minutes=int(self._s.prop_time_cooldown_minutes or 0),
        )

    def applied_risk_pct(self, now: datetime) -> tuple[float, str]:
        st = self.load(now)
        self._roll_day(st, now)
        reason = "base"
        if self._is_locked(st, now):
            reason = "locked"
        self.save(st)
        return st.risk_pct, reason

    def allow_new_trade(self, now: datetime) -> tuple[bool, str | None]:
        block = self.block_reason(now)
        if block:
            return False, block.code
        return True, None

    def on_trade_close(self, net_r: float, now: datetime) -> None:
        st = self.load(now)
        self._roll_day(st, now)
        st.daily_net_r += net_r
        st.daily_trades += 1
        if net_r < 0:
            st.daily_losses += 1
            st.consecutive_losses += 1
            st.trades_since_loss = 0
            if self._s.prop_stepdown_after_loss:
                st.risk_pct = max(self._s.prop_risk_min_pct, st.risk_pct * self._s.prop_stepdown_factor)
            st.locked_until_ts = now.isoformat()
        else:
            st.consecutive_losses = 0
            st.trades_since_loss += 1
            if self._s.prop_stepup_after_win and st.trades_since_loss >= self._s.prop_stepup_cooldown_trades:
                st.risk_pct = min(self._s.prop_risk_max_pct, st.risk_pct * self._s.prop_stepup_factor)
        self.save(st)

    def _is_locked(self, st: GovernorState, now: datetime) -> bool:
        if not st.locked_until_ts:
            return False
        try:
            lock_dt = datetime.fromisoformat(st.locked_until_ts)
        except ValueError:
            return False
        return (now - lock_dt).total_seconds() < self._s.prop_time_cooldown_minutes * 60

    def _roll_day(self, st: GovernorState, now: datetime) -> None:
        day_key = trading_day_key(now)
        if st.day_key == day_key:
            return
        st.day_key = day_key
        st.daily_net_r = 0.0
        st.daily_losses = 0
        st.daily_trades = 0
        if self._s.prop_reset_consec_losses_on_day_rollover:
            st.consecutive_losses = 0
            st.trades_since_loss = 999
        st.locked_until_ts = None
