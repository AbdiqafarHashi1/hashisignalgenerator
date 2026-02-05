from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

from pydantic import BaseModel

from .config import Settings
from .models import PostureSnapshot, Status, TradePlan


class DailyState(BaseModel):
    date: str
    trades: int = 0
    losses: int = 0
    pnl_usd: float = 0.0
    last_loss_ts: datetime | None = None
    latest_decision: TradePlan | None = None


class StateStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._daily: dict[str, DailyState] = {}
        self._posture_cache: dict[tuple[str, str], PostureSnapshot] = {}
        self._last_processed_close_time_ms: int | None = None
        self._last_notified_key: str | None = None

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def get_daily_state(self, symbol: str) -> DailyState:
        date_key = self._today_key()
        with self._lock:
            key = f"{symbol}:{date_key}"
            if key not in self._daily:
                self._daily[key] = DailyState(date=date_key)
            return self._daily[key]

    def get_posture(self, symbol: str) -> PostureSnapshot | None:
        date_key = self._today_key()
        with self._lock:
            return self._posture_cache.get((symbol, date_key))

    def set_posture(self, snapshot: PostureSnapshot) -> None:
        with self._lock:
            self._posture_cache[(snapshot.symbol, snapshot.date)] = snapshot

    def set_latest_decision(self, symbol: str, plan: TradePlan) -> None:
        state = self.get_daily_state(symbol)
        with self._lock:
            state.latest_decision = plan

    def get_last_processed_close_time_ms(self) -> int | None:
        with self._lock:
            return self._last_processed_close_time_ms

    def set_last_processed_close_time_ms(self, value: int | None) -> None:
        with self._lock:
            self._last_processed_close_time_ms = value

    def get_last_notified_key(self) -> str | None:
        with self._lock:
            return self._last_notified_key

    def set_last_notified_key(self, value: str | None) -> None:
        with self._lock:
            self._last_notified_key = value

    def record_trade(self, symbol: str) -> None:
        state = self.get_daily_state(symbol)
        with self._lock:
            state.trades += 1

    def record_outcome(self, symbol: str, pnl_usd: float, win: bool, timestamp: datetime) -> None:
        state = self.get_daily_state(symbol)
        with self._lock:
            state.pnl_usd += pnl_usd
            if not win:
                state.losses += 1
                state.last_loss_ts = timestamp

    def check_limits(self, symbol: str, cfg: Settings, now: datetime) -> tuple[bool, Status, list[str]]:
        state = self.get_daily_state(symbol)
        rationale: list[str] = []
        account_loss_limit = cfg.account_size * cfg.max_daily_loss_pct

        if state.pnl_usd <= -account_loss_limit:
            rationale.append("daily_loss_limit_exceeded")
            return False, Status.RISK_OFF, rationale
        if state.losses >= cfg.max_losses_per_day:
            rationale.append("max_losses_reached")
            return False, Status.RISK_OFF, rationale
        if state.trades >= cfg.max_trades_per_day:
            rationale.append("max_trades_reached")
            return False, Status.NO_TRADE, rationale
        if state.last_loss_ts is not None:
            minutes_since = (now - state.last_loss_ts).total_seconds() / 60.0
            if minutes_since < cfg.cooldown_minutes_after_loss:
                rationale.append("cooldown_after_loss")
                return False, Status.NO_TRADE, rationale
        return True, Status.NO_TRADE, rationale
