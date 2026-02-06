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
    consecutive_losses: int = 0
    pnl_usd: float = 0.0
    last_loss_ts: datetime | None = None
    latest_decision: TradePlan | None = None


class StateStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._daily: dict[str, DailyState] = {}
        self._posture_cache: dict[tuple[str, str], PostureSnapshot] = {}
        self._last_processed_close_time_ms: dict[str, int] = {}
        self._last_notified_key: dict[str, str] = {}
        self._symbols: list[str] = []
        self._last_heartbeat_ts: datetime | None = None
        self._last_telegram_update_id: int | None = None

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

    def get_last_processed_close_time_ms(self, symbol: str) -> int | None:
        with self._lock:
            return self._last_processed_close_time_ms.get(symbol)

    def set_last_processed_close_time_ms(self, symbol: str, value: int | None) -> None:
        with self._lock:
            if value is None:
                self._last_processed_close_time_ms.pop(symbol, None)
            else:
                self._last_processed_close_time_ms[symbol] = value

    def get_last_notified_key(self, symbol: str) -> str | None:
        with self._lock:
            return self._last_notified_key.get(symbol)

    def set_last_notified_key(self, symbol: str, value: str | None) -> None:
        with self._lock:
            if value is None:
                self._last_notified_key.pop(symbol, None)
            else:
                self._last_notified_key[symbol] = value

    def get_symbols(self) -> list[str]:
        with self._lock:
            return list(self._symbols)

    def set_symbols(self, symbols: list[str]) -> None:
        with self._lock:
            self._symbols = list(symbols)

    def get_last_heartbeat_ts(self) -> datetime | None:
        with self._lock:
            return self._last_heartbeat_ts

    def set_last_heartbeat_ts(self, value: datetime | None) -> None:
        with self._lock:
            self._last_heartbeat_ts = value

    def get_last_telegram_update_id(self) -> int | None:
        with self._lock:
            return self._last_telegram_update_id

    def set_last_telegram_update_id(self, value: int | None) -> None:
        with self._lock:
            self._last_telegram_update_id = value

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
                state.consecutive_losses += 1
                state.last_loss_ts = timestamp
            else:
                state.consecutive_losses = 0

    def check_limits(self, symbol: str, cfg: Settings, now: datetime) -> tuple[bool, Status, list[str]]:
        state = self.get_daily_state(symbol)
        rationale: list[str] = []
        account_loss_limit = cfg.account_size * cfg.max_daily_loss_pct
        profit_target = cfg.account_size * (cfg.daily_profit_target_pct or 0.0)

        if state.pnl_usd <= -account_loss_limit:
            rationale.append("daily_loss_limit")
            return False, Status.RISK_OFF, rationale
        if profit_target > 0 and state.pnl_usd >= profit_target:
            rationale.append("daily_profit_target")
            return False, Status.RISK_OFF, rationale
        if cfg.max_consecutive_losses and state.consecutive_losses >= cfg.max_consecutive_losses:
            rationale.append("max_losses")
            return False, Status.RISK_OFF, rationale
        if state.trades >= cfg.max_trades_per_day:
            rationale.append("max_trades")
            return False, Status.RISK_OFF, rationale
        if state.last_loss_ts is not None:
            minutes_since = (now - state.last_loss_ts).total_seconds() / 60.0
            if minutes_since < cfg.cooldown_minutes_after_loss:
                rationale.append("cooldown")
                return False, Status.RISK_OFF, rationale
        return True, Status.NO_TRADE, rationale
