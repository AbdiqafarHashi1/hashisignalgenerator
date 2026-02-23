from __future__ import annotations

from collections import deque
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
        self._last_trade_key: dict[str, str] = {}
        self._last_decision_ts: dict[str, datetime] = {}
        self._symbols: list[str] = []
        self._last_heartbeat_ts: datetime | None = None
        self._last_telegram_update_id: int | None = None
        self._decision_meta: dict[str, dict[str, object]] = {}
        self._skip_reason_counts: dict[str, int] = {}
        self._global_realized_pnl_usd: float = 0.0
        self._global_peak_equity_usd: float | None = None
        self._market_data_errors: dict[str, int] = {}
        self._gate_events: dict[str, deque[dict[str, object]]] = {}
        self._tick_timing: dict[str, float | None] = {
            "last_tick_started_ts": None,
            "last_tick_finished_ts": None,
            "last_tick_compute_ms": None,
            "last_sleep_s": None,
        }
        self._clock = lambda: datetime.now(timezone.utc)

    def set_clock(self, clock_fn) -> None:
        self._clock = clock_fn

    def _now(self) -> datetime:
        return self._clock()

    def _today_key(self, now: datetime | None = None) -> str:
        return (now or self._now()).date().isoformat()

    def get_daily_state(self, symbol: str, now: datetime | None = None) -> DailyState:
        date_key = self._today_key(now)
        with self._lock:
            key = f"{symbol}:{date_key}"
            if key not in self._daily:
                self._daily[key] = DailyState(date=date_key)
            return self._daily[key]

    def get_posture(self, symbol: str, now: datetime | None = None) -> PostureSnapshot | None:
        date_key = self._today_key(now)
        with self._lock:
            return self._posture_cache.get((symbol, date_key))

    def set_posture(self, snapshot: PostureSnapshot) -> None:
        with self._lock:
            self._posture_cache[(snapshot.symbol, snapshot.date)] = snapshot

    def set_latest_decision(self, symbol: str, plan: TradePlan) -> None:
        now = self._now()
        state = self.get_daily_state(symbol, now)
        with self._lock:
            state.latest_decision = plan
            self._last_decision_ts[symbol] = now

    def get_last_decision_ts(self, symbol: str) -> datetime | None:
        with self._lock:
            return self._last_decision_ts.get(symbol)

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

    def get_last_trade_key(self, symbol: str) -> str | None:
        with self._lock:
            return self._last_trade_key.get(symbol)

    def set_last_trade_key(self, symbol: str, value: str | None) -> None:
        with self._lock:
            if value is None:
                self._last_trade_key.pop(symbol, None)
            else:
                self._last_trade_key[symbol] = value

    def get_symbols(self) -> list[str]:
        with self._lock:
            return list(self._symbols)

    def set_symbols(self, symbols: list[str]) -> None:
        normalized = [item.strip().upper() for item in symbols if item and item.strip()]
        with self._lock:
            self._symbols = normalized

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


    def set_decision_meta(self, symbol: str, meta: dict[str, object]) -> None:
        with self._lock:
            self._decision_meta[symbol] = dict(meta)

    def record_skip_reason(self, reason: str | None) -> None:
        if not reason:
            return
        with self._lock:
            self._skip_reason_counts[reason] = self._skip_reason_counts.get(reason, 0) + 1

    def skip_reason_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._skip_reason_counts)

    def get_decision_meta(self, symbol: str) -> dict[str, object]:
        with self._lock:
            return dict(self._decision_meta.get(symbol, {}))

    def record_trade(self, symbol: str) -> None:
        state = self.get_daily_state(symbol, self._now())
        with self._lock:
            state.trades += 1

    def record_outcome(self, symbol: str, pnl_usd: float, win: bool, timestamp: datetime) -> None:
        state = self.get_daily_state(symbol, timestamp)
        with self._lock:
            self._global_realized_pnl_usd += pnl_usd
            state.pnl_usd += pnl_usd
            if not win:
                state.losses += 1
                state.consecutive_losses += 1
                state.last_loss_ts = timestamp
            else:
                state.consecutive_losses = 0

    def set_global_equity(self, equity_usd: float) -> None:
        with self._lock:
            if self._global_peak_equity_usd is None or equity_usd > self._global_peak_equity_usd:
                self._global_peak_equity_usd = equity_usd

    def global_drawdown(self, account_size: float) -> tuple[float, float]:
        with self._lock:
            peak = self._global_peak_equity_usd if self._global_peak_equity_usd is not None else account_size
            current = account_size + self._global_realized_pnl_usd
        dd_usd = max(0.0, peak - current)
        dd_pct = (dd_usd / peak) if peak > 0 else 0.0
        return dd_usd, dd_pct


    def record_market_data_error(self, reason: str) -> None:
        if not reason:
            return
        with self._lock:
            self._market_data_errors[reason] = self._market_data_errors.get(reason, 0) + 1

    def market_data_error_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._market_data_errors)

    def record_gate_event(self, symbol: str, payload: dict[str, object], max_events: int = 50) -> None:
        with self._lock:
            queue = self._gate_events.setdefault(symbol, deque(maxlen=max_events))
            queue.append(dict(payload))

    def gate_events(self, symbol: str, limit: int = 50) -> list[dict[str, object]]:
        with self._lock:
            events = list(self._gate_events.get(symbol, deque()))
        return events[-max(1, limit):]

    def set_tick_timing(
        self,
        *,
        last_tick_started_ts: float | None,
        last_tick_finished_ts: float | None,
        last_tick_compute_ms: float | None,
        last_sleep_s: float | None,
    ) -> None:
        with self._lock:
            self._tick_timing = {
                "last_tick_started_ts": last_tick_started_ts,
                "last_tick_finished_ts": last_tick_finished_ts,
                "last_tick_compute_ms": last_tick_compute_ms,
                "last_sleep_s": last_sleep_s,
            }

    def tick_timing(self) -> dict[str, float | None]:
        with self._lock:
            return dict(self._tick_timing)

    def reset(self) -> None:
        with self._lock:
            self._daily.clear()
            self._posture_cache.clear()
            self._last_processed_close_time_ms.clear()
            self._last_notified_key.clear()
            self._last_trade_key.clear()
            self._last_decision_ts.clear()
            self._last_heartbeat_ts = None
            self._last_telegram_update_id = None
            self._decision_meta.clear()
            self._skip_reason_counts.clear()
            self._global_realized_pnl_usd = 0.0
            self._global_peak_equity_usd = None
            self._market_data_errors.clear()
            self._gate_events.clear()
            self._tick_timing = {
                "last_tick_started_ts": None,
                "last_tick_finished_ts": None,
                "last_tick_compute_ms": None,
                "last_sleep_s": None,
            }

    def risk_snapshot(self, symbol: str, cfg: Settings, now: datetime) -> dict[str, object]:
        state = self.get_daily_state(symbol, now)
        account_loss_limit = cfg.account_size * cfg.max_daily_loss_pct
        cooldown_active = False
        cooldown_remaining_minutes = 0
        if state.last_loss_ts is not None:
            minutes_since = (now - state.last_loss_ts).total_seconds() / 60.0
            if minutes_since < cfg.cooldown_minutes_after_loss:
                cooldown_active = True
                cooldown_remaining_minutes = int(cfg.cooldown_minutes_after_loss - minutes_since)
        daily_loss_remaining = max(0.0, account_loss_limit + state.pnl_usd)
        global_dd_usd, global_dd_pct = self.global_drawdown(float(cfg.account_size or 0.0))
        return {
            "trades_today": state.trades,
            "trades_remaining": max(0, cfg.max_trades_per_day - state.trades),
            "consecutive_losses": state.consecutive_losses,
            "cooldown_active": cooldown_active,
            "cooldown_remaining_minutes": cooldown_remaining_minutes,
            "daily_loss_remaining_usd": daily_loss_remaining,
            "daily_loss_limit_usd": account_loss_limit,
            "daily_loss_cap_hit": state.pnl_usd <= -account_loss_limit,
            "global_drawdown_usd": global_dd_usd,
            "global_drawdown_pct": global_dd_pct,
            "global_drawdown_cap_hit": global_dd_pct >= float(cfg.global_drawdown_limit_pct or 0.0),
        }

    def risk_check(
        self,
        symbol: str,
        cfg: Settings,
        now: datetime,
        trades_today_closed: int | None = None,
    ) -> tuple[bool, str | None]:
        state = self.get_daily_state(symbol, now)
        account_loss_limit = cfg.account_size * cfg.max_daily_loss_pct

        if not cfg.debug_disable_hard_risk_gates and state.pnl_usd <= -account_loss_limit:
            return False, "daily_loss_limit_hit"

        _, global_dd_pct = self.global_drawdown(float(cfg.account_size or 0.0))
        if not cfg.debug_disable_hard_risk_gates and global_dd_pct >= float(cfg.global_drawdown_limit_pct or 0.0):
            return False, "global_dd_limit_hit"

        trades_today = state.trades if trades_today_closed is None else int(trades_today_closed)
        if trades_today >= cfg.max_trades_per_day:
            return False, "max_trades_exceeded"

        if not cfg.debug_loosen and state.last_loss_ts is not None:
            minutes_since = (now - state.last_loss_ts).total_seconds() / 60.0
            if minutes_since < cfg.cooldown_minutes_after_loss:
                return False, "cooldown"

        if cfg.max_consecutive_losses and state.consecutive_losses >= cfg.max_consecutive_losses:
            return False, "max_consecutive_losses"

        return True, None

    def check_limits(self, symbol: str, cfg: Settings, now: datetime) -> tuple[bool, Status, list[str]]:
        state = self.get_daily_state(symbol, now)
        rationale: list[str] = []
        profit_target = cfg.account_size * (cfg.daily_profit_target_pct or 0.0)

        allowed, gate_reason = self.risk_check(symbol, cfg, now)
        if not allowed and gate_reason:
            rationale.append(gate_reason)
            if gate_reason == "global_dd_limit_hit":
                rationale.append("global_drawdown_limit")
            return False, Status.RISK_OFF, rationale

        if cfg.manual_kill_switch:
            rationale.append("manual_kill_switch")
            return False, Status.RISK_OFF, rationale
        if profit_target > 0 and state.pnl_usd >= profit_target:
            rationale.append("daily_profit_target")
            return False, Status.RISK_OFF, rationale
        if not cfg.debug_disable_hard_risk_gates and cfg.max_losses_per_day and state.losses >= cfg.max_losses_per_day:
            rationale.append("max_losses_per_day")
            return False, Status.RISK_OFF, rationale
        return True, Status.NO_TRADE, rationale
