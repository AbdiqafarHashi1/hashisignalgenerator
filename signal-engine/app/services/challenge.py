from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ..config import Settings
from ..utils.trading_day import trading_day_key
from .database import Database

STATE_KEY = "challenge.state"
CHALLENGE_START_KEY = "accounting.challenge_start_ts"


class ChallengeStatus(str, Enum):
    RUNNING = "RUNNING"
    STOPPED_DAILY_TARGET = "STOPPED_DAILY_TARGET"
    STOPPED_COOLDOWN = "STOPPED_COOLDOWN"
    FAILED_DRAWDOWN = "FAILED_DRAWDOWN"
    FAILED_DAILY = "FAILED_DAILY"
    PASSED = "PASSED"


@dataclass
class ChallengeState:
    status: str = ChallengeStatus.RUNNING.value
    status_reason: str = "Challenge running normally."
    start_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    profit_pct: float = 0.0
    daily_loss_pct: float = 0.0
    global_dd_pct: float = 0.0
    days_traded_count: int = 0
    days_elapsed_count: int = 0
    violation_reason: str | None = None  # deprecated
    passed_ts: str | None = None  # deprecated
    failed_ts: str | None = None  # deprecated
    pass_at_ts: str | None = None
    pass_at_equity: float | None = None
    failed_at_ts: str | None = None
    failed_at_equity: float | None = None
    last_candle_date: str | None = None
    last_traded_day: str | None = None


class ChallengeService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db

    def _new_state(self, now: datetime, start_equity: float) -> ChallengeState:
        return ChallengeState(
            start_equity=start_equity,
            current_equity=start_equity,
            peak_equity=start_equity,
            daily_start_equity=start_equity,
            last_candle_date=trading_day_key(now),
            days_elapsed_count=1,
        )

    def _normalize_legacy_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = str(payload.get("status", ChallengeStatus.RUNNING.value)).upper()
        if status == "IN_PROGRESS":
            payload["status"] = ChallengeStatus.RUNNING.value
        elif status == "FAILED":
            reason = str(payload.get("violation_reason") or "")
            payload["status"] = ChallengeStatus.FAILED_DRAWDOWN.value if "global" in reason else ChallengeStatus.FAILED_DAILY.value
        elif status == "PASSED":
            payload["status"] = ChallengeStatus.PASSED.value
        payload.setdefault("status_reason", "Challenge running normally.")
        payload.setdefault("pass_at_ts", payload.get("passed_ts"))
        payload.setdefault("failed_at_ts", payload.get("failed_ts"))
        payload.setdefault("pass_at_equity", payload.get("current_equity"))
        payload.setdefault("failed_at_equity", payload.get("current_equity"))
        return payload

    def ensure_challenge_start_ts(self, now: datetime) -> str:
        row = self._db.get_runtime_state(CHALLENGE_START_KEY)
        if row and row.value_text:
            return str(row.value_text)
        value = now.isoformat()
        self._db.set_runtime_state(CHALLENGE_START_KEY, value_text=value)
        return value

    def load(self, now: datetime | None = None, start_equity: float | None = None) -> ChallengeState:
        row = self._db.get_runtime_state(STATE_KEY)
        ts = now or datetime.now(timezone.utc)
        eq = float(start_equity or self._settings.account_size)
        self.ensure_challenge_start_ts(ts)
        if row and row.value_text:
            payload = self._normalize_legacy_payload(self._db.loads_json(row.value_text))
            state = ChallengeState(**payload)
            self.save(state)
            return state
        state = self._new_state(ts, eq)
        self.save(state)
        return state

    def save(self, state: ChallengeState) -> None:
        self._db.set_runtime_state(STATE_KEY, value_text=self._db.dumps_json(asdict(state)))

    def reset(self, now: datetime, start_equity: float) -> ChallengeState:
        self._db.set_runtime_state(CHALLENGE_START_KEY, value_text=now.isoformat())
        state = self._new_state(now, start_equity)
        self.save(state)
        return state

    def update(self, *, equity: float, daily_start_equity: float, now: datetime, traded_today: bool) -> ChallengeState:
        state = self.load(now=now, start_equity=self._settings.account_size)
        date_key = trading_day_key(now)
        if state.last_candle_date != date_key:
            state.last_candle_date = date_key
            state.days_elapsed_count += 1
            state.daily_start_equity = equity
        state.current_equity = equity
        state.peak_equity = max(state.peak_equity, equity)
        state.daily_start_equity = daily_start_equity
        if traded_today and state.last_traded_day != date_key:
            state.days_traded_count += 1
            state.last_traded_day = date_key

        if state.start_equity > 0:
            state.profit_pct = (equity - state.start_equity) / state.start_equity
        if state.daily_start_equity > 0:
            state.daily_loss_pct = max(0.0, (state.daily_start_equity - equity) / state.daily_start_equity)
        if state.peak_equity > 0:
            state.global_dd_pct = max(0.0, (state.peak_equity - equity) / state.peak_equity)

        if state.status == ChallengeStatus.RUNNING.value:
            if state.global_dd_pct >= self._settings.prop_max_global_dd_pct:
                state.status = ChallengeStatus.FAILED_DRAWDOWN.value
                state.status_reason = f"FAILED: Global DD {self._settings.prop_max_global_dd_pct * 100:.1f}% breached"
                state.violation_reason = "global_drawdown_limit"
                state.failed_ts = now.isoformat()
                state.failed_at_ts = state.failed_ts
                state.failed_at_equity = equity
            elif state.daily_loss_pct >= self._settings.prop_max_daily_loss_pct:
                state.status = ChallengeStatus.FAILED_DAILY.value
                state.status_reason = f"FAILED: Daily loss {self._settings.prop_max_daily_loss_pct * 100:.1f}% breached"
                state.violation_reason = "daily_loss_limit"
                state.failed_ts = now.isoformat()
                state.failed_at_ts = state.failed_ts
                state.failed_at_equity = equity
            elif state.profit_pct >= self._settings.prop_profit_target_pct and state.days_traded_count >= self._settings.prop_min_trading_days:
                state.status = ChallengeStatus.PASSED.value
                state.status_reason = "PASSED: Profit target reached"
                state.passed_ts = now.isoformat()
                state.pass_at_ts = state.passed_ts
                state.pass_at_equity = equity
                state.violation_reason = "profit_target_reached"

        self.save(state)
        return state

    def status_payload(self, now: datetime | None = None) -> dict[str, Any]:
        ts = now or datetime.now(timezone.utc)
        state = self.load(now=ts, start_equity=self._settings.account_size)
        payload = asdict(state)
        payload["server_ts"] = datetime.now(timezone.utc).isoformat()
        payload["candle_ts"] = ts.isoformat()
        payload["replay_cursor_ts"] = payload["candle_ts"]
        payload["challenge_start_ts"] = self.ensure_challenge_start_ts(ts)
        return payload
