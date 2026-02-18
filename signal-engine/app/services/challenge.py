from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ..config import Settings
from .database import Database

STATE_KEY = "challenge.state"


class ChallengeStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"


@dataclass
class ChallengeState:
    status: str = ChallengeStatus.IN_PROGRESS.value
    start_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    profit_pct: float = 0.0
    daily_loss_pct: float = 0.0
    global_dd_pct: float = 0.0
    days_traded_count: int = 0
    days_elapsed_count: int = 0
    violation_reason: str | None = None
    passed_ts: str | None = None
    failed_ts: str | None = None
    last_candle_date: str | None = None
    last_traded_day: str | None = None


class ChallengeService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db

    def load(self, now: datetime | None = None, start_equity: float | None = None) -> ChallengeState:
        row = self._db.get_runtime_state(STATE_KEY)
        if row and row.value_text:
            payload = self._db.loads_json(row.value_text)
            return ChallengeState(**payload)
        ts = now or datetime.now(timezone.utc)
        eq = float(start_equity or self._settings.account_size)
        state = ChallengeState(
            start_equity=eq,
            current_equity=eq,
            peak_equity=eq,
            daily_start_equity=eq,
            last_candle_date=ts.date().isoformat(),
            days_elapsed_count=1,
        )
        self.save(state)
        return state

    def save(self, state: ChallengeState) -> None:
        self._db.set_runtime_state(STATE_KEY, value_text=self._db.dumps_json(asdict(state)))

    def reset(self, now: datetime, start_equity: float) -> ChallengeState:
        state = ChallengeState(
            start_equity=start_equity,
            current_equity=start_equity,
            peak_equity=start_equity,
            daily_start_equity=start_equity,
            last_candle_date=now.date().isoformat(),
            days_elapsed_count=1,
        )
        self.save(state)
        return state

    def update(
        self,
        *,
        equity: float,
        daily_start_equity: float,
        now: datetime,
        traded_today: bool,
    ) -> ChallengeState:
        state = self.load(now=now, start_equity=self._settings.account_size)
        date_key = now.date().isoformat()
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

        if state.status == ChallengeStatus.IN_PROGRESS.value:
            if state.daily_loss_pct >= self._settings.prop_max_daily_loss_pct:
                state.status = ChallengeStatus.FAILED.value
                state.violation_reason = "daily_loss_limit"
                state.failed_ts = now.isoformat()
            elif state.global_dd_pct >= self._settings.prop_max_global_dd_pct:
                state.status = ChallengeStatus.FAILED.value
                state.violation_reason = "global_drawdown_limit"
                state.failed_ts = now.isoformat()
            elif (
                state.profit_pct >= self._settings.prop_profit_target_pct
                and state.days_traded_count >= self._settings.prop_min_trading_days
            ):
                state.status = ChallengeStatus.PASSED.value
                state.passed_ts = now.isoformat()
                state.violation_reason = "profit_target_reached"
        self.save(state)
        return state

    def status_payload(self, now: datetime | None = None) -> dict[str, Any]:
        state = self.load(now=now, start_equity=self._settings.account_size)
        payload = asdict(state)
        payload["server_ts"] = datetime.now(timezone.utc).isoformat()
        payload["candle_ts"] = (now or datetime.now(timezone.utc)).isoformat()
        payload["replay_cursor_ts"] = payload["candle_ts"]
        return payload
