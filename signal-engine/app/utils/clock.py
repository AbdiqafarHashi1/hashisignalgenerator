from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone


class Clock:
    def now_ts(self) -> int:
        raise NotImplementedError

    def now_dt(self) -> datetime:
        raise NotImplementedError

    async def sleep(self, seconds: float) -> None:
        raise NotImplementedError


class RealClock(Clock):
    def now_ts(self) -> int:
        return int(time.time())

    def now_dt(self) -> datetime:
        return datetime.now(timezone.utc)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, float(seconds)))


class ReplayClock(Clock):
    def __init__(self) -> None:
        self.current_ts: int = int(time.time())

    def set_ts(self, ts: int) -> None:
        self.current_ts = int(ts)

    def now_ts(self) -> int:
        return self.current_ts

    def now_dt(self) -> datetime:
        return datetime.fromtimestamp(self.current_ts, tz=timezone.utc)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(0)

