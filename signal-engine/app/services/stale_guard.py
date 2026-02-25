from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from ..config import Settings
from ..utils.intervals import interval_to_ms


@dataclass(frozen=True)
class FreshnessStatus:
    last_candle_ts: str | None
    computed_now_ts: str | None
    last_candle_age_seconds: float | None
    stale_threshold_seconds: float
    stale_blocked: bool
    stale_clock_source: str
    replay_pointer_now_ts: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def replay_mode_active(settings: Settings) -> bool:
    run_mode = str(getattr(settings, "run_mode", "")).strip().lower()
    market_data_provider = str(getattr(settings, "market_data_provider", "")).strip().lower()
    market_provider = str(getattr(settings, "market_provider", "")).strip().lower()
    return run_mode == "replay" or market_data_provider == "replay" or market_provider == "replay"


def compute_freshness_status(
    *,
    settings: Settings,
    interval: str,
    last_candle_ts_ms: int | None,
    replay_now_ts: datetime | None = None,
    wall_now_ts: datetime | None = None,
) -> FreshnessStatus:
    stale_threshold_seconds = (interval_to_ms(interval) / 1000.0) + float(settings.market_data_allow_stale or 0)
    replay_mode = replay_mode_active(settings)

    if replay_mode:
        if replay_now_ts is None:
            return FreshnessStatus(
                last_candle_ts=_iso_from_ms(last_candle_ts_ms),
                computed_now_ts=None,
                last_candle_age_seconds=None,
                stale_threshold_seconds=stale_threshold_seconds,
                stale_blocked=False,
                stale_clock_source="always_fresh_fallback",
                replay_pointer_now_ts=None,
            )
        now_ts_ms = int(replay_now_ts.timestamp() * 1000)
        age_seconds = _age_seconds(now_ts_ms, last_candle_ts_ms)
        enforce_stale = bool(getattr(settings, "replay_enforce_stale", False))
        stale_blocked = bool(enforce_stale and _stale_enabled(last_candle_ts_ms) and age_seconds is not None and age_seconds > stale_threshold_seconds)
        return FreshnessStatus(
            last_candle_ts=_iso_from_ms(last_candle_ts_ms),
            computed_now_ts=replay_now_ts.isoformat(),
            last_candle_age_seconds=age_seconds,
            stale_threshold_seconds=stale_threshold_seconds,
            stale_blocked=stale_blocked,
            stale_clock_source="replay_clock",
            replay_pointer_now_ts=replay_now_ts.isoformat(),
        )

    wall_now = wall_now_ts or datetime.now(timezone.utc)
    now_ts_ms = int(wall_now.timestamp() * 1000)
    age_seconds = _age_seconds(now_ts_ms, last_candle_ts_ms)
    stale_blocked = bool(_stale_enabled(last_candle_ts_ms) and age_seconds is not None and age_seconds > stale_threshold_seconds)
    return FreshnessStatus(
        last_candle_ts=_iso_from_ms(last_candle_ts_ms),
        computed_now_ts=wall_now.isoformat(),
        last_candle_age_seconds=age_seconds,
        stale_threshold_seconds=stale_threshold_seconds,
        stale_blocked=stale_blocked,
        stale_clock_source="wall_clock",
        replay_pointer_now_ts=None,
    )


def _stale_enabled(last_candle_ts_ms: int | None) -> bool:
    return bool(last_candle_ts_ms is not None and int(last_candle_ts_ms) >= 1_600_000_000_000)


def _age_seconds(now_ts_ms: int, last_candle_ts_ms: int | None) -> float | None:
    if last_candle_ts_ms is None:
        return None
    return max(0.0, (now_ts_ms - int(last_candle_ts_ms)) / 1000.0)


def _iso_from_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
