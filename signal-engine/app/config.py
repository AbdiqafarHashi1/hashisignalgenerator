from __future__ import annotations

from datetime import datetime, time
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",

    )

    MODE: Literal["prop_cfd", "personal_crypto", "paper", "signal_only"] = "prop_cfd"
    engine_mode: Literal["paper", "signal_only"] = "signal_only"
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT"], validation_alias=AliasChoices("SYMBOLS", "symbols"))
    heartbeat_minutes: int = Field(30, validation_alias=AliasChoices("HEARTBEAT_MINUTES", "heartbeat_minutes"))

    account_size: float | None = None
    base_risk_pct: float | None = None
    max_risk_pct: float | None = None
    max_trades_per_day: int | None = None
    max_daily_loss_pct: float | None = None
    min_signal_score: int | None = None

    funding_extreme_abs: float = 0.03
    funding_elevated_abs: float = 0.02
    leverage_extreme: float = 3.0
    leverage_elevated: float = 2.5
    oi_spike_pct: float = 0.18
    trend_strength_min: float = 0.6

    cooldown_minutes_after_loss: int = 30
    max_losses_per_day: int = 2

    news_blackouts: str = ""
    data_dir: str = "data"
    telegram_enabled: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def apply_mode_defaults(self) -> "Settings":
        if self.MODE in {"paper", "signal_only"}:
            self.engine_mode = self.MODE
            self.MODE = "prop_cfd"
        if self.MODE == "prop_cfd":
            defaults = {
                "account_size": 25000,
                "base_risk_pct": 0.0035,
                "max_risk_pct": 0.005,
                "max_trades_per_day": 2,
                "max_daily_loss_pct": 0.01,
                "min_signal_score": 75,
            }
        else:
            defaults = {
                "account_size": 2000,
                "base_risk_pct": 0.0075,
                "max_risk_pct": 0.01,
                "max_trades_per_day": 2,
                "max_daily_loss_pct": 0.02,
                "min_signal_score": 65,
            }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        return self

    def blackout_windows(self) -> list[tuple[time, time]]:
        windows: list[tuple[time, time]] = []
        if not self.news_blackouts.strip():
            return windows
        for part in self.news_blackouts.split(","):
            part = part.strip()
            if not part:
                continue
            start_s, end_s = part.split("-")
            start_t = time.fromisoformat(start_s.strip())
            end_t = time.fromisoformat(end_s.strip())
            windows.append((start_t, end_t))
        return windows

    def is_blackout(self, dt: datetime) -> bool:
        for start_t, end_t in self.blackout_windows():
            if start_t <= dt.time() <= end_t:
                return True
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings()
