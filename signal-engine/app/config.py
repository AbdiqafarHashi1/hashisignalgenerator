from __future__ import annotations

from datetime import datetime, time
from functools import lru_cache
import json
import logging
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
    PROFILE: Literal["profit", "diag"] = "profit"
    engine_mode: Literal["paper", "signal_only"] = "signal_only"
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT"], validation_alias=AliasChoices("SYMBOLS", "symbols"))
    heartbeat_minutes: int = Field(30, validation_alias=AliasChoices("HEARTBEAT_MINUTES", "heartbeat_minutes"))
    strategy: Literal["scalper", "baseline"] = Field(
        "scalper",
        validation_alias=AliasChoices("STRATEGY", "strategy"),
    )
    debug_loosen: bool = Field(False, validation_alias=AliasChoices("DEBUG_LOOSEN", "debug_loosen"))
    debug_disable_hard_risk_gates: bool = Field(
        False,
        validation_alias=AliasChoices("DEBUG_DISABLE_HARD_RISK_GATES", "debug_disable_hard_risk_gates"),
    )
    market_provider: str = Field("binance_public", validation_alias=AliasChoices("MARKET_PROVIDER", "market_provider"))
    market_data_enabled: bool = Field(True, validation_alias=AliasChoices("MARKET_DATA_ENABLED", "market_data_enabled"))

    account_size: float | None = None
    base_risk_pct: float | None = None
    max_risk_pct: float | None = None
    max_trades_per_day: int | None = None
    max_daily_loss_pct: float | None = None
    min_signal_score: int | None = None
    daily_profit_target_pct: float | None = None
    max_consecutive_losses: int | None = None
    cooldown_minutes_after_loss: int | None = None

    funding_extreme_abs: float = 0.03
    funding_elevated_abs: float = 0.02
    leverage_extreme: float = 3.0
    leverage_elevated: float = 2.5
    oi_spike_pct: float = 0.18
    trend_strength_min: float | None = None
    candle_interval: str | None = None
    candle_history_limit: int = 120
    tick_interval_seconds: int = Field(60, validation_alias=AliasChoices("TICK_INTERVAL_SECONDS", "tick_interval_seconds"))
    ema_length: int = 50
    momentum_mode: Literal["adx", "atr"] = "adx"
    adx_period: int = 14
    adx_threshold: float = 20.0
    atr_period: int = 14
    atr_sma_period: int = 20
    ema_pullback_pct: float = 0.0015
    engulfing_wick_ratio: float = 0.5
    volume_confirm_enabled: bool = True
    volume_sma_period: int = 20
    volume_confirm_multiplier: float = 1.0
    max_stop_pct: float = 0.0025
    take_profit_pct: float = 0.003

    max_losses_per_day: int = 3

    news_blackouts: str = ""
    data_dir: str = "data"
    telegram_enabled: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    smoke_test_force_trade: bool = Field(False, validation_alias=AliasChoices("SMOKE_TEST_FORCE_TRADE", "smoke_test_force_trade"))

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"SYMBOLS must be a JSON list string, got {value!r}") from exc
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"SYMBOLS must be a list of strings, got {value!r}")
        return [item.strip().upper() for item in value if item.strip()]

    @model_validator(mode="after")
    def apply_mode_defaults(self) -> "Settings":
        logger = logging.getLogger(__name__)
        if self.MODE in {"paper", "signal_only"}:
            self.engine_mode = self.MODE
            self.MODE = "prop_cfd"
        if self.PROFILE == "diag":
            self.engine_mode = "signal_only"
        if self.MODE == "prop_cfd":
            defaults = {
                "account_size": 25000,
                "base_risk_pct": 0.0025,
                "max_risk_pct": 0.0025,
                "max_daily_loss_pct": 0.02,
                "daily_profit_target_pct": 0.015,
                "max_consecutive_losses": 3,
            }
        else:
            defaults = {
                "account_size": 2000,
                "base_risk_pct": 0.0025,
                "max_risk_pct": 0.0025,
                "max_daily_loss_pct": 0.02,
                "daily_profit_target_pct": 0.015,
                "max_consecutive_losses": 3,
            }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        profile_defaults = self._profile_defaults()
        for key, value in profile_defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        if self.debug_loosen:
            changes: list[str] = []
            if self.min_signal_score is not None:
                original = self.min_signal_score
                self.min_signal_score = max(0, self.min_signal_score - 8)
                if self.min_signal_score != original:
                    changes.append(f"min_signal_score {original}->{self.min_signal_score}")
            if self.trend_strength_min is not None:
                original = self.trend_strength_min
                self.trend_strength_min = max(0.0, self.trend_strength_min - 0.05)
                if self.trend_strength_min != original:
                    changes.append(f"trend_strength_min {original}->{self.trend_strength_min}")
            if self.adx_threshold is not None:
                original = self.adx_threshold
                self.adx_threshold = max(0.0, self.adx_threshold - 3.0)
                if self.adx_threshold != original:
                    changes.append(f"adx_threshold {original}->{self.adx_threshold}")
            if changes:
                logger.info("debug_loosen applied: %s", ", ".join(changes))
        return self

    def _profile_defaults(self) -> dict[str, object]:
        if self.PROFILE == "diag":
            return {
                "candle_interval": "1m",
                "min_signal_score": 35,
                "trend_strength_min": 0.30,
                "cooldown_minutes_after_loss": 0,
                "max_trades_per_day": 50,
            }
        return {
            "candle_interval": "1m",
            "min_signal_score": 60,
            "trend_strength_min": 0.45,
            "cooldown_minutes_after_loss": 10,
            "max_trades_per_day": 8,
        }

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
        if self.debug_loosen:
            return False
        for start_t, end_t in self.blackout_windows():
            if start_t <= dt.time() <= end_t:
                return True
        return False

    def resolved_settings(self) -> dict[str, object]:
        return {
            "profile": self.PROFILE,
            "mode": self.MODE,
            "engine_mode": self.engine_mode,
            "symbols": list(self.symbols),
            "candle_interval": self.candle_interval,
            "tick_interval_seconds": self.tick_interval_seconds,
            "min_signal_score": self.min_signal_score,
            "trend_strength_min": self.trend_strength_min,
            "cooldown_minutes_after_loss": self.cooldown_minutes_after_loss,
            "max_trades_per_day": self.max_trades_per_day,
            "max_losses_per_day": self.max_losses_per_day,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "base_risk_pct": self.base_risk_pct,
            "max_risk_pct": self.max_risk_pct,
            "news_blackouts": self.news_blackouts,
            "debug_loosen": self.debug_loosen,
            "debug_disable_hard_risk_gates": self.debug_disable_hard_risk_gates,
            "strategy": self.strategy,
            "market_provider": self.market_provider,
            "market_data_enabled": self.market_data_enabled,
            "smoke_test_force_trade": self.smoke_test_force_trade,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
