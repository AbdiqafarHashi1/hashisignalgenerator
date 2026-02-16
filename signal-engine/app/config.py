from __future__ import annotations

from datetime import datetime, time
from functools import lru_cache
import json
import logging
import os
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",

    )

    MODE: Literal["prop_cfd", "personal_crypto", "paper", "signal_only", "live"] = Field("prop_cfd", validation_alias=AliasChoices("MODE", "mode"))
    PROFILE: Literal["profit", "diag"] = Field("profit", validation_alias=AliasChoices("PROFILE", "profile"))
    strategy_profile: Literal["SCALPER_FAST", "SCALPER_STABLE", "RANGE_MEAN_REVERT"] = Field("SCALPER_STABLE", validation_alias=AliasChoices("STRATEGY_PROFILE", "strategy_profile"))
    engine_mode: Literal["paper", "signal_only", "live"] = Field("signal_only", validation_alias=AliasChoices("ENGINE_MODE", "engine_mode"))
    symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["BTCUSDT"],
        validation_alias=AliasChoices("SYMBOLS", "symbols"),
    )
    heartbeat_minutes: int = Field(30, validation_alias=AliasChoices("HEARTBEAT_MINUTES", "heartbeat_minutes"))
    strategy: Literal["scalper", "baseline"] = Field(
        "scalper",
        validation_alias=AliasChoices("STRATEGY", "strategy"),
    )
    current_mode: Literal["SWING", "SCALP"] = Field(
        "SWING",
        validation_alias=AliasChoices("CURRENT_MODE", "current_mode"),
    )
    scalp_tp_pct: float = Field(0.005, validation_alias=AliasChoices("SCALP_TP_PCT", "scalp_tp_pct"))
    scalp_sl_pct: float = Field(0.0025, validation_alias=AliasChoices("SCALP_SL_PCT", "scalp_sl_pct"))
    scalp_max_hold_minutes: int = Field(
        60,
        validation_alias=AliasChoices("SCALP_MAX_HOLD_MINUTES", "scalp_max_hold_minutes"),
    )
    scalp_reentry_cooldown_minutes: int = Field(
        20,
        validation_alias=AliasChoices("SCALP_REENTRY_COOLDOWN_MINUTES", "scalp_reentry_cooldown_minutes"),
    )
    scalp_min_score: int = Field(78, validation_alias=AliasChoices("SCALP_MIN_SCORE", "scalp_min_score"))
    scalp_trend_filter_enabled: bool = Field(
        True,
        validation_alias=AliasChoices("SCALP_TREND_FILTER_ENABLED", "scalp_trend_filter_enabled"),
    )
    daily_max_dd_pct: float = Field(3.0, validation_alias=AliasChoices("DAILY_MAX_DD_PCT", "daily_max_dd_pct"))
    scalp_regime_enabled: bool = Field(
        True,
        validation_alias=AliasChoices("SCALP_REGIME_ENABLED", "scalp_regime_enabled"),
    )
    scalp_ema_fast: int = Field(20, validation_alias=AliasChoices("SCALP_EMA_FAST", "scalp_ema_fast"))
    scalp_ema_slow: int = Field(50, validation_alias=AliasChoices("SCALP_EMA_SLOW", "scalp_ema_slow"))
    scalp_ema_trend: int = Field(200, validation_alias=AliasChoices("SCALP_EMA_TREND", "scalp_ema_trend"))
    scalp_atr_period: int = Field(14, validation_alias=AliasChoices("SCALP_ATR_PERIOD", "scalp_atr_period"))
    scalp_atr_pct_min: float = Field(0.0015, validation_alias=AliasChoices("SCALP_ATR_PCT_MIN", "scalp_atr_pct_min"))
    scalp_atr_pct_max: float = Field(0.0120, validation_alias=AliasChoices("SCALP_ATR_PCT_MAX", "scalp_atr_pct_max"))
    scalp_trend_slope_min: float = Field(
        0.0002,
        validation_alias=AliasChoices("SCALP_TREND_SLOPE_MIN", "scalp_trend_slope_min"),
    )
    scalp_setup_mode: Literal["pullback_engulfing", "breakout_retest", "either"] = Field(
        "pullback_engulfing",
        validation_alias=AliasChoices("SCALP_SETUP_MODE", "scalp_setup_mode"),
    )
    scalp_pullback_ema: int = Field(20, validation_alias=AliasChoices("SCALP_PULLBACK_EMA", "scalp_pullback_ema"))
    scalp_pullback_max_dist_pct: float = Field(
        0.0020,
        validation_alias=AliasChoices("SCALP_PULLBACK_MAX_DIST_PCT", "scalp_pullback_max_dist_pct"),
    )
    scalp_engulfing_min_body_pct: float = Field(
        0.0006,
        validation_alias=AliasChoices("SCALP_ENGULFING_MIN_BODY_PCT", "scalp_engulfing_min_body_pct"),
    )
    scalp_rsi_period: int = Field(14, validation_alias=AliasChoices("SCALP_RSI_PERIOD", "scalp_rsi_period"))
    scalp_rsi_confirm: bool = Field(True, validation_alias=AliasChoices("SCALP_RSI_CONFIRM", "scalp_rsi_confirm"))
    scalp_rsi_long_min: float = Field(45, validation_alias=AliasChoices("SCALP_RSI_LONG_MIN", "scalp_rsi_long_min"))
    scalp_rsi_short_max: float = Field(55, validation_alias=AliasChoices("SCALP_RSI_SHORT_MAX", "scalp_rsi_short_max"))
    scalp_breakout_lookback: int = Field(
        20,
        validation_alias=AliasChoices("SCALP_BREAKOUT_LOOKBACK", "scalp_breakout_lookback"),
    )
    scalp_retest_max_bars: int = Field(
        6,
        validation_alias=AliasChoices("SCALP_RETEST_MAX_BARS", "scalp_retest_max_bars"),
    )
    min_signal_score_trend: int = Field(78, validation_alias=AliasChoices("MIN_SIGNAL_SCORE_TREND", "min_signal_score_trend"))
    min_signal_score_range: int = Field(70, validation_alias=AliasChoices("MIN_SIGNAL_SCORE_RANGE", "min_signal_score_range"))
    be_trigger_r_mult: float = Field(0.6, validation_alias=AliasChoices("BE_TRIGGER_R_MULT", "be_trigger_r_mult"))
    exit_score_min: int = Field(55, validation_alias=AliasChoices("EXIT_SCORE_MIN", "exit_score_min"))
    pullback_atr_mult: float = Field(0.5, validation_alias=AliasChoices("PULLBACK_ATR_MULT", "pullback_atr_mult"))
    sl_atr_mult: float = Field(1.0, validation_alias=AliasChoices("SL_ATR_MULT", "sl_atr_mult"))
    tp_atr_mult: float = Field(1.2, validation_alias=AliasChoices("TP_ATR_MULT", "tp_atr_mult"))
    dev_atr_mult: float = Field(1.0, validation_alias=AliasChoices("DEV_ATR_MULT", "dev_atr_mult"))
    range_sl_atr_mult: float = Field(0.8, validation_alias=AliasChoices("RANGE_SL_ATR_MULT", "range_sl_atr_mult"))
    range_tp_atr_mult: float = Field(1.0, validation_alias=AliasChoices("RANGE_TP_ATR_MULT", "range_tp_atr_mult"))
    symbol_cooldown_min: int = Field(5, validation_alias=AliasChoices("SYMBOL_COOLDOWN_MIN", "symbol_cooldown_min"))
    debug_loosen: bool = Field(False, validation_alias=AliasChoices("DEBUG_LOOSEN", "debug_loosen"))
    debug_disable_hard_risk_gates: bool = Field(
        False,
        validation_alias=AliasChoices("DEBUG_DISABLE_HARD_RISK_GATES", "debug_disable_hard_risk_gates"),
    )
    market_provider: str = Field("bybit", validation_alias=AliasChoices("MARKET_PROVIDER", "market_provider"))
    market_data_provider: str = Field("bybit", validation_alias=AliasChoices("MARKET_DATA_PROVIDER", "market_data_provider"))
    market_data_fallbacks: str = Field("binance,okx,replay", validation_alias=AliasChoices("MARKET_DATA_FALLBACKS", "market_data_fallbacks"))
    market_data_failover_threshold: int = Field(
        3,
        validation_alias=AliasChoices("MARKET_DATA_FAILOVER_THRESHOLD", "market_data_failover_threshold"),
    )
    market_data_backoff_base_ms: int = Field(
        500,
        validation_alias=AliasChoices("MARKET_DATA_BACKOFF_BASE_MS", "market_data_backoff_base_ms"),
    )
    market_data_backoff_max_ms: int = Field(
        15000,
        validation_alias=AliasChoices("MARKET_DATA_BACKOFF_MAX_MS", "market_data_backoff_max_ms"),
    )
    market_data_replay_path: str = Field(
        "data/replay",
        validation_alias=AliasChoices("MARKET_DATA_REPLAY_PATH", "market_data_replay_path"),
    )
    market_data_replay_speed: float = Field(
        1.0,
        validation_alias=AliasChoices("MARKET_DATA_REPLAY_SPEED", "market_data_replay_speed"),
    )
    market_data_allow_stale: int = Field(
        60,
        validation_alias=AliasChoices("MARKET_DATA_ALLOW_STALE", "market_data_allow_stale"),
    )
    market_data_enabled: bool = Field(True, validation_alias=AliasChoices("MARKET_DATA_ENABLED", "market_data_enabled"))

    bybit_testnet: bool = Field(False, validation_alias=AliasChoices("BYBIT_TESTNET", "bybit_testnet"))
    bybit_api_key: str = Field("", validation_alias=AliasChoices("BYBIT_API_KEY", "bybit_api_key"))
    bybit_api_secret: str = Field("", validation_alias=AliasChoices("BYBIT_API_SECRET", "bybit_api_secret"))
    bybit_rest_base: str = Field("https://api.bybit.com", validation_alias=AliasChoices("BYBIT_REST_BASE", "bybit_rest_base"))
    bybit_ws_public_linear: str = Field("wss://stream.bybit.com/v5/public/linear", validation_alias=AliasChoices("BYBIT_WS_PUBLIC_LINEAR", "bybit_ws_public_linear"))

    account_size: float | None = Field(None, validation_alias=AliasChoices("ACCOUNT_SIZE", "account_size"))
    base_risk_pct: float | None = Field(None, validation_alias=AliasChoices("BASE_RISK_PCT", "base_risk_pct"))
    risk_per_trade_usd: float | None = Field(None, validation_alias=AliasChoices("RISK_PER_TRADE_USD", "risk_per_trade_usd"))
    max_risk_pct: float | None = Field(None, validation_alias=AliasChoices("MAX_RISK_PCT", "max_risk_pct"))
    max_trades_per_day: int | None = Field(None, validation_alias=AliasChoices("MAX_TRADES_PER_DAY", "max_trades_per_day"))
    max_daily_loss_pct: float | None = Field(None, validation_alias=AliasChoices("MAX_DAILY_LOSS_PCT", "max_daily_loss_pct"))
    min_signal_score: int | None = Field(None, validation_alias=AliasChoices("MIN_SIGNAL_SCORE", "min_signal_score"))
    daily_profit_target_pct: float | None = Field(None, validation_alias=AliasChoices("DAILY_PROFIT_TARGET_PCT", "daily_profit_target_pct"))
    max_consecutive_losses: int | None = Field(None, validation_alias=AliasChoices("MAX_CONSECUTIVE_LOSSES", "max_consecutive_losses"))
    cooldown_minutes_after_loss: int | None = Field(None, validation_alias=AliasChoices("COOLDOWN_MINUTES_AFTER_LOSS", "cooldown_minutes_after_loss"))
    global_drawdown_limit_pct: float = Field(
        0.08,
        validation_alias=AliasChoices("GLOBAL_DD_LIMIT_PCT", "global_drawdown_limit_pct"),
    )
    manual_kill_switch: bool = Field(
        False,
        validation_alias=AliasChoices("MANUAL_KILL_SWITCH", "manual_kill_switch"),
    )

    funding_extreme_abs: float = Field(0.03, validation_alias=AliasChoices("FUNDING_EXTREME_ABS", "funding_extreme_abs"))
    funding_elevated_abs: float = Field(0.02, validation_alias=AliasChoices("FUNDING_ELEVATED_ABS", "funding_elevated_abs"))
    leverage_extreme: float = Field(3.0, validation_alias=AliasChoices("LEVERAGE_EXTREME", "leverage_extreme"))
    leverage_elevated: float = Field(2.5, validation_alias=AliasChoices("LEVERAGE_ELEVATED", "leverage_elevated"))
    oi_spike_pct: float = Field(0.18, validation_alias=AliasChoices("OI_SPIKE_PCT", "oi_spike_pct"))
    trend_strength_min: float | None = Field(None, validation_alias=AliasChoices("TREND_STRENGTH_MIN", "trend_strength_min"))
    candle_interval: Literal["1m", "3m", "5m"] | None = Field(None, validation_alias=AliasChoices("CANDLE_INTERVAL", "candle_interval"))
    candle_history_limit: int = Field(120, validation_alias=AliasChoices("CANDLE_HISTORY_LIMIT", "candle_history_limit"))
    tick_interval_seconds: int = Field(
        60,
        validation_alias=AliasChoices(
            "TICK_INTERVAL_SECONDS",
            "SCHEDULER_TICK_INTERVAL_SECONDS",
            "tick_interval_seconds",
        ),
    )

    trend_rsi_midline: float = Field(50.0, validation_alias=AliasChoices("TREND_RSI_MIDLINE", "trend_rsi_midline"))
    range_rsi_long_max: float = Field(35.0, validation_alias=AliasChoices("RANGE_RSI_LONG_MAX", "range_rsi_long_max"))
    range_rsi_short_min: float = Field(65.0, validation_alias=AliasChoices("RANGE_RSI_SHORT_MIN", "range_rsi_short_min"))
    trend_strength_scale: float = Field(2000.0, validation_alias=AliasChoices("TREND_STRENGTH_SCALE", "trend_strength_scale"))
    trend_score_slope_scale: float = Field(40000.0, validation_alias=AliasChoices("TREND_SCORE_SLOPE_SCALE", "trend_score_slope_scale"))
    trend_score_max_boost: int = Field(25, validation_alias=AliasChoices("TREND_SCORE_MAX_BOOST", "trend_score_max_boost"))
    atr_score_scale: float = Field(3000.0, validation_alias=AliasChoices("ATR_SCORE_SCALE", "atr_score_scale"))
    atr_score_max_boost: int = Field(15, validation_alias=AliasChoices("ATR_SCORE_MAX_BOOST", "atr_score_max_boost"))
    trend_confirm_score_boost: int = Field(10, validation_alias=AliasChoices("TREND_CONFIRM_SCORE_BOOST", "trend_confirm_score_boost"))
    range_base_score_boost: int = Field(8, validation_alias=AliasChoices("RANGE_BASE_SCORE_BOOST", "range_base_score_boost"))
    min_breakout_window: int = Field(25, validation_alias=AliasChoices("MIN_BREAKOUT_WINDOW", "min_breakout_window"))
    trend_bias_lookback: int = Field(20, validation_alias=AliasChoices("TREND_BIAS_LOOKBACK", "trend_bias_lookback"))
    setup_min_candles: int = Field(3, validation_alias=AliasChoices("SETUP_MIN_CANDLES", "setup_min_candles"))
    trend_min_candles: int = Field(4, validation_alias=AliasChoices("TREND_MIN_CANDLES", "trend_min_candles"))
    funding_guard_tail_minute: int = Field(0, validation_alias=AliasChoices("FUNDING_GUARD_TAIL_MINUTE", "funding_guard_tail_minute"))
    tp_cap_long_pct: float = Field(0.02, validation_alias=AliasChoices("TP_CAP_LONG_PCT", "tp_cap_long_pct"))
    tp_cap_short_pct: float = Field(0.02, validation_alias=AliasChoices("TP_CAP_SHORT_PCT", "tp_cap_short_pct"))
    max_notional_account_multiplier: float = Field(3.0, validation_alias=AliasChoices("MAX_NOTIONAL_ACCOUNT_MULTIPLIER", "max_notional_account_multiplier"))
    move_to_breakeven_trigger_r: float = Field(
        0.6,
        validation_alias=AliasChoices(
            "MOVE_TO_BREAKEVEN_TRIGGER_R",
            "BE_TRIGGER_R_MULT",
            "move_to_breakeven_trigger_r",
        ),
    )
    move_to_breakeven_buffer_r: float = Field(
        0.05,
        validation_alias=AliasChoices("BE_BUFFER_R", "move_to_breakeven_buffer_r"),
    )
    move_to_breakeven_buffer_bps: float = Field(
        0.0,
        validation_alias=AliasChoices("BE_BUFFER_BPS", "move_to_breakeven_buffer_bps"),
    )
    move_to_breakeven_min_seconds_open: int = Field(
        60,
        validation_alias=AliasChoices("BE_MIN_SECONDS_OPEN", "move_to_breakeven_min_seconds_open"),
    )
    move_to_breakeven_offset_bps: float = Field(
        0.0,
        validation_alias=AliasChoices("BE_OFFSET_BPS", "move_to_breakeven_offset_bps"),
    )
    breakout_tp_multiplier: float = Field(1.2, validation_alias=AliasChoices("BREAKOUT_TP_MULTIPLIER", "breakout_tp_multiplier"))
    regime_entry_buffer_pct: float = Field(0.0002, validation_alias=AliasChoices("REGIME_ENTRY_BUFFER_PCT", "regime_entry_buffer_pct"))
    prop_score_threshold: int = Field(80, validation_alias=AliasChoices("PROP_SCORE_THRESHOLD", "prop_score_threshold"))
    personal_score_threshold: int = Field(75, validation_alias=AliasChoices("PERSONAL_SCORE_THRESHOLD", "personal_score_threshold"))

    force_trade_mode: bool = Field(False, validation_alias=AliasChoices("FORCE_TRADE_MODE", "force_trade_mode"))
    force_trade_every_seconds: int = Field(
        5,
        validation_alias=AliasChoices("FORCE_TRADE_EVERY_SECONDS", "force_trade_every_seconds"),
    )
    force_trade_cooldown_seconds: int = Field(
        0,
        validation_alias=AliasChoices("FORCE_TRADE_COOLDOWN_SECONDS", "force_trade_cooldown_seconds"),
    )
    force_trade_auto_close_seconds: int = Field(
        0,
        validation_alias=AliasChoices("FORCE_TRADE_AUTO_CLOSE_SECONDS", "force_trade_auto_close_seconds"),
    )
    force_trade_random_direction: bool = Field(
        True,
        validation_alias=AliasChoices("FORCE_TRADE_RANDOM_DIRECTION", "force_trade_random_direction"),
    )
    sweet8_enabled: bool = Field(False, validation_alias=AliasChoices("SWEET8_ENABLED", "sweet8_enabled"))
    sweet8_mode: Literal["auto", "scalper", "swing"] = Field(
        "auto",
        validation_alias=AliasChoices("SWEET8_MODE", "sweet8_mode"),
    )
    sweet8_base_risk_pct: float = Field(0.0025, validation_alias=AliasChoices("SWEET8_BASE_RISK_PCT", "sweet8_base_risk_pct"))
    sweet8_max_risk_pct: float = Field(0.005, validation_alias=AliasChoices("SWEET8_MAX_RISK_PCT", "sweet8_max_risk_pct"))
    sweet8_max_daily_loss_pct: float = Field(
        0.015,
        validation_alias=AliasChoices("SWEET8_MAX_DAILY_LOSS_PCT", "sweet8_max_daily_loss_pct"),
    )
    sweet8_disable_premature_exits: bool = Field(
        True,
        validation_alias=AliasChoices("SWEET8_DISABLE_PREMATURE_EXITS", "sweet8_disable_premature_exits"),
    )
    sweet8_disable_time_stop: bool = Field(True, validation_alias=AliasChoices("SWEET8_DISABLE_TIME_STOP", "sweet8_disable_time_stop"))
    sweet8_disable_force_auto_close: bool = Field(
        True,
        validation_alias=AliasChoices("SWEET8_DISABLE_FORCE_AUTO_CLOSE", "sweet8_disable_force_auto_close"),
    )
    sweet8_max_open_positions_total: int = Field(
        1,
        validation_alias=AliasChoices("SWEET8_MAX_OPEN_POSITIONS_TOTAL", "sweet8_max_open_positions_total"),
    )
    sweet8_max_open_positions_per_symbol: int = Field(
        1,
        validation_alias=AliasChoices("SWEET8_MAX_OPEN_POSITIONS_PER_SYMBOL", "sweet8_max_open_positions_per_symbol"),
    )
    sweet8_scalp_atr_sl_mult: float = Field(
        1.0,
        validation_alias=AliasChoices("SWEET8_SCALP_ATR_SL_MULT", "sweet8_scalp_atr_sl_mult"),
    )
    sweet8_scalp_atr_tp_mult: float = Field(
        1.4,
        validation_alias=AliasChoices("SWEET8_SCALP_ATR_TP_MULT", "sweet8_scalp_atr_tp_mult"),
    )
    sweet8_scalp_min_score: int = Field(
        75,
        validation_alias=AliasChoices("SWEET8_SCALP_MIN_SCORE", "sweet8_scalp_min_score"),
    )
    sweet8_swing_atr_sl_mult: float = Field(
        1.5,
        validation_alias=AliasChoices("SWEET8_SWING_ATR_SL_MULT", "sweet8_swing_atr_sl_mult"),
    )
    sweet8_swing_atr_tp_mult: float = Field(
        2.5,
        validation_alias=AliasChoices("SWEET8_SWING_ATR_TP_MULT", "sweet8_swing_atr_tp_mult"),
    )
    sweet8_swing_min_score: int = Field(
        85,
        validation_alias=AliasChoices("SWEET8_SWING_MIN_SCORE", "sweet8_swing_min_score"),
    )
    sweet8_regime_adx_threshold: int = Field(
        25,
        validation_alias=AliasChoices("SWEET8_REGIME_ADX_THRESHOLD", "sweet8_regime_adx_threshold"),
    )
    sweet8_regime_vol_threshold: float = Field(
        1.2,
        validation_alias=AliasChoices("SWEET8_REGIME_VOL_THRESHOLD", "sweet8_regime_vol_threshold"),
    )
    sweet8_blocked_close_total: int = 0
    sweet8_blocked_close_time_stop: int = 0
    sweet8_blocked_close_force: int = 0
    sweet8_current_mode: Literal["auto", "scalper", "swing"] = "auto"
    ema_length: int = 50
    momentum_mode: Literal["adx", "atr"] = "adx"
    adx_period: int = Field(14, validation_alias=AliasChoices("ADX_PERIOD", "adx_period"))
    adx_threshold: float = Field(20.0, validation_alias=AliasChoices("ADX_THRESHOLD", "adx_threshold"))
    atr_period: int = Field(14, validation_alias=AliasChoices("ATR_PERIOD", "atr_period"))
    atr_sma_period: int = 20
    ema_pullback_pct: float = Field(0.0015, validation_alias=AliasChoices("EMA_PULLBACK_PCT", "ema_pullback_pct"))
    engulfing_wick_ratio: float = Field(0.5, validation_alias=AliasChoices("ENGULFING_WICK_RATIO", "engulfing_wick_ratio"))
    volume_confirm_enabled: bool = True
    volume_sma_period: int = 20
    volume_confirm_multiplier: float = 1.0
    max_stop_pct: float = 0.0025
    take_profit_pct: float = 0.003
    fee_rate_bps: float = Field(5.5, validation_alias=AliasChoices("FEE_BPS", "fee_rate_bps"))
    spread_bps: float = Field(1.5, validation_alias=AliasChoices("SPREAD_BPS", "spread_bps"))
    slippage_bps: float = Field(1.5, validation_alias=AliasChoices("SLIPPAGE_BPS", "slippage_bps"))
    breakout_volume_multiplier: float = 1.5
    breakout_atr_multiplier: float = 1.2
    max_hold_minutes: int = Field(
        720,
        validation_alias=AliasChoices(
            "MAX_HOLD_MINUTES",
            "max_hold_minutes",
            "TIME_STOP_MINUTES",
            "time_stop_minutes",
        ),
    )
    reentry_cooldown_minutes: int = Field(
        30,
        validation_alias=AliasChoices("REENTRY_COOLDOWN_MINUTES", "reentry_cooldown_minutes"),
    )
    funding_block_before_minutes: int = Field(
        10,
        validation_alias=AliasChoices("FUNDING_BLOCK_BEFORE_MINUTES", "funding_block_before_minutes"),
    )
    funding_close_before_minutes: int = Field(
        2,
        validation_alias=AliasChoices("FUNDING_CLOSE_BEFORE_MINUTES", "funding_close_before_minutes"),
    )
    funding_interval_minutes: int = Field(
        480,
        validation_alias=AliasChoices("FUNDING_INTERVAL_MINUTES", "funding_interval_minutes"),
    )
    funding_blackout_force_close: bool = Field(
        False,
        validation_alias=AliasChoices("FUNDING_BLACKOUT_FORCE_CLOSE", "funding_blackout_force_close"),
    )
    funding_blackout_max_util_pct: float = Field(
        70,
        validation_alias=AliasChoices("FUNDING_BLACKOUT_MAX_UTIL_PCT", "funding_blackout_max_util_pct"),
    )
    funding_blackout_max_loss_usd: float = Field(
        100,
        validation_alias=AliasChoices("FUNDING_BLACKOUT_MAX_LOSS_USD", "funding_blackout_max_loss_usd"),
    )

    max_losses_per_day: int = Field(3, validation_alias=AliasChoices("MAX_LOSSES_PER_DAY", "max_losses_per_day"))

    news_blackouts: str = ""
    data_dir: str = "data"
    telegram_debug_skips: bool = Field(False, validation_alias=AliasChoices("TELEGRAM_DEBUG_SKIPS", "telegram_debug_skips"))
    telegram_enabled: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    smoke_test_force_trade: bool = Field(False, validation_alias=AliasChoices("SMOKE_TEST_FORCE_TRADE", "smoke_test_force_trade"))
    next_public_api_base: str | None = Field(None, validation_alias=AliasChoices("NEXT_PUBLIC_API_BASE", "next_public_api_base"))
    internal_api_base_url: str | None = Field(None, validation_alias=AliasChoices("INTERNAL_API_BASE_URL", "internal_api_base_url"))
    database_url: str | None = Field(None, validation_alias=AliasChoices("DATABASE_URL", "database_url"))

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"SYMBOLS must be a JSON list string or comma-separated list, got {value!r}") from exc
            else:
                value = [item.strip() for item in raw.split(",") if item.strip()]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"SYMBOLS must be a list of strings, got {value!r}")
        return [item.strip().upper() for item in value if item.strip()]

    @field_validator("candle_interval", mode="before")
    @classmethod
    def validate_candle_interval(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        normalized = str(value).strip().lower()
        if normalized not in {"1m", "3m", "5m"}:
            raise ValueError("CANDLE_INTERVAL must be one of: 1m, 3m, 5m")
        return normalized

    @model_validator(mode="after")
    def apply_mode_defaults(self) -> "Settings":
        logger = logging.getLogger(__name__)
        profile_defaults = self._profile_defaults()
        for key, value in profile_defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        if "ENGINE_MODE" not in os.environ and self.MODE in {"paper", "signal_only", "live"}:
            self.engine_mode = self.MODE

        if self.sweet8_enabled:
            self.debug_loosen = False
            self.debug_disable_hard_risk_gates = False
            if "BASE_RISK_PCT" not in os.environ:
                self.base_risk_pct = self.sweet8_base_risk_pct
            if "MAX_RISK_PCT" not in os.environ:
                self.max_risk_pct = self.sweet8_max_risk_pct
            if "MAX_DAILY_LOSS_PCT" not in os.environ:
                self.max_daily_loss_pct = self.sweet8_max_daily_loss_pct
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
                "account_size": 25000,
                "base_risk_pct": 0.0025,
                "max_risk_pct": 0.0025,
                "max_daily_loss_pct": 0.015,
                "daily_profit_target_pct": 0.015,
                "max_consecutive_losses": 2,
                "candle_interval": "1m",
                "min_signal_score": 35,
                "trend_strength_min": 0.30,
                "cooldown_minutes_after_loss": 0,
                "max_trades_per_day": 50,
            }
        profiles: dict[str, dict[str, object]] = {
            "SCALPER_FAST": {
                "account_size": 25000,
                "base_risk_pct": 0.0025,
                "max_risk_pct": 0.0025,
                "max_daily_loss_pct": 0.015,
                "daily_profit_target_pct": 0.015,
                "max_consecutive_losses": 2,
                "candle_interval": "1m",
                "tick_interval_seconds": 30,
                "min_signal_score": 55,
                "trend_strength_min": 0.35,
                "cooldown_minutes_after_loss": 0,
                "max_trades_per_day": 50,
            },
            "SCALPER_STABLE": {
                "account_size": 25000,
                "base_risk_pct": 0.0025,
                "max_risk_pct": 0.0025,
                "max_daily_loss_pct": 0.015,
                "daily_profit_target_pct": 0.015,
                "max_consecutive_losses": 2,
                "candle_interval": "1m",
                "tick_interval_seconds": 60,
                "min_signal_score": 60,
                "trend_strength_min": 0.45,
                "cooldown_minutes_after_loss": 10,
                "max_trades_per_day": 8,
            },
            "RANGE_MEAN_REVERT": {
                "account_size": 25000,
                "base_risk_pct": 0.0020,
                "max_risk_pct": 0.0025,
                "max_daily_loss_pct": 0.012,
                "daily_profit_target_pct": 0.012,
                "max_consecutive_losses": 2,
                "candle_interval": "5m",
                "tick_interval_seconds": 60,
                "min_signal_score": 58,
                "trend_strength_min": 0.35,
                "cooldown_minutes_after_loss": 10,
                "max_trades_per_day": 6,
            },
        }
        return profiles[self.strategy_profile]

    def config_sources(self) -> dict[str, str]:
        profile_keys = set(self._profile_defaults().keys())
        aliases = self.env_alias_map()
        sources: dict[str, str] = {}
        for name in self.__class__.model_fields:
            env_key = aliases.get(name, name.upper())
            if env_key in os.environ:
                sources[name] = "env"
            elif name in profile_keys:
                sources[name] = "profile_default"
            else:
                sources[name] = "default"
        return sources

    def env_alias_map(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for name, field in self.__class__.model_fields.items():
            alias = field.validation_alias
            if isinstance(alias, AliasChoices) and alias.choices:
                aliases[name] = str(alias.choices[0])
            else:
                aliases[name] = name.upper()
        return aliases

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
            "strategy_profile": self.strategy_profile,
            "mode": self.MODE,
            "engine_mode": self.engine_mode,
            "account_size": self.account_size,
            "symbols": list(self.symbols),
            "candle_interval": self.candle_interval,
            "tick_interval_seconds": self.tick_interval_seconds,
            "force_trade_mode": self.force_trade_mode,
            "force_trade_every_seconds": self.force_trade_every_seconds,
            "force_trade_cooldown_seconds": self.force_trade_cooldown_seconds,
            "force_trade_auto_close_seconds": self.force_trade_auto_close_seconds,
            "force_trade_random_direction": self.force_trade_random_direction,
            "min_signal_score": self.min_signal_score,
            "trend_strength_min": self.trend_strength_min,
            "cooldown_minutes_after_loss": self.cooldown_minutes_after_loss,
            "max_trades_per_day": self.max_trades_per_day,
            "max_losses_per_day": self.max_losses_per_day,
            "current_mode": self.current_mode,
            "scalp_tp_pct": self.scalp_tp_pct,
            "scalp_sl_pct": self.scalp_sl_pct,
            "scalp_max_hold_minutes": self.scalp_max_hold_minutes,
            "scalp_reentry_cooldown_minutes": self.scalp_reentry_cooldown_minutes,
            "scalp_min_score": self.scalp_min_score,
            "scalp_trend_filter_enabled": self.scalp_trend_filter_enabled,
            "scalp_regime_enabled": self.scalp_regime_enabled,
            "scalp_ema_fast": self.scalp_ema_fast,
            "scalp_ema_slow": self.scalp_ema_slow,
            "scalp_ema_trend": self.scalp_ema_trend,
            "scalp_atr_period": self.scalp_atr_period,
            "scalp_atr_pct_min": self.scalp_atr_pct_min,
            "scalp_atr_pct_max": self.scalp_atr_pct_max,
            "scalp_trend_slope_min": self.scalp_trend_slope_min,
            "scalp_setup_mode": self.scalp_setup_mode,
            "scalp_pullback_ema": self.scalp_pullback_ema,
            "scalp_pullback_max_dist_pct": self.scalp_pullback_max_dist_pct,
            "scalp_engulfing_min_body_pct": self.scalp_engulfing_min_body_pct,
            "scalp_rsi_period": self.scalp_rsi_period,
            "scalp_rsi_confirm": self.scalp_rsi_confirm,
            "scalp_rsi_long_min": self.scalp_rsi_long_min,
            "scalp_rsi_short_max": self.scalp_rsi_short_max,
            "scalp_breakout_lookback": self.scalp_breakout_lookback,
            "scalp_retest_max_bars": self.scalp_retest_max_bars,
            "min_signal_score_trend": self.min_signal_score_trend,
            "min_signal_score_range": self.min_signal_score_range,
            "be_trigger_r_mult": self.be_trigger_r_mult,
            "move_to_breakeven_trigger_r": self.move_to_breakeven_trigger_r,
            "move_to_breakeven_buffer_r": self.move_to_breakeven_buffer_r,
            "move_to_breakeven_buffer_bps": self.move_to_breakeven_buffer_bps,
            "move_to_breakeven_min_seconds_open": self.move_to_breakeven_min_seconds_open,
            "move_to_breakeven_offset_bps": self.move_to_breakeven_offset_bps,
            "exit_score_min": self.exit_score_min,
            "pullback_atr_mult": self.pullback_atr_mult,
            "sl_atr_mult": self.sl_atr_mult,
            "tp_atr_mult": self.tp_atr_mult,
            "dev_atr_mult": self.dev_atr_mult,
            "range_sl_atr_mult": self.range_sl_atr_mult,
            "range_tp_atr_mult": self.range_tp_atr_mult,
            "symbol_cooldown_min": self.symbol_cooldown_min,
            "daily_max_dd_pct": self.daily_max_dd_pct,
            "telegram_debug_skips": self.telegram_debug_skips,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "global_drawdown_limit_pct": self.global_drawdown_limit_pct,
            "base_risk_pct": self.base_risk_pct,
            "risk_per_trade_usd": self.risk_per_trade_usd,
            "max_risk_pct": self.max_risk_pct,
            "news_blackouts": self.news_blackouts,
            "debug_loosen": self.debug_loosen,
            "debug_disable_hard_risk_gates": self.debug_disable_hard_risk_gates,
            "strategy": self.strategy,
            "market_provider": self.market_provider,
            "market_data_provider": self.market_data_provider,
            "market_data_fallbacks": self.market_data_fallbacks,
            "market_data_replay_path": self.market_data_replay_path,
            "market_data_replay_speed": self.market_data_replay_speed,
            "market_data_allow_stale": self.market_data_allow_stale,
            "market_data_failover_threshold": self.market_data_failover_threshold,
            "market_data_backoff_base_ms": self.market_data_backoff_base_ms,
            "market_data_backoff_max_ms": self.market_data_backoff_max_ms,
            "market_data_enabled": self.market_data_enabled,
            "bybit_testnet": self.bybit_testnet,
            "bybit_rest_base": self.bybit_rest_base,
            "bybit_ws_public_linear": self.bybit_ws_public_linear,
            "manual_kill_switch": self.manual_kill_switch,
            "smoke_test_force_trade": self.smoke_test_force_trade,
            "database_url": self.database_url,
            "sweet8_enabled": self.sweet8_enabled,
            "sweet8_mode": self.sweet8_mode,
            "sweet8_current_mode": self.sweet8_current_mode,
            "sweet8_blocked_close_total": self.sweet8_blocked_close_total,
            "sweet8_blocked_close_time_stop": self.sweet8_blocked_close_time_stop,
            "sweet8_blocked_close_force": self.sweet8_blocked_close_force,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
