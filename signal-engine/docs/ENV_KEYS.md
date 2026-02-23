# Environment Keys Audit

Generated at: `2026-02-23T00:45:50.545190+00:00`

## How env is loaded
1. BaseSettings reads .env file first (env_file=.env)
1. Process environment variables override matching keys
1. Field defaults apply when env keys are absent
1. Profile defaults (_profile_defaults) fill None fields
1. Runtime overrides in apply_mode_defaults adjust derived values

## Table 1: Canonical Keys

| Key | Type | Default | Component | Used For | Source Locations | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| ACCOUNT_SIZE | float | `None` | Settings | Config field `account_size` | app/config.py:226 (Settings.account_size) | active |
| ADX_PERIOD | int | `14` | Settings | Config field `adx_period` | app/config.py:454 (Settings.adx_period) | active |
| ADX_THRESHOLD | float | `20.0` | Settings | Config field `adx_threshold` | app/config.py:455 (Settings.adx_threshold) | active |
| ATR_PERIOD | int | `14` | Settings | Config field `atr_period` | app/config.py:456 (Settings.atr_period) | active |
| ATR_SCORE_MAX_BOOST | int | `15` | Settings | Config field `atr_score_max_boost` | app/config.py:279 (Settings.atr_score_max_boost) | active |
| ATR_SCORE_SCALE | float | `3000.0` | Settings | Config field `atr_score_scale` | app/config.py:278 (Settings.atr_score_scale) | active |
| ATR_SMA_PERIOD | int | `20` | Settings | Config field `atr_sma_period` | app/config.py:457 (Settings.atr_sma_period) | active |
| BASE_RISK_PCT | float | `None` | Settings | Config field `base_risk_pct` | app/config.py:227 (Settings.base_risk_pct) | active |
| BE_BUFFER_BPS | float | `0.0` | Settings | Config field `move_to_breakeven_buffer_bps` | app/config.py:303 (Settings.move_to_breakeven_buffer_bps) | dead/legacy |
| BE_BUFFER_R | float | `0.05` | Settings | Config field `move_to_breakeven_buffer_r` | app/config.py:299 (Settings.move_to_breakeven_buffer_r) | dead/legacy |
| BE_ENABLED | bool | `True` | Settings | Config field `be_enabled` | app/config.py:351 (Settings.be_enabled) | dead/legacy |
| BE_MIN_SECONDS_OPEN | int | `60` | Settings | Config field `move_to_breakeven_min_seconds_open` | app/config.py:307 (Settings.move_to_breakeven_min_seconds_open) | active |
| BE_OFFSET_BPS | float | `0.0` | Settings | Config field `move_to_breakeven_offset_bps` | app/config.py:311 (Settings.move_to_breakeven_offset_bps) | dead/legacy |
| BE_TRIGGER_R_MULT | float | `0.6` | Settings | Config field `be_trigger_r_mult` | app/config.py:146 (Settings.be_trigger_r_mult) | dead/legacy |
| BREAKOUT_ATR_MULTIPLIER | float | `1.2` | Settings | Config field `breakout_atr_multiplier` | app/config.py:469 (Settings.breakout_atr_multiplier) | active |
| BREAKOUT_TP_MULTIPLIER | float | `1.2` | Settings | Config field `breakout_tp_multiplier` | app/config.py:315 (Settings.breakout_tp_multiplier) | active |
| BREAKOUT_VOLUME_MULTIPLIER | float | `1.5` | Settings | Config field `breakout_volume_multiplier` | app/config.py:468 (Settings.breakout_volume_multiplier) | active |
| BYBIT_API_KEY | str | `""` | Settings | Config field `bybit_api_key` | app/config.py:221 (Settings.bybit_api_key) | dead/legacy |
| BYBIT_API_SECRET | str | `""` | Settings | Config field `bybit_api_secret` | app/config.py:222 (Settings.bybit_api_secret) | dead/legacy |
| BYBIT_REST_BASE | str | `https://api.bybit.com` | Settings | Config field `bybit_rest_base` | app/config.py:223 (Settings.bybit_rest_base) | active |
| BYBIT_TESTNET | bool | `False` | Settings | Config field `bybit_testnet` | app/config.py:220 (Settings.bybit_testnet) | active |
| BYBIT_WS_PUBLIC_LINEAR | str | `wss://stream.bybit.com/v5/public/linear` | Settings | Config field `bybit_ws_public_linear` | app/config.py:224 (Settings.bybit_ws_public_linear) | dead/legacy |
| CANDLE_HISTORY_LIMIT | int | `120` | Settings | Config field `candle_history_limit` | app/config.py:262 (Settings.candle_history_limit) | dead/legacy |
| CANDLE_INTERVAL | str | `None` | Settings | Config field `candle_interval` | app/config.py:261 (Settings.candle_interval) | active |
| COOLDOWN_MINUTES_AFTER_LOSS | int | `None` | Settings | Config field `cooldown_minutes_after_loss` | app/config.py:245 (Settings.cooldown_minutes_after_loss) | active |
| CURRENT_MODE | str | `SWING` | Settings | Config field `current_mode` | app/config.py:50 (Settings.current_mode) | active |
| DAILY_MAX_DD_PCT | float | `3.0` | Settings | Config field `daily_max_dd_pct` | app/config.py:69 (Settings.daily_max_dd_pct) | dead/legacy |
| DAILY_PROFIT_TARGET_PCT | float | `None` | Settings | Config field `daily_profit_target_pct` | app/config.py:243 (Settings.daily_profit_target_pct) | active |
| DATABASE_URL | str | `None` | Settings | Config field `database_url` | app/config.py:519 (Settings.database_url) | active |
| DATA_DIR | str | `data` | Settings | Config field `data_dir` | app/config.py:511 (Settings.data_dir) | active |
| DEBUG_DISABLE_HARD_RISK_GATES | bool | `False` | Settings | Config field `debug_disable_hard_risk_gates` | app/config.py:177 (Settings.debug_disable_hard_risk_gates) | active |
| DEBUG_LOOSEN | bool | `False` | Settings | Config field `debug_loosen` | app/config.py:176 (Settings.debug_loosen) | active |
| DEBUG_REPLAY_TRACE | str | `None` | Runtime | Direct environment lookup | app/providers/replay.py:157 (os.getenv)<br>app/services/scheduler.py:124 (os.getenv) | active |
| DEBUG_REPLAY_TRACE_EVERY | str | `None` | Runtime | Direct environment lookup | app/providers/replay.py:160 (os.getenv)<br>app/services/scheduler.py:127 (os.getenv) | active |
| DEBUG_RUNTIME_DIAG | str | `None` | Runtime | Direct environment lookup | app/main.py:413 (os.getenv) | active |
| DEV_ATR_MULT | float | `1.0` | Settings | Config field `dev_atr_mult` | app/config.py:172 (Settings.dev_atr_mult) | active |
| DISABLE_BREAKOUT_CHASE | bool | `False` | Settings | Config field `disable_breakout_chase` | app/config.py:380 (Settings.disable_breakout_chase) | active |
| EMA_LENGTH | int | `50` | Settings | Config field `ema_length` | app/config.py:452 (Settings.ema_length) | active |
| EMA_PULLBACK_PCT | float | `0.0015` | Settings | Config field `ema_pullback_pct` | app/config.py:458 (Settings.ema_pullback_pct) | active |
| ENGINE_MODE | str | `signal_only` | Settings | Config field `engine_mode` | app/config.py:40 (Settings.engine_mode) | active |
| ENGULFING_WICK_RATIO | float | `0.5` | Settings | Config field `engulfing_wick_ratio` | app/config.py:459 (Settings.engulfing_wick_ratio) | dead/legacy |
| EXIT_SCORE_MIN | int | `55` | Settings | Config field `exit_score_min` | app/config.py:168 (Settings.exit_score_min) | dead/legacy |
| FEE_BPS | float | `5.5` | Settings | Config field `fee_rate_bps` | app/config.py:465 (Settings.fee_rate_bps) | active |
| FORCE_TRADE_AUTO_CLOSE_SECONDS | int | `0` | Settings | Config field `force_trade_auto_close_seconds` | app/config.py:368 (Settings.force_trade_auto_close_seconds) | active |
| FORCE_TRADE_COOLDOWN_SECONDS | int | `0` | Settings | Config field `force_trade_cooldown_seconds` | app/config.py:364 (Settings.force_trade_cooldown_seconds) | active |
| FORCE_TRADE_EVERY_SECONDS | int | `5` | Settings | Config field `force_trade_every_seconds` | app/config.py:360 (Settings.force_trade_every_seconds) | active |
| FORCE_TRADE_MODE | bool | `False` | Settings | Config field `force_trade_mode` | app/config.py:359 (Settings.force_trade_mode) | active |
| FORCE_TRADE_RANDOM_DIRECTION | bool | `True` | Settings | Config field `force_trade_random_direction` | app/config.py:372 (Settings.force_trade_random_direction) | active |
| FUNDING_BLACKOUT_FORCE_CLOSE | bool | `False` | Settings | Config field `funding_blackout_force_close` | app/config.py:495 (Settings.funding_blackout_force_close) | dead/legacy |
| FUNDING_BLACKOUT_MAX_LOSS_USD | float | `100` | Settings | Config field `funding_blackout_max_loss_usd` | app/config.py:503 (Settings.funding_blackout_max_loss_usd) | dead/legacy |
| FUNDING_BLACKOUT_MAX_UTIL_PCT | float | `70` | Settings | Config field `funding_blackout_max_util_pct` | app/config.py:499 (Settings.funding_blackout_max_util_pct) | dead/legacy |
| FUNDING_BLOCK_BEFORE_MINUTES | int | `10` | Settings | Config field `funding_block_before_minutes` | app/config.py:483 (Settings.funding_block_before_minutes) | active |
| FUNDING_CLOSE_BEFORE_MINUTES | int | `2` | Settings | Config field `funding_close_before_minutes` | app/config.py:487 (Settings.funding_close_before_minutes) | active |
| FUNDING_ELEVATED_ABS | float | `0.02` | Settings | Config field `funding_elevated_abs` | app/config.py:256 (Settings.funding_elevated_abs) | active |
| FUNDING_EXTREME_ABS | float | `0.03` | Settings | Config field `funding_extreme_abs` | app/config.py:255 (Settings.funding_extreme_abs) | active |
| FUNDING_GUARD_TAIL_MINUTE | int | `0` | Settings | Config field `funding_guard_tail_minute` | app/config.py:286 (Settings.funding_guard_tail_minute) | active |
| FUNDING_INTERVAL_MINUTES | int | `480` | Settings | Config field `funding_interval_minutes` | app/config.py:491 (Settings.funding_interval_minutes) | active |
| GLOBAL_DD_LIMIT_PCT | float | `0.08` | Settings | Config field `global_drawdown_limit_pct` | app/config.py:246 (Settings.global_drawdown_limit_pct) | active |
| HEARTBEAT_MINUTES | int | `30` | Settings | Config field `heartbeat_minutes` | app/config.py:45 (Settings.heartbeat_minutes) | dead/legacy |
| HTF_BIAS_ENABLED | bool | `False` | Settings | Config field `htf_bias_enabled` | app/config.py:115 (Settings.htf_bias_enabled) | active |
| HTF_BIAS_REQUIRE_SLOPE | bool | `False` | Settings | Config field `htf_bias_require_slope` | app/config.py:119 (Settings.htf_bias_require_slope) | active |
| HTF_EMA_FAST | int | `50` | Settings | Config field `htf_ema_fast` | app/config.py:117 (Settings.htf_ema_fast) | active |
| HTF_EMA_SLOW | int | `200` | Settings | Config field `htf_ema_slow` | app/config.py:118 (Settings.htf_ema_slow) | active |
| HTF_INTERVAL | str | `1h` | Settings | Config field `htf_interval` | app/config.py:116 (Settings.htf_interval) | active |
| INTERNAL_API_BASE_URL | str | `None` | Settings | Config field `internal_api_base_url` | app/config.py:518 (Settings.internal_api_base_url) | dead/legacy |
| LEGACY_ENV_KEYS | list | `["BE_TRIGGER_R", "MAX_TRADES_PER_DAY", "BASE_RISK_PCT", "MAX_RISK_PCT", "MAX_DAILY_LOSS_PCT"]` | Settings | Config field `LEGACY_ENV_KEYS` | app/config.py:522 (Settings.LEGACY_ENV_KEYS) | dead/legacy |
| LEVERAGE_ELEVATED | float | `2.5` | Settings | Config field `leverage_elevated` | app/config.py:258 (Settings.leverage_elevated) | active |
| LEVERAGE_EXTREME | float | `3.0` | Settings | Config field `leverage_extreme` | app/config.py:257 (Settings.leverage_extreme) | active |
| MANUAL_KILL_SWITCH | bool | `False` | Settings | Config field `manual_kill_switch` | app/config.py:250 (Settings.manual_kill_switch) | active |
| MARKET_DATA_ALLOW_STALE | int | `60` | Settings | Config field `market_data_allow_stale` | app/config.py:214 (Settings.market_data_allow_stale) | dead/legacy |
| MARKET_DATA_BACKOFF_BASE_MS | int | `500` | Settings | Config field `market_data_backoff_base_ms` | app/config.py:188 (Settings.market_data_backoff_base_ms) | active |
| MARKET_DATA_BACKOFF_MAX_MS | int | `15000` | Settings | Config field `market_data_backoff_max_ms` | app/config.py:192 (Settings.market_data_backoff_max_ms) | active |
| MARKET_DATA_ENABLED | bool | `True` | Settings | Config field `market_data_enabled` | app/config.py:218 (Settings.market_data_enabled) | dead/legacy |
| MARKET_DATA_FAILOVER_THRESHOLD | int | `3` | Settings | Config field `market_data_failover_threshold` | app/config.py:184 (Settings.market_data_failover_threshold) | active |
| MARKET_DATA_FALLBACKS | str | `binance,okx,replay` | Settings | Config field `market_data_fallbacks` | app/config.py:183 (Settings.market_data_fallbacks) | active |
| MARKET_DATA_PROVIDER | str | `bybit` | Settings | Config field `market_data_provider` | app/config.py:182 (Settings.market_data_provider) | active |
| MARKET_DATA_REPLAY_PATH | str | `data/replay` | Settings | Config field `market_data_replay_path` | app/config.py:196 (Settings.market_data_replay_path) | active |
| MARKET_DATA_REPLAY_SPEED | float | `1.0` | Settings | Config field `market_data_replay_speed` | app/config.py:200 (Settings.market_data_replay_speed) | active |
| MARKET_PROVIDER | str | `bybit` | Settings | Config field `market_provider` | app/config.py:181 (Settings.market_provider) | dead/legacy |
| MAX_CONSECUTIVE_LOSSES | int | `None` | Settings | Config field `max_consecutive_losses` | app/config.py:244 (Settings.max_consecutive_losses) | active |
| MAX_DAILY_LOSS_PCT | float | `None` | Settings | Config field `max_daily_loss_pct` | app/config.py:241 (Settings.max_daily_loss_pct) | active |
| MAX_HOLD_MINUTES | int | `720` | Settings | Config field `max_hold_minutes` | app/config.py:470 (Settings.max_hold_minutes) | dead/legacy |
| MAX_LOSSES_PER_DAY | int | `3` | Settings | Config field `max_losses_per_day` | app/config.py:508 (Settings.max_losses_per_day) | active |
| MAX_NOTIONAL_ACCOUNT_MULTIPLIER | float | `3.0` | Settings | Config field `max_notional_account_multiplier` | app/config.py:289 (Settings.max_notional_account_multiplier) | active |
| MAX_OPEN_POSITIONS_PER_DIRECTION | int | `0` | Settings | Config field `max_open_positions_per_direction` | app/config.py:384 (Settings.max_open_positions_per_direction) | active |
| MAX_RISK_PCT | float | `None` | Settings | Config field `max_risk_pct` | app/config.py:239 (Settings.max_risk_pct) | active |
| MAX_STOP_PCT | float | `0.0025` | Settings | Config field `max_stop_pct` | app/config.py:463 (Settings.max_stop_pct) | active |
| MAX_TRADES_PER_DAY | int | `None` | Settings | Config field `max_trades_per_day` | app/config.py:240 (Settings.max_trades_per_day)<br>app/config.py:587 (environ.get)<br>app/config.py:587 (os.environ.get) | active |
| MIN_BREAKOUT_WINDOW | int | `25` | Settings | Config field `min_breakout_window` | app/config.py:282 (Settings.min_breakout_window) | active |
| MIN_SIGNAL_SCORE | int | `None` | Settings | Config field `min_signal_score` | app/config.py:242 (Settings.min_signal_score) | active |
| MIN_SIGNAL_SCORE_RANGE | int | `70` | Settings | Config field `min_signal_score_range` | app/config.py:145 (Settings.min_signal_score_range) | active |
| MIN_SIGNAL_SCORE_TREND | int | `78` | Settings | Config field `min_signal_score_trend` | app/config.py:144 (Settings.min_signal_score_trend) | active |
| MODE | str | `prop_cfd` | Settings | Config field `MODE` | app/config.py:23 (Settings.MODE) | active |
| MOMENTUM_MODE | str | `adx` | Settings | Config field `momentum_mode` | app/config.py:453 (Settings.momentum_mode) | active |
| MOVE_TO_BREAKEVEN_TRIGGER_R | float | `0.6` | Settings | Config field `move_to_breakeven_trigger_r` | app/config.py:290 (Settings.move_to_breakeven_trigger_r) | active |
| NEWS_BLACKOUTS | str | `""` | Settings | Config field `news_blackouts` | app/config.py:510 (Settings.news_blackouts) | dead/legacy |
| NEXT_PUBLIC_API_BASE | str | `None` | Settings | Config field `next_public_api_base` | app/config.py:517 (Settings.next_public_api_base) | dead/legacy |
| OI_SPIKE_PCT | float | `0.18` | Settings | Config field `oi_spike_pct` | app/config.py:259 (Settings.oi_spike_pct) | active |
| PARTIAL_TP_CLOSE_PCT | float | `0.35` | Settings | Config field `partial_tp_close_pct` | app/config.py:354 (Settings.partial_tp_close_pct) | dead/legacy |
| PARTIAL_TP_ENABLED | bool | `True` | Settings | Config field `partial_tp_enabled` | app/config.py:352 (Settings.partial_tp_enabled) | dead/legacy |
| PARTIAL_TP_R | float | `1.0` | Settings | Config field `partial_tp_r` | app/config.py:353 (Settings.partial_tp_r) | dead/legacy |
| PERSONAL_SCORE_THRESHOLD | int | `75` | Settings | Config field `personal_score_threshold` | app/config.py:318 (Settings.personal_score_threshold) | dead/legacy |
| POSITION_SIZE_USD_CAP | float | `None` | Settings | Config field `position_size_usd_cap` | app/config.py:229 (Settings.position_size_usd_cap) | active |
| PROFILE | str | `profit` | Settings | Config field `PROFILE` | app/config.py:25 (Settings.PROFILE) | active |
| PROP_DAILY_STOP_AFTER_LOSSES | int | `2` | Settings | Config field `prop_daily_stop_after_losses` | app/config.py:339 (Settings.prop_daily_stop_after_losses) | dead/legacy |
| PROP_DAILY_STOP_AFTER_NET_R | float | `2.0` | Settings | Config field `prop_daily_stop_after_net_r` | app/config.py:338 (Settings.prop_daily_stop_after_net_r) | dead/legacy |
| PROP_DD_INCLUDES_UNREALIZED | bool | `True` | Settings | Config field `prop_dd_includes_unrealized` | app/config.py:327 (Settings.prop_dd_includes_unrealized) | dead/legacy |
| PROP_ENABLED | bool | `True` | Settings | Config field `prop_enabled` | app/config.py:321 (Settings.prop_enabled) | dead/legacy |
| PROP_GOVERNOR_ENABLED | bool | `True` | Settings | Config field `prop_governor_enabled` | app/config.py:329 (Settings.prop_governor_enabled) | dead/legacy |
| PROP_MAX_CONSEC_LOSSES | int | `2` | Settings | Config field `prop_max_consec_losses` | app/config.py:341 (Settings.prop_max_consec_losses) | dead/legacy |
| PROP_MAX_DAILY_LOSS_PCT | float | `0.05` | Settings | Config field `prop_max_daily_loss_pct` | app/config.py:325 (Settings.prop_max_daily_loss_pct) | dead/legacy |
| PROP_MAX_DAYS | int | `60` | Settings | Config field `prop_max_days` | app/config.py:324 (Settings.prop_max_days) | dead/legacy |
| PROP_MAX_GLOBAL_DD_PCT | float | `0.1` | Settings | Config field `prop_max_global_dd_pct` | app/config.py:326 (Settings.prop_max_global_dd_pct) | dead/legacy |
| PROP_MAX_TRADES_PER_DAY | int | `4` | Settings | Config field `prop_max_trades_per_day` | app/config.py:340 (Settings.prop_max_trades_per_day)<br>app/config.py:586 (environ.get)<br>app/config.py:586 (os.environ.get) | active |
| PROP_MIN_TRADING_DAYS | int | `5` | Settings | Config field `prop_min_trading_days` | app/config.py:323 (Settings.prop_min_trading_days) | dead/legacy |
| PROP_PROFIT_TARGET_PCT | float | `0.08` | Settings | Config field `prop_profit_target_pct` | app/config.py:322 (Settings.prop_profit_target_pct) | active |
| PROP_RESET_CONSEC_LOSSES_ON_DAY_ROLLOVER | bool | `True` | Settings | Config field `prop_reset_consec_losses_on_day_rollover` | app/config.py:342 (Settings.prop_reset_consec_losses_on_day_rollover) | dead/legacy |
| PROP_RISK_BASE_PCT | float | `0.0015` | Settings | Config field `prop_risk_base_pct` | app/config.py:330 (Settings.prop_risk_base_pct) | active |
| PROP_RISK_MAX_PCT | float | `0.002` | Settings | Config field `prop_risk_max_pct` | app/config.py:331 (Settings.prop_risk_max_pct) | active |
| PROP_RISK_MIN_PCT | float | `0.0007` | Settings | Config field `prop_risk_min_pct` | app/config.py:332 (Settings.prop_risk_min_pct) | active |
| PROP_SCORE_THRESHOLD | int | `80` | Settings | Config field `prop_score_threshold` | app/config.py:317 (Settings.prop_score_threshold) | dead/legacy |
| PROP_STEPDOWN_AFTER_LOSS | bool | `True` | Settings | Config field `prop_stepdown_after_loss` | app/config.py:333 (Settings.prop_stepdown_after_loss) | dead/legacy |
| PROP_STEPDOWN_FACTOR | float | `0.5` | Settings | Config field `prop_stepdown_factor` | app/config.py:334 (Settings.prop_stepdown_factor) | dead/legacy |
| PROP_STEPUP_AFTER_WIN | bool | `True` | Settings | Config field `prop_stepup_after_win` | app/config.py:335 (Settings.prop_stepup_after_win) | dead/legacy |
| PROP_STEPUP_COOLDOWN_TRADES | int | `2` | Settings | Config field `prop_stepup_cooldown_trades` | app/config.py:337 (Settings.prop_stepup_cooldown_trades) | dead/legacy |
| PROP_STEPUP_FACTOR | float | `1.15` | Settings | Config field `prop_stepup_factor` | app/config.py:336 (Settings.prop_stepup_factor) | dead/legacy |
| PROP_TIME_COOLDOWN_MINUTES | int | `60` | Settings | Config field `prop_time_cooldown_minutes` | app/config.py:349 (Settings.prop_time_cooldown_minutes) | dead/legacy |
| PULLBACK_ATR_MULT | float | `0.5` | Settings | Config field `pullback_atr_mult` | app/config.py:169 (Settings.pullback_atr_mult) | active |
| RANGE_BASE_SCORE_BOOST | int | `8` | Settings | Config field `range_base_score_boost` | app/config.py:281 (Settings.range_base_score_boost) | active |
| RANGE_RSI_LONG_MAX | float | `35.0` | Settings | Config field `range_rsi_long_max` | app/config.py:273 (Settings.range_rsi_long_max) | active |
| RANGE_RSI_SHORT_MIN | float | `65.0` | Settings | Config field `range_rsi_short_min` | app/config.py:274 (Settings.range_rsi_short_min) | active |
| RANGE_SL_ATR_MULT | float | `0.8` | Settings | Config field `range_sl_atr_mult` | app/config.py:173 (Settings.range_sl_atr_mult) | active |
| RANGE_TP_ATR_MULT | float | `1.0` | Settings | Config field `range_tp_atr_mult` | app/config.py:174 (Settings.range_tp_atr_mult) | active |
| REENTRY_COOLDOWN_MINUTES | int | `30` | Settings | Config field `reentry_cooldown_minutes` | app/config.py:479 (Settings.reentry_cooldown_minutes) | active |
| REGIME_ENTRY_BUFFER_PCT | float | `0.0002` | Settings | Config field `regime_entry_buffer_pct` | app/config.py:316 (Settings.regime_entry_buffer_pct) | dead/legacy |
| REPLAY_END_TS | str | `None` | Settings | Config field `replay_end_ts` | app/config.py:211 (Settings.replay_end_ts) | active |
| REPLAY_MAX_BARS | int | `5000` | Settings | Config field `replay_max_bars` | app/config.py:209 (Settings.replay_max_bars) | dead/legacy |
| REPLAY_MAX_TRADES | int | `120` | Settings | Config field `replay_max_trades` | app/config.py:208 (Settings.replay_max_trades) | dead/legacy |
| REPLAY_PAUSE_SECONDS | float | `None` | Settings | Config field `replay_pause_seconds` | app/config.py:204 (Settings.replay_pause_seconds) | dead/legacy |
| REPLAY_RESUME | bool | `False` | Settings | Config field `replay_resume` | app/config.py:212 (Settings.replay_resume) | active |
| REPLAY_SEED | int | `None` | Settings | Config field `replay_seed` | app/config.py:213 (Settings.replay_seed) | active |
| REPLAY_START_TS | str | `None` | Settings | Config field `replay_start_ts` | app/config.py:210 (Settings.replay_start_ts) | active |
| REQUIRE_CANDLE_CLOSE_CONFIRM | bool | `True` | Settings | Config field `require_candle_close_confirm` | app/config.py:376 (Settings.require_candle_close_confirm) | dead/legacy |
| RISK_PER_TRADE_USD | float | `None` | Settings | Config field `risk_per_trade_usd` | app/config.py:228 (Settings.risk_per_trade_usd) | active |
| RISK_REDUCTION_ENABLED | bool | `True` | Settings | Config field `risk_reduction_enabled` | app/config.py:161 (Settings.risk_reduction_enabled) | active |
| RISK_REDUCTION_TARGET_R | float | `0.7` | Settings | Config field `risk_reduction_target_r` | app/config.py:154 (Settings.risk_reduction_target_r) | active |
| RISK_REDUCTION_TRIGGER_R | float | `0.6` | Settings | Config field `risk_reduction_trigger_r` | app/config.py:147 (Settings.risk_reduction_trigger_r) | active |
| RUN_MODE | str | `live` | Settings | Config field `run_mode` | app/config.py:24 (Settings.run_mode) | active |
| SCALP_ATR_PCT_MAX | float | `0.012` | Settings | Config field `scalp_atr_pct_max` | app/config.py:79 (Settings.scalp_atr_pct_max) | active |
| SCALP_ATR_PCT_MIN | float | `0.0015` | Settings | Config field `scalp_atr_pct_min` | app/config.py:78 (Settings.scalp_atr_pct_min) | active |
| SCALP_ATR_PERIOD | int | `14` | Settings | Config field `scalp_atr_period` | app/config.py:77 (Settings.scalp_atr_period) | active |
| SCALP_BREAKOUT_LOOKBACK | int | `20` | Settings | Config field `scalp_breakout_lookback` | app/config.py:107 (Settings.scalp_breakout_lookback) | active |
| SCALP_EMA_FAST | int | `20` | Settings | Config field `scalp_ema_fast` | app/config.py:74 (Settings.scalp_ema_fast) | active |
| SCALP_EMA_SLOW | int | `50` | Settings | Config field `scalp_ema_slow` | app/config.py:75 (Settings.scalp_ema_slow) | active |
| SCALP_EMA_TREND | int | `200` | Settings | Config field `scalp_ema_trend` | app/config.py:76 (Settings.scalp_ema_trend) | active |
| SCALP_ENGULFING_MIN_BODY_PCT | float | `0.0006` | Settings | Config field `scalp_engulfing_min_body_pct` | app/config.py:97 (Settings.scalp_engulfing_min_body_pct) | active |
| SCALP_MAX_HOLD_MINUTES | int | `60` | Settings | Config field `scalp_max_hold_minutes` | app/config.py:56 (Settings.scalp_max_hold_minutes) | dead/legacy |
| SCALP_MIN_SCORE | int | `78` | Settings | Config field `scalp_min_score` | app/config.py:64 (Settings.scalp_min_score) | active |
| SCALP_PULLBACK_EMA | int | `20` | Settings | Config field `scalp_pullback_ema` | app/config.py:88 (Settings.scalp_pullback_ema) | active |
| SCALP_PULLBACK_MAX_DIST_PCT | float | `0.002` | Settings | Config field `scalp_pullback_max_dist_pct` | app/config.py:89 (Settings.scalp_pullback_max_dist_pct) | active |
| SCALP_PULLBACK_MIN_DIST_PCT | float | `0.0` | Settings | Config field `scalp_pullback_min_dist_pct` | app/config.py:93 (Settings.scalp_pullback_min_dist_pct) | active |
| SCALP_REENTRY_COOLDOWN_MINUTES | int | `20` | Settings | Config field `scalp_reentry_cooldown_minutes` | app/config.py:60 (Settings.scalp_reentry_cooldown_minutes) | active |
| SCALP_REGIME_ENABLED | bool | `True` | Settings | Config field `scalp_regime_enabled` | app/config.py:70 (Settings.scalp_regime_enabled) | active |
| SCALP_RETEST_MAX_BARS | int | `6` | Settings | Config field `scalp_retest_max_bars` | app/config.py:111 (Settings.scalp_retest_max_bars) | active |
| SCALP_RSI_CONFIRM | bool | `True` | Settings | Config field `scalp_rsi_confirm` | app/config.py:102 (Settings.scalp_rsi_confirm) | active |
| SCALP_RSI_LONG_MAX | int | `100` | Settings | Config field `scalp_rsi_long_max` | app/config.py:105 (Settings.scalp_rsi_long_max) | active |
| SCALP_RSI_LONG_MIN | float | `45` | Settings | Config field `scalp_rsi_long_min` | app/config.py:103 (Settings.scalp_rsi_long_min) | active |
| SCALP_RSI_PERIOD | int | `14` | Settings | Config field `scalp_rsi_period` | app/config.py:101 (Settings.scalp_rsi_period) | active |
| SCALP_RSI_SHORT_MAX | float | `55` | Settings | Config field `scalp_rsi_short_max` | app/config.py:104 (Settings.scalp_rsi_short_max) | active |
| SCALP_RSI_SHORT_MIN | int | `0` | Settings | Config field `scalp_rsi_short_min` | app/config.py:106 (Settings.scalp_rsi_short_min) | active |
| SCALP_SETUP_MODE | str | `pullback_engulfing` | Settings | Config field `scalp_setup_mode` | app/config.py:84 (Settings.scalp_setup_mode) | active |
| SCALP_SL_PCT | float | `0.005` | Settings | Config field `scalp_sl_pct` | app/config.py:55 (Settings.scalp_sl_pct) | active |
| SCALP_TP_PCT | float | `0.01` | Settings | Config field `scalp_tp_pct` | app/config.py:54 (Settings.scalp_tp_pct) | active |
| SCALP_TREND_FILTER_ENABLED | bool | `True` | Settings | Config field `scalp_trend_filter_enabled` | app/config.py:65 (Settings.scalp_trend_filter_enabled) | active |
| SCALP_TREND_SLOPE_MIN | float | `0.0002` | Settings | Config field `scalp_trend_slope_min` | app/config.py:80 (Settings.scalp_trend_slope_min) | active |
| SETTINGS_ENABLE_LEGACY | bool | `False` | Settings | Config field `settings_enable_legacy` | app/config.py:520 (Settings.settings_enable_legacy) | dead/legacy |
| SETUP_MIN_CANDLES | int | `3` | Settings | Config field `setup_min_candles` | app/config.py:284 (Settings.setup_min_candles) | active |
| SLIPPAGE_BPS | float | `1.5` | Settings | Config field `slippage_bps` | app/config.py:467 (Settings.slippage_bps) | active |
| SL_ATR_MULT | float | `1.4` | Settings | Config field `sl_atr_mult` | app/config.py:170 (Settings.sl_atr_mult) | active |
| SMOKE_TEST_FORCE_TRADE | bool | `False` | Settings | Config field `smoke_test_force_trade` | app/config.py:516 (Settings.smoke_test_force_trade) | active |
| SPREAD_BPS | float | `1.5` | Settings | Config field `spread_bps` | app/config.py:466 (Settings.spread_bps) | active |
| STRATEGY | str | `scalper` | Settings | Config field `strategy` | app/config.py:46 (Settings.strategy) | active |
| STRATEGY_ADX_THRESHOLD | float | `22.0` | Settings | Config field `strategy_adx_threshold` | app/config.py:30 (Settings.strategy_adx_threshold) | active |
| STRATEGY_ATR_STOP_MULT | float | `1.6` | Settings | Config field `strategy_atr_stop_mult` | app/config.py:34 (Settings.strategy_atr_stop_mult) | active |
| STRATEGY_BIAS_EMA_FAST | int | `50` | Settings | Config field `strategy_bias_ema_fast` | app/config.py:27 (Settings.strategy_bias_ema_fast) | active |
| STRATEGY_BIAS_EMA_SLOW | int | `200` | Settings | Config field `strategy_bias_ema_slow` | app/config.py:28 (Settings.strategy_bias_ema_slow) | active |
| STRATEGY_MAX_ATR_PCT | float | `0.012` | Settings | Config field `strategy_max_atr_pct` | app/config.py:32 (Settings.strategy_max_atr_pct) | active |
| STRATEGY_MAX_STOP_PCT | float | `0.015` | Settings | Config field `strategy_max_stop_pct` | app/config.py:37 (Settings.strategy_max_stop_pct) | active |
| STRATEGY_MIN_ATR_PCT | float | `0.0015` | Settings | Config field `strategy_min_atr_pct` | app/config.py:31 (Settings.strategy_min_atr_pct) | active |
| STRATEGY_MIN_STOP_PCT | float | `0.002` | Settings | Config field `strategy_min_stop_pct` | app/config.py:36 (Settings.strategy_min_stop_pct) | active |
| STRATEGY_PARTIAL_R | float | `1.8` | Settings | Config field `strategy_partial_r` | app/config.py:39 (Settings.strategy_partial_r) | active |
| STRATEGY_PROFILE | str | `SCALPER_STABLE` | Settings | Config field `strategy_profile` | app/config.py:26 (Settings.strategy_profile) | active |
| STRATEGY_SWING_LOOKBACK | int | `30` | Settings | Config field `strategy_swing_lookback` | app/config.py:29 (Settings.strategy_swing_lookback) | active |
| STRATEGY_SWING_STOP_BUFFER_BPS | float | `8.0` | Settings | Config field `strategy_swing_stop_buffer_bps` | app/config.py:35 (Settings.strategy_swing_stop_buffer_bps) | active |
| STRATEGY_TARGET_R | float | `2.2` | Settings | Config field `strategy_target_r` | app/config.py:38 (Settings.strategy_target_r) | active |
| STRATEGY_TREND_SLOPE_THRESHOLD | float | `0.0002` | Settings | Config field `strategy_trend_slope_threshold` | app/config.py:33 (Settings.strategy_trend_slope_threshold) | active |
| SWEET8_BASE_RISK_PCT | float | `0.0025` | Settings | Config field `sweet8_base_risk_pct` | app/config.py:393 (Settings.sweet8_base_risk_pct) | dead/legacy |
| SWEET8_BLOCKED_CLOSE_FORCE | int | `0` | Settings | Config field `sweet8_blocked_close_force` | app/config.py:450 (Settings.sweet8_blocked_close_force) | dead/legacy |
| SWEET8_BLOCKED_CLOSE_TIME_STOP | int | `0` | Settings | Config field `sweet8_blocked_close_time_stop` | app/config.py:449 (Settings.sweet8_blocked_close_time_stop) | dead/legacy |
| SWEET8_BLOCKED_CLOSE_TOTAL | int | `0` | Settings | Config field `sweet8_blocked_close_total` | app/config.py:448 (Settings.sweet8_blocked_close_total) | active |
| SWEET8_CURRENT_MODE | str | `auto` | Settings | Config field `sweet8_current_mode` | app/config.py:451 (Settings.sweet8_current_mode) | active |
| SWEET8_DISABLE_FORCE_AUTO_CLOSE | bool | `True` | Settings | Config field `sweet8_disable_force_auto_close` | app/config.py:404 (Settings.sweet8_disable_force_auto_close) | dead/legacy |
| SWEET8_DISABLE_PREMATURE_EXITS | bool | `True` | Settings | Config field `sweet8_disable_premature_exits` | app/config.py:399 (Settings.sweet8_disable_premature_exits) | dead/legacy |
| SWEET8_DISABLE_TIME_STOP | bool | `True` | Settings | Config field `sweet8_disable_time_stop` | app/config.py:403 (Settings.sweet8_disable_time_stop) | dead/legacy |
| SWEET8_ENABLED | bool | `False` | Settings | Config field `sweet8_enabled` | app/config.py:388 (Settings.sweet8_enabled) | active |
| SWEET8_MAX_DAILY_LOSS_PCT | float | `0.015` | Settings | Config field `sweet8_max_daily_loss_pct` | app/config.py:395 (Settings.sweet8_max_daily_loss_pct) | dead/legacy |
| SWEET8_MAX_OPEN_POSITIONS_PER_SYMBOL | int | `1` | Settings | Config field `sweet8_max_open_positions_per_symbol` | app/config.py:412 (Settings.sweet8_max_open_positions_per_symbol) | dead/legacy |
| SWEET8_MAX_OPEN_POSITIONS_TOTAL | int | `2` | Settings | Config field `sweet8_max_open_positions_total` | app/config.py:408 (Settings.sweet8_max_open_positions_total) | dead/legacy |
| SWEET8_MAX_RISK_PCT | float | `0.005` | Settings | Config field `sweet8_max_risk_pct` | app/config.py:394 (Settings.sweet8_max_risk_pct) | dead/legacy |
| SWEET8_MODE | str | `auto` | Settings | Config field `sweet8_mode` | app/config.py:389 (Settings.sweet8_mode) | dead/legacy |
| SWEET8_REGIME_ADX_THRESHOLD | int | `25` | Settings | Config field `sweet8_regime_adx_threshold` | app/config.py:440 (Settings.sweet8_regime_adx_threshold) | dead/legacy |
| SWEET8_REGIME_VOL_THRESHOLD | float | `1.2` | Settings | Config field `sweet8_regime_vol_threshold` | app/config.py:444 (Settings.sweet8_regime_vol_threshold) | dead/legacy |
| SWEET8_SCALP_ATR_SL_MULT | float | `1.0` | Settings | Config field `sweet8_scalp_atr_sl_mult` | app/config.py:416 (Settings.sweet8_scalp_atr_sl_mult) | dead/legacy |
| SWEET8_SCALP_ATR_TP_MULT | float | `1.4` | Settings | Config field `sweet8_scalp_atr_tp_mult` | app/config.py:420 (Settings.sweet8_scalp_atr_tp_mult) | dead/legacy |
| SWEET8_SCALP_MIN_SCORE | int | `75` | Settings | Config field `sweet8_scalp_min_score` | app/config.py:424 (Settings.sweet8_scalp_min_score) | dead/legacy |
| SWEET8_SWING_ATR_SL_MULT | float | `1.5` | Settings | Config field `sweet8_swing_atr_sl_mult` | app/config.py:428 (Settings.sweet8_swing_atr_sl_mult) | dead/legacy |
| SWEET8_SWING_ATR_TP_MULT | float | `2.5` | Settings | Config field `sweet8_swing_atr_tp_mult` | app/config.py:432 (Settings.sweet8_swing_atr_tp_mult) | dead/legacy |
| SWEET8_SWING_MIN_SCORE | int | `85` | Settings | Config field `sweet8_swing_min_score` | app/config.py:436 (Settings.sweet8_swing_min_score) | dead/legacy |
| SYMBOLS | list | `["BTCUSDT"]` | Settings | Config field `symbols` | app/config.py:41 (Settings.symbols) | active |
| SYMBOL_COOLDOWN_MIN | int | `5` | Settings | Config field `symbol_cooldown_min` | app/config.py:175 (Settings.symbol_cooldown_min) | dead/legacy |
| TAKE_PROFIT_PCT | float | `0.003` | Settings | Config field `take_profit_pct` | app/config.py:464 (Settings.take_profit_pct) | active |
| TELEGRAM_BOT_TOKEN | str | `None` | Settings | Config field `telegram_bot_token` | app/config.py:514 (Settings.telegram_bot_token) | active |
| TELEGRAM_CHAT_ID | str | `None` | Settings | Config field `telegram_chat_id` | app/config.py:515 (Settings.telegram_chat_id) | active |
| TELEGRAM_DEBUG_SKIPS | bool | `False` | Settings | Config field `telegram_debug_skips` | app/config.py:512 (Settings.telegram_debug_skips) | dead/legacy |
| TELEGRAM_ENABLED | bool | `False` | Settings | Config field `telegram_enabled` | app/config.py:513 (Settings.telegram_enabled) | active |
| TICK_INTERVAL_SECONDS | int | `60` | Settings | Config field `tick_interval_seconds` | app/config.py:263 (Settings.tick_interval_seconds) | active |
| TP_ATR_MULT | float | `2.2` | Settings | Config field `tp_atr_mult` | app/config.py:171 (Settings.tp_atr_mult) | active |
| TP_CAP_LONG_PCT | float | `0.02` | Settings | Config field `tp_cap_long_pct` | app/config.py:287 (Settings.tp_cap_long_pct) | dead/legacy |
| TP_CAP_SHORT_PCT | float | `0.02` | Settings | Config field `tp_cap_short_pct` | app/config.py:288 (Settings.tp_cap_short_pct) | dead/legacy |
| TRAIL_ATR_MULT | float | `1.1` | Settings | Config field `trail_atr_mult` | app/config.py:357 (Settings.trail_atr_mult) | dead/legacy |
| TRAIL_ENABLED | bool | `True` | Settings | Config field `trail_enabled` | app/config.py:355 (Settings.trail_enabled) | dead/legacy |
| TRAIL_START_R | float | `1.5` | Settings | Config field `trail_start_r` | app/config.py:356 (Settings.trail_start_r) | dead/legacy |
| TRAIL_STEP_R | float | `0.5` | Settings | Config field `trail_step_r` | app/config.py:358 (Settings.trail_step_r) | dead/legacy |
| TREND_BIAS_LOOKBACK | int | `20` | Settings | Config field `trend_bias_lookback` | app/config.py:283 (Settings.trend_bias_lookback) | active |
| TREND_CONFIRM_SCORE_BOOST | int | `10` | Settings | Config field `trend_confirm_score_boost` | app/config.py:280 (Settings.trend_confirm_score_boost) | active |
| TREND_MIN_CANDLES | int | `4` | Settings | Config field `trend_min_candles` | app/config.py:285 (Settings.trend_min_candles) | active |
| TREND_RSI_MIDLINE | float | `50.0` | Settings | Config field `trend_rsi_midline` | app/config.py:272 (Settings.trend_rsi_midline) | active |
| TREND_SCORE_MAX_BOOST | int | `25` | Settings | Config field `trend_score_max_boost` | app/config.py:277 (Settings.trend_score_max_boost) | active |
| TREND_SCORE_SLOPE_SCALE | float | `40000.0` | Settings | Config field `trend_score_slope_scale` | app/config.py:276 (Settings.trend_score_slope_scale) | active |
| TREND_STRENGTH_MIN | float | `None` | Settings | Config field `trend_strength_min` | app/config.py:260 (Settings.trend_strength_min) | active |
| TREND_STRENGTH_SCALE | float | `2000.0` | Settings | Config field `trend_strength_scale` | app/config.py:275 (Settings.trend_strength_scale) | active |
| TRIGGER_BODY_RATIO_MIN | float | `0.0` | Settings | Config field `trigger_body_ratio_min` | app/config.py:139 (Settings.trigger_body_ratio_min) | active |
| TRIGGER_CLOSE_LOCATION_MIN | float | `0.0` | Settings | Config field `trigger_close_location_min` | app/config.py:140 (Settings.trigger_close_location_min) | active |
| VOLUME_CONFIRM_ENABLED | bool | `True` | Settings | Config field `volume_confirm_enabled` | app/config.py:460 (Settings.volume_confirm_enabled) | active |
| VOLUME_CONFIRM_MULTIPLIER | float | `1.0` | Settings | Config field `volume_confirm_multiplier` | app/config.py:462 (Settings.volume_confirm_multiplier) | active |
| VOLUME_SMA_PERIOD | int | `20` | Settings | Config field `volume_sma_period` | app/config.py:461 (Settings.volume_sma_period) | active |
| WARMUP_IGNORE_HTF_IF_DISABLED | bool | `True` | Settings | Config field `warmup_ignore_htf_if_disabled` | app/config.py:131 (Settings.warmup_ignore_htf_if_disabled) | active |
| WARMUP_MIN_BARS_1H | int | `220` | Settings | Config field `warmup_min_bars_1h` | app/config.py:127 (Settings.warmup_min_bars_1h) | active |
| WARMUP_MIN_BARS_5M | int | `250` | Settings | Config field `warmup_min_bars_5m` | app/config.py:123 (Settings.warmup_min_bars_5m) | active |
| WARMUP_REQUIRE_REPLAY_READY | bool | `True` | Settings | Config field `warmup_require_replay_ready` | app/config.py:135 (Settings.warmup_require_replay_ready) | active |

## Table 2: Legacy/Alias Keys

| Legacy Key | Maps To | Status (accepted/ignored/forbidden) | Notes | Locations |
| --- | --- | --- | --- | --- |
| BE_TRIGGER_R | MOVE_TO_BREAKEVEN_TRIGGER_R | forbidden | Alias for MOVE_TO_BREAKEVEN_TRIGGER_R; precedence order: MOVE_TO_BREAKEVEN_TRIGGER_R > BE_TRIGGER_R > BE_TRIGGER_R_MULT > move_to_breakeven_trigger_r; rejected when SETTINGS_ENABLE_LEGACY=false | app/config.py:290 (Settings.move_to_breakeven_trigger_r)<br>app/config.py:580 (os.environ[]) |
| BE_TRIGGER_R_MULT | MOVE_TO_BREAKEVEN_TRIGGER_R | accepted | Alias for MOVE_TO_BREAKEVEN_TRIGGER_R; precedence order: MOVE_TO_BREAKEVEN_TRIGGER_R > BE_TRIGGER_R > BE_TRIGGER_R_MULT > move_to_breakeven_trigger_r | app/config.py:290 (Settings.move_to_breakeven_trigger_r) |
| MAX_POSITION_SIZE_USD | POSITION_SIZE_USD_CAP | accepted | Alias for POSITION_SIZE_USD_CAP; precedence order: POSITION_SIZE_USD_CAP > MAX_POSITION_SIZE_USD > POSITION_SIZE_USD_MAX > MAX_POSITION_USD > position_size_usd_cap | app/config.py:229 (Settings.position_size_usd_cap) |
| MAX_POSITION_USD | POSITION_SIZE_USD_CAP | accepted | Alias for POSITION_SIZE_USD_CAP; precedence order: POSITION_SIZE_USD_CAP > MAX_POSITION_SIZE_USD > POSITION_SIZE_USD_MAX > MAX_POSITION_USD > position_size_usd_cap | app/config.py:229 (Settings.position_size_usd_cap) |
| POSITION_SIZE_USD_MAX | POSITION_SIZE_USD_CAP | accepted | Alias for POSITION_SIZE_USD_CAP; precedence order: POSITION_SIZE_USD_CAP > MAX_POSITION_SIZE_USD > POSITION_SIZE_USD_MAX > MAX_POSITION_USD > position_size_usd_cap | app/config.py:229 (Settings.position_size_usd_cap) |
| REPLAY_SPEED | MARKET_DATA_REPLAY_SPEED | accepted | Alias for MARKET_DATA_REPLAY_SPEED; precedence order: MARKET_DATA_REPLAY_SPEED > REPLAY_SPEED > market_data_replay_speed | app/config.py:200 (Settings.market_data_replay_speed) |
| SCHEDULER_TICK_INTERVAL_SECONDS | TICK_INTERVAL_SECONDS | accepted | Alias for TICK_INTERVAL_SECONDS; precedence order: TICK_INTERVAL_SECONDS > SCHEDULER_TICK_INTERVAL_SECONDS > tick_interval_seconds | app/config.py:263 (Settings.tick_interval_seconds) |
| TIME_STOP_MINUTES | MAX_HOLD_MINUTES | accepted | Alias for MAX_HOLD_MINUTES; precedence order: MAX_HOLD_MINUTES > max_hold_minutes > TIME_STOP_MINUTES > time_stop_minutes | app/config.py:470 (Settings.max_hold_minutes) |
| account_size | ACCOUNT_SIZE | accepted | Alias for ACCOUNT_SIZE; precedence order: ACCOUNT_SIZE > account_size | app/config.py:226 (Settings.account_size) |
| adx_period | ADX_PERIOD | accepted | Alias for ADX_PERIOD; precedence order: ADX_PERIOD > adx_period | app/config.py:454 (Settings.adx_period) |
| adx_threshold | ADX_THRESHOLD | accepted | Alias for ADX_THRESHOLD; precedence order: ADX_THRESHOLD > adx_threshold | app/config.py:455 (Settings.adx_threshold) |
| atr_period | ATR_PERIOD | accepted | Alias for ATR_PERIOD; precedence order: ATR_PERIOD > atr_period | app/config.py:456 (Settings.atr_period) |
| atr_score_max_boost | ATR_SCORE_MAX_BOOST | accepted | Alias for ATR_SCORE_MAX_BOOST; precedence order: ATR_SCORE_MAX_BOOST > atr_score_max_boost | app/config.py:279 (Settings.atr_score_max_boost) |
| atr_score_scale | ATR_SCORE_SCALE | accepted | Alias for ATR_SCORE_SCALE; precedence order: ATR_SCORE_SCALE > atr_score_scale | app/config.py:278 (Settings.atr_score_scale) |
| base_risk_pct | BASE_RISK_PCT | accepted | Alias for BASE_RISK_PCT; precedence order: BASE_RISK_PCT > base_risk_pct | app/config.py:227 (Settings.base_risk_pct) |
| be_enabled | BE_ENABLED | accepted | Alias for BE_ENABLED; precedence order: BE_ENABLED > be_enabled | app/config.py:351 (Settings.be_enabled) |
| be_trigger_r_mult | BE_TRIGGER_R_MULT | accepted | Alias for BE_TRIGGER_R_MULT; precedence order: BE_TRIGGER_R_MULT > be_trigger_r_mult | app/config.py:146 (Settings.be_trigger_r_mult) |
| breakout_tp_multiplier | BREAKOUT_TP_MULTIPLIER | accepted | Alias for BREAKOUT_TP_MULTIPLIER; precedence order: BREAKOUT_TP_MULTIPLIER > breakout_tp_multiplier | app/config.py:315 (Settings.breakout_tp_multiplier) |
| bybit_api_key | BYBIT_API_KEY | accepted | Alias for BYBIT_API_KEY; precedence order: BYBIT_API_KEY > bybit_api_key | app/config.py:221 (Settings.bybit_api_key) |
| bybit_api_secret | BYBIT_API_SECRET | accepted | Alias for BYBIT_API_SECRET; precedence order: BYBIT_API_SECRET > bybit_api_secret | app/config.py:222 (Settings.bybit_api_secret) |
| bybit_rest_base | BYBIT_REST_BASE | accepted | Alias for BYBIT_REST_BASE; precedence order: BYBIT_REST_BASE > bybit_rest_base | app/config.py:223 (Settings.bybit_rest_base) |
| bybit_testnet | BYBIT_TESTNET | accepted | Alias for BYBIT_TESTNET; precedence order: BYBIT_TESTNET > bybit_testnet | app/config.py:220 (Settings.bybit_testnet) |
| bybit_ws_public_linear | BYBIT_WS_PUBLIC_LINEAR | accepted | Alias for BYBIT_WS_PUBLIC_LINEAR; precedence order: BYBIT_WS_PUBLIC_LINEAR > bybit_ws_public_linear | app/config.py:224 (Settings.bybit_ws_public_linear) |
| candle_history_limit | CANDLE_HISTORY_LIMIT | accepted | Alias for CANDLE_HISTORY_LIMIT; precedence order: CANDLE_HISTORY_LIMIT > candle_history_limit | app/config.py:262 (Settings.candle_history_limit) |
| candle_interval | CANDLE_INTERVAL | accepted | Alias for CANDLE_INTERVAL; precedence order: CANDLE_INTERVAL > candle_interval | app/config.py:261 (Settings.candle_interval) |
| cooldown_minutes_after_loss | COOLDOWN_MINUTES_AFTER_LOSS | accepted | Alias for COOLDOWN_MINUTES_AFTER_LOSS; precedence order: COOLDOWN_MINUTES_AFTER_LOSS > cooldown_minutes_after_loss | app/config.py:245 (Settings.cooldown_minutes_after_loss) |
| current_mode | CURRENT_MODE | accepted | Alias for CURRENT_MODE; precedence order: CURRENT_MODE > current_mode | app/config.py:50 (Settings.current_mode) |
| daily_max_dd_pct | DAILY_MAX_DD_PCT | accepted | Alias for DAILY_MAX_DD_PCT; precedence order: DAILY_MAX_DD_PCT > daily_max_dd_pct | app/config.py:69 (Settings.daily_max_dd_pct) |
| daily_profit_target_pct | DAILY_PROFIT_TARGET_PCT | accepted | Alias for DAILY_PROFIT_TARGET_PCT; precedence order: DAILY_PROFIT_TARGET_PCT > daily_profit_target_pct | app/config.py:243 (Settings.daily_profit_target_pct) |
| database_url | DATABASE_URL | accepted | Alias for DATABASE_URL; precedence order: DATABASE_URL > database_url | app/config.py:519 (Settings.database_url) |
| debug_disable_hard_risk_gates | DEBUG_DISABLE_HARD_RISK_GATES | accepted | Alias for DEBUG_DISABLE_HARD_RISK_GATES; precedence order: DEBUG_DISABLE_HARD_RISK_GATES > debug_disable_hard_risk_gates | app/config.py:177 (Settings.debug_disable_hard_risk_gates) |
| debug_loosen | DEBUG_LOOSEN | accepted | Alias for DEBUG_LOOSEN; precedence order: DEBUG_LOOSEN > debug_loosen | app/config.py:176 (Settings.debug_loosen) |
| dev_atr_mult | DEV_ATR_MULT | accepted | Alias for DEV_ATR_MULT; precedence order: DEV_ATR_MULT > dev_atr_mult | app/config.py:172 (Settings.dev_atr_mult) |
| disable_breakout_chase | DISABLE_BREAKOUT_CHASE | accepted | Alias for DISABLE_BREAKOUT_CHASE; precedence order: DISABLE_BREAKOUT_CHASE > disable_breakout_chase | app/config.py:380 (Settings.disable_breakout_chase) |
| ema_pullback_pct | EMA_PULLBACK_PCT | accepted | Alias for EMA_PULLBACK_PCT; precedence order: EMA_PULLBACK_PCT > ema_pullback_pct | app/config.py:458 (Settings.ema_pullback_pct) |
| engine_mode | ENGINE_MODE | accepted | Alias for ENGINE_MODE; precedence order: ENGINE_MODE > engine_mode | app/config.py:40 (Settings.engine_mode) |
| engulfing_wick_ratio | ENGULFING_WICK_RATIO | accepted | Alias for ENGULFING_WICK_RATIO; precedence order: ENGULFING_WICK_RATIO > engulfing_wick_ratio | app/config.py:459 (Settings.engulfing_wick_ratio) |
| exit_score_min | EXIT_SCORE_MIN | accepted | Alias for EXIT_SCORE_MIN; precedence order: EXIT_SCORE_MIN > exit_score_min | app/config.py:168 (Settings.exit_score_min) |
| fee_rate_bps | FEE_BPS | accepted | Alias for FEE_BPS; precedence order: FEE_BPS > fee_rate_bps | app/config.py:465 (Settings.fee_rate_bps) |
| force_trade_auto_close_seconds | FORCE_TRADE_AUTO_CLOSE_SECONDS | accepted | Alias for FORCE_TRADE_AUTO_CLOSE_SECONDS; precedence order: FORCE_TRADE_AUTO_CLOSE_SECONDS > force_trade_auto_close_seconds | app/config.py:368 (Settings.force_trade_auto_close_seconds) |
| force_trade_cooldown_seconds | FORCE_TRADE_COOLDOWN_SECONDS | accepted | Alias for FORCE_TRADE_COOLDOWN_SECONDS; precedence order: FORCE_TRADE_COOLDOWN_SECONDS > force_trade_cooldown_seconds | app/config.py:364 (Settings.force_trade_cooldown_seconds) |
| force_trade_every_seconds | FORCE_TRADE_EVERY_SECONDS | accepted | Alias for FORCE_TRADE_EVERY_SECONDS; precedence order: FORCE_TRADE_EVERY_SECONDS > force_trade_every_seconds | app/config.py:360 (Settings.force_trade_every_seconds) |
| force_trade_mode | FORCE_TRADE_MODE | accepted | Alias for FORCE_TRADE_MODE; precedence order: FORCE_TRADE_MODE > force_trade_mode | app/config.py:359 (Settings.force_trade_mode) |
| force_trade_random_direction | FORCE_TRADE_RANDOM_DIRECTION | accepted | Alias for FORCE_TRADE_RANDOM_DIRECTION; precedence order: FORCE_TRADE_RANDOM_DIRECTION > force_trade_random_direction | app/config.py:372 (Settings.force_trade_random_direction) |
| funding_blackout_force_close | FUNDING_BLACKOUT_FORCE_CLOSE | accepted | Alias for FUNDING_BLACKOUT_FORCE_CLOSE; precedence order: FUNDING_BLACKOUT_FORCE_CLOSE > funding_blackout_force_close | app/config.py:495 (Settings.funding_blackout_force_close) |
| funding_blackout_max_loss_usd | FUNDING_BLACKOUT_MAX_LOSS_USD | accepted | Alias for FUNDING_BLACKOUT_MAX_LOSS_USD; precedence order: FUNDING_BLACKOUT_MAX_LOSS_USD > funding_blackout_max_loss_usd | app/config.py:503 (Settings.funding_blackout_max_loss_usd) |
| funding_blackout_max_util_pct | FUNDING_BLACKOUT_MAX_UTIL_PCT | accepted | Alias for FUNDING_BLACKOUT_MAX_UTIL_PCT; precedence order: FUNDING_BLACKOUT_MAX_UTIL_PCT > funding_blackout_max_util_pct | app/config.py:499 (Settings.funding_blackout_max_util_pct) |
| funding_block_before_minutes | FUNDING_BLOCK_BEFORE_MINUTES | accepted | Alias for FUNDING_BLOCK_BEFORE_MINUTES; precedence order: FUNDING_BLOCK_BEFORE_MINUTES > funding_block_before_minutes | app/config.py:483 (Settings.funding_block_before_minutes) |
| funding_close_before_minutes | FUNDING_CLOSE_BEFORE_MINUTES | accepted | Alias for FUNDING_CLOSE_BEFORE_MINUTES; precedence order: FUNDING_CLOSE_BEFORE_MINUTES > funding_close_before_minutes | app/config.py:487 (Settings.funding_close_before_minutes) |
| funding_elevated_abs | FUNDING_ELEVATED_ABS | accepted | Alias for FUNDING_ELEVATED_ABS; precedence order: FUNDING_ELEVATED_ABS > funding_elevated_abs | app/config.py:256 (Settings.funding_elevated_abs) |
| funding_extreme_abs | FUNDING_EXTREME_ABS | accepted | Alias for FUNDING_EXTREME_ABS; precedence order: FUNDING_EXTREME_ABS > funding_extreme_abs | app/config.py:255 (Settings.funding_extreme_abs) |
| funding_guard_tail_minute | FUNDING_GUARD_TAIL_MINUTE | accepted | Alias for FUNDING_GUARD_TAIL_MINUTE; precedence order: FUNDING_GUARD_TAIL_MINUTE > funding_guard_tail_minute | app/config.py:286 (Settings.funding_guard_tail_minute) |
| funding_interval_minutes | FUNDING_INTERVAL_MINUTES | accepted | Alias for FUNDING_INTERVAL_MINUTES; precedence order: FUNDING_INTERVAL_MINUTES > funding_interval_minutes | app/config.py:491 (Settings.funding_interval_minutes) |
| global_drawdown_limit_pct | GLOBAL_DD_LIMIT_PCT | accepted | Alias for GLOBAL_DD_LIMIT_PCT; precedence order: GLOBAL_DD_LIMIT_PCT > global_drawdown_limit_pct | app/config.py:246 (Settings.global_drawdown_limit_pct) |
| heartbeat_minutes | HEARTBEAT_MINUTES | accepted | Alias for HEARTBEAT_MINUTES; precedence order: HEARTBEAT_MINUTES > heartbeat_minutes | app/config.py:45 (Settings.heartbeat_minutes) |
| htf_bias_enabled | HTF_BIAS_ENABLED | accepted | Alias for HTF_BIAS_ENABLED; precedence order: HTF_BIAS_ENABLED > htf_bias_enabled | app/config.py:115 (Settings.htf_bias_enabled) |
| htf_bias_require_slope | HTF_BIAS_REQUIRE_SLOPE | accepted | Alias for HTF_BIAS_REQUIRE_SLOPE; precedence order: HTF_BIAS_REQUIRE_SLOPE > htf_bias_require_slope | app/config.py:119 (Settings.htf_bias_require_slope) |
| htf_ema_fast | HTF_EMA_FAST | accepted | Alias for HTF_EMA_FAST; precedence order: HTF_EMA_FAST > htf_ema_fast | app/config.py:117 (Settings.htf_ema_fast) |
| htf_ema_slow | HTF_EMA_SLOW | accepted | Alias for HTF_EMA_SLOW; precedence order: HTF_EMA_SLOW > htf_ema_slow | app/config.py:118 (Settings.htf_ema_slow) |
| htf_interval | HTF_INTERVAL | accepted | Alias for HTF_INTERVAL; precedence order: HTF_INTERVAL > htf_interval | app/config.py:116 (Settings.htf_interval) |
| internal_api_base_url | INTERNAL_API_BASE_URL | accepted | Alias for INTERNAL_API_BASE_URL; precedence order: INTERNAL_API_BASE_URL > internal_api_base_url | app/config.py:518 (Settings.internal_api_base_url) |
| leverage_elevated | LEVERAGE_ELEVATED | accepted | Alias for LEVERAGE_ELEVATED; precedence order: LEVERAGE_ELEVATED > leverage_elevated | app/config.py:258 (Settings.leverage_elevated) |
| leverage_extreme | LEVERAGE_EXTREME | accepted | Alias for LEVERAGE_EXTREME; precedence order: LEVERAGE_EXTREME > leverage_extreme | app/config.py:257 (Settings.leverage_extreme) |
| manual_kill_switch | MANUAL_KILL_SWITCH | accepted | Alias for MANUAL_KILL_SWITCH; precedence order: MANUAL_KILL_SWITCH > manual_kill_switch | app/config.py:250 (Settings.manual_kill_switch) |
| market_data_allow_stale | MARKET_DATA_ALLOW_STALE | accepted | Alias for MARKET_DATA_ALLOW_STALE; precedence order: MARKET_DATA_ALLOW_STALE > market_data_allow_stale | app/config.py:214 (Settings.market_data_allow_stale) |
| market_data_backoff_base_ms | MARKET_DATA_BACKOFF_BASE_MS | accepted | Alias for MARKET_DATA_BACKOFF_BASE_MS; precedence order: MARKET_DATA_BACKOFF_BASE_MS > market_data_backoff_base_ms | app/config.py:188 (Settings.market_data_backoff_base_ms) |
| market_data_backoff_max_ms | MARKET_DATA_BACKOFF_MAX_MS | accepted | Alias for MARKET_DATA_BACKOFF_MAX_MS; precedence order: MARKET_DATA_BACKOFF_MAX_MS > market_data_backoff_max_ms | app/config.py:192 (Settings.market_data_backoff_max_ms) |
| market_data_enabled | MARKET_DATA_ENABLED | accepted | Alias for MARKET_DATA_ENABLED; precedence order: MARKET_DATA_ENABLED > market_data_enabled | app/config.py:218 (Settings.market_data_enabled) |
| market_data_failover_threshold | MARKET_DATA_FAILOVER_THRESHOLD | accepted | Alias for MARKET_DATA_FAILOVER_THRESHOLD; precedence order: MARKET_DATA_FAILOVER_THRESHOLD > market_data_failover_threshold | app/config.py:184 (Settings.market_data_failover_threshold) |
| market_data_fallbacks | MARKET_DATA_FALLBACKS | accepted | Alias for MARKET_DATA_FALLBACKS; precedence order: MARKET_DATA_FALLBACKS > market_data_fallbacks | app/config.py:183 (Settings.market_data_fallbacks) |
| market_data_provider | MARKET_DATA_PROVIDER | accepted | Alias for MARKET_DATA_PROVIDER; precedence order: MARKET_DATA_PROVIDER > market_data_provider | app/config.py:182 (Settings.market_data_provider) |
| market_data_replay_path | MARKET_DATA_REPLAY_PATH | accepted | Alias for MARKET_DATA_REPLAY_PATH; precedence order: MARKET_DATA_REPLAY_PATH > market_data_replay_path | app/config.py:196 (Settings.market_data_replay_path) |
| market_data_replay_speed | MARKET_DATA_REPLAY_SPEED | accepted | Alias for MARKET_DATA_REPLAY_SPEED; precedence order: MARKET_DATA_REPLAY_SPEED > REPLAY_SPEED > market_data_replay_speed | app/config.py:200 (Settings.market_data_replay_speed) |
| market_provider | MARKET_PROVIDER | accepted | Alias for MARKET_PROVIDER; precedence order: MARKET_PROVIDER > market_provider | app/config.py:181 (Settings.market_provider) |
| max_consecutive_losses | MAX_CONSECUTIVE_LOSSES | accepted | Alias for MAX_CONSECUTIVE_LOSSES; precedence order: MAX_CONSECUTIVE_LOSSES > max_consecutive_losses | app/config.py:244 (Settings.max_consecutive_losses) |
| max_daily_loss_pct | MAX_DAILY_LOSS_PCT | accepted | Alias for MAX_DAILY_LOSS_PCT; precedence order: MAX_DAILY_LOSS_PCT > max_daily_loss_pct | app/config.py:241 (Settings.max_daily_loss_pct) |
| max_hold_minutes | MAX_HOLD_MINUTES | accepted | Alias for MAX_HOLD_MINUTES; precedence order: MAX_HOLD_MINUTES > max_hold_minutes > TIME_STOP_MINUTES > time_stop_minutes | app/config.py:470 (Settings.max_hold_minutes) |
| max_losses_per_day | MAX_LOSSES_PER_DAY | accepted | Alias for MAX_LOSSES_PER_DAY; precedence order: MAX_LOSSES_PER_DAY > max_losses_per_day | app/config.py:508 (Settings.max_losses_per_day) |
| max_notional_account_multiplier | MAX_NOTIONAL_ACCOUNT_MULTIPLIER | accepted | Alias for MAX_NOTIONAL_ACCOUNT_MULTIPLIER; precedence order: MAX_NOTIONAL_ACCOUNT_MULTIPLIER > max_notional_account_multiplier | app/config.py:289 (Settings.max_notional_account_multiplier) |
| max_open_positions_per_direction | MAX_OPEN_POSITIONS_PER_DIRECTION | accepted | Alias for MAX_OPEN_POSITIONS_PER_DIRECTION; precedence order: MAX_OPEN_POSITIONS_PER_DIRECTION > max_open_positions_per_direction | app/config.py:384 (Settings.max_open_positions_per_direction) |
| max_risk_pct | MAX_RISK_PCT | accepted | Alias for MAX_RISK_PCT; precedence order: MAX_RISK_PCT > max_risk_pct | app/config.py:239 (Settings.max_risk_pct) |
| max_trades_per_day | MAX_TRADES_PER_DAY | accepted | Alias for MAX_TRADES_PER_DAY; precedence order: MAX_TRADES_PER_DAY > max_trades_per_day | app/config.py:240 (Settings.max_trades_per_day) |
| min_breakout_window | MIN_BREAKOUT_WINDOW | accepted | Alias for MIN_BREAKOUT_WINDOW; precedence order: MIN_BREAKOUT_WINDOW > min_breakout_window | app/config.py:282 (Settings.min_breakout_window) |
| min_signal_score | MIN_SIGNAL_SCORE | accepted | Alias for MIN_SIGNAL_SCORE; precedence order: MIN_SIGNAL_SCORE > min_signal_score | app/config.py:242 (Settings.min_signal_score) |
| min_signal_score_range | MIN_SIGNAL_SCORE_RANGE | accepted | Alias for MIN_SIGNAL_SCORE_RANGE; precedence order: MIN_SIGNAL_SCORE_RANGE > min_signal_score_range | app/config.py:145 (Settings.min_signal_score_range) |
| min_signal_score_trend | MIN_SIGNAL_SCORE_TREND | accepted | Alias for MIN_SIGNAL_SCORE_TREND; precedence order: MIN_SIGNAL_SCORE_TREND > min_signal_score_trend | app/config.py:144 (Settings.min_signal_score_trend) |
| mode | MODE | accepted | Alias for MODE; precedence order: MODE > mode | app/config.py:23 (Settings.MODE) |
| move_to_breakeven_buffer_bps | BE_BUFFER_BPS | accepted | Alias for BE_BUFFER_BPS; precedence order: BE_BUFFER_BPS > move_to_breakeven_buffer_bps | app/config.py:303 (Settings.move_to_breakeven_buffer_bps) |
| move_to_breakeven_buffer_r | BE_BUFFER_R | accepted | Alias for BE_BUFFER_R; precedence order: BE_BUFFER_R > move_to_breakeven_buffer_r | app/config.py:299 (Settings.move_to_breakeven_buffer_r) |
| move_to_breakeven_min_seconds_open | BE_MIN_SECONDS_OPEN | accepted | Alias for BE_MIN_SECONDS_OPEN; precedence order: BE_MIN_SECONDS_OPEN > move_to_breakeven_min_seconds_open | app/config.py:307 (Settings.move_to_breakeven_min_seconds_open) |
| move_to_breakeven_offset_bps | BE_OFFSET_BPS | accepted | Alias for BE_OFFSET_BPS; precedence order: BE_OFFSET_BPS > move_to_breakeven_offset_bps | app/config.py:311 (Settings.move_to_breakeven_offset_bps) |
| move_to_breakeven_trigger_r | MOVE_TO_BREAKEVEN_TRIGGER_R | accepted | Alias for MOVE_TO_BREAKEVEN_TRIGGER_R; precedence order: MOVE_TO_BREAKEVEN_TRIGGER_R > BE_TRIGGER_R > BE_TRIGGER_R_MULT > move_to_breakeven_trigger_r | app/config.py:290 (Settings.move_to_breakeven_trigger_r) |
| next_public_api_base | NEXT_PUBLIC_API_BASE | accepted | Alias for NEXT_PUBLIC_API_BASE; precedence order: NEXT_PUBLIC_API_BASE > next_public_api_base | app/config.py:517 (Settings.next_public_api_base) |
| oi_spike_pct | OI_SPIKE_PCT | accepted | Alias for OI_SPIKE_PCT; precedence order: OI_SPIKE_PCT > oi_spike_pct | app/config.py:259 (Settings.oi_spike_pct) |
| partial_tp_close_pct | PARTIAL_TP_CLOSE_PCT | accepted | Alias for PARTIAL_TP_CLOSE_PCT; precedence order: PARTIAL_TP_CLOSE_PCT > partial_tp_close_pct | app/config.py:354 (Settings.partial_tp_close_pct) |
| partial_tp_enabled | PARTIAL_TP_ENABLED | accepted | Alias for PARTIAL_TP_ENABLED; precedence order: PARTIAL_TP_ENABLED > partial_tp_enabled | app/config.py:352 (Settings.partial_tp_enabled) |
| partial_tp_r | PARTIAL_TP_R | accepted | Alias for PARTIAL_TP_R; precedence order: PARTIAL_TP_R > partial_tp_r | app/config.py:353 (Settings.partial_tp_r) |
| personal_score_threshold | PERSONAL_SCORE_THRESHOLD | accepted | Alias for PERSONAL_SCORE_THRESHOLD; precedence order: PERSONAL_SCORE_THRESHOLD > personal_score_threshold | app/config.py:318 (Settings.personal_score_threshold) |
| position_size_usd_cap | POSITION_SIZE_USD_CAP | accepted | Alias for POSITION_SIZE_USD_CAP; precedence order: POSITION_SIZE_USD_CAP > MAX_POSITION_SIZE_USD > POSITION_SIZE_USD_MAX > MAX_POSITION_USD > position_size_usd_cap | app/config.py:229 (Settings.position_size_usd_cap) |
| profile | PROFILE | accepted | Alias for PROFILE; precedence order: PROFILE > profile | app/config.py:25 (Settings.PROFILE) |
| prop_daily_stop_after_losses | PROP_DAILY_STOP_AFTER_LOSSES | accepted | Alias for PROP_DAILY_STOP_AFTER_LOSSES; precedence order: PROP_DAILY_STOP_AFTER_LOSSES > prop_daily_stop_after_losses | app/config.py:339 (Settings.prop_daily_stop_after_losses) |
| prop_daily_stop_after_net_r | PROP_DAILY_STOP_AFTER_NET_R | accepted | Alias for PROP_DAILY_STOP_AFTER_NET_R; precedence order: PROP_DAILY_STOP_AFTER_NET_R > prop_daily_stop_after_net_r | app/config.py:338 (Settings.prop_daily_stop_after_net_r) |
| prop_dd_includes_unrealized | PROP_DD_INCLUDES_UNREALIZED | accepted | Alias for PROP_DD_INCLUDES_UNREALIZED; precedence order: PROP_DD_INCLUDES_UNREALIZED > prop_dd_includes_unrealized | app/config.py:327 (Settings.prop_dd_includes_unrealized) |
| prop_enabled | PROP_ENABLED | accepted | Alias for PROP_ENABLED; precedence order: PROP_ENABLED > prop_enabled | app/config.py:321 (Settings.prop_enabled) |
| prop_governor_enabled | PROP_GOVERNOR_ENABLED | accepted | Alias for PROP_GOVERNOR_ENABLED; precedence order: PROP_GOVERNOR_ENABLED > prop_governor_enabled | app/config.py:329 (Settings.prop_governor_enabled) |
| prop_max_consec_losses | PROP_MAX_CONSEC_LOSSES | accepted | Alias for PROP_MAX_CONSEC_LOSSES; precedence order: PROP_MAX_CONSEC_LOSSES > prop_max_consec_losses | app/config.py:341 (Settings.prop_max_consec_losses) |
| prop_max_daily_loss_pct | PROP_MAX_DAILY_LOSS_PCT | accepted | Alias for PROP_MAX_DAILY_LOSS_PCT; precedence order: PROP_MAX_DAILY_LOSS_PCT > prop_max_daily_loss_pct | app/config.py:325 (Settings.prop_max_daily_loss_pct) |
| prop_max_days | PROP_MAX_DAYS | accepted | Alias for PROP_MAX_DAYS; precedence order: PROP_MAX_DAYS > prop_max_days | app/config.py:324 (Settings.prop_max_days) |
| prop_max_global_dd_pct | PROP_MAX_GLOBAL_DD_PCT | accepted | Alias for PROP_MAX_GLOBAL_DD_PCT; precedence order: PROP_MAX_GLOBAL_DD_PCT > prop_max_global_dd_pct | app/config.py:326 (Settings.prop_max_global_dd_pct) |
| prop_max_trades_per_day | PROP_MAX_TRADES_PER_DAY | accepted | Alias for PROP_MAX_TRADES_PER_DAY; precedence order: PROP_MAX_TRADES_PER_DAY > prop_max_trades_per_day | app/config.py:340 (Settings.prop_max_trades_per_day) |
| prop_min_trading_days | PROP_MIN_TRADING_DAYS | accepted | Alias for PROP_MIN_TRADING_DAYS; precedence order: PROP_MIN_TRADING_DAYS > prop_min_trading_days | app/config.py:323 (Settings.prop_min_trading_days) |
| prop_profit_target_pct | PROP_PROFIT_TARGET_PCT | accepted | Alias for PROP_PROFIT_TARGET_PCT; precedence order: PROP_PROFIT_TARGET_PCT > prop_profit_target_pct | app/config.py:322 (Settings.prop_profit_target_pct) |
| prop_reset_consec_losses_on_day_rollover | PROP_RESET_CONSEC_LOSSES_ON_DAY_ROLLOVER | accepted | Alias for PROP_RESET_CONSEC_LOSSES_ON_DAY_ROLLOVER; precedence order: PROP_RESET_CONSEC_LOSSES_ON_DAY_ROLLOVER > prop_reset_consec_losses_on_day_rollover | app/config.py:342 (Settings.prop_reset_consec_losses_on_day_rollover) |
| prop_risk_base_pct | PROP_RISK_BASE_PCT | accepted | Alias for PROP_RISK_BASE_PCT; precedence order: PROP_RISK_BASE_PCT > prop_risk_base_pct | app/config.py:330 (Settings.prop_risk_base_pct) |
| prop_risk_max_pct | PROP_RISK_MAX_PCT | accepted | Alias for PROP_RISK_MAX_PCT; precedence order: PROP_RISK_MAX_PCT > prop_risk_max_pct | app/config.py:331 (Settings.prop_risk_max_pct) |
| prop_risk_min_pct | PROP_RISK_MIN_PCT | accepted | Alias for PROP_RISK_MIN_PCT; precedence order: PROP_RISK_MIN_PCT > prop_risk_min_pct | app/config.py:332 (Settings.prop_risk_min_pct) |
| prop_score_threshold | PROP_SCORE_THRESHOLD | accepted | Alias for PROP_SCORE_THRESHOLD; precedence order: PROP_SCORE_THRESHOLD > prop_score_threshold | app/config.py:317 (Settings.prop_score_threshold) |
| prop_stepdown_after_loss | PROP_STEPDOWN_AFTER_LOSS | accepted | Alias for PROP_STEPDOWN_AFTER_LOSS; precedence order: PROP_STEPDOWN_AFTER_LOSS > prop_stepdown_after_loss | app/config.py:333 (Settings.prop_stepdown_after_loss) |
| prop_stepdown_factor | PROP_STEPDOWN_FACTOR | accepted | Alias for PROP_STEPDOWN_FACTOR; precedence order: PROP_STEPDOWN_FACTOR > prop_stepdown_factor | app/config.py:334 (Settings.prop_stepdown_factor) |
| prop_stepup_after_win | PROP_STEPUP_AFTER_WIN | accepted | Alias for PROP_STEPUP_AFTER_WIN; precedence order: PROP_STEPUP_AFTER_WIN > prop_stepup_after_win | app/config.py:335 (Settings.prop_stepup_after_win) |
| prop_stepup_cooldown_trades | PROP_STEPUP_COOLDOWN_TRADES | accepted | Alias for PROP_STEPUP_COOLDOWN_TRADES; precedence order: PROP_STEPUP_COOLDOWN_TRADES > prop_stepup_cooldown_trades | app/config.py:337 (Settings.prop_stepup_cooldown_trades) |
| prop_stepup_factor | PROP_STEPUP_FACTOR | accepted | Alias for PROP_STEPUP_FACTOR; precedence order: PROP_STEPUP_FACTOR > prop_stepup_factor | app/config.py:336 (Settings.prop_stepup_factor) |
| prop_time_cooldown_minutes | PROP_TIME_COOLDOWN_MINUTES | accepted | Alias for PROP_TIME_COOLDOWN_MINUTES; precedence order: PROP_TIME_COOLDOWN_MINUTES > prop_time_cooldown_minutes | app/config.py:349 (Settings.prop_time_cooldown_minutes) |
| pullback_atr_mult | PULLBACK_ATR_MULT | accepted | Alias for PULLBACK_ATR_MULT; precedence order: PULLBACK_ATR_MULT > pullback_atr_mult | app/config.py:169 (Settings.pullback_atr_mult) |
| range_base_score_boost | RANGE_BASE_SCORE_BOOST | accepted | Alias for RANGE_BASE_SCORE_BOOST; precedence order: RANGE_BASE_SCORE_BOOST > range_base_score_boost | app/config.py:281 (Settings.range_base_score_boost) |
| range_rsi_long_max | RANGE_RSI_LONG_MAX | accepted | Alias for RANGE_RSI_LONG_MAX; precedence order: RANGE_RSI_LONG_MAX > range_rsi_long_max | app/config.py:273 (Settings.range_rsi_long_max) |
| range_rsi_short_min | RANGE_RSI_SHORT_MIN | accepted | Alias for RANGE_RSI_SHORT_MIN; precedence order: RANGE_RSI_SHORT_MIN > range_rsi_short_min | app/config.py:274 (Settings.range_rsi_short_min) |
| range_sl_atr_mult | RANGE_SL_ATR_MULT | accepted | Alias for RANGE_SL_ATR_MULT; precedence order: RANGE_SL_ATR_MULT > range_sl_atr_mult | app/config.py:173 (Settings.range_sl_atr_mult) |
| range_tp_atr_mult | RANGE_TP_ATR_MULT | accepted | Alias for RANGE_TP_ATR_MULT; precedence order: RANGE_TP_ATR_MULT > range_tp_atr_mult | app/config.py:174 (Settings.range_tp_atr_mult) |
| reentry_cooldown_minutes | REENTRY_COOLDOWN_MINUTES | accepted | Alias for REENTRY_COOLDOWN_MINUTES; precedence order: REENTRY_COOLDOWN_MINUTES > reentry_cooldown_minutes | app/config.py:479 (Settings.reentry_cooldown_minutes) |
| regime_entry_buffer_pct | REGIME_ENTRY_BUFFER_PCT | accepted | Alias for REGIME_ENTRY_BUFFER_PCT; precedence order: REGIME_ENTRY_BUFFER_PCT > regime_entry_buffer_pct | app/config.py:316 (Settings.regime_entry_buffer_pct) |
| replay_end_ts | REPLAY_END_TS | accepted | Alias for REPLAY_END_TS; precedence order: REPLAY_END_TS > replay_end_ts | app/config.py:211 (Settings.replay_end_ts) |
| replay_max_bars | REPLAY_MAX_BARS | accepted | Alias for REPLAY_MAX_BARS; precedence order: REPLAY_MAX_BARS > replay_max_bars | app/config.py:209 (Settings.replay_max_bars) |
| replay_max_trades | REPLAY_MAX_TRADES | accepted | Alias for REPLAY_MAX_TRADES; precedence order: REPLAY_MAX_TRADES > replay_max_trades | app/config.py:208 (Settings.replay_max_trades) |
| replay_pause_seconds | REPLAY_PAUSE_SECONDS | accepted | Alias for REPLAY_PAUSE_SECONDS; precedence order: REPLAY_PAUSE_SECONDS > replay_pause_seconds | app/config.py:204 (Settings.replay_pause_seconds) |
| replay_resume | REPLAY_RESUME | accepted | Alias for REPLAY_RESUME; precedence order: REPLAY_RESUME > replay_resume | app/config.py:212 (Settings.replay_resume) |
| replay_seed | REPLAY_SEED | accepted | Alias for REPLAY_SEED; precedence order: REPLAY_SEED > replay_seed | app/config.py:213 (Settings.replay_seed) |
| replay_start_ts | REPLAY_START_TS | accepted | Alias for REPLAY_START_TS; precedence order: REPLAY_START_TS > replay_start_ts | app/config.py:210 (Settings.replay_start_ts) |
| require_candle_close_confirm | REQUIRE_CANDLE_CLOSE_CONFIRM | accepted | Alias for REQUIRE_CANDLE_CLOSE_CONFIRM; precedence order: REQUIRE_CANDLE_CLOSE_CONFIRM > require_candle_close_confirm | app/config.py:376 (Settings.require_candle_close_confirm) |
| risk_per_trade_usd | RISK_PER_TRADE_USD | accepted | Alias for RISK_PER_TRADE_USD; precedence order: RISK_PER_TRADE_USD > risk_per_trade_usd | app/config.py:228 (Settings.risk_per_trade_usd) |
| risk_reduction_enabled | RISK_REDUCTION_ENABLED | accepted | Alias for RISK_REDUCTION_ENABLED; precedence order: RISK_REDUCTION_ENABLED > risk_reduction_enabled | app/config.py:161 (Settings.risk_reduction_enabled) |
| risk_reduction_target_r | RISK_REDUCTION_TARGET_R | accepted | Alias for RISK_REDUCTION_TARGET_R; precedence order: RISK_REDUCTION_TARGET_R > risk_reduction_target_r | app/config.py:154 (Settings.risk_reduction_target_r) |
| risk_reduction_trigger_r | RISK_REDUCTION_TRIGGER_R | accepted | Alias for RISK_REDUCTION_TRIGGER_R; precedence order: RISK_REDUCTION_TRIGGER_R > risk_reduction_trigger_r | app/config.py:147 (Settings.risk_reduction_trigger_r) |
| run_mode | RUN_MODE | accepted | Alias for RUN_MODE; precedence order: RUN_MODE > run_mode | app/config.py:24 (Settings.run_mode) |
| scalp_atr_pct_max | SCALP_ATR_PCT_MAX | accepted | Alias for SCALP_ATR_PCT_MAX; precedence order: SCALP_ATR_PCT_MAX > scalp_atr_pct_max | app/config.py:79 (Settings.scalp_atr_pct_max) |
| scalp_atr_pct_min | SCALP_ATR_PCT_MIN | accepted | Alias for SCALP_ATR_PCT_MIN; precedence order: SCALP_ATR_PCT_MIN > scalp_atr_pct_min | app/config.py:78 (Settings.scalp_atr_pct_min) |
| scalp_atr_period | SCALP_ATR_PERIOD | accepted | Alias for SCALP_ATR_PERIOD; precedence order: SCALP_ATR_PERIOD > scalp_atr_period | app/config.py:77 (Settings.scalp_atr_period) |
| scalp_breakout_lookback | SCALP_BREAKOUT_LOOKBACK | accepted | Alias for SCALP_BREAKOUT_LOOKBACK; precedence order: SCALP_BREAKOUT_LOOKBACK > scalp_breakout_lookback | app/config.py:107 (Settings.scalp_breakout_lookback) |
| scalp_ema_fast | SCALP_EMA_FAST | accepted | Alias for SCALP_EMA_FAST; precedence order: SCALP_EMA_FAST > scalp_ema_fast | app/config.py:74 (Settings.scalp_ema_fast) |
| scalp_ema_slow | SCALP_EMA_SLOW | accepted | Alias for SCALP_EMA_SLOW; precedence order: SCALP_EMA_SLOW > scalp_ema_slow | app/config.py:75 (Settings.scalp_ema_slow) |
| scalp_ema_trend | SCALP_EMA_TREND | accepted | Alias for SCALP_EMA_TREND; precedence order: SCALP_EMA_TREND > scalp_ema_trend | app/config.py:76 (Settings.scalp_ema_trend) |
| scalp_engulfing_min_body_pct | SCALP_ENGULFING_MIN_BODY_PCT | accepted | Alias for SCALP_ENGULFING_MIN_BODY_PCT; precedence order: SCALP_ENGULFING_MIN_BODY_PCT > scalp_engulfing_min_body_pct | app/config.py:97 (Settings.scalp_engulfing_min_body_pct) |
| scalp_max_hold_minutes | SCALP_MAX_HOLD_MINUTES | accepted | Alias for SCALP_MAX_HOLD_MINUTES; precedence order: SCALP_MAX_HOLD_MINUTES > scalp_max_hold_minutes | app/config.py:56 (Settings.scalp_max_hold_minutes) |
| scalp_min_score | SCALP_MIN_SCORE | accepted | Alias for SCALP_MIN_SCORE; precedence order: SCALP_MIN_SCORE > scalp_min_score | app/config.py:64 (Settings.scalp_min_score) |
| scalp_pullback_ema | SCALP_PULLBACK_EMA | accepted | Alias for SCALP_PULLBACK_EMA; precedence order: SCALP_PULLBACK_EMA > scalp_pullback_ema | app/config.py:88 (Settings.scalp_pullback_ema) |
| scalp_pullback_max_dist_pct | SCALP_PULLBACK_MAX_DIST_PCT | accepted | Alias for SCALP_PULLBACK_MAX_DIST_PCT; precedence order: SCALP_PULLBACK_MAX_DIST_PCT > scalp_pullback_max_dist_pct | app/config.py:89 (Settings.scalp_pullback_max_dist_pct) |
| scalp_pullback_min_dist_pct | SCALP_PULLBACK_MIN_DIST_PCT | accepted | Alias for SCALP_PULLBACK_MIN_DIST_PCT; precedence order: SCALP_PULLBACK_MIN_DIST_PCT > scalp_pullback_min_dist_pct | app/config.py:93 (Settings.scalp_pullback_min_dist_pct) |
| scalp_reentry_cooldown_minutes | SCALP_REENTRY_COOLDOWN_MINUTES | accepted | Alias for SCALP_REENTRY_COOLDOWN_MINUTES; precedence order: SCALP_REENTRY_COOLDOWN_MINUTES > scalp_reentry_cooldown_minutes | app/config.py:60 (Settings.scalp_reentry_cooldown_minutes) |
| scalp_regime_enabled | SCALP_REGIME_ENABLED | accepted | Alias for SCALP_REGIME_ENABLED; precedence order: SCALP_REGIME_ENABLED > scalp_regime_enabled | app/config.py:70 (Settings.scalp_regime_enabled) |
| scalp_retest_max_bars | SCALP_RETEST_MAX_BARS | accepted | Alias for SCALP_RETEST_MAX_BARS; precedence order: SCALP_RETEST_MAX_BARS > scalp_retest_max_bars | app/config.py:111 (Settings.scalp_retest_max_bars) |
| scalp_rsi_confirm | SCALP_RSI_CONFIRM | accepted | Alias for SCALP_RSI_CONFIRM; precedence order: SCALP_RSI_CONFIRM > scalp_rsi_confirm | app/config.py:102 (Settings.scalp_rsi_confirm) |
| scalp_rsi_long_max | SCALP_RSI_LONG_MAX | accepted | Alias for SCALP_RSI_LONG_MAX; precedence order: SCALP_RSI_LONG_MAX > scalp_rsi_long_max | app/config.py:105 (Settings.scalp_rsi_long_max) |
| scalp_rsi_long_min | SCALP_RSI_LONG_MIN | accepted | Alias for SCALP_RSI_LONG_MIN; precedence order: SCALP_RSI_LONG_MIN > scalp_rsi_long_min | app/config.py:103 (Settings.scalp_rsi_long_min) |
| scalp_rsi_period | SCALP_RSI_PERIOD | accepted | Alias for SCALP_RSI_PERIOD; precedence order: SCALP_RSI_PERIOD > scalp_rsi_period | app/config.py:101 (Settings.scalp_rsi_period) |
| scalp_rsi_short_max | SCALP_RSI_SHORT_MAX | accepted | Alias for SCALP_RSI_SHORT_MAX; precedence order: SCALP_RSI_SHORT_MAX > scalp_rsi_short_max | app/config.py:104 (Settings.scalp_rsi_short_max) |
| scalp_rsi_short_min | SCALP_RSI_SHORT_MIN | accepted | Alias for SCALP_RSI_SHORT_MIN; precedence order: SCALP_RSI_SHORT_MIN > scalp_rsi_short_min | app/config.py:106 (Settings.scalp_rsi_short_min) |
| scalp_setup_mode | SCALP_SETUP_MODE | accepted | Alias for SCALP_SETUP_MODE; precedence order: SCALP_SETUP_MODE > scalp_setup_mode | app/config.py:84 (Settings.scalp_setup_mode) |
| scalp_sl_pct | SCALP_SL_PCT | accepted | Alias for SCALP_SL_PCT; precedence order: SCALP_SL_PCT > scalp_sl_pct | app/config.py:55 (Settings.scalp_sl_pct) |
| scalp_tp_pct | SCALP_TP_PCT | accepted | Alias for SCALP_TP_PCT; precedence order: SCALP_TP_PCT > scalp_tp_pct | app/config.py:54 (Settings.scalp_tp_pct) |
| scalp_trend_filter_enabled | SCALP_TREND_FILTER_ENABLED | accepted | Alias for SCALP_TREND_FILTER_ENABLED; precedence order: SCALP_TREND_FILTER_ENABLED > scalp_trend_filter_enabled | app/config.py:65 (Settings.scalp_trend_filter_enabled) |
| scalp_trend_slope_min | SCALP_TREND_SLOPE_MIN | accepted | Alias for SCALP_TREND_SLOPE_MIN; precedence order: SCALP_TREND_SLOPE_MIN > scalp_trend_slope_min | app/config.py:80 (Settings.scalp_trend_slope_min) |
| settings_enable_legacy | SETTINGS_ENABLE_LEGACY | accepted | Alias for SETTINGS_ENABLE_LEGACY; precedence order: SETTINGS_ENABLE_LEGACY > settings_enable_legacy | app/config.py:520 (Settings.settings_enable_legacy) |
| setup_min_candles | SETUP_MIN_CANDLES | accepted | Alias for SETUP_MIN_CANDLES; precedence order: SETUP_MIN_CANDLES > setup_min_candles | app/config.py:284 (Settings.setup_min_candles) |
| sl_atr_mult | SL_ATR_MULT | accepted | Alias for SL_ATR_MULT; precedence order: SL_ATR_MULT > sl_atr_mult | app/config.py:170 (Settings.sl_atr_mult) |
| slippage_bps | SLIPPAGE_BPS | accepted | Alias for SLIPPAGE_BPS; precedence order: SLIPPAGE_BPS > slippage_bps | app/config.py:467 (Settings.slippage_bps) |
| smoke_test_force_trade | SMOKE_TEST_FORCE_TRADE | accepted | Alias for SMOKE_TEST_FORCE_TRADE; precedence order: SMOKE_TEST_FORCE_TRADE > smoke_test_force_trade | app/config.py:516 (Settings.smoke_test_force_trade) |
| spread_bps | SPREAD_BPS | accepted | Alias for SPREAD_BPS; precedence order: SPREAD_BPS > spread_bps | app/config.py:466 (Settings.spread_bps) |
| strategy | STRATEGY | accepted | Alias for STRATEGY; precedence order: STRATEGY > strategy | app/config.py:46 (Settings.strategy) |
| strategy_adx_threshold | STRATEGY_ADX_THRESHOLD | accepted | Alias for STRATEGY_ADX_THRESHOLD; precedence order: STRATEGY_ADX_THRESHOLD > strategy_adx_threshold | app/config.py:30 (Settings.strategy_adx_threshold) |
| strategy_atr_stop_mult | STRATEGY_ATR_STOP_MULT | accepted | Alias for STRATEGY_ATR_STOP_MULT; precedence order: STRATEGY_ATR_STOP_MULT > strategy_atr_stop_mult | app/config.py:34 (Settings.strategy_atr_stop_mult) |
| strategy_bias_ema_fast | STRATEGY_BIAS_EMA_FAST | accepted | Alias for STRATEGY_BIAS_EMA_FAST; precedence order: STRATEGY_BIAS_EMA_FAST > strategy_bias_ema_fast | app/config.py:27 (Settings.strategy_bias_ema_fast) |
| strategy_bias_ema_slow | STRATEGY_BIAS_EMA_SLOW | accepted | Alias for STRATEGY_BIAS_EMA_SLOW; precedence order: STRATEGY_BIAS_EMA_SLOW > strategy_bias_ema_slow | app/config.py:28 (Settings.strategy_bias_ema_slow) |
| strategy_max_atr_pct | STRATEGY_MAX_ATR_PCT | accepted | Alias for STRATEGY_MAX_ATR_PCT; precedence order: STRATEGY_MAX_ATR_PCT > strategy_max_atr_pct | app/config.py:32 (Settings.strategy_max_atr_pct) |
| strategy_max_stop_pct | STRATEGY_MAX_STOP_PCT | accepted | Alias for STRATEGY_MAX_STOP_PCT; precedence order: STRATEGY_MAX_STOP_PCT > strategy_max_stop_pct | app/config.py:37 (Settings.strategy_max_stop_pct) |
| strategy_min_atr_pct | STRATEGY_MIN_ATR_PCT | accepted | Alias for STRATEGY_MIN_ATR_PCT; precedence order: STRATEGY_MIN_ATR_PCT > strategy_min_atr_pct | app/config.py:31 (Settings.strategy_min_atr_pct) |
| strategy_min_stop_pct | STRATEGY_MIN_STOP_PCT | accepted | Alias for STRATEGY_MIN_STOP_PCT; precedence order: STRATEGY_MIN_STOP_PCT > strategy_min_stop_pct | app/config.py:36 (Settings.strategy_min_stop_pct) |
| strategy_partial_r | STRATEGY_PARTIAL_R | accepted | Alias for STRATEGY_PARTIAL_R; precedence order: STRATEGY_PARTIAL_R > strategy_partial_r | app/config.py:39 (Settings.strategy_partial_r) |
| strategy_profile | STRATEGY_PROFILE | accepted | Alias for STRATEGY_PROFILE; precedence order: STRATEGY_PROFILE > strategy_profile | app/config.py:26 (Settings.strategy_profile) |
| strategy_swing_lookback | STRATEGY_SWING_LOOKBACK | accepted | Alias for STRATEGY_SWING_LOOKBACK; precedence order: STRATEGY_SWING_LOOKBACK > strategy_swing_lookback | app/config.py:29 (Settings.strategy_swing_lookback) |
| strategy_swing_stop_buffer_bps | STRATEGY_SWING_STOP_BUFFER_BPS | accepted | Alias for STRATEGY_SWING_STOP_BUFFER_BPS; precedence order: STRATEGY_SWING_STOP_BUFFER_BPS > strategy_swing_stop_buffer_bps | app/config.py:35 (Settings.strategy_swing_stop_buffer_bps) |
| strategy_target_r | STRATEGY_TARGET_R | accepted | Alias for STRATEGY_TARGET_R; precedence order: STRATEGY_TARGET_R > strategy_target_r | app/config.py:38 (Settings.strategy_target_r) |
| strategy_trend_slope_threshold | STRATEGY_TREND_SLOPE_THRESHOLD | accepted | Alias for STRATEGY_TREND_SLOPE_THRESHOLD; precedence order: STRATEGY_TREND_SLOPE_THRESHOLD > strategy_trend_slope_threshold | app/config.py:33 (Settings.strategy_trend_slope_threshold) |
| sweet8_base_risk_pct | SWEET8_BASE_RISK_PCT | accepted | Alias for SWEET8_BASE_RISK_PCT; precedence order: SWEET8_BASE_RISK_PCT > sweet8_base_risk_pct | app/config.py:393 (Settings.sweet8_base_risk_pct) |
| sweet8_disable_force_auto_close | SWEET8_DISABLE_FORCE_AUTO_CLOSE | accepted | Alias for SWEET8_DISABLE_FORCE_AUTO_CLOSE; precedence order: SWEET8_DISABLE_FORCE_AUTO_CLOSE > sweet8_disable_force_auto_close | app/config.py:404 (Settings.sweet8_disable_force_auto_close) |
| sweet8_disable_premature_exits | SWEET8_DISABLE_PREMATURE_EXITS | accepted | Alias for SWEET8_DISABLE_PREMATURE_EXITS; precedence order: SWEET8_DISABLE_PREMATURE_EXITS > sweet8_disable_premature_exits | app/config.py:399 (Settings.sweet8_disable_premature_exits) |
| sweet8_disable_time_stop | SWEET8_DISABLE_TIME_STOP | accepted | Alias for SWEET8_DISABLE_TIME_STOP; precedence order: SWEET8_DISABLE_TIME_STOP > sweet8_disable_time_stop | app/config.py:403 (Settings.sweet8_disable_time_stop) |
| sweet8_enabled | SWEET8_ENABLED | accepted | Alias for SWEET8_ENABLED; precedence order: SWEET8_ENABLED > sweet8_enabled | app/config.py:388 (Settings.sweet8_enabled) |
| sweet8_max_daily_loss_pct | SWEET8_MAX_DAILY_LOSS_PCT | accepted | Alias for SWEET8_MAX_DAILY_LOSS_PCT; precedence order: SWEET8_MAX_DAILY_LOSS_PCT > sweet8_max_daily_loss_pct | app/config.py:395 (Settings.sweet8_max_daily_loss_pct) |
| sweet8_max_open_positions_per_symbol | SWEET8_MAX_OPEN_POSITIONS_PER_SYMBOL | accepted | Alias for SWEET8_MAX_OPEN_POSITIONS_PER_SYMBOL; precedence order: SWEET8_MAX_OPEN_POSITIONS_PER_SYMBOL > sweet8_max_open_positions_per_symbol | app/config.py:412 (Settings.sweet8_max_open_positions_per_symbol) |
| sweet8_max_open_positions_total | SWEET8_MAX_OPEN_POSITIONS_TOTAL | accepted | Alias for SWEET8_MAX_OPEN_POSITIONS_TOTAL; precedence order: SWEET8_MAX_OPEN_POSITIONS_TOTAL > sweet8_max_open_positions_total | app/config.py:408 (Settings.sweet8_max_open_positions_total) |
| sweet8_max_risk_pct | SWEET8_MAX_RISK_PCT | accepted | Alias for SWEET8_MAX_RISK_PCT; precedence order: SWEET8_MAX_RISK_PCT > sweet8_max_risk_pct | app/config.py:394 (Settings.sweet8_max_risk_pct) |
| sweet8_mode | SWEET8_MODE | accepted | Alias for SWEET8_MODE; precedence order: SWEET8_MODE > sweet8_mode | app/config.py:389 (Settings.sweet8_mode) |
| sweet8_regime_adx_threshold | SWEET8_REGIME_ADX_THRESHOLD | accepted | Alias for SWEET8_REGIME_ADX_THRESHOLD; precedence order: SWEET8_REGIME_ADX_THRESHOLD > sweet8_regime_adx_threshold | app/config.py:440 (Settings.sweet8_regime_adx_threshold) |
| sweet8_regime_vol_threshold | SWEET8_REGIME_VOL_THRESHOLD | accepted | Alias for SWEET8_REGIME_VOL_THRESHOLD; precedence order: SWEET8_REGIME_VOL_THRESHOLD > sweet8_regime_vol_threshold | app/config.py:444 (Settings.sweet8_regime_vol_threshold) |
| sweet8_scalp_atr_sl_mult | SWEET8_SCALP_ATR_SL_MULT | accepted | Alias for SWEET8_SCALP_ATR_SL_MULT; precedence order: SWEET8_SCALP_ATR_SL_MULT > sweet8_scalp_atr_sl_mult | app/config.py:416 (Settings.sweet8_scalp_atr_sl_mult) |
| sweet8_scalp_atr_tp_mult | SWEET8_SCALP_ATR_TP_MULT | accepted | Alias for SWEET8_SCALP_ATR_TP_MULT; precedence order: SWEET8_SCALP_ATR_TP_MULT > sweet8_scalp_atr_tp_mult | app/config.py:420 (Settings.sweet8_scalp_atr_tp_mult) |
| sweet8_scalp_min_score | SWEET8_SCALP_MIN_SCORE | accepted | Alias for SWEET8_SCALP_MIN_SCORE; precedence order: SWEET8_SCALP_MIN_SCORE > sweet8_scalp_min_score | app/config.py:424 (Settings.sweet8_scalp_min_score) |
| sweet8_swing_atr_sl_mult | SWEET8_SWING_ATR_SL_MULT | accepted | Alias for SWEET8_SWING_ATR_SL_MULT; precedence order: SWEET8_SWING_ATR_SL_MULT > sweet8_swing_atr_sl_mult | app/config.py:428 (Settings.sweet8_swing_atr_sl_mult) |
| sweet8_swing_atr_tp_mult | SWEET8_SWING_ATR_TP_MULT | accepted | Alias for SWEET8_SWING_ATR_TP_MULT; precedence order: SWEET8_SWING_ATR_TP_MULT > sweet8_swing_atr_tp_mult | app/config.py:432 (Settings.sweet8_swing_atr_tp_mult) |
| sweet8_swing_min_score | SWEET8_SWING_MIN_SCORE | accepted | Alias for SWEET8_SWING_MIN_SCORE; precedence order: SWEET8_SWING_MIN_SCORE > sweet8_swing_min_score | app/config.py:436 (Settings.sweet8_swing_min_score) |
| symbol_cooldown_min | SYMBOL_COOLDOWN_MIN | accepted | Alias for SYMBOL_COOLDOWN_MIN; precedence order: SYMBOL_COOLDOWN_MIN > symbol_cooldown_min | app/config.py:175 (Settings.symbol_cooldown_min) |
| symbols | SYMBOLS | accepted | Alias for SYMBOLS; precedence order: SYMBOLS > symbols | app/config.py:41 (Settings.symbols) |
| telegram_debug_skips | TELEGRAM_DEBUG_SKIPS | accepted | Alias for TELEGRAM_DEBUG_SKIPS; precedence order: TELEGRAM_DEBUG_SKIPS > telegram_debug_skips | app/config.py:512 (Settings.telegram_debug_skips) |
| tick_interval_seconds | TICK_INTERVAL_SECONDS | accepted | Alias for TICK_INTERVAL_SECONDS; precedence order: TICK_INTERVAL_SECONDS > SCHEDULER_TICK_INTERVAL_SECONDS > tick_interval_seconds | app/config.py:263 (Settings.tick_interval_seconds) |
| time_stop_minutes | MAX_HOLD_MINUTES | accepted | Alias for MAX_HOLD_MINUTES; precedence order: MAX_HOLD_MINUTES > max_hold_minutes > TIME_STOP_MINUTES > time_stop_minutes | app/config.py:470 (Settings.max_hold_minutes) |
| tp_atr_mult | TP_ATR_MULT | accepted | Alias for TP_ATR_MULT; precedence order: TP_ATR_MULT > tp_atr_mult | app/config.py:171 (Settings.tp_atr_mult) |
| tp_cap_long_pct | TP_CAP_LONG_PCT | accepted | Alias for TP_CAP_LONG_PCT; precedence order: TP_CAP_LONG_PCT > tp_cap_long_pct | app/config.py:287 (Settings.tp_cap_long_pct) |
| tp_cap_short_pct | TP_CAP_SHORT_PCT | accepted | Alias for TP_CAP_SHORT_PCT; precedence order: TP_CAP_SHORT_PCT > tp_cap_short_pct | app/config.py:288 (Settings.tp_cap_short_pct) |
| trail_atr_mult | TRAIL_ATR_MULT | accepted | Alias for TRAIL_ATR_MULT; precedence order: TRAIL_ATR_MULT > trail_atr_mult | app/config.py:357 (Settings.trail_atr_mult) |
| trail_enabled | TRAIL_ENABLED | accepted | Alias for TRAIL_ENABLED; precedence order: TRAIL_ENABLED > trail_enabled | app/config.py:355 (Settings.trail_enabled) |
| trail_start_r | TRAIL_START_R | accepted | Alias for TRAIL_START_R; precedence order: TRAIL_START_R > trail_start_r | app/config.py:356 (Settings.trail_start_r) |
| trail_step_r | TRAIL_STEP_R | accepted | Alias for TRAIL_STEP_R; precedence order: TRAIL_STEP_R > trail_step_r | app/config.py:358 (Settings.trail_step_r) |
| trend_bias_lookback | TREND_BIAS_LOOKBACK | accepted | Alias for TREND_BIAS_LOOKBACK; precedence order: TREND_BIAS_LOOKBACK > trend_bias_lookback | app/config.py:283 (Settings.trend_bias_lookback) |
| trend_confirm_score_boost | TREND_CONFIRM_SCORE_BOOST | accepted | Alias for TREND_CONFIRM_SCORE_BOOST; precedence order: TREND_CONFIRM_SCORE_BOOST > trend_confirm_score_boost | app/config.py:280 (Settings.trend_confirm_score_boost) |
| trend_min_candles | TREND_MIN_CANDLES | accepted | Alias for TREND_MIN_CANDLES; precedence order: TREND_MIN_CANDLES > trend_min_candles | app/config.py:285 (Settings.trend_min_candles) |
| trend_rsi_midline | TREND_RSI_MIDLINE | accepted | Alias for TREND_RSI_MIDLINE; precedence order: TREND_RSI_MIDLINE > trend_rsi_midline | app/config.py:272 (Settings.trend_rsi_midline) |
| trend_score_max_boost | TREND_SCORE_MAX_BOOST | accepted | Alias for TREND_SCORE_MAX_BOOST; precedence order: TREND_SCORE_MAX_BOOST > trend_score_max_boost | app/config.py:277 (Settings.trend_score_max_boost) |
| trend_score_slope_scale | TREND_SCORE_SLOPE_SCALE | accepted | Alias for TREND_SCORE_SLOPE_SCALE; precedence order: TREND_SCORE_SLOPE_SCALE > trend_score_slope_scale | app/config.py:276 (Settings.trend_score_slope_scale) |
| trend_strength_min | TREND_STRENGTH_MIN | accepted | Alias for TREND_STRENGTH_MIN; precedence order: TREND_STRENGTH_MIN > trend_strength_min | app/config.py:260 (Settings.trend_strength_min) |
| trend_strength_scale | TREND_STRENGTH_SCALE | accepted | Alias for TREND_STRENGTH_SCALE; precedence order: TREND_STRENGTH_SCALE > trend_strength_scale | app/config.py:275 (Settings.trend_strength_scale) |
| trigger_body_ratio_min | TRIGGER_BODY_RATIO_MIN | accepted | Alias for TRIGGER_BODY_RATIO_MIN; precedence order: TRIGGER_BODY_RATIO_MIN > trigger_body_ratio_min | app/config.py:139 (Settings.trigger_body_ratio_min) |
| trigger_close_location_min | TRIGGER_CLOSE_LOCATION_MIN | accepted | Alias for TRIGGER_CLOSE_LOCATION_MIN; precedence order: TRIGGER_CLOSE_LOCATION_MIN > trigger_close_location_min | app/config.py:140 (Settings.trigger_close_location_min) |
| warmup_ignore_htf_if_disabled | WARMUP_IGNORE_HTF_IF_DISABLED | accepted | Alias for WARMUP_IGNORE_HTF_IF_DISABLED; precedence order: WARMUP_IGNORE_HTF_IF_DISABLED > warmup_ignore_htf_if_disabled | app/config.py:131 (Settings.warmup_ignore_htf_if_disabled) |
| warmup_min_bars_1h | WARMUP_MIN_BARS_1H | accepted | Alias for WARMUP_MIN_BARS_1H; precedence order: WARMUP_MIN_BARS_1H > warmup_min_bars_1h | app/config.py:127 (Settings.warmup_min_bars_1h) |
| warmup_min_bars_5m | WARMUP_MIN_BARS_5M | accepted | Alias for WARMUP_MIN_BARS_5M; precedence order: WARMUP_MIN_BARS_5M > warmup_min_bars_5m | app/config.py:123 (Settings.warmup_min_bars_5m) |
| warmup_require_replay_ready | WARMUP_REQUIRE_REPLAY_READY | accepted | Alias for WARMUP_REQUIRE_REPLAY_READY; precedence order: WARMUP_REQUIRE_REPLAY_READY > warmup_require_replay_ready | app/config.py:135 (Settings.warmup_require_replay_ready) |

## Table 3: Profile-driven defaults

## Table 4: Keys referenced in .env.example but not used (drift report)

- _None_

## Table 5: Keys used in code but missing from .env.example (missing doc report)

- ADX_PERIOD
- ADX_THRESHOLD
- ATR_PERIOD
- ATR_SCORE_MAX_BOOST
- ATR_SCORE_SCALE
- ATR_SMA_PERIOD
- BASE_RISK_PCT
- BE_BUFFER_R
- BE_OFFSET_BPS
- BE_TRIGGER_R
- BE_TRIGGER_R_MULT
- BREAKOUT_ATR_MULTIPLIER
- BREAKOUT_TP_MULTIPLIER
- BREAKOUT_VOLUME_MULTIPLIER
- BYBIT_API_KEY
- BYBIT_API_SECRET
- BYBIT_REST_BASE
- BYBIT_TESTNET
- BYBIT_WS_PUBLIC_LINEAR
- COOLDOWN_MINUTES_AFTER_LOSS
- CURRENT_MODE
- DAILY_MAX_DD_PCT
- DAILY_PROFIT_TARGET_PCT
- DEBUG_DISABLE_HARD_RISK_GATES
- DEBUG_LOOSEN
- DEBUG_RUNTIME_DIAG
- DEV_ATR_MULT
- DISABLE_BREAKOUT_CHASE
- EMA_LENGTH
- EMA_PULLBACK_PCT
- ENGULFING_WICK_RATIO
- EXIT_SCORE_MIN
- FEE_BPS
- FORCE_TRADE_AUTO_CLOSE_SECONDS
- FORCE_TRADE_COOLDOWN_SECONDS
- FORCE_TRADE_EVERY_SECONDS
- FORCE_TRADE_MODE
- FORCE_TRADE_RANDOM_DIRECTION
- FUNDING_BLACKOUT_FORCE_CLOSE
- FUNDING_BLACKOUT_MAX_LOSS_USD
- FUNDING_BLACKOUT_MAX_UTIL_PCT
- FUNDING_BLOCK_BEFORE_MINUTES
- FUNDING_CLOSE_BEFORE_MINUTES
- FUNDING_ELEVATED_ABS
- FUNDING_EXTREME_ABS
- FUNDING_GUARD_TAIL_MINUTE
- FUNDING_INTERVAL_MINUTES
- GLOBAL_DD_LIMIT_PCT
- HEARTBEAT_MINUTES
- HTF_BIAS_REQUIRE_SLOPE
- HTF_EMA_FAST
- HTF_EMA_SLOW
- LEGACY_ENV_KEYS
- LEVERAGE_ELEVATED
- LEVERAGE_EXTREME
- MANUAL_KILL_SWITCH
- MARKET_DATA_ALLOW_STALE
- MARKET_DATA_BACKOFF_BASE_MS
- MARKET_DATA_BACKOFF_MAX_MS
- MARKET_DATA_ENABLED
- MARKET_DATA_FAILOVER_THRESHOLD
- MARKET_DATA_FALLBACKS
- MARKET_PROVIDER
- MAX_CONSECUTIVE_LOSSES
- MAX_DAILY_LOSS_PCT
- MAX_HOLD_MINUTES
- MAX_LOSSES_PER_DAY
- MAX_NOTIONAL_ACCOUNT_MULTIPLIER
- MAX_OPEN_POSITIONS_PER_DIRECTION
- MAX_POSITION_SIZE_USD
- MAX_POSITION_USD
- MAX_RISK_PCT
- MAX_STOP_PCT
- MAX_TRADES_PER_DAY
- MIN_BREAKOUT_WINDOW
- MIN_SIGNAL_SCORE
- MIN_SIGNAL_SCORE_RANGE
- MIN_SIGNAL_SCORE_TREND
- MOMENTUM_MODE
- NEWS_BLACKOUTS
- OI_SPIKE_PCT
- PERSONAL_SCORE_THRESHOLD
- POSITION_SIZE_USD_CAP
- POSITION_SIZE_USD_MAX
- PROP_SCORE_THRESHOLD
- PROP_STEPDOWN_AFTER_LOSS
- PROP_STEPDOWN_FACTOR
- PROP_STEPUP_AFTER_WIN
- PROP_STEPUP_COOLDOWN_TRADES
- PROP_STEPUP_FACTOR
- PULLBACK_ATR_MULT
- RANGE_BASE_SCORE_BOOST
- RANGE_RSI_LONG_MAX
- RANGE_RSI_SHORT_MIN
- RANGE_SL_ATR_MULT
- RANGE_TP_ATR_MULT
- REENTRY_COOLDOWN_MINUTES
- REGIME_ENTRY_BUFFER_PCT
- REPLAY_MAX_BARS
- REPLAY_MAX_TRADES
- REPLAY_PAUSE_SECONDS
- REPLAY_RESUME
- REPLAY_SEED
- REPLAY_SPEED
- REQUIRE_CANDLE_CLOSE_CONFIRM
- RISK_PER_TRADE_USD
- RISK_REDUCTION_ENABLED
- RISK_REDUCTION_TARGET_R
- RISK_REDUCTION_TRIGGER_R
- SCALP_ATR_PCT_MAX
- SCALP_ATR_PCT_MIN
- SCALP_ATR_PERIOD
- SCALP_BREAKOUT_LOOKBACK
- SCALP_EMA_FAST
- SCALP_EMA_SLOW
- SCALP_EMA_TREND
- SCALP_ENGULFING_MIN_BODY_PCT
- SCALP_MAX_HOLD_MINUTES
- SCALP_MIN_SCORE
- SCALP_PULLBACK_EMA
- SCALP_PULLBACK_MAX_DIST_PCT
- SCALP_PULLBACK_MIN_DIST_PCT
- SCALP_REENTRY_COOLDOWN_MINUTES
- SCALP_REGIME_ENABLED
- SCALP_RETEST_MAX_BARS
- SCALP_RSI_CONFIRM
- SCALP_RSI_LONG_MAX
- SCALP_RSI_LONG_MIN
- SCALP_RSI_PERIOD
- SCALP_RSI_SHORT_MAX
- SCALP_RSI_SHORT_MIN
- SCALP_SETUP_MODE
- SCALP_SL_PCT
- SCALP_TP_PCT
- SCALP_TREND_FILTER_ENABLED
- SCALP_TREND_SLOPE_MIN
- SCHEDULER_TICK_INTERVAL_SECONDS
- SETUP_MIN_CANDLES
- SLIPPAGE_BPS
- SL_ATR_MULT
- SMOKE_TEST_FORCE_TRADE
- SPREAD_BPS
- STRATEGY
- STRATEGY_ADX_THRESHOLD
- STRATEGY_ATR_STOP_MULT
- STRATEGY_BIAS_EMA_FAST
- STRATEGY_BIAS_EMA_SLOW
- STRATEGY_MAX_ATR_PCT
- STRATEGY_MAX_STOP_PCT
- STRATEGY_MIN_ATR_PCT
- STRATEGY_MIN_STOP_PCT
- STRATEGY_PARTIAL_R
- STRATEGY_SWING_LOOKBACK
- STRATEGY_SWING_STOP_BUFFER_BPS
- STRATEGY_TARGET_R
- STRATEGY_TREND_SLOPE_THRESHOLD
- SWEET8_BASE_RISK_PCT
- SWEET8_BLOCKED_CLOSE_FORCE
- SWEET8_BLOCKED_CLOSE_TIME_STOP
- SWEET8_BLOCKED_CLOSE_TOTAL
- SWEET8_CURRENT_MODE
- SWEET8_DISABLE_FORCE_AUTO_CLOSE
- SWEET8_DISABLE_PREMATURE_EXITS
- SWEET8_DISABLE_TIME_STOP
- SWEET8_ENABLED
- SWEET8_MAX_DAILY_LOSS_PCT
- SWEET8_MAX_OPEN_POSITIONS_PER_SYMBOL
- SWEET8_MAX_OPEN_POSITIONS_TOTAL
- SWEET8_MAX_RISK_PCT
- SWEET8_MODE
- SWEET8_REGIME_ADX_THRESHOLD
- SWEET8_REGIME_VOL_THRESHOLD
- SWEET8_SCALP_ATR_SL_MULT
- SWEET8_SCALP_ATR_TP_MULT
- SWEET8_SCALP_MIN_SCORE
- SWEET8_SWING_ATR_SL_MULT
- SWEET8_SWING_ATR_TP_MULT
- SWEET8_SWING_MIN_SCORE
- SYMBOL_COOLDOWN_MIN
- TAKE_PROFIT_PCT
- TELEGRAM_DEBUG_SKIPS
- TIME_STOP_MINUTES
- TP_ATR_MULT
- TP_CAP_LONG_PCT
- TP_CAP_SHORT_PCT
- TRAIL_STEP_R
- TREND_BIAS_LOOKBACK
- TREND_CONFIRM_SCORE_BOOST
- TREND_MIN_CANDLES
- TREND_RSI_MIDLINE
- TREND_SCORE_MAX_BOOST
- TREND_SCORE_SLOPE_SCALE
- TREND_STRENGTH_MIN
- TREND_STRENGTH_SCALE
- TRIGGER_BODY_RATIO_MIN
- TRIGGER_CLOSE_LOCATION_MIN
- VOLUME_CONFIRM_ENABLED
- VOLUME_CONFIRM_MULTIPLIER
- VOLUME_SMA_PERIOD
- WARMUP_IGNORE_HTF_IF_DISABLED
- WARMUP_REQUIRE_REPLAY_READY
- account_size
- adx_period
- adx_threshold
- atr_period
- atr_score_max_boost
- atr_score_scale
- base_risk_pct
- be_enabled
- be_trigger_r_mult
- breakout_tp_multiplier
- bybit_api_key
- bybit_api_secret
- bybit_rest_base
- bybit_testnet
- bybit_ws_public_linear
- candle_history_limit
- candle_interval
- cooldown_minutes_after_loss
- current_mode
- daily_max_dd_pct
- daily_profit_target_pct
- database_url
- debug_disable_hard_risk_gates
- debug_loosen
- dev_atr_mult
- disable_breakout_chase
- ema_pullback_pct
- engine_mode
- engulfing_wick_ratio
- exit_score_min
- fee_rate_bps
- force_trade_auto_close_seconds
- force_trade_cooldown_seconds
- force_trade_every_seconds
- force_trade_mode
- force_trade_random_direction
- funding_blackout_force_close
- funding_blackout_max_loss_usd
- funding_blackout_max_util_pct
- funding_block_before_minutes
- funding_close_before_minutes
- funding_elevated_abs
- funding_extreme_abs
- funding_guard_tail_minute
- funding_interval_minutes
- global_drawdown_limit_pct
- heartbeat_minutes
- htf_bias_enabled
- htf_bias_require_slope
- htf_ema_fast
- htf_ema_slow
- htf_interval
- internal_api_base_url
- leverage_elevated
- leverage_extreme
- manual_kill_switch
- market_data_allow_stale
- market_data_backoff_base_ms
- market_data_backoff_max_ms
- market_data_enabled
- market_data_failover_threshold
- market_data_fallbacks
- market_data_provider
- market_data_replay_path
- market_data_replay_speed
- market_provider
- max_consecutive_losses
- max_daily_loss_pct
- max_hold_minutes
- max_losses_per_day
- max_notional_account_multiplier
- max_open_positions_per_direction
- max_risk_pct
- max_trades_per_day
- min_breakout_window
- min_signal_score
- min_signal_score_range
- min_signal_score_trend
- mode
- move_to_breakeven_buffer_bps
- move_to_breakeven_buffer_r
- move_to_breakeven_min_seconds_open
- move_to_breakeven_offset_bps
- move_to_breakeven_trigger_r
- next_public_api_base
- oi_spike_pct
- partial_tp_close_pct
- partial_tp_enabled
- partial_tp_r
- personal_score_threshold
- position_size_usd_cap
- profile
- prop_daily_stop_after_losses
- prop_daily_stop_after_net_r
- prop_dd_includes_unrealized
- prop_enabled
- prop_governor_enabled
- prop_max_consec_losses
- prop_max_daily_loss_pct
- prop_max_days
- prop_max_global_dd_pct
- prop_max_trades_per_day
- prop_min_trading_days
- prop_profit_target_pct
- prop_reset_consec_losses_on_day_rollover
- prop_risk_base_pct
- prop_risk_max_pct
- prop_risk_min_pct
- prop_score_threshold
- prop_stepdown_after_loss
- prop_stepdown_factor
- prop_stepup_after_win
- prop_stepup_cooldown_trades
- prop_stepup_factor
- prop_time_cooldown_minutes
- pullback_atr_mult
- range_base_score_boost
- range_rsi_long_max
- range_rsi_short_min
- range_sl_atr_mult
- range_tp_atr_mult
- reentry_cooldown_minutes
- regime_entry_buffer_pct
- replay_end_ts
- replay_max_bars
- replay_max_trades
- replay_pause_seconds
- replay_resume
- replay_seed
- replay_start_ts
- require_candle_close_confirm
- risk_per_trade_usd
- risk_reduction_enabled
- risk_reduction_target_r
- risk_reduction_trigger_r
- run_mode
- scalp_atr_pct_max
- scalp_atr_pct_min
- scalp_atr_period
- scalp_breakout_lookback
- scalp_ema_fast
- scalp_ema_slow
- scalp_ema_trend
- scalp_engulfing_min_body_pct
- scalp_max_hold_minutes
- scalp_min_score
- scalp_pullback_ema
- scalp_pullback_max_dist_pct
- scalp_pullback_min_dist_pct
- scalp_reentry_cooldown_minutes
- scalp_regime_enabled
- scalp_retest_max_bars
- scalp_rsi_confirm
- scalp_rsi_long_max
- scalp_rsi_long_min
- scalp_rsi_period
- scalp_rsi_short_max
- scalp_rsi_short_min
- scalp_setup_mode
- scalp_sl_pct
- scalp_tp_pct
- scalp_trend_filter_enabled
- scalp_trend_slope_min
- settings_enable_legacy
- setup_min_candles
- sl_atr_mult
- slippage_bps
- smoke_test_force_trade
- spread_bps
- strategy
- strategy_adx_threshold
- strategy_atr_stop_mult
- strategy_bias_ema_fast
- strategy_bias_ema_slow
- strategy_max_atr_pct
- strategy_max_stop_pct
- strategy_min_atr_pct
- strategy_min_stop_pct
- strategy_partial_r
- strategy_profile
- strategy_swing_lookback
- strategy_swing_stop_buffer_bps
- strategy_target_r
- strategy_trend_slope_threshold
- sweet8_base_risk_pct
- sweet8_disable_force_auto_close
- sweet8_disable_premature_exits
- sweet8_disable_time_stop
- sweet8_enabled
- sweet8_max_daily_loss_pct
- sweet8_max_open_positions_per_symbol
- sweet8_max_open_positions_total
- sweet8_max_risk_pct
- sweet8_mode
- sweet8_regime_adx_threshold
- sweet8_regime_vol_threshold
- sweet8_scalp_atr_sl_mult
- sweet8_scalp_atr_tp_mult
- sweet8_scalp_min_score
- sweet8_swing_atr_sl_mult
- sweet8_swing_atr_tp_mult
- sweet8_swing_min_score
- symbol_cooldown_min
- symbols
- telegram_debug_skips
- tick_interval_seconds
- time_stop_minutes
- tp_atr_mult
- tp_cap_long_pct
- tp_cap_short_pct
- trail_atr_mult
- trail_enabled
- trail_start_r
- trail_step_r
- trend_bias_lookback
- trend_confirm_score_boost
- trend_min_candles
- trend_rsi_midline
- trend_score_max_boost
- trend_score_slope_scale
- trend_strength_min
- trend_strength_scale
- trigger_body_ratio_min
- trigger_close_location_min
- warmup_ignore_htf_if_disabled
- warmup_min_bars_1h
- warmup_min_bars_5m
- warmup_require_replay_ready

## Forbidden when SETTINGS_ENABLE_LEGACY=false
- BASE_RISK_PCT
- BE_TRIGGER_R
- MAX_DAILY_LOSS_PCT
- MAX_RISK_PCT
- MAX_TRADES_PER_DAY
