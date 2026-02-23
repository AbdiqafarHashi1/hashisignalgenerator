# FINAL EXPOSURE (single source of truth)

## 1) Strategy profiles present (names + files + entrypoints)

### Profile names present
Defined in config typing + defaults map:
- `CORE_STRATEGY`
- `SCALPER_FAST`
- `SCALPER_STABLE`
- `RANGE_MEAN_REVERT`
- `INTRADAY_TREND_SELECTIVE`
- `PROP_PASS`
- `CRYPTO_SCALP_25`
- `FTMO_PASS_25`
- `INSTANT_FUNDED`

Source of truth:
- Enum/typing + env key binding: `app/config.py` (`Settings.strategy_profile`).
- Profile value map: `app/config.py` (`_profile_defaults`).

### Runtime strategy entrypoints
The decision pipeline calls a single strategy entrypoint:
1. Scheduler calls `decide(...)` via strategy decision pipeline.
2. `decide(...)` uses `_decide_legacy(...)` whenever `STRATEGY` is `scalper` or `baseline` (default = `scalper`).
3. `_decide_legacy(...)` uses the scalper module helpers (`app/strategy/scalper.py`).

Primary files:
- `app/services/scheduler.py`
- `app/strategy/decision.py`
- `app/strategy/scalper.py`

## 2) SINGLE strategy used by default

- **Prop pass mode default**: `PROFILE=profit` + `STRATEGY_PROFILE=PROP_PASS` resolves into the same decision entrypoint (`decide -> _decide_legacy`, strategy=`scalper`).
- **Instant funded mode default**: `PROFILE=instant_funded` forces `STRATEGY_PROFILE=INSTANT_FUNDED`, but still uses the same strategy entrypoint (`decide -> _decide_legacy`, strategy=`scalper`) unless `STRATEGY` is explicitly changed.

So: **one default execution strategy path** for both modes.

## 3) Risk/governor systems and where enforced

### Prop governor
- State model + evaluation service: `app/services/prop_governor.py`.
- Enforcement during scheduling: `DecisionScheduler._governor_blockers(...)` in `app/services/scheduler.py`.
- Runtime updates on close: `PaperTrader._update_governor_after_close(...)` in `app/services/paper_trader.py`.
- Wired into app lifecycle + dashboard contract in `app/main.py`.

### Instant funded constraints
- Enforced in config normalization: `Settings.apply_mode_defaults()` in `app/config.py`:
  - `PROFILE=instant_funded` forces `STRATEGY_PROFILE=INSTANT_FUNDED`.
  - Applies `INSTANT_*` limits (risk, DD caps, trades/day, cooldown).
  - Disables prop governor by default for instant profile unless explicitly overridden.

### `SAFE_75` / `FAST_25`
- Canonical selector: `RISK_MODE` in `Settings`.
- Normalization/enforcement: `Settings.apply_mode_defaults()` sets tighter/faster risk envelopes based on strategy profile or explicit `RISK_MODE`.
- Runtime gate enforcement uses normalized risk values inside `StateStore.risk_check(...)` and `StateStore.check_limits(...)` in `app/state.py`.

## 4) Canonical metrics pipeline (equity/fees/dd/progress)

1. **Compute canonical accounting**: `app/services/metrics.py::compute_metrics(...)`
   - Computes `equity_start`, `equity_now`, `realized_net`, `unrealized_net`, `fees_today`, `fees_total`, `daily_dd_pct`, `global_dd_pct`, progress fields.
   - Uses runtime-state anchors:
     - `accounting.challenge_start_ts`
     - `accounting.day_start_equity`
     - `accounting.equity_high_watermark`
2. **Dashboard metrics adapter**: `app/services/dashboard_metrics.py::build_dashboard_metrics(...)`
   - Adds compatibility aliases (`equity_now_usd`, `global_drawdown_pct`, etc.) from canonical values.
3. **API exposure**:
   - `/dashboard/overview` in `app/main.py` calls `build_dashboard_metrics(...)`.
4. **Dashboard client read path**:
   - `dashboard/lib/dashboardClient.ts` calls `/dashboard/overview` and maps account/challenge/governor/risk/trades/equity series.

## 5) Env keys actually read at runtime (grouped)

Source of truth:
- `Settings` env aliases in `app/config.py`.
- Extra direct runtime env reads (`DEBUG_RUNTIME_DIAG`, `DEBUG_REPLAY_TRACE`, `DEBUG_REPLAY_TRACE_EVERY`) in `app/main.py`, `app/services/scheduler.py`, and `app/providers/replay.py`.

### Core runtime identity
`ASSET_CLASS`, `ENGINE_MODE`, `MODE`, `PROFILE`, `RISK_MODE`, `RUN_MODE`, `SETTINGS_ENABLE_LEGACY`, `STRATEGY_PROFILE`, `SYMBOLS`

### Market data / replay
`CANDLE_HISTORY_LIMIT`, `CANDLE_INTERVAL`, `HTF_BIAS_ENABLED`, `HTF_BIAS_REQUIRE_SLOPE`, `HTF_EMA_FAST`, `HTF_EMA_SLOW`, `HTF_INTERVAL`, `MARKET_DATA_ALLOW_STALE`, `MARKET_DATA_BACKOFF_BASE_MS`, `MARKET_DATA_BACKOFF_MAX_MS`, `MARKET_DATA_ENABLED`, `MARKET_DATA_FAILOVER_THRESHOLD`, `MARKET_DATA_FALLBACKS`, `MARKET_DATA_PROVIDER`, `MARKET_DATA_REPLAY_PATH`, `MARKET_DATA_REPLAY_SPEED`, `MARKET_PROVIDER`, `REPLAY_END_TS`, `REPLAY_MAX_BARS`, `REPLAY_MAX_TRADES`, `REPLAY_PAUSE_SECONDS`, `REPLAY_RESUME`, `REPLAY_SEED`, `REPLAY_START_TS`, `TICK_INTERVAL_SECONDS`, `WARMUP_IGNORE_HTF_IF_DISABLED`, `WARMUP_MIN_BARS_1H`, `WARMUP_MIN_BARS_5M`, `WARMUP_REQUIRE_REPLAY_READY`

### Provider credentials/endpoints
`BYBIT_API_KEY`, `BYBIT_API_SECRET`, `BYBIT_REST_BASE`, `BYBIT_TESTNET`, `BYBIT_WS_PUBLIC_LINEAR`, `OANDA_ACCOUNT_ID`, `OANDA_API_TOKEN`, `OANDA_ENV`, `OANDA_INSTRUMENTS`

### Risk / governor / account limits
`ACCOUNT_SIZE`, `BASE_RISK_PCT`, `COOLDOWN_MINUTES_AFTER_LOSS`, `DAILY_MAX_DD_PCT`, `DAILY_PROFIT_TARGET_PCT`, `GLOBAL_DD_LIMIT_PCT`, `INSTANT_COOLDOWN_MINUTES`, `INSTANT_MAX_DAILY_DD_PCT`, `INSTANT_MAX_GLOBAL_DD_PCT`, `INSTANT_MAX_TRADES_PER_DAY`, `INSTANT_MIN_SIGNAL_SCORE`, `INSTANT_MONTHLY_TARGET_PCT`, `INSTANT_RISK_BASE_PCT`, `INSTANT_RISK_MAX_PCT`, `MAX_CONSECUTIVE_LOSSES`, `MAX_DAILY_LOSS_PCT`, `MAX_LOSSES_PER_DAY`, `MAX_RISK_PCT`, `MAX_TRADES_PER_DAY`, `PROP_DAILY_STOP_AFTER_LOSSES`, `PROP_DAILY_STOP_AFTER_NET_R`, `PROP_DD_INCLUDES_UNREALIZED`, `PROP_ENABLED`, `PROP_GOVERNOR_ENABLED`, `PROP_MAX_CONSEC_LOSSES`, `PROP_MAX_DAILY_LOSS_PCT`, `PROP_MAX_DAYS`, `PROP_MAX_GLOBAL_DD_PCT`, `PROP_MAX_TRADES_PER_DAY`, `PROP_MIN_TRADING_DAYS`, `PROP_PROFIT_TARGET_PCT`, `PROP_RESET_CONSEC_LOSSES_ON_DAY_ROLLOVER`, `PROP_RISK_BASE_PCT`, `PROP_RISK_MAX_PCT`, `PROP_RISK_MIN_PCT`, `PROP_SCORE_THRESHOLD`, `PROP_STEPDOWN_AFTER_LOSS`, `PROP_STEPDOWN_FACTOR`, `PROP_STEPUP_AFTER_WIN`, `PROP_STEPUP_COOLDOWN_TRADES`, `PROP_STEPUP_FACTOR`, `PROP_TIME_COOLDOWN_MINUTES`, `RISK_PER_TRADE_USD`, `RISK_REDUCTION_ENABLED`, `RISK_REDUCTION_TARGET_R`, `RISK_REDUCTION_TRIGGER_R`, `SWEET8_BASE_RISK_PCT`, `SWEET8_BLOCKED_CLOSE_FORCE`, `SWEET8_BLOCKED_CLOSE_TIME_STOP`, `SWEET8_BLOCKED_CLOSE_TOTAL`, `SWEET8_CURRENT_MODE`, `SWEET8_DISABLE_FORCE_AUTO_CLOSE`, `SWEET8_DISABLE_PREMATURE_EXITS`, `SWEET8_DISABLE_TIME_STOP`, `SWEET8_ENABLED`, `SWEET8_MAX_DAILY_LOSS_PCT`, `SWEET8_MAX_OPEN_POSITIONS_PER_SYMBOL`, `SWEET8_MAX_OPEN_POSITIONS_TOTAL`, `SWEET8_MAX_RISK_PCT`, `SWEET8_MODE`, `SWEET8_REGIME_ADX_THRESHOLD`, `SWEET8_REGIME_VOL_THRESHOLD`, `SWEET8_SCALP_ATR_SL_MULT`, `SWEET8_SCALP_ATR_TP_MULT`, `SWEET8_SCALP_MIN_SCORE`, `SWEET8_SWING_ATR_SL_MULT`, `SWEET8_SWING_ATR_TP_MULT`, `SWEET8_SWING_MIN_SCORE`

### Strategy & execution tuning
`ADX_PERIOD`, `ADX_THRESHOLD`, `ATR_PERIOD`, `ATR_SCORE_MAX_BOOST`, `ATR_SCORE_SCALE`, `ATR_SMA_PERIOD`, `BE_BUFFER_BPS`, `BE_BUFFER_R`, `BE_ENABLED`, `BE_MIN_SECONDS_OPEN`, `BE_OFFSET_BPS`, `BE_TRIGGER_R_MULT`, `BREAKOUT_ATR_MULTIPLIER`, `BREAKOUT_TP_MULTIPLIER`, `BREAKOUT_VOLUME_MULTIPLIER`, `CURRENT_MODE`, `DEV_ATR_MULT`, `DISABLE_BREAKOUT_CHASE`, `EMA_LENGTH`, `EMA_PULLBACK_PCT`, `EXIT_SCORE_MIN`, `FUNDING_BLACKOUT_FORCE_CLOSE`, `FUNDING_BLACKOUT_MAX_LOSS_USD`, `FUNDING_BLACKOUT_MAX_UTIL_PCT`, `FUNDING_BLOCK_BEFORE_MINUTES`, `FUNDING_CLOSE_BEFORE_MINUTES`, `FUNDING_ELEVATED_ABS`, `FUNDING_EXTREME_ABS`, `FUNDING_GUARD_TAIL_MINUTE`, `FUNDING_INTERVAL_MINUTES`, `MAX_HOLD_MINUTES`, `MAX_OPEN_POSITIONS_PER_DIRECTION`, `MAX_STOP_PCT`, `MIN_SIGNAL_SCORE`, `MIN_SIGNAL_SCORE_RANGE`, `MIN_SIGNAL_SCORE_TREND`, `MOMENTUM_MODE`, `MOVE_TO_BREAKEVEN_TRIGGER_R`, `NEWS_BLACKOUTS`, `PARTIAL_TP_CLOSE_PCT`, `PARTIAL_TP_ENABLED`, `PARTIAL_TP_R`, `PERSONAL_SCORE_THRESHOLD`, `POSITION_SIZE_USD_CAP`, `PULLBACK_ATR_MULT`, `RANGE_BASE_SCORE_BOOST`, `RANGE_RSI_LONG_MAX`, `RANGE_RSI_SHORT_MIN`, `RANGE_SL_ATR_MULT`, `RANGE_TP_ATR_MULT`, `REENTRY_COOLDOWN_MINUTES`, `REQUIRE_CANDLE_CLOSE_CONFIRM`, `SCALP_ATR_PCT_MAX`, `SCALP_ATR_PCT_MIN`, `SCALP_ATR_PERIOD`, `SCALP_BREAKOUT_LOOKBACK`, `SCALP_EMA_FAST`, `SCALP_EMA_SLOW`, `SCALP_EMA_TREND`, `SCALP_ENGULFING_MIN_BODY_PCT`, `SCALP_MAX_HOLD_MINUTES`, `SCALP_MIN_SCORE`, `SCALP_PULLBACK_EMA`, `SCALP_PULLBACK_MAX_DIST_PCT`, `SCALP_PULLBACK_MIN_DIST_PCT`, `SCALP_REENTRY_COOLDOWN_MINUTES`, `SCALP_REGIME_ENABLED`, `SCALP_RETEST_MAX_BARS`, `SCALP_RSI_CONFIRM`, `SCALP_RSI_LONG_MAX`, `SCALP_RSI_LONG_MIN`, `SCALP_RSI_PERIOD`, `SCALP_RSI_SHORT_MAX`, `SCALP_RSI_SHORT_MIN`, `SCALP_SETUP_MODE`, `SCALP_SL_PCT`, `SCALP_TP_PCT`, `SCALP_TREND_FILTER_ENABLED`, `SCALP_TREND_SLOPE_MIN`, `SL_ATR_MULT`, `STRATEGY_ADX_THRESHOLD`, `STRATEGY_ATR_STOP_MULT`, `STRATEGY_BIAS_EMA_FAST`, `STRATEGY_BIAS_EMA_SLOW`, `STRATEGY_MAX_ATR_PCT`, `STRATEGY_MAX_STOP_PCT`, `STRATEGY_MIN_ATR_PCT`, `STRATEGY_MIN_STOP_PCT`, `STRATEGY_PARTIAL_R`, `STRATEGY_SWING_LOOKBACK`, `STRATEGY_SWING_STOP_BUFFER_BPS`, `STRATEGY_TARGET_R`, `STRATEGY_TREND_SLOPE_THRESHOLD`, `SYMBOL_COOLDOWN_MIN`, `TAKE_PROFIT_PCT`, `TP_ATR_MULT`, `TP_CAP_LONG_PCT`, `TP_CAP_SHORT_PCT`, `TRAIL_ATR_MULT`, `TRAIL_ENABLED`, `TRAIL_START_R`, `TRAIL_STEP_R`, `TREND_BIAS_LOOKBACK`, `TREND_CONFIRM_SCORE_BOOST`, `TREND_MIN_CANDLES`, `TREND_RSI_MIDLINE`, `TREND_SCORE_MAX_BOOST`, `TREND_SCORE_SLOPE_SCALE`, `TREND_STRENGTH_MIN`, `TREND_STRENGTH_SCALE`, `TRIGGER_BODY_RATIO_MIN`, `TRIGGER_CLOSE_LOCATION_MIN`, `VOLUME_CONFIRM_ENABLED`, `VOLUME_CONFIRM_MULTIPLIER`, `VOLUME_SMA_PERIOD`

### Storage / integrations
`DATABASE_URL`, `DATA_DIR`, `HEARTBEAT_MINUTES`, `INTERNAL_API_BASE_URL`, `NEXT_PUBLIC_API_BASE`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_DEBUG_SKIPS`, `TELEGRAM_ENABLED`

### Debug / diagnostics / forcing
`DEBUG_DISABLE_HARD_RISK_GATES`, `DEBUG_LOOSEN`, `DEBUG_REPLAY_TRACE`, `DEBUG_REPLAY_TRACE_EVERY`, `DEBUG_RUNTIME_DIAG`, `FORCE_TRADE_AUTO_CLOSE_SECONDS`, `FORCE_TRADE_COOLDOWN_SECONDS`, `FORCE_TRADE_EVERY_SECONDS`, `FORCE_TRADE_MODE`, `FORCE_TRADE_RANDOM_DIRECTION`, `SMOKE_TEST_FORCE_TRADE`

### Other runtime keys
`ENGULFING_WICK_RATIO`, `FEE_BPS`, `LEVERAGE_ELEVATED`, `LEVERAGE_EXTREME`, `MANUAL_KILL_SWITCH`, `MAX_NOTIONAL_ACCOUNT_MULTIPLIER`, `MIN_BREAKOUT_WINDOW`, `OI_SPIKE_PCT`, `REGIME_ENTRY_BUFFER_PCT`, `SETUP_MIN_CANDLES`, `SLIPPAGE_BPS`, `SPREAD_BPS`, `STRATEGY`

## 6) Deprecated / unused env keys still present

### Clearly deprecated legacy compatibility keys
- `BE_TRIGGER_R`
- `MAX_TRADES_PER_DAY`
- `BASE_RISK_PCT`
- `MAX_RISK_PCT`
- `MAX_DAILY_LOSS_PCT`

Status:
- Kept only for legacy compatibility when `SETTINGS_ENABLE_LEGACY=true`.
- Runtime already rejects them when legacy mode is off.

**Safe deletion proposal (env templates/docs only):** remove these from user-facing templates and use canonical replacements:
- `MOVE_TO_BREAKEVEN_TRIGGER_R`
- `PROP_MAX_TRADES_PER_DAY` / `INSTANT_MAX_TRADES_PER_DAY`
- `PROP_RISK_BASE_PCT` / `INSTANT_RISK_BASE_PCT`
- `PROP_RISK_MAX_PCT` / `INSTANT_RISK_MAX_PCT`
- `PROP_MAX_DAILY_LOSS_PCT` / `INSTANT_MAX_DAILY_DD_PCT`

## 7) How to verify everything works

```bash
# 1) reset to final template
cd signal-engine
cp .env.final .env

# 2) clean state (optional but recommended)
rm -f data/trades.db data/performance.json data/runtime_state.json data/replay_runtime_state.json

# 3) run tests
pytest -q

# 4) start stack
docker compose up -d --build

# 5) start replay engine and inspect overview
curl -s -X POST http://localhost:8000/engine/replay/start | jq
curl -s http://localhost:8000/dashboard/overview | jq

# 6) open dashboard
# http://localhost:3000

# 7) confirm canonical metrics
curl -s http://localhost:8000/dashboard/overview | jq '.account | {equity_start, equity_now, realized_net, unrealized_net, fees_today, fees_total_usd, daily_dd_pct, global_dd_pct}'
```

## 8) Forex replay downloader status

Already implemented and tested:
- Downloader CLI: `tools/forex_replay_download.py`
- Existing downloader test: `tests/test_forex_replay_download.py`
- README includes command examples.

Minimal smoke coverage for loading forex replay dataset now included in:
- `tests/test_replay_provider.py::test_replay_provider_loads_forex_pair_when_dataset_exists`
