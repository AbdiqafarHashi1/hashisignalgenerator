# signal-engine

A conservative, signal-only decision engine for crypto/CFD setups. The engine targets high expectancy and asymmetry over frequency. It produces three outcomes only: `TRADE`, `NO_TRADE`, or `RISK_OFF`.

## Philosophy
- Trade frequency is irrelevant. Quality and R:R dominate.
- Losses are small and capped by strict risk rules.
- Winners are asymmetric and only taken when multiple filters align.
- Manual execution only. The engine never places orders.

## Architecture
The decision pipeline is a strict, multi-tier filter:
1. Daily posture (cached per symbol/day)
2. State/risk gate (daily limits + cooldowns)
3. Trend filter (EMA50 on 5m candles)
4. Momentum filter (ADX or ATR-over-ATR-SMA)
5. Engulfing trigger at EMA pullback/break-retest with optional volume confirmation
6. Fixed scalper SL/TP distances with capped risk
7. Final TradePlan output

## Quick start
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Hetzner Ubuntu deployment (API + Dashboard)
1. Install Docker + Compose plugin.
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose-plugin
   sudo usermod -aG docker $USER
   newgrp docker
   ```
2. Clone the repo and create your `.env` file.
   ```bash
   git clone <YOUR_REPO_URL> signal-engine
   cd signal-engine
   cp .env.example .env
   ```
   Update TELEGRAM settings, MODE, SYMBOLS, and any risk settings as needed.
3. Build and start the services.
   ```bash
   docker compose up -d --build
   ```
4. Access:
   - Dashboard: `http://SERVER_IP/`
   - API docs: `http://SERVER_IP:8000/docs`

No domain is required for the initial deployment. Add a domain + HTTPS later when ready.

## API
- `GET /health`
- `GET /heartbeat`
- `GET /run?force=true`
- `GET /start`
- `GET /stop`
- `GET /stats`
- `GET /trades`
- `GET /positions`
- `GET /equity`
- `GET /symbols`
- `POST /symbols`
- `GET /paper/reset`
- `POST /webhook/tradingview`
- `GET /decision/latest?symbol=BTCUSDT`
- `POST /trade_outcome`
- `GET /state`
- `GET /state/today?symbol=BTCUSDT`
- `GET /test/telegram`
- `GET /debug/runtime`
- `POST /debug/force_signal`
- `POST /debug/smoke/run_full_cycle`
- `POST /debug/storage/reset`

## TradingView webhook payload
Send a single JSON payload that includes the TradingView structure plus normalized market + bias inputs.

```json
{
  "tradingview": {
    "symbol": "BTCUSDT",
    "direction_hint": "long",
    "entry_low": 67000,
    "entry_high": 67250,
    "sl_hint": 66350,
    "setup_type": "sweep_reclaim",
    "tf_bias": "4h",
    "tf_entry": "5m"
  },
  "market": {
    "funding_rate": 0.0012,
    "oi_change_24h": 0.04,
    "leverage_ratio": 1.8,
    "trend_strength": 0.72
  },
  "bias": {
    "direction": "long",
    "confidence": 0.81
  },
  "interval": "5m",
  "candles": [
    { "open": 67000, "high": 67200, "low": 66950, "close": 67150, "volume": 120.5 },
    { "open": 67150, "high": 67320, "low": 67020, "close": 67280, "volume": 141.2 }
  ]
}
```
The engine expects 5m candles for scalper logic. If you omit `candles`, the engine will return a `RISK_OFF` plan with a `no_candles` rationale.

## Environment variables
Modes are strict and can only be `prop_cfd` or `personal_crypto`.

Profiles tune strategy thresholds only (risk gates stay on):
- `PROFILE=profit` (default) — stricter, real-usage bias
- `PROFILE=diag` — frequent-signal diagnostics, forced to notification-only mode

Key settings (defaults depend on MODE):
- `MODE`
- `account_size`
- `base_risk_pct`
- `max_risk_pct`
- `max_trades_per_day`
- `max_daily_loss_pct`
- `daily_profit_target_pct`
- `max_consecutive_losses`
- `min_signal_score`
- `TICK_INTERVAL_SECONDS`
- `SCHEDULER_TICK_INTERVAL_SECONDS`
- `SMOKE_TEST_FORCE_TRADE`
- `FORCE_TRADE_MODE`
- `FORCE_TRADE_EVERY_SECONDS`
- `FORCE_TRADE_COOLDOWN_SECONDS`
- `FORCE_TRADE_AUTO_CLOSE_SECONDS`
- `FORCE_TRADE_RANDOM_DIRECTION`

Risk environment thresholds:
- `funding_extreme_abs`
- `funding_elevated_abs`
- `leverage_extreme`
- `leverage_elevated`
- `oi_spike_pct`
- `trend_strength_min`

Safety controls:
- `cooldown_minutes_after_loss`
- `max_losses_per_day`
- `news_blackouts` (UTC windows like `12:00-13:00,19:30-20:15`)

Scalper controls:
- `candle_interval`
- `candle_history_limit`
- `ema_length`
- `momentum_mode`
- `adx_period`
- `adx_threshold`
- `atr_period`
- `atr_sma_period`
- `ema_pullback_pct`
- `engulfing_wick_ratio`
- `volume_confirm_enabled`
- `volume_sma_period`
- `volume_confirm_multiplier`
- `max_stop_pct`
- `take_profit_pct`

## Logging
Every webhook and decision is logged as JSONL in `./data/logs/YYYY-MM-DD.jsonl` with a correlation id.

## DIAG proof-fire workflow
1. Set DIAG profile and enable debug endpoints in your `.env`:
   ```bash
   PROFILE=diag
   DEBUG_LOOSEN=true
   TELEGRAM_ENABLED=true
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```
2. Start the stack:
   ```bash
   docker compose up -d --build
   ```
3. Verify runtime settings:
   ```bash
   curl -s http://localhost:8000/debug/runtime | jq
   ```
4. Force a diagnostic decision (bypasses soft gates only):
   ```bash
   curl -s -X POST http://localhost:8000/debug/force_signal \
     -H "Content-Type: application/json" \
     -d '{"symbol":"BTCUSDT","direction":"long","strategy":"scalper","bypass_soft_gates":true}' | jq
   ```
5. Confirm the decision was stored:
   ```bash
   curl -s "http://localhost:8000/decision/latest?symbol=BTCUSDT" | jq
   ```

## Smoke test force-trade workflow (paper)
Use this to validate the full trade pipeline (decision -> paper trade -> storage) end-to-end.

1. Enable smoke-test forced trading and start the service:
   ```bash
   export SMOKE_TEST_FORCE_TRADE=true
   export MODE=paper
   uvicorn app.main:app --reload
   ```
2. Start the scheduler (optional if you already have snapshots):
   ```bash
   curl -s http://localhost:8000/engine/start | jq
   ```
3. Force a trade decision and execution:
   ```bash
   curl -s -X POST http://localhost:8000/debug/force_signal \
     -H "Content-Type: application/json" \
     -d '{"symbol":"ETHUSDT","direction":"long","bypass_soft_gates":true}' | jq
   ```
4. Confirm the decision is stored:
   ```bash
   curl -s "http://localhost:8000/decision/latest?symbol=ETHUSDT" | jq
   ```
5. Confirm the paper position and trade record exist:
   ```bash
   curl -s http://localhost:8000/positions | jq
   curl -s http://localhost:8000/trades | jq
   ```

## Force-trade mode (firehose) workflow
Use this mode to spam paper trades on a short interval for end-to-end validation. Set
`FORCE_TRADE_AUTO_CLOSE_SECONDS` to close any open paper trades after that many seconds;
leave it as `0` to allow multiple concurrent forced trades.

Example `.env`:
```bash
MODE=paper
FORCE_TRADE_MODE=true
FORCE_TRADE_EVERY_SECONDS=5
FORCE_TRADE_COOLDOWN_SECONDS=0
FORCE_TRADE_AUTO_CLOSE_SECONDS=0
FORCE_TRADE_RANDOM_DIRECTION=true
```

1. Enable force-trade mode and start the service:
   ```bash
   export MODE=paper
   export FORCE_TRADE_MODE=true
   export FORCE_TRADE_EVERY_SECONDS=5
   export FORCE_TRADE_COOLDOWN_SECONDS=0
   export FORCE_TRADE_AUTO_CLOSE_SECONDS=0
   export FORCE_TRADE_RANDOM_DIRECTION=true
   uvicorn app.main:app --reload
   ```
2. Start the scheduler:
   ```bash
   curl -s http://localhost:8000/engine/start | jq
   ```
3. Verify trade count increases within 30 seconds:
   ```bash
   curl -s http://localhost:8000/trades | jq
   sleep 30
   curl -s http://localhost:8000/trades | jq
   ```
   Or run:
   ```bash
   ./scripts/force_trade_smoke_test.sh
   ```

## Deterministic full-cycle smoke test (no market data required)
Use this endpoint to force a full open → close → PnL cycle in one call.

1. Reset stored state (clears `./data` logs and trade rows):
   ```bash
   curl -s -X POST http://localhost:8000/debug/storage/reset | jq
   ```
2. Run the full cycle:
   ```bash
   curl -s -X POST http://localhost:8000/debug/smoke/run_full_cycle \
     -H "Content-Type: application/json" \
     -d '{"symbol":"ETHUSDT","direction":"long","hold_seconds":2,"entry_price":100.0}' | jq
   ```
3. Verify trades and stats reflect the closed trade:
   ```bash
   curl -s http://localhost:8000/trades | jq
   curl -s http://localhost:8000/stats | jq
   ```

## Switching back to PROFIT
Update your `.env`:
```bash
PROFILE=profit
DEBUG_LOOSEN=false
```
Then restart:
```bash
docker compose up -d --build
```

## Tests
```bash
pytest -q
```

## Safety disclaimer
This project is for research and signal generation only. It does not place trades or provide financial advice. Use at your own risk.

## Local verification checklist (docker compose)
1. Build and start:
   ```bash
   docker compose up -d --build
   ```
2. Confirm engine status:
   ```bash
   curl -s http://localhost:8000/engine/status | jq
   ```
3. Inspect runtime settings:
   ```bash
   curl -s http://localhost:8000/debug/runtime | jq
   ```
4. Force a diagnostic signal:
   ```bash
   curl -s -X POST http://localhost:8000/debug/force_signal \
     -H "Content-Type: application/json" \
     -d '{"symbol":"BTCUSDT","direction":"long","strategy":"scalper","bypass_soft_gates":true}' | jq
   ```
5. Fetch the latest decision:
   ```bash
   curl -s "http://localhost:8000/decision/latest?symbol=BTCUSDT" | jq
   ```
6. Switch to PROFIT profile and rebuild:
   ```bash
   PROFILE=profit docker compose up -d --build
   ```
