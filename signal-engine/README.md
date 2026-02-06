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

## Tests
```bash
pytest -q
```

## Safety disclaimer
This project is for research and signal generation only. It does not place trades or provide financial advice. Use at your own risk.
