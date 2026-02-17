# Replay dataset tooling

This folder contains tooling to build deterministic replay datasets for `MARKET_DATA_PROVIDER=replay`.

## Replay provider format (discovered from `app/providers/replay.py`)

### Directory structure

Replay files are discovered under:

- `Settings.market_data_replay_path` (default `data/replay`)
- `{REPLAY_PATH}/{SYMBOL}/{INTERVAL}.csv`
- `{REPLAY_PATH}/{SYMBOL}/{INTERVAL}.jsonl` (also supported)

Examples:

- `data/replay/ETHUSDT/1m.csv`
- `data/replay/ETHUSDT/3m.csv`
- `data/replay/ETHUSDT/5m.csv`

`SYMBOL` matching is upper-cased in the provider (`ETHUSDT`, `BTCUSDT`, ...).

### Filename pattern (symbol + interval)

The interval in the filename must match your runtime `CANDLE_INTERVAL` value (e.g. `1m`, `3m`, `5m`).

- If runtime uses `CANDLE_INTERVAL=5m`, replay file should be `5m.csv`.
- For downloader CLI convenience, Bybit interval code `5` is saved as `5m.csv`.

### Required CSV columns and timestamp expectations

CSV must include header:

```csv
timestamp,open,high,low,close,volume,close_time
```

Column notes:

- Required numeric fields: `open`, `high`, `low`, `close`.
- `volume` is optional logically (defaults to `0.0` if blank/missing), but this tooling always writes it.
- `close_time` is optional logically (defaults to `timestamp` if missing), but this tooling always writes it.

Timestamp parsing accepts:

- ISO-8601 strings (`2024-01-01T00:00:00+00:00`, with optional trailing `Z`)
- Unix epoch seconds (s)
- Unix epoch milliseconds (ms)

Loaded candles are sorted by `close_time` ascending before replay.

---

## Downloader: `download_bybit_klines.py`

Downloads Bybit V5 `category=linear` klines and writes replay-ready CSV files.

### Usage

```bash
python tools/replay/download_bybit_klines.py --symbol ETHUSDT --interval 5 --days 45 --out data/replay
```

Or explicit dates:

```bash
python tools/replay/download_bybit_klines.py --symbol ETHUSDT --interval 5m --start 2024-01-01 --end 2024-03-01 --out data/replay
```

### Idempotency

- Existing CSV rows are loaded.
- New downloads are merged by `timestamp` key.
- Re-runs do not duplicate rows.
- Final output is sorted ascending by timestamp.

### API used

`GET {BYBIT_REST_BASE}/v5/market/kline`

Params sent:

- `category=linear`
- `symbol`
- `interval`
- `start`
- `end`
- `limit`

The script pages until the end time is reached.
