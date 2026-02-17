#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import httpx

BYBIT_REST_BASE = "https://api.bybit.com"
BYBIT_KLINE_ENDPOINT = "/v5/market/kline"
BYBIT_MAX_LIMIT = 1000


@dataclass(frozen=True)
class Candle:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Bybit V5 klines to replay CSV format")
    parser.add_argument("--symbol", required=True, help="Bybit symbol, e.g. ETHUSDT")
    parser.add_argument("--interval", required=True, help="Bybit interval code (1,3,5,15) or shorthand like 5m")
    parser.add_argument("--days", type=int, help="Lookback window in days (mutually exclusive with --start/--end)")
    parser.add_argument("--start", help="Start date (UTC) YYYY-MM-DD")
    parser.add_argument("--end", help="End date (UTC) YYYY-MM-DD")
    parser.add_argument("--out", default="data/replay", help="Replay output root path")
    parser.add_argument("--base-url", default=BYBIT_REST_BASE, help="Bybit REST base URL")
    args = parser.parse_args()

    if args.days and (args.start or args.end):
        parser.error("Use either --days or --start/--end, not both")
    if not args.days and not (args.start and args.end):
        parser.error("Provide --days or both --start and --end")
    if (args.start and not args.end) or (args.end and not args.start):
        parser.error("--start and --end must be provided together")
    if args.days is not None and args.days <= 0:
        parser.error("--days must be > 0")
    return args


def normalize_interval(raw: str) -> tuple[str, str, int]:
    value = (raw or "").strip().lower()
    if value.endswith("m"):
        value = value[:-1]
    if not value.isdigit():
        raise ValueError(f"Unsupported interval: {raw}")
    minutes = int(value)
    if minutes <= 0:
        raise ValueError(f"Unsupported interval: {raw}")
    return value, f"{minutes}m", minutes * 60_000


def parse_date_utc(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)


def range_from_args(args: argparse.Namespace) -> tuple[int, int]:
    now = datetime.now(tz=UTC)
    if args.days:
        end_dt = now
        start_dt = end_dt - timedelta(days=args.days)
        return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

    start_dt = parse_date_utc(args.start)
    end_dt = parse_date_utc(args.end) + timedelta(days=1)
    if end_dt <= start_dt:
        raise ValueError("--end must be on/after --start")
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


def to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


def parse_bybit_rows(rows: Iterable[list[str]], interval_ms: int) -> list[Candle]:
    parsed: list[Candle] = []
    for row in rows:
        if len(row) < 6:
            continue
        open_time_ms = int(row[0])
        parsed.append(
            Candle(
                timestamp_ms=open_time_ms,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time_ms=open_time_ms + interval_ms,
            )
        )
    return parsed


def load_existing(path: Path) -> dict[int, Candle]:
    if not path.exists():
        return {}
    candles: dict[int, Candle] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = int(datetime.fromisoformat(row["timestamp"]).timestamp() * 1000)
            close_ms = int(datetime.fromisoformat(row["close_time"]).timestamp() * 1000)
            candles[ts] = Candle(
                timestamp_ms=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0) or 0.0),
                close_time_ms=close_ms,
            )
    return candles


def write_csv(path: Path, candles: list[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume", "close_time"])
        writer.writeheader()
        for c in candles:
            writer.writerow(
                {
                    "timestamp": to_iso(c.timestamp_ms),
                    "open": f"{c.open:.8f}",
                    "high": f"{c.high:.8f}",
                    "low": f"{c.low:.8f}",
                    "close": f"{c.close:.8f}",
                    "volume": f"{c.volume:.8f}",
                    "close_time": to_iso(c.close_time_ms),
                }
            )


def fetch_bybit_klines(base_url: str, symbol: str, bybit_interval: str, start_ms: int, end_ms: int, interval_ms: int) -> list[Candle]:
    collected: dict[int, Candle] = {}
    cursor_ms = start_ms
    page = 0
    with httpx.Client(timeout=20.0, base_url=base_url) as client:
        while cursor_ms < end_ms:
            page += 1
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "start": cursor_ms,
                "end": end_ms,
                "limit": BYBIT_MAX_LIMIT,
            }
            response = client.get(BYBIT_KLINE_ENDPOINT, params=params)
            response.raise_for_status()
            payload = response.json()
            if payload.get("retCode") != 0:
                raise RuntimeError(f"Bybit error: {payload.get('retMsg')}")
            rows = payload.get("result", {}).get("list", [])
            candles = parse_bybit_rows(rows, interval_ms)
            if not candles:
                print(f"[page {page}] no rows returned; stopping")
                break

            min_ts = min(c.timestamp_ms for c in candles)
            max_ts = max(c.timestamp_ms for c in candles)
            for candle in candles:
                if start_ms <= candle.timestamp_ms < end_ms:
                    collected[candle.timestamp_ms] = candle

            print(
                f"[page {page}] fetched={len(candles)} window={to_iso(min_ts)}..{to_iso(max_ts)} total_unique={len(collected)}"
            )

            next_cursor_ms = max_ts + interval_ms
            if next_cursor_ms <= cursor_ms:
                break
            cursor_ms = next_cursor_ms

    return sorted(collected.values(), key=lambda c: c.timestamp_ms)


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper()
    bybit_interval, file_interval, interval_ms = normalize_interval(args.interval)
    start_ms, end_ms = range_from_args(args)

    output_path = Path(args.out) / symbol / f"{file_interval}.csv"
    print(f"Downloading {symbol} interval={bybit_interval} ({file_interval}) start={to_iso(start_ms)} end={to_iso(end_ms)}")
    fresh = fetch_bybit_klines(args.base_url, symbol, bybit_interval, start_ms, end_ms, interval_ms)
    if not fresh:
        print("No klines downloaded for requested range.")
        return 1

    existing = load_existing(output_path)
    merged = {**existing, **{c.timestamp_ms: c for c in fresh}}
    ordered = sorted(merged.values(), key=lambda c: c.timestamp_ms)
    write_csv(output_path, ordered)

    added = len(ordered) - len(existing)
    print(f"Wrote {output_path} rows={len(ordered)} added={added} updated_or_existing={len(existing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
