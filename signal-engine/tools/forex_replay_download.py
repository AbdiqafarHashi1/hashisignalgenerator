from __future__ import annotations

import argparse
import csv
import lzma
import struct
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import httpx

PAIR_MAP = {
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "XAUUSD": "xauusd",
}

TIMEFRAME_MINUTES = {"5m": 5, "1h": 60}


@dataclass
class Tick:
    ts: datetime
    bid: float
    ask: float


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def dukascopy_symbol(pair: str) -> str:
    code = PAIR_MAP.get(pair.upper())
    if not code:
        raise ValueError(f"unsupported_pair:{pair}")
    return code


def hour_url(pair: str, hour_dt: datetime) -> str:
    sym = dukascopy_symbol(pair)
    return (
        f"https://datafeed.dukascopy.com/datafeed/{sym}/"
        f"{hour_dt.year}/{hour_dt.month - 1:02d}/{hour_dt.day:02d}/{hour_dt.hour:02d}h_ticks.bi5"
    )


def parse_bi5(content: bytes, hour_dt: datetime, pair: str) -> list[Tick]:
    raw = lzma.decompress(content)
    ticks: list[Tick] = []
    for offset in range(0, len(raw), 20):
        chunk = raw[offset: offset + 20]
        if len(chunk) < 20:
            continue
        millis, ask_i, bid_i, *_ = struct.unpack(">IIIff", chunk)
        ts = hour_dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc) + timedelta(milliseconds=millis)
        scale = 100000.0 if "JPY" not in pair.upper() else 1000.0
        bid = bid_i / scale
        ask = ask_i / scale
        ticks.append(Tick(ts=ts, bid=bid, ask=ask))
    return ticks




def aggregate_ticks(ticks: Iterable[Tick], timeframe: str) -> list[Candle]:
    minutes = TIMEFRAME_MINUTES[timeframe]
    buckets: dict[datetime, list[Tick]] = defaultdict(list)
    for tick in ticks:
        ts = tick.ts.astimezone(timezone.utc)
        minute = (ts.minute // minutes) * minutes
        bucket_ts = ts.replace(minute=minute, second=0, microsecond=0)
        buckets[bucket_ts].append(tick)
    candles: list[Candle] = []
    for bucket_ts in sorted(buckets.keys()):
        rows = buckets[bucket_ts]
        mids = [((item.bid + item.ask) / 2.0) for item in rows]
        candles.append(Candle(timestamp=bucket_ts, open=mids[0], high=max(mids), low=min(mids), close=mids[-1], volume=0.0))
    return candles


def write_csv(path: Path, candles: list[Candle], timeframe: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    delta = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "close_time"])
        for candle in candles:
            close_time = candle.timestamp + delta
            writer.writerow([
                candle.timestamp.isoformat(),
                f"{candle.open:.6f}",
                f"{candle.high:.6f}",
                f"{candle.low:.6f}",
                f"{candle.close:.6f}",
                "0",
                close_time.isoformat(),
            ])


async def download_ticks(pair: str, start: date, end: date) -> list[Tick]:
    ticks: list[Tick] = []
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    async with httpx.AsyncClient(timeout=30.0) as client:
        current = start_dt
        while current < end_dt:
            url = hour_url(pair, current)
            response = await client.get(url)
            if response.status_code == 200:
                ticks.extend(parse_bi5(response.content, current, pair))
            current += timedelta(hours=1)
    return ticks


async def run(pair: str, timeframe: str, start: date, end: date, out_dir: Path, all_timeframes: bool = False) -> None:
    ticks = await download_ticks(pair, start, end)
    timeframes = list(TIMEFRAME_MINUTES.keys()) if all_timeframes else [timeframe]
    for tf in timeframes:
        candles = aggregate_ticks(ticks, tf)
        out = out_dir / pair.upper() / f"{tf}.csv"
        write_csv(out, candles, tf)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download free Dukascopy forex ticks and write replay CSV files")
    parser.add_argument("--pair", required=True, help="EURUSD, GBPUSD, USDJPY, XAUUSD")
    parser.add_argument("--timeframe", default="5m", choices=["5m", "1h"])
    parser.add_argument("--all-timeframes", action="store_true")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="data/replay", help="Output replay root")
    return parser.parse_args()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


if __name__ == "__main__":
    import asyncio

    args = _parse_args()
    asyncio.run(
        run(
            pair=args.pair.upper(),
            timeframe=args.timeframe,
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            out_dir=Path(args.out),
            all_timeframes=bool(args.all_timeframes),
        )
    )
