#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.providers.bybit import fetch_symbol_klines


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate replay candles from live providers")
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out", default="data/replay")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        snapshot = await fetch_symbol_klines(
            symbol=symbol,
            interval=args.interval,
            limit=args.limit,
            provider="bybit",
            fallback_provider="binance,okx",
            failover_threshold=3,
            backoff_base_ms=500,
            backoff_max_ms=15000,
        )
        if snapshot is None:
            continue
        rows = [c.model_dump(mode="json") for c in snapshot.candles]
        (out / f"{symbol}_{args.interval}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {symbol}_{args.interval}.json provider={snapshot.provider_name} candles={len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
