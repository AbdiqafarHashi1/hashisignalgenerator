from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

from tools import forex_replay_download as downloader


def test_downloader_writes_expected_paths_and_header(monkeypatch, tmp_path: Path) -> None:
    async def fake_download_ticks(pair: str, start: date, end: date):
        assert pair == "EURUSD"
        return [
            downloader.Tick(ts=downloader.datetime(2024, 6, 1, 0, 0, tzinfo=downloader.timezone.utc), bid=1.1, ask=1.2),
            downloader.Tick(ts=downloader.datetime(2024, 6, 1, 0, 3, tzinfo=downloader.timezone.utc), bid=1.2, ask=1.3),
            downloader.Tick(ts=downloader.datetime(2024, 6, 1, 1, 1, tzinfo=downloader.timezone.utc), bid=1.0, ask=1.1),
        ]

    monkeypatch.setattr(downloader, "download_ticks", fake_download_ticks)

    asyncio.run(
        downloader.run(
            pair="EURUSD",
            timeframe="5m",
            start=date(2024, 6, 1),
            end=date(2024, 6, 1),
            out_dir=tmp_path,
            all_timeframes=True,
        )
    )

    five = tmp_path / "EURUSD" / "5m.csv"
    oneh = tmp_path / "EURUSD" / "1h.csv"
    assert five.exists()
    assert oneh.exists()
    header = five.read_text(encoding="utf-8").splitlines()[0]
    assert header == "timestamp,open,high,low,close,volume,close_time"
