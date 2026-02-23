from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient


def _write_replay_csv(path: Path, bars: int = 1200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume,close_time\n")
        price = 1.1000
        for i in range(bars):
            ts = start + timedelta(minutes=5 * i)
            price += 0.0002 if i % 5 else -0.00015
            f.write(f"{ts.isoformat()},{price:.5f},{price+0.0004:.5f},{price-0.0004:.5f},{price:.5f},10,{ts.isoformat()}\n")


def test_dashboard_overview_200_requests_under_replay(monkeypatch, tmp_path: Path):
    from app import config as config_module
    from app import main as main_module

    replay_file = tmp_path / "replay" / "EURUSD" / "5m.csv"
    _write_replay_csv(replay_file)

    monkeypatch.setenv("RUN_MODE", "replay")
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("ENGINE_MODE", "paper")
    monkeypatch.setenv("ASSET_CLASS", "forex")
    monkeypatch.setenv("SYMBOLS", "EURUSD")
    monkeypatch.setenv("MARKET_DATA_REPLAY_PATH", str(tmp_path / "replay"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'load.db'}")
    config_module.get_settings.cache_clear()
    main_module = reload(main_module)

    with TestClient(main_module.app) as client:
        for _ in range(300):
            assert client.get("/engine/run_once", params={"force": "true"}).status_code == 200

        slowest = 0.0
        for _ in range(200):
            t0 = time.perf_counter()
            res = client.get("/dashboard/overview")
            dt = time.perf_counter() - t0
            slowest = max(slowest, dt)
            assert res.status_code == 200
        assert slowest < 5.0
