from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient


def _write_replay_csv(path: Path, bars: int = 2100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume,close_time\n")
        price = 2000.0
        for i in range(bars):
            ts = start + timedelta(minutes=5 * i)
            price += 0.5 if i % 7 else -0.8
            f.write(f"{ts.isoformat()},{price:.2f},{price+2:.2f},{price-2:.2f},{price+0.2:.2f},10,{ts.isoformat()}\n")


def _load_app(monkeypatch, tmp_path: Path):
    from app import config as config_module
    from app import main as main_module

    replay_file = tmp_path / "replay" / "ETHUSDT" / "5m.csv"
    _write_replay_csv(replay_file)

    monkeypatch.setenv("RUN_MODE", "replay")
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("ENGINE_MODE", "paper")
    monkeypatch.setenv("SYMBOLS", "ETHUSDT")
    monkeypatch.setenv("CANDLE_INTERVAL", "5m")
    monkeypatch.setenv("MARKET_DATA_REPLAY_PATH", str(tmp_path / "replay"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'load.db'}")
    monkeypatch.setenv("REPLAY_MAX_BARS", "2200")
    config_module.get_settings.cache_clear()
    return reload(main_module)


def test_replay_and_dashboard_endpoints_under_load(monkeypatch, tmp_path: Path):
    main_module = _load_app(monkeypatch, tmp_path)
    with TestClient(main_module.app) as client:
        for _ in range(2000):
            r = client.get("/engine/run_once", params={"force": "true"})
            assert r.status_code == 200

        for _ in range(20):
            assert client.get("/dashboard/overview").status_code == 200
            assert client.get("/dashboard/metrics").status_code == 200
            assert client.get("/replay/progress").status_code == 200

        progress = client.get("/replay/progress").json()
        assert progress["bars_processed"] >= 1000
