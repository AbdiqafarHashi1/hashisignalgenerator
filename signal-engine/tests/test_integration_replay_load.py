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


def test_replay_overview_drawdown_fields_track_equity(monkeypatch, tmp_path: Path):
    main_module = _load_app(monkeypatch, tmp_path)
    observed: list[tuple[float, float, float, float]] = []
    with TestClient(main_module.app) as client:
        main_module.database.set_runtime_state("accounting.challenge_start_ts", value_text="1970-01-01T00:00:00+00:00")
        for i in range(320):
            r = client.get("/engine/run_once", params={"force": "true"})
            assert r.status_code == 200
            if i in {80, 160, 240}:
                now = main_module._runtime_now()
                trade_id = main_module.database.open_trade("ETHUSDT", 100.0, 95.0, 110.0, 1.0, "long", now, trade_mode="paper")
                main_module.database.close_trade(trade_id, 96.0, -4.0, -0.8, now, "sl_close", fees=0.0)
            payload = client.get("/dashboard/overview").json()
            acct = payload["account"]
            equity_start = float(acct["equity_start"])
            equity_now = float(acct["equity_now"])
            global_dd = float(acct["global_dd_pct"])
            daily_dd = float(acct["daily_dd_pct"])
            observed.append((equity_start, equity_now, global_dd, daily_dd))

    assert any(abs(cur[1] - prev[1]) > 1e-9 for prev, cur in zip(observed, observed[1:]))
    assert any(abs(cur[2] - prev[2]) > 1e-12 for prev, cur in zip(observed, observed[1:]))
    for equity_start, equity_now, global_dd, daily_dd in observed:
        expected_global = max(0.0, (equity_start - equity_now) / equity_start) if equity_start > 0 else 0.0
        assert abs(global_dd - expected_global) < 1e-9
        assert daily_dd >= 0.0
