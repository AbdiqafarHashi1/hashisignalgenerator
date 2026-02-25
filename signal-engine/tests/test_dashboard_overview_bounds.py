from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib import reload

from fastapi.testclient import TestClient


def _fresh_main(monkeypatch, tmp_path):
    from app import config as config_module
    from app import main as main_module

    monkeypatch.setenv("RUN_MODE", "replay")
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("ENGINE_MODE", "paper")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'overview-bounds.db'}")
    config_module.get_settings.cache_clear()
    return reload(main_module)


def test_overview_default_limits_are_bounded(monkeypatch, tmp_path) -> None:
    main_module = _fresh_main(monkeypatch, tmp_path)
    with TestClient(main_module.app) as client:
        db = client.app.state.database
        now = datetime.now(timezone.utc)
        for idx in range(350):
            opened = now - timedelta(minutes=idx + 2)
            trade_id = db.open_trade("BTCUSDT", 100.0, 90.0, 120.0, 1.0, "long", opened, trade_mode="paper")
            db.close_trade(trade_id, 101.0, 1.0, 0.5, opened + timedelta(minutes=1), "tp_close", fees=0.1)
            db.log_event("execution_fill", {"symbol": "BTCUSDT", "side": "long", "price": 101.0, "qty": 1.0, "fee": 0.1, "reason": "filled"}, f"corr-{idx}")

        payload = client.get("/dashboard/overview").json()
        assert len(payload["recent_trades"]) <= 200
        assert len(payload["account"]["executions"]) <= 200


def test_dashboard_separate_endpoints_respect_limits(monkeypatch, tmp_path) -> None:
    main_module = _fresh_main(monkeypatch, tmp_path)
    with TestClient(main_module.app) as client:
        db = client.app.state.database
        now = datetime.now(timezone.utc)
        for idx in range(120):
            opened = now - timedelta(minutes=idx + 2)
            trade_id = db.open_trade("BTCUSDT", 100.0, 90.0, 120.0, 1.0, "long", opened, trade_mode="paper")
            db.close_trade(trade_id, 101.0, 1.0, 0.5, opened + timedelta(minutes=1), "tp_close", fees=0.1)
            db.log_event("execution_fill", {"symbol": "BTCUSDT", "side": "long", "price": 101.0, "qty": 1.0, "fee": 0.1, "reason": "filled"}, f"corr-{idx}")

        trades = client.get("/dashboard/trades", params={"limit": 25, "offset": 10}).json()["items"]
        executions = client.get("/dashboard/executions", params={"limit": 25, "offset": 10}).json()["items"]

        assert len(trades) == 25
        assert len(executions) == 25
