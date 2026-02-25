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
        assert len(payload["recent_trades"]) <= 50
        assert len(payload["account"]["executions"]) <= 50
        assert len(payload["account"]["open_orders"]) <= 50
        assert "trades" not in payload
        assert "executions_all" not in payload


def test_dashboard_separate_endpoints_respect_limits(monkeypatch, tmp_path) -> None:
    main_module = _fresh_main(monkeypatch, tmp_path)
    with TestClient(main_module.app) as client:
        db = client.app.state.database
        now = datetime.now(timezone.utc)
        for idx in range(5000):
            opened = now - timedelta(minutes=idx + 2)
            trade_id = db.open_trade("BTCUSDT", 100.0, 90.0, 120.0, 1.0, "long", opened, trade_mode="paper")
            db.close_trade(trade_id, 101.0, 1.0, 0.5, opened + timedelta(minutes=1), "tp_close", fees=0.1)
            db.log_event("execution_fill", {"symbol": "BTCUSDT", "side": "long", "price": 101.0, "qty": 1.0, "fee": 0.1, "reason": "filled"}, f"corr-{idx}")

        overview = client.get("/dashboard/overview", params={"executions_limit": 5000, "trades_limit": 5000, "open_orders_limit": 5000})
        assert overview.status_code == 200
        ov_payload = overview.json()
        assert len(ov_payload["recent_trades"]) == 200
        assert len(ov_payload["account"]["executions"]) == 200

        trades_payload = client.get("/dashboard/trades", params={"limit": 5000, "offset": 10}).json()
        executions_payload = client.get("/dashboard/executions", params={"limit": 5000, "offset": 10}).json()

        assert trades_payload["limit"] == 200
        assert executions_payload["limit"] == 200
        assert len(trades_payload["items"]) == 200
        assert len(executions_payload["items"]) == 200


def test_overview_perf_and_debug_perf(monkeypatch, tmp_path) -> None:
    import time

    main_module = _fresh_main(monkeypatch, tmp_path)
    with TestClient(main_module.app) as client:
        db = client.app.state.database
        now = datetime.now(timezone.utc)
        for idx in range(3000):
            opened = now - timedelta(minutes=idx + 2)
            trade_id = db.open_trade("BTCUSDT", 100.0, 90.0, 120.0, 1.0, "long", opened, trade_mode="paper")
            db.close_trade(trade_id, 101.0, 1.0, 0.5, opened + timedelta(minutes=1), "tp_close", fees=0.1)
            db.log_event("execution_fill", {"symbol": "BTCUSDT", "side": "long", "price": 101.0, "qty": 1.0, "fee": 0.1, "reason": "filled"}, f"corr-{idx}")

        start = time.perf_counter()
        response = client.get("/dashboard/overview")
        elapsed = time.perf_counter() - start
        assert response.status_code == 200
        assert elapsed < 2.0

        perf = client.get("/debug/perf").json()
        assert perf["last_overview_ms"] is not None
        assert perf["last_trades_query_ms"] is not None
        assert perf["last_exec_query_ms"] is not None
