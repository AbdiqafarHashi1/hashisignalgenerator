from __future__ import annotations

from importlib import reload
from pathlib import Path
import shutil

from fastapi.testclient import TestClient


def _load_main(monkeypatch, tmp_path: Path):
    from app import config as config_module
    from app import main as main_module

    fixture_replay = Path(__file__).parent / "fixtures" / "replay"
    replay_path = tmp_path / "replay"
    shutil.copytree(fixture_replay, replay_path)
    monkeypatch.setenv("RUN_MODE", "replay")
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("ENGINE_MODE", "paper")
    monkeypatch.setenv("SYMBOLS", "ETHUSDT")
    monkeypatch.setenv("CANDLE_INTERVAL", "3m")
    monkeypatch.setenv("MARKET_DATA_REPLAY_PATH", str(replay_path))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    config_module.get_settings.cache_clear()
    return reload(main_module)


def test_replay_progress_endpoint(monkeypatch, tmp_path: Path) -> None:
    main_module = _load_main(monkeypatch, tmp_path)
    with TestClient(main_module.app) as client:
        run = client.get("/engine/run_once", params={"force": "true"})
        assert run.status_code == 200
        resp = client.get("/replay/progress")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["bars_processed"] >= 1
        assert payload["total_bars"] >= payload["bars_processed"]


def test_health_has_provider_summary(monkeypatch, tmp_path: Path) -> None:
    main_module = _load_main(monkeypatch, tmp_path)
    with TestClient(main_module.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["provider_type"]
        assert payload["config_summary"]["run_mode"] == "replay"
