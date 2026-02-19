from __future__ import annotations

from importlib import reload
from pathlib import Path
import shutil

from fastapi.testclient import TestClient


def _load_main(monkeypatch, tmp_path: Path, replay_speed: str = "1"):

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
    monkeypatch.setenv("FORCE_TRADE_MODE", "true")
    monkeypatch.setenv("FORCE_TRADE_RANDOM_DIRECTION", "false")
    monkeypatch.setenv("FORCE_TRADE_AUTO_CLOSE_SECONDS", "1")
    monkeypatch.setenv("REPLAY_MAX_BARS", "40")
    monkeypatch.setenv("REPLAY_MAX_TRADES", "12")
    monkeypatch.setenv("MARKET_DATA_REPLAY_SPEED", replay_speed)
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'replay.db'}")

    config_module.get_settings.cache_clear()
    return reload(main_module)


def _run_and_collect(main_module) -> tuple[int, float, list[tuple[str, str, float, float | None, str, str, str]]]:
    with TestClient(main_module.app) as client:
        for _ in range(40):
            resp = client.get("/engine/run_once", params={"force": "true"})
            assert resp.status_code == 200
        state = client.get("/state").json()
        trades = client.get("/trades").json()["trades"]
        normalized = [
            (
                str(t["symbol"]),
                str(t["side"]),
                float(t["entry"]),
                None if t.get("exit") is None else float(t["exit"]),
                str(t.get("opened_at")),
                str(t.get("closed_at")),
                str(t.get("result")),
            )
            for t in trades
        ]
        return len(trades), float(state["equity"]), normalized


def test_replay_produces_trade(monkeypatch, tmp_path: Path) -> None:
    main_module = _load_main(monkeypatch, tmp_path)
    trades, _equity, _details = _run_and_collect(main_module)
    assert trades >= 1


def test_replay_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    main_module = _load_main(monkeypatch, tmp_path / "a")
    first_trades, first_equity, first_details = _run_and_collect(main_module)

    main_module = _load_main(monkeypatch, tmp_path / "b")
    second_trades, second_equity, second_details = _run_and_collect(main_module)

    assert first_trades == second_trades
    assert first_equity == second_equity
    assert first_details == second_details


def test_replay_speed_does_not_change_trade_outcomes(monkeypatch, tmp_path: Path) -> None:
    main_module = _load_main(monkeypatch, tmp_path / "speed1", replay_speed="1")
    first_trades, first_equity, first_details = _run_and_collect(main_module)

    main_module = _load_main(monkeypatch, tmp_path / "speed5", replay_speed="5")
    second_trades, second_equity, second_details = _run_and_collect(main_module)

    assert first_trades == second_trades
    assert first_equity == second_equity
    assert first_details == second_details
