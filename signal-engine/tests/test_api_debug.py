from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.models import Direction, Posture, Status, TradePlan
from app.providers.binance import BinanceCandle, BinanceKlineSnapshot
from app.services import scheduler as scheduler_module


def _build_snapshot(close_time_ms: int, closed: bool) -> BinanceKlineSnapshot:
    open_time = datetime.fromtimestamp((close_time_ms - 300_000) / 1000, tz=timezone.utc)
    close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
    candle = BinanceCandle(
        open_time=open_time,
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=123.0,
        close_time=close_time,
    )
    return BinanceKlineSnapshot(
        symbol="BTCUSDT",
        interval="5m",
        price=candle.close,
        volume=candle.volume,
        kline_open_time_ms=close_time_ms - 300_000,
        kline_close_time_ms=close_time_ms,
        kline_is_closed=closed,
        candle=candle,
        candles=[candle, candle, candle, candle, candle],
    )


def _trade_plan() -> TradePlan:
    return TradePlan(
        status=Status.NO_TRADE,
        direction=Direction.long,
        entry_zone=None,
        stop_loss=None,
        take_profit=None,
        risk_pct_used=None,
        position_size_usd=None,
        signal_score=None,
        posture=Posture.NORMAL,
        rationale=["waiting"],
        raw_input_snapshot={},
    )


def test_decision_persists_after_run_force(monkeypatch) -> None:
    from app import main as main_module

    snapshot = _build_snapshot(close_time_ms=2_000_000, closed=True)

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    with TestClient(main_module.app) as client:
        response = client.get("/run", params={"force": "true"})
        assert response.status_code == 200

        decision_response = client.get("/decision/latest", params={"symbol": "BTCUSDT"})
        assert decision_response.status_code == 200
        payload = decision_response.json()
        assert payload["symbol"] == "BTCUSDT"
        assert payload["decision"]["status"] in {"NO_TRADE", "RISK_OFF", "TRADE"}


def test_telegram_test_endpoint(monkeypatch) -> None:
    from app import main as main_module

    async def fake_send(*args, **kwargs):
        return True

    monkeypatch.setattr(main_module, "send_telegram_message", fake_send)
    with TestClient(main_module.app) as client:
        response = client.get("/test/telegram")
        assert response.status_code == 200
        assert response.json()["status"] == "sent"


def _fresh_main(monkeypatch, env: dict[str, str]):
    from importlib import reload

    from app import config as config_module
    from app import main as main_module

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config_module.get_settings.cache_clear()
    return reload(main_module)


def test_debug_runtime_includes_profile(monkeypatch) -> None:
    main_module = _fresh_main(monkeypatch, {"PROFILE": "diag", "MIN_SIGNAL_SCORE": "35"})
    with TestClient(main_module.app) as client:
        response = client.get("/debug/runtime")
        assert response.status_code == 200
        payload = response.json()
        assert payload["settings"]["profile"] == "diag"
        assert payload["settings"]["min_signal_score"] == 35


def test_force_signal_respects_hard_gates(monkeypatch) -> None:
    main_module = _fresh_main(
        monkeypatch,
        {"DEBUG_LOOSEN": "true", "ACCOUNT_SIZE": "1000", "MAX_DAILY_LOSS_PCT": "0.01"},
    )
    snapshot = _build_snapshot(close_time_ms=2_000_000, closed=True)

    with TestClient(main_module.app) as client:
        scheduler = client.app.state.scheduler
        state_store = client.app.state.state_store
        monkeypatch.setattr(scheduler, "last_snapshot", lambda symbol: snapshot)
        daily_state = state_store.get_daily_state("BTCUSDT")
        daily_state.pnl_usd = -20.0

        response = client.post(
            "/debug/force_signal",
            json={"symbol": "BTCUSDT", "direction": "long", "bypass_soft_gates": True},
        )
        assert response.status_code == 200
        decision = response.json()["decision"]
        assert decision["status"] == "RISK_OFF"
        assert "daily_loss_limit_hit" in decision["rationale"]


def test_scheduler_state_persists_across_requests(monkeypatch) -> None:
    from app import main as main_module

    snapshot = _build_snapshot(close_time_ms=5_000_000, closed=True)

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    with TestClient(main_module.app) as client:
        response = client.get("/engine/run_once", params={"force": "true"})
        assert response.status_code == 200

        debug = client.get("/debug/runtime")
        assert debug.status_code == 200
        payload = debug.json()
        assert payload["scheduler"]["last_tick_time"] is not None
        symbol = payload["symbols"][0]
        assert symbol["candles_fetched_count"] == len(snapshot.candles)


def test_symbols_endpoint_normalizes_values() -> None:
    from app import main as main_module

    with TestClient(main_module.app) as client:
        response = client.post("/symbols", json={"symbols": [" btcusdt ", "Ethusdt"]})
        assert response.status_code == 200
        assert response.json()["symbols"] == ["BTCUSDT", "ETHUSDT"]


def test_dashboard_overview_contract() -> None:
    from app import main as main_module

    with TestClient(main_module.app) as client:
        response = client.get("/dashboard/overview")
        assert response.status_code == 200
        payload = response.json()
        assert "account" in payload
        assert "risk" in payload
        assert "activity" in payload
        assert "symbols" in payload
        assert "recent_trades" in payload


def test_kill_switch_endpoint() -> None:
    from app import main as main_module

    with TestClient(main_module.app) as client:
        response = client.post("/engine/kill-switch", params={"enabled": "true"})
        assert response.status_code == 200
        assert response.json()["manual_kill_switch"] is True


def test_debug_kline_includes_provider(monkeypatch) -> None:
    from app import main as main_module
    from app.providers.bybit import BybitCandle, BybitKlineSnapshot

    async def fake_fetch(**kwargs):
        candle = BybitCandle(
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=5.0,
        )
        return BybitKlineSnapshot(
            symbol=kwargs["symbol"],
            interval=kwargs.get("interval", "1m"),
            price=100.5,
            volume=5.0,
            kline_open_time_ms=1,
            kline_close_time_ms=2,
            kline_is_closed=True,
            candle=candle,
            candles=[candle],
            provider_name="binance",
            provider_category="spot",
            provider_endpoint="/api/v3/klines",
        )

    monkeypatch.setattr(main_module, "fetch_symbol_klines", fake_fetch)
    with TestClient(main_module.app) as client:
        response = client.get("/debug/kline", params={"interval": "1m"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["symbols"][0]["provider"] in {"bybit", "binance"}


def test_debug_config_sources(monkeypatch) -> None:
    main_module = _fresh_main(monkeypatch, {"ACCOUNT_SIZE": "9999", "STRATEGY_PROFILE": "SCALPER_STABLE"})
    with TestClient(main_module.app) as client:
        response = client.get("/debug/config")
        assert response.status_code == 200
        payload = response.json()
        assert payload["effective"]["account_size"] == 9999.0
        assert payload["sources"]["account_size"] == "env"
        assert payload["env_keys"]["account_size"] == "ACCOUNT_SIZE"


def test_debug_db_endpoint() -> None:
    from app import main as main_module

    with TestClient(main_module.app) as client:
        response = client.get("/debug/db")
        assert response.status_code == 200
        payload = response.json()
        assert payload["connected"] is True
        assert "database_type" in payload


def test_state_snapshot_survives_missing_challenge_service() -> None:
    from app import main as main_module

    with TestClient(main_module.app) as client:
        original_challenge_service = main_module.challenge_service
        try:
            main_module.challenge_service = None
            response = client.get("/state")
            assert response.status_code == 200
            payload = response.json()
            assert payload["challenge"] is None
            assert payload["challenge_ready"] is False
            assert payload["challenge_error"] == "engine_not_ready"
        finally:
            main_module.challenge_service = original_challenge_service


def test_challenge_service_initialized_on_startup() -> None:
    from app import main as main_module

    with TestClient(main_module.app) as client:
        response = client.get("/challenge/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in {"IN_PROGRESS", "PASSED", "FAILED"}

        state_response = client.get("/state")
        assert state_response.status_code == 200
        state_payload = state_response.json()
        assert state_payload["challenge_ready"] is True
        assert isinstance(state_payload["challenge"], dict)
        assert state_payload["challenge_error"] is None


def test_replay_run_once_with_prop_enabled_does_not_crash_when_challenge_unavailable(monkeypatch) -> None:
    main_module = _fresh_main(monkeypatch, {"RUN_MODE": "replay", "PROP_ENABLED": "true"})

    with TestClient(main_module.app) as client:
        async def fake_run_once(*args, **kwargs):
            return []

        monkeypatch.setattr(client.app.state.scheduler, "run_once", fake_run_once)
        original_challenge_service = main_module.challenge_service
        try:
            main_module.challenge_service = None
            response = client.get("/engine/run_once", params={"force": "true"})
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

            state_response = client.get("/state")
            assert state_response.status_code == 200
            payload = state_response.json()
            assert payload["challenge_ready"] is False
        finally:
            main_module.challenge_service = original_challenge_service
