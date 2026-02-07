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

    client = TestClient(main_module.app)
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
    client = TestClient(main_module.app)
    response = client.get("/test/telegram")
    assert response.status_code == 200
    assert response.json()["status"] == "sent"
