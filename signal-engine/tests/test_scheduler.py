from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.config import Settings
from app.models import Direction, Posture, Status, TradePlan
from app.providers.binance import BinanceCandle, BinanceKlineSnapshot
from app.services import scheduler as scheduler_module
from app.services.scheduler import DecisionScheduler
from app.services.database import Database
from app.services.paper_trader import PaperTrader
from app.state import StateStore


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
        candles=[candle],
    )


def _trade_plan() -> TradePlan:
    return TradePlan(
        status=Status.TRADE,
        direction=Direction.long,
        entry_zone=(100.0, 101.0),
        stop_loss=98.0,
        take_profit=110.0,
        risk_pct_used=0.01,
        position_size_usd=100.0,
        signal_score=80,
        posture=Posture.NORMAL,
        rationale=["qualified_trade"],
        raw_input_snapshot={},
    )


def test_scheduler_skips_open_candle(monkeypatch) -> None:
    snapshot = _build_snapshot(close_time_ms=1_000_000, closed=False)

    async def fake_fetch(*args, **kwargs):
        return snapshot

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)

    settings = Settings(telegram_enabled=False, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        results = await scheduler.run_once()
        assert results[0]["plan"] is None
        assert results[0]["reason"] == "candle_open"
        assert state.get_last_processed_close_time_ms("BTCUSDT") is None

    asyncio.run(run())


def test_scheduler_runs_once_per_candle(monkeypatch) -> None:
    snapshot = _build_snapshot(close_time_ms=2_000_000, closed=True)
    calls = {"decide": 0}

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        calls["decide"] += 1
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(telegram_enabled=False, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        results = await scheduler.run_once()
        assert results[0]["plan"] is not None
        assert results[0]["reason"] is None
        assert state.get_last_processed_close_time_ms("BTCUSDT") == snapshot.kline_close_time_ms
        results = await scheduler.run_once()
        assert results[0]["plan"] is None
        assert results[0]["reason"] == "candle_already_processed"
        assert calls["decide"] == 1

    asyncio.run(run())


def test_scheduler_dedupes_notifications(monkeypatch) -> None:
    snapshot = _build_snapshot(close_time_ms=3_000_000, closed=True)
    sent = {"count": 0}

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    async def fake_send(message: str, settings: Settings):
        sent["count"] += 1
        return True

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)
    monkeypatch.setattr(scheduler_module, "send_telegram_message", fake_send)

    settings = Settings(
        telegram_enabled=True,
        telegram_bot_token="token",
        telegram_chat_id="chat",
        _env_file=None,
    )
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        await scheduler.run_once(force=True)
        await scheduler.run_once(force=True)
        assert sent["count"] == 1

    asyncio.run(run())


def test_scheduler_trade_pipeline_creates_paper_trade(monkeypatch, tmp_path) -> None:
    snapshot = _build_snapshot(close_time_ms=4_000_000, closed=True)
    sent = {"count": 0}

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    async def fake_send(message: str, settings: Settings):
        sent["count"] += 1
        return True

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)
    monkeypatch.setattr(scheduler_module, "send_telegram_message", fake_send)

    settings = Settings(
        MODE="paper",
        telegram_enabled=True,
        telegram_bot_token="token",
        telegram_chat_id="chat",
        data_dir=str(tmp_path),
        _env_file=None,
    )
    state = StateStore()
    database = Database(settings)
    paper_trader = PaperTrader(settings, database)
    scheduler = DecisionScheduler(settings, state, database=database, paper_trader=paper_trader)

    async def run():
        await scheduler.run_once(force=True)
        assert sent["count"] == 1
        assert len(database.fetch_trades()) == 1
        assert len(database.fetch_open_trades()) == 1

    asyncio.run(run())


def test_scheduler_dedupes_trades_per_candle(monkeypatch, tmp_path) -> None:
    snapshot = _build_snapshot(close_time_ms=5_000_000, closed=True)

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(
        MODE="paper",
        data_dir=str(tmp_path),
        _env_file=None,
    )
    state = StateStore()
    database = Database(settings)
    paper_trader = PaperTrader(settings, database)
    scheduler = DecisionScheduler(settings, state, database=database, paper_trader=paper_trader)

    async def run():
        await scheduler.run_once(force=True)
        await scheduler.run_once(force=True)
        assert len(database.fetch_trades()) == 1

    asyncio.run(run())
