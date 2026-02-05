from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.models import Direction, Posture, Status, TradePlan
from app.providers.binance import BinanceCandle, BinanceKlineSnapshot
from app.services import scheduler as scheduler_module
from app.services.database import Database, TradeRecord
from app.services.paper_trader import PaperTrader
from app.services.scheduler import DecisionScheduler
from app.services.stats import compute_stats
from app.state import StateStore


def _build_snapshot(symbol: str, close_time_ms: int, closed: bool, price: float) -> BinanceKlineSnapshot:
    open_time = datetime.fromtimestamp((close_time_ms - 900_000) / 1000, tz=timezone.utc)
    close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
    candle = BinanceCandle(
        open_time=open_time,
        open=price - 5,
        high=price + 5,
        low=price - 10,
        close=price,
        volume=123.0,
        close_time=close_time,
    )
    return BinanceKlineSnapshot(
        symbol=symbol,
        interval="15m",
        price=candle.close,
        volume=candle.volume,
        kline_open_time_ms=close_time_ms - 900_000,
        kline_close_time_ms=close_time_ms,
        kline_is_closed=closed,
        candle=candle,
    )


def _trade_plan() -> TradePlan:
    return TradePlan(
        status=Status.TRADE,
        direction=Direction.long,
        entry_zone=(100.0, 101.0),
        stop_loss=95.0,
        take_profit=110.0,
        risk_pct_used=0.01,
        position_size_usd=100.0,
        signal_score=80,
        posture=Posture.NORMAL,
        rationale=["qualified_trade"],
        raw_input_snapshot={},
    )


def _build_scheduler(tmp_path: Path, settings: Settings, state: StateStore) -> DecisionScheduler:
    settings.data_dir = str(tmp_path)
    database = Database(settings)
    paper_trader = PaperTrader(settings, database)
    return DecisionScheduler(settings, state, database, paper_trader)


def test_scheduler_per_symbol_dedupe(monkeypatch, tmp_path) -> None:
    snapshots = {
        "BTCUSDT": _build_snapshot("BTCUSDT", 1_000_000, True, 100.0),
        "ETHUSDT": _build_snapshot("ETHUSDT", 2_000_000, True, 200.0),
    }

    async def fake_fetch(symbol: str, interval: str = "15m"):
        return snapshots[symbol]

    call_count = {"value": 0}

    def fake_decide(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return _trade_plan()
        return TradePlan(
            status=Status.NO_TRADE,
            direction=Direction.long,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=50,
            posture=Posture.NORMAL,
            rationale=["no_trade"],
            raw_input_snapshot={},
        )

    monkeypatch.setattr(scheduler_module, "fetch_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(telegram_enabled=False, symbols=["BTCUSDT", "ETHUSDT"])
    state = StateStore()
    state.set_symbols(settings.symbols)
    state.set_last_processed_close_time_ms("BTCUSDT", 1_000_000)
    scheduler = _build_scheduler(tmp_path, settings, state)

    async def run():
        results = await scheduler.run_once()
        assert results["BTCUSDT"]["status"] == "skipped"
        assert results["ETHUSDT"]["status"] == "ok"

    asyncio.run(run())


def test_scheduler_dedupes_notifications(monkeypatch, tmp_path) -> None:
    snapshot = _build_snapshot("BTCUSDT", 3_000_000, True, 105.0)
    sent = {"count": 0}

    async def fake_fetch(symbol: str, interval: str = "15m"):
        return snapshot

    call_count = {"value": 0}

    def fake_decide(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return _trade_plan()
        return TradePlan(
            status=Status.NO_TRADE,
            direction=Direction.long,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=50,
            posture=Posture.NORMAL,
            rationale=["no_trade"],
            raw_input_snapshot={},
        )

    async def fake_send(message: str, settings: Settings):
        sent["count"] += 1
        return True

    monkeypatch.setattr(scheduler_module, "fetch_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)
    monkeypatch.setattr(scheduler_module, "send_telegram_message", fake_send)

    settings = Settings(
        telegram_enabled=True,
        telegram_bot_token="token",
        telegram_chat_id="chat",
        symbols=["BTCUSDT"],
    )
    state = StateStore()
    state.set_symbols(settings.symbols)
    scheduler = _build_scheduler(tmp_path, settings, state)

    async def run():
        await scheduler.run_once(force=True)
        await scheduler.run_once(force=True)
        assert sent["count"] == 1

    asyncio.run(run())


def test_paper_trade_open_close(monkeypatch, tmp_path) -> None:
    snapshots = [
        _build_snapshot("BTCUSDT", 4_000_000, True, 100.0),
        _build_snapshot("BTCUSDT", 4_900_000, True, 110.0),
    ]
    index = {"value": 0}

    async def fake_fetch(symbol: str, interval: str = "15m"):
        snap = snapshots[index["value"]]
        index["value"] += 1
        return snap

    call_count = {"value": 0}

    def fake_decide(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return _trade_plan()
        return TradePlan(
            status=Status.NO_TRADE,
            direction=Direction.long,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_pct_used=None,
            position_size_usd=None,
            signal_score=50,
            posture=Posture.NORMAL,
            rationale=["no_trade"],
            raw_input_snapshot={},
        )

    monkeypatch.setattr(scheduler_module, "fetch_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(symbols=["BTCUSDT"], telegram_enabled=False, engine_mode="paper")
    state = StateStore()
    state.set_symbols(settings.symbols)
    scheduler = _build_scheduler(tmp_path, settings, state)

    async def run():
        await scheduler.run_once(force=True)
        await scheduler.run_once(force=True)
        trades = scheduler._database.fetch_trades()
        assert len(trades) == 1
        assert trades[0].closed_at is not None
        assert trades[0].result == "win"

    asyncio.run(run())


def test_stats_calculation() -> None:
    trades = [
        TradeRecord(
            id=1,
            symbol="BTCUSDT",
            entry=100.0,
            exit=110.0,
            stop=95.0,
            take_profit=110.0,
            size=100.0,
            pnl_usd=10.0,
            pnl_r=1.0,
            side="long",
            opened_at="2024-01-01T00:00:00Z",
            closed_at="2024-01-01T01:00:00Z",
            result="win",
        ),
        TradeRecord(
            id=2,
            symbol="BTCUSDT",
            entry=100.0,
            exit=90.0,
            stop=95.0,
            take_profit=110.0,
            size=100.0,
            pnl_usd=-10.0,
            pnl_r=-1.0,
            side="long",
            opened_at="2024-01-02T00:00:00Z",
            closed_at="2024-01-02T01:00:00Z",
            result="loss",
        ),
    ]
    summary = compute_stats(trades)
    assert summary.total_trades == 2
    assert summary.wins == 1
    assert summary.losses == 1
    assert summary.total_pnl == 0.0
