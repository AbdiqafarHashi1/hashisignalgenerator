from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.config import Settings
from app.models import Direction, Posture, Status, TradePlan
from app.providers.binance import BinanceCandle, BinanceKlineSnapshot
from app.services import scheduler as scheduler_module
from app.services.scheduler import DecisionScheduler, classify_scalp_regime, is_scalp_setup_confirmed
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


def test_scheduler_waits_when_provider_not_ready_then_processes_closed_candle(monkeypatch) -> None:
    snapshot = _build_snapshot(close_time_ms=7_000_000, closed=True)
    calls = {"fetch": 0, "decide": 0}

    async def fake_fetch(*args, **kwargs):
        calls["fetch"] += 1
        if calls["fetch"] == 1:
            return None
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
        first = await scheduler.run_once()
        assert first[0]["plan"] is None
        assert first[0]["reason"] == "candle_open"

        second = await scheduler.run_once()
        assert second[0]["plan"] is not None
        assert second[0]["reason"] is None
        assert calls["decide"] == 1

    asyncio.run(run())



def test_scheduler_logs_transition_from_candle_open_to_ready(monkeypatch, caplog) -> None:
    snapshot = _build_snapshot(close_time_ms=7_000_000, closed=True)
    calls = {"fetch": 0}

    async def fake_fetch(*args, **kwargs):
        calls["fetch"] += 1
        if calls["fetch"] == 1:
            return None
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(telegram_enabled=False, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        await scheduler.run_once()
        await scheduler.run_once()

    caplog.set_level("INFO")
    asyncio.run(run())

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "status=not_ready reason=candle_open" in logs
    assert "candle_fetch symbol=BTCUSDT candles=" in logs


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
        force_trade_mode=False,
        smoke_test_force_trade=False,
        _env_file=None,
    )
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        await scheduler.run_once()
        await scheduler.run_once()
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


def test_scheduler_symbol_error_isolated(monkeypatch) -> None:
    snapshot = _build_snapshot(close_time_ms=6_000_000, closed=True)

    async def fake_fetch(*args, **kwargs):
        if kwargs.get("symbol") == "BTCUSDT":
            raise RuntimeError("temporary_data_source_failure")
        return snapshot.model_copy(update={"symbol": "ETHUSDT"})

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)

    settings = Settings(symbols=["BTCUSDT", "ETHUSDT"], telegram_enabled=False, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        results = await scheduler.run_once()
        by_symbol = {item["symbol"]: item for item in results}
        assert by_symbol["BTCUSDT"]["reason"] == "symbol_error:RuntimeError"
        assert by_symbol["ETHUSDT"]["reason"] in {"candle_already_processed", None}

    asyncio.run(run())


def test_scheduler_loop_survives_tick_exception(monkeypatch) -> None:
    settings = Settings(telegram_enabled=False, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state, interval_seconds=1)
    calls = {"count": 0}

    async def fake_run_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("network_timeout")
        scheduler._stop_event.set()
        return []

    monkeypatch.setattr(scheduler, "run_once", fake_run_once)

    async def run():
        started = await scheduler.start()
        assert started is True
        await asyncio.wait_for(scheduler._task, timeout=3)
        assert calls["count"] >= 2
        assert scheduler.stop_reason == "TimeoutError: network_timeout"

    asyncio.run(run())


def test_scheduler_time_stop_closes_old_trade(monkeypatch, tmp_path) -> None:
    snapshot = _build_snapshot(close_time_ms=8_000_000, closed=True)
    snapshot.candle.close = 101.0
    snapshot.candle.high = 101.2
    snapshot.candle.low = 100.8
    snapshot.price = 101.0

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(MODE="paper", data_dir=str(tmp_path), max_hold_minutes=1, _env_file=None)
    state = StateStore()
    database = Database(settings)
    paper_trader = PaperTrader(settings, database)
    scheduler = DecisionScheduler(settings, state, database=database, paper_trader=paper_trader)

    trade_id = paper_trader.maybe_open_trade("BTCUSDT", _trade_plan(), allow_multiple=True)
    assert trade_id is not None
    with database._conn:
        database._conn.execute("UPDATE trades SET opened_at = ? WHERE id = ?", ("2000-01-01T00:00:00+00:00", trade_id))

    async def run():
        await scheduler.run_once(force=True)
        closed = [t for t in database.fetch_trades() if t.id == trade_id][0]
        assert closed.closed_at is not None
        assert closed.result == "time_stop_close"

    asyncio.run(run())


def test_funding_blackout_blocks_entries_without_auto_close(monkeypatch, tmp_path) -> None:
    snapshot = _build_snapshot(close_time_ms=9_000_000, closed=True)
    snapshot.candle.close = 101.0
    snapshot.candle.high = 101.2
    snapshot.candle.low = 100.8
    snapshot.price = 101.0

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)
    monkeypatch.setattr(
        scheduler_module,
        "_funding_blackout_state",
        lambda *args, **kwargs: {"block_new_entries": True, "close_positions": True},
    )

    settings = Settings(MODE="paper", data_dir=str(tmp_path), funding_blackout_force_close=False, _env_file=None)
    state = StateStore()
    database = Database(settings)
    paper_trader = PaperTrader(settings, database)
    scheduler = DecisionScheduler(settings, state, database=database, paper_trader=paper_trader)

    trade_id = paper_trader.maybe_open_trade("BTCUSDT", _trade_plan(), allow_multiple=True)
    assert trade_id is not None

    async def run():
        results = await scheduler.run_once(force=True)
        assert results[0]["reason"] == "funding_blackout_entries_blocked"
        still_open = [t for t in database.fetch_open_trades() if t.id == trade_id]
        assert len(still_open) == 1

    asyncio.run(run())


def test_scalp_mode_overrides_tp_sl_from_executed_entry(monkeypatch, tmp_path) -> None:
    snapshot = _build_snapshot(close_time_ms=10_000_000, closed=True)
    snapshot.candle.close = 2000.0
    snapshot.candle.high = 2002.0
    snapshot.candle.low = 1998.0

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan().model_copy(update={"entry_zone": (1500.0, 1500.0), "stop_loss": 1490.0, "take_profit": 1520.0, "signal_score": 90})

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)
    monkeypatch.setattr(scheduler_module, "is_scalp_setup_confirmed", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        scheduler_module,
        "classify_scalp_regime",
        lambda *args, **kwargs: {"regime_label": "bull", "allowed_side": "long", "atr_pct": 0.003, "ema_fast": 1.0, "ema_slow": 1.0, "ema_trend": 1.0},
    )
    monkeypatch.setattr(
        scheduler_module,
        "_funding_blackout_state",
        lambda *args, **kwargs: {"block_new_entries": False, "close_positions": False},
    )

    settings = Settings(MODE="paper", data_dir=str(tmp_path), current_mode="SCALP", _env_file=None)
    state = StateStore()
    database = Database(settings)
    paper_trader = PaperTrader(settings, database)
    scheduler = DecisionScheduler(settings, state, database=database, paper_trader=paper_trader)

    async def run():
        await scheduler.run_once(force=False)
        open_trade = database.fetch_open_trades("BTCUSDT")[0]
        expected_tp = 2000.0 * (1 + settings.scalp_tp_pct)
        expected_sl = 2000.0 * (1 - settings.scalp_sl_pct)
        assert round(open_trade.take_profit, 4) == round(expected_tp, 4)
        assert round(open_trade.stop, 4) == round(expected_sl, 4)

    asyncio.run(run())


def test_scalp_time_stop_uses_scalp_max_hold_minutes(monkeypatch, tmp_path) -> None:
    snapshot = _build_snapshot(close_time_ms=11_000_000, closed=True)
    snapshot.candle.high = 101.2
    snapshot.candle.low = 100.8

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)
    monkeypatch.setattr(
        scheduler_module,
        "_funding_blackout_state",
        lambda *args, **kwargs: {"block_new_entries": False, "close_positions": False},
    )

    settings = Settings(MODE="paper", data_dir=str(tmp_path), current_mode="SCALP", scalp_max_hold_minutes=1, _env_file=None)
    state = StateStore()
    database = Database(settings)
    paper_trader = PaperTrader(settings, database)
    scheduler = DecisionScheduler(settings, state, database=database, paper_trader=paper_trader)

    trade_id = paper_trader.maybe_open_trade("BTCUSDT", _trade_plan(), allow_multiple=True)
    assert trade_id is not None
    with database._conn:
        database._conn.execute("UPDATE trades SET opened_at = ? WHERE id = ?", ("2000-01-01T00:00:00+00:00", trade_id))

    async def run():
        await scheduler.run_once(force=False)
        closed = [t for t in database.fetch_trades() if t.id == trade_id][0]
        assert closed.result == "time_stop_close"

    asyncio.run(run())


def _snapshot_from_closes(closes: list[float], start_ms: int = 20_000_000) -> BinanceKlineSnapshot:
    candles: list[BinanceCandle] = []
    for idx, close in enumerate(closes):
        open_price = closes[idx - 1] if idx > 0 else close
        high = max(open_price, close) * 1.001
        low = min(open_price, close) * 0.999
        open_time_ms = start_ms + (idx * 300_000)
        close_time_ms = open_time_ms + 300_000
        candles.append(
            BinanceCandle(
                open_time=datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=100.0,
                close_time=datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc),
            )
        )
    last = candles[-1]
    return BinanceKlineSnapshot(
        symbol="BTCUSDT",
        interval="5m",
        price=last.close,
        volume=last.volume,
        kline_open_time_ms=int(last.open_time.timestamp() * 1000),
        kline_close_time_ms=int(last.close_time.timestamp() * 1000),
        kline_is_closed=True,
        candle=last,
        candles=candles,
    )


def test_scalp_regime_bull_allows_long():
    closes = [100 + (i * 0.4) for i in range(240)]
    snapshot = _snapshot_from_closes(closes)
    settings = Settings(current_mode="SCALP", _env_file=None)

    regime = classify_scalp_regime(snapshot, settings)

    assert regime["regime_label"] == "bull"
    assert regime["allowed_side"] == "long"


def test_scalp_regime_chop_blocks_side():
    closes = [100 + ((-1) ** i) * 0.05 for i in range(240)]
    snapshot = _snapshot_from_closes(closes)
    settings = Settings(current_mode="SCALP", _env_file=None)

    regime = classify_scalp_regime(snapshot, settings)

    assert regime["regime_label"] in {"chop", "dead"}
    if regime["regime_label"] == "chop":
        assert regime["allowed_side"] is None


def test_scalp_regime_atr_too_low_blocks():
    closes = [100 + (i * 0.01) for i in range(240)]
    snapshot = _snapshot_from_closes(closes)
    settings = Settings(current_mode="SCALP", scalp_atr_pct_min=0.01, _env_file=None)

    regime = classify_scalp_regime(snapshot, settings)

    assert regime["regime_label"] == "dead"
    assert regime["allowed_side"] is None


def test_scalp_pullback_engulfing_confirmation_true():
    closes = [100 + (i * 0.2) for i in range(40)]
    snapshot = _snapshot_from_closes(closes)
    prev = snapshot.candles[-2]
    prev.open = prev.close + 0.12
    prev.close = prev.close - 0.10
    prev.high = prev.close + 0.02
    current = snapshot.candles[-1]
    current.open = prev.close - 0.20
    current.close = prev.high + 0.20
    current.high = current.close + 0.03
    current.low = current.open - 0.03
    snapshot.price = current.close
    snapshot.candle = current

    settings = Settings(current_mode="SCALP", scalp_rsi_confirm=False, scalp_pullback_max_dist_pct=0.05, _env_file=None)

    assert is_scalp_setup_confirmed(snapshot, settings, Direction.long) is True


def test_scalp_pullback_engulfing_fails_when_too_far_from_ema():
    closes = [100 + (i * 0.2) for i in range(40)]
    snapshot = _snapshot_from_closes(closes)
    current = snapshot.candles[-1]
    current.close = current.close * 1.02
    current.high = current.close * 1.001
    current.low = current.close * 0.999
    snapshot.candle = current
    snapshot.price = current.close

    settings = Settings(
        current_mode="SCALP",
        scalp_rsi_confirm=False,
        scalp_pullback_max_dist_pct=0.0001,
        _env_file=None,
    )

    assert is_scalp_setup_confirmed(snapshot, settings, Direction.long) is False


def test_skip_reason_is_atr_too_low_when_regime_blocks(monkeypatch, tmp_path):
    snapshot = _snapshot_from_closes([100 + (i * 0.01) for i in range(240)], start_ms=50_000_000)

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan().model_copy(update={"signal_score": 95})

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)
    monkeypatch.setattr(
        scheduler_module,
        "_funding_blackout_state",
        lambda *args, **kwargs: {"block_new_entries": False, "close_positions": False},
    )

    settings = Settings(
        MODE="paper",
        data_dir=str(tmp_path),
        current_mode="SCALP",
        scalp_atr_pct_min=0.01,
        scalp_rsi_confirm=False,
        _env_file=None,
    )
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        results = await scheduler.run_once(force=False)
        assert results[0]["decision"] == "skip"
        assert results[0]["skip_reason"] == "atr_too_low"

    asyncio.run(run())
