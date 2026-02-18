from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.config import Settings
from app.models import Direction, Posture, Status, TradePlan
from app.providers.binance import BinanceCandle, BinanceKlineSnapshot
from app.services import scheduler as scheduler_module
from app.services.scheduler import DecisionScheduler, classify_scalp_regime, is_scalp_setup_confirmed, scalp_setup_gate_reason
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




def test_force_run_bypasses_dedupe_without_force_trade_mode(monkeypatch) -> None:
    snapshot = _build_snapshot(close_time_ms=8_000_000, closed=True)
    calls = {"decide": 0}

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        calls["decide"] += 1
        return _trade_plan().model_copy(update={"rationale": ["qualified_trade"]})

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(force_trade_mode=False, smoke_test_force_trade=False, telegram_enabled=False, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        first = await scheduler.run_once(force=False)
        assert first[0]["reason"] is None
        second = await scheduler.run_once(force=True)
        assert second[0]["reason"] is None
        assert calls["decide"] == 2
        assert "force_trade_mode" not in (second[0]["plan"].rationale if second[0]["plan"] else [])

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




def test_scheduler_uses_cached_snapshot_before_next_close(monkeypatch) -> None:
    from time import time
    close_time_ms = int(time() // 300 * 300 * 1000)
    snapshot = _build_snapshot(close_time_ms=close_time_ms, closed=True)
    calls = {"fetch": 0, "decide": 0}

    async def fake_fetch(*args, **kwargs):
        calls["fetch"] += 1
        return snapshot

    def fake_decide(*args, **kwargs):
        calls["decide"] += 1
        return _trade_plan().model_copy(update={"status": Status.NO_TRADE, "rationale": ["no_valid_setup"]})

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(telegram_enabled=False, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        await scheduler.run_once(force=False)
        await scheduler.run_once(force=False)
        assert calls["fetch"] == 1
        assert calls["decide"] == 1

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
        assert results[0]["final_entry_gate"] == "atr_too_low"
        assert state.get_decision_meta("BTCUSDT").get("final_entry_gate") == "atr_too_low"

    asyncio.run(run())


def test_scheduler_blocks_stale_market_data(monkeypatch) -> None:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    snapshot = _build_snapshot(close_time_ms=now_ms - 700_000, closed=True)

    async def fake_fetch(*args, **kwargs):
        return snapshot

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)

    settings = Settings(telegram_enabled=False, market_data_allow_stale=1, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run() -> None:
        results = await scheduler.run_once()
        assert results[0]["reason"] == "MARKET_DATA_STALE"
        assert results[0]["trade_opened"] is False
        assert state.get_decision_meta("BTCUSDT").get("market_data_status") == "STALE"

    asyncio.run(run())


def test_scheduler_can_process_open_candle_when_close_confirm_disabled(monkeypatch) -> None:
    snapshot = _build_snapshot(close_time_ms=9_000_000, closed=False)

    async def fake_fetch(*args, **kwargs):
        return snapshot

    def fake_decide(*args, **kwargs):
        return _trade_plan()

    monkeypatch.setattr(scheduler_module, "fetch_symbol_klines", fake_fetch)
    monkeypatch.setattr(scheduler_module, "decide", fake_decide)

    settings = Settings(telegram_enabled=False, require_candle_close_confirm=False, _env_file=None)
    state = StateStore()
    scheduler = DecisionScheduler(settings, state)

    async def run():
        results = await scheduler.run_once()
        assert results[0]["plan"] is not None
        assert results[0]["reason"] is None

    asyncio.run(run())


def test_scalp_pullback_min_depth_gate_blocks_and_allows():
    closes = [100 + (i * 0.2) for i in range(40)]
    snapshot = _snapshot_from_closes(closes)
    prev = snapshot.candles[-2]
    prev.open = prev.close + 0.12
    prev.close = prev.close - 0.10
    prev.high = prev.close + 0.02
    current = snapshot.candles[-1]
    current.open = prev.close - 0.02
    current.close = prev.high + 0.03
    current.high = current.close + 0.03
    current.low = current.open - 0.03
    snapshot.price = current.close
    snapshot.candle = current

    block_settings = Settings(
        current_mode="SCALP",
        scalp_rsi_confirm=False,
        scalp_pullback_max_dist_pct=0.05,
        scalp_pullback_min_dist_pct=0.05,
        adx_threshold=0,
        _env_file=None,
    )
    allow_settings = block_settings.model_copy(update={"scalp_pullback_min_dist_pct": 0.0})

    assert scalp_setup_gate_reason(snapshot, block_settings, Direction.long) == (False, "pullback_too_shallow")
    assert is_scalp_setup_confirmed(snapshot, allow_settings, Direction.long) is True


def test_scalp_rsi_exhaustion_caps_block_long_and_short():
    long_snapshot = _snapshot_from_closes([100 + (i * 0.3) for i in range(50)])
    prev = long_snapshot.candles[-2]
    prev.open = prev.close + 0.1
    prev.close = prev.close - 0.05
    long_cur = long_snapshot.candles[-1]
    long_cur.open = prev.close - 0.08
    long_cur.close = prev.high + 0.12
    long_snapshot.candle = long_cur
    long_snapshot.price = long_cur.close

    long_settings = Settings(
        current_mode="SCALP",
        scalp_rsi_confirm=True,
        scalp_pullback_max_dist_pct=0.05,
        scalp_rsi_long_max=60,
        adx_threshold=0,
        _env_file=None,
    )

    assert scalp_setup_gate_reason(long_snapshot, long_settings, Direction.long) == (False, "rsi_exhausted_long")

    short_snapshot = _snapshot_from_closes([200 - (i * 0.3) for i in range(50)])
    prev_short = short_snapshot.candles[-2]
    prev_short.open = prev_short.close - 0.1
    prev_short.close = prev_short.close + 0.05
    short_cur = short_snapshot.candles[-1]
    short_cur.open = prev_short.close + 0.08
    short_cur.close = prev_short.low - 0.12
    short_snapshot.candle = short_cur
    short_snapshot.price = short_cur.close

    short_settings = Settings(
        current_mode="SCALP",
        scalp_rsi_confirm=True,
        scalp_pullback_max_dist_pct=0.05,
        scalp_rsi_short_min=40,
        adx_threshold=0,
        _env_file=None,
    )

    assert scalp_setup_gate_reason(short_snapshot, short_settings, Direction.short) == (False, "rsi_exhausted_short")


def test_htf_bias_gate_blocks_misaligned_and_allows_aligned(monkeypatch):
    long_plan = _trade_plan().model_copy(update={"direction": Direction.long, "signal_score": 90})

    down_snapshot = _snapshot_from_closes([300 - (i * 0.5) for i in range(300)])
    up_snapshot = _snapshot_from_closes([100 + (i * 0.5) for i in range(300)])

    monkeypatch.setattr(scheduler_module, "is_scalp_setup_confirmed", lambda *args, **kwargs: True)

    settings = Settings(
        current_mode="SCALP",
        scalp_regime_enabled=False,
        scalp_trend_filter_enabled=False,
        scalp_rsi_confirm=False,
        htf_bias_enabled=True,
        htf_interval="1h",
        htf_ema_fast=10,
        htf_ema_slow=20,
        _env_file=None,
    )

    blocked_plan, blocked_reason, blocked_meta = scheduler_module._apply_mode_overrides(long_plan, down_snapshot, settings)
    allowed_plan, allowed_reason, allowed_meta = scheduler_module._apply_mode_overrides(long_plan, up_snapshot, settings)

    assert blocked_plan.status == Status.NO_TRADE
    assert blocked_reason == "htf_bias_reject"
    assert blocked_meta["htf_bias_reject"] is True

    assert allowed_plan.status == Status.TRADE
    assert allowed_reason is None
    assert allowed_meta["htf_bias_reject"] is False


def test_trigger_quality_filter_rejects_weak_candle(monkeypatch):
    snapshot = _snapshot_from_closes([100 + (i * 0.2) for i in range(80)])
    current = snapshot.candle
    current.open = current.close - 0.02
    current.high = current.close + 0.20
    current.low = current.close - 0.20

    monkeypatch.setattr(scheduler_module, "is_scalp_setup_confirmed", lambda *args, **kwargs: True)

    settings = Settings(
        current_mode="SCALP",
        scalp_regime_enabled=False,
        scalp_trend_filter_enabled=False,
        trigger_body_ratio_min=0.4,
        trigger_close_location_min=0.9,
        _env_file=None,
    )

    _, reason, meta = scheduler_module._apply_mode_overrides(_trade_plan(), snapshot, settings)

    assert reason in {"trigger_body_ratio_reject", "trigger_close_location_reject"}
    assert meta["trigger_body_ratio_reject"] or meta["trigger_close_location_reject"]


def test_new_filters_disabled_preserve_default_trade_path(monkeypatch):
    snapshot = _snapshot_from_closes([100 + (i * 0.2) for i in range(80)])
    monkeypatch.setattr(scheduler_module, "is_scalp_setup_confirmed", lambda *args, **kwargs: True)

    settings = Settings(
        current_mode="SCALP",
        scalp_regime_enabled=False,
        scalp_trend_filter_enabled=False,
        htf_bias_enabled=False,
        trigger_body_ratio_min=0.0,
        trigger_close_location_min=0.0,
        _env_file=None,
    )

    plan, reason, meta = scheduler_module._apply_mode_overrides(_trade_plan(), snapshot, settings)
    assert plan.status == Status.TRADE
    assert reason is None
    assert meta["htf_bias_reject"] is False
    assert meta["trigger_body_ratio_reject"] is False
    assert meta["trigger_close_location_reject"] is False


def test_tick_listener_error_logs_are_rate_limited(monkeypatch, caplog):
    settings = Settings(symbols=["BTCUSDT"], _env_file=None)
    scheduler = DecisionScheduler(settings, StateStore())

    def broken_listener() -> None:
        raise RuntimeError("boom")

    scheduler.add_tick_listener(broken_listener)

    values = iter([0.0, 1.0, 12.0])
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: next(values))

    caplog.set_level("WARNING")
    scheduler._notify_tick_listeners()
    scheduler._notify_tick_listeners()
    scheduler._notify_tick_listeners()

    full_errors = [record for record in caplog.records if record.message.startswith("scheduler_tick_listener_error error=")]
    suppressed = [record for record in caplog.records if "scheduler_tick_listener_error_suppressed" in record.message]

    assert len(full_errors) == 2
    assert len(suppressed) == 1
