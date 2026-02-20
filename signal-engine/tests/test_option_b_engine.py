from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import Settings
from app.providers.replay import ReplayProvider
from app.services.database import Database
from app.services.prop_governor import PropRiskGovernor
from signal_engine.accounting import compute_accounting_snapshot
from signal_engine.features import compute_features
from signal_engine.governor.prop import evaluate_prop_block
from signal_engine.regime import Regime, classify_regime
from signal_engine.strategy.entries import evaluate_entries


def _sample_rows(n: int = 320, start: float = 100.0) -> list[dict[str, object]]:
    rows = []
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = start
    for i in range(n):
        price += 0.12 if i % 7 else -0.05
        rows.append({"timestamp": ts + timedelta(minutes=5 * i), "open": price - 0.05, "high": price + 0.6, "low": price - 0.6, "close": price, "volume": 1000 + i})
    return rows


def test_replay_speed_determinism(tmp_path: Path) -> None:
    data_dir = tmp_path / "replay" / "ETHUSDT"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "5m.csv"
    rows = _sample_rows(80)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume,close_time\n")
        for row in rows:
            ts = row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(f"{ts},{row['open']},{row['high']},{row['low']},{row['close']},{row['volume']},{ts}\n")

    p1 = ReplayProvider(str(tmp_path / "replay"))
    p2 = ReplayProvider(str(tmp_path / "replay"))

    async def _run(provider: ReplayProvider, speed: float) -> list[float]:
        closes = []
        for _ in range(15):
            snap = await provider.fetch_symbol_klines("ETHUSDT", "5m", limit=20, speed=speed)
            closes.append(snap.candle.close)
        return closes

    import asyncio

    seq1 = asyncio.run(_run(p1, 1.0))
    seq2 = asyncio.run(_run(p2, 20.0))
    assert seq1 == seq2


def test_day_rollover_resets_governor(tmp_path: Path) -> None:
    cfg = Settings(_env_file=None, data_dir=str(tmp_path), database_url=f"sqlite:///{tmp_path}/t.db")
    db = Database(cfg)
    gov = PropRiskGovernor(cfg, db)
    day1 = datetime(2024, 1, 1, 23, 55, tzinfo=timezone.utc)
    gov.on_trade_close(-1.0, day1)
    state_before = gov.load(day1)
    assert state_before.daily_trades == 1
    day2 = datetime(2024, 1, 2, 0, 5, tzinfo=timezone.utc)
    gov.applied_risk_pct(day2)
    state_after = gov.load(day2)
    assert state_after.daily_trades == 0


def test_governor_blocks_dd_including_unrealized() -> None:
    acct = compute_accounting_snapshot(
        equity_start=1000,
        realized_pnl=-20,
        unrealized_pnl=-40,
        fees=0,
        day_start_equity=1000,
        hwm=1000,
        trade_close_dates=[],
        profit_target_pct=0.08,
    )
    block = evaluate_prop_block(
        now=datetime(2024, 1, 1, tzinfo=timezone.utc),
        day_key="2024-01-01",
        stored_day_key="2024-01-01",
        daily_trades=0,
        consecutive_losses=0,
        last_loss_at=None,
        accounting=acct,
        max_daily_loss_pct=0.05,
        max_global_dd_pct=0.10,
        max_trades_per_day=5,
        max_consecutive_losses=2,
        cooldown_minutes=30,
    )
    assert block is not None
    assert block.code == "daily_loss_limit"


def test_entry_engine_blocks_dead_and_requires_expansion_bos() -> None:
    rows = _sample_rows()
    feats = compute_features(rows)
    dead = classify_regime(feats, min_atr_pct=0.05)
    assert dead.regime == Regime.DEAD
    last = rows[-1]
    blocked = evaluate_entries(regime=dead, features=feats, close=float(last["close"]), high=float(last["high"]), low=float(last["low"]), equity_state="NEUTRAL")
    assert blocked.signal is None

    trend = classify_regime(feats)
    weak = evaluate_entries(regime=trend, features=feats, close=float(last["close"]), high=float(last["close"]) + 0.01, low=float(last["close"]) - 0.01, equity_state="NEUTRAL")
    assert weak.signal is None
