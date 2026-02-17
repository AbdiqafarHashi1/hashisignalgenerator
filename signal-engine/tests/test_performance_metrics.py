from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.services.database import TradeRecord
from app.services.performance import build_performance_snapshot


def _trade(
    trade_id: int,
    *,
    entry: float,
    exit_price: float,
    stop: float,
    side: str,
    pnl_usd: float,
    pnl_r: float,
    opened_at: datetime,
    closed_at: datetime,
    result: str,
) -> TradeRecord:
    return TradeRecord(
        id=trade_id,
        symbol="BTCUSDT",
        entry=entry,
        exit=exit_price,
        stop=stop,
        take_profit=entry + 20,
        size=1.0,
        pnl_usd=pnl_usd,
        pnl_r=pnl_r,
        side=side,
        opened_at=opened_at.isoformat(),
        closed_at=closed_at.isoformat(),
        result=result,
        trade_mode="paper",
    )


def test_build_performance_snapshot_computes_metrics() -> None:
    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    trades = [
        _trade(
            1,
            entry=100,
            exit_price=110,
            stop=95,
            side="long",
            pnl_usd=8,
            pnl_r=2,
            opened_at=now - timedelta(hours=3),
            closed_at=now - timedelta(hours=2),
            result="tp_close",
        ),
        _trade(
            2,
            entry=100,
            exit_price=95,
            stop=105,
            side="short",
            pnl_usd=4,
            pnl_r=1,
            opened_at=now - timedelta(hours=2),
            closed_at=now - timedelta(hours=1),
            result="tp_close",
        ),
        _trade(
            3,
            entry=100,
            exit_price=97,
            stop=95,
            side="long",
            pnl_usd=-4,
            pnl_r=-0.6,
            opened_at=now - timedelta(minutes=90),
            closed_at=now - timedelta(minutes=30),
            result="sl_close",
        ),
    ]

    snapshot = build_performance_snapshot(trades, account_size=1000.0, skip_reason_counts={"cooldown": 2})

    assert snapshot.trades_today == 3
    assert snapshot.win_rate == 2 / 3
    assert snapshot.profit_factor == 3.0
    assert snapshot.expectancy_r == 0.8
    assert snapshot.max_drawdown_pct >= 0
    assert snapshot.skip_reason_counts == {"cooldown": 2}
    assert snapshot.win_loss_distribution["wins"] == 2
    assert snapshot.win_loss_distribution["losses"] == 1


def test_performance_metrics_endpoint_persists_file() -> None:
    from app import main as main_module

    with TestClient(main_module.app) as client:
        response = client.get("/metrics/performance")
        assert response.status_code == 200
        payload = response.json()
        assert "trades_today" in payload
        assert "equity_curve" in payload

        data_dir = client.app.state.settings.data_dir
        performance_path = f"{data_dir}/performance.json"
        with open(performance_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        assert "trades_today" in content
