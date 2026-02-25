from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import Settings
from ..state import StateStore
from .database import Database
from .metrics import compute_metrics
from .paper_trader import PaperTrader


def build_dashboard_metrics(
    cfg: Settings,
    db: Database,
    trader: PaperTrader,
    state_store: StateStore,
    now: datetime,
    *,
    persist_runtime_state: bool = True,
) -> dict[str, Any]:
    payload = compute_metrics(cfg, db, trader, state_store, now, persist_runtime_state=persist_runtime_state)

    # Backward-compatible aliases expected by existing endpoints/UI/tests.
    payload.update(
        {
            "metrics_version": 2,
            "trades_all": payload["trades"],
            "equity_start_usd": payload["equity_start"],
            "equity_now_usd": payload["equity_now"],
            "balance": payload["equity_start"] + payload["realized_net"],
            "pnl_realized_total": payload["realized_gross"],
            "pnl_realized_net_total": payload["realized_net"],
            "pnl_unrealized": payload["unrealized_net"],
            "realized_gross_usd": payload["realized_gross"],
            "realized_net_usd": payload["realized_net"],
            "fees_total_usd": payload["fees_total"],
            "unrealized_usd": payload["unrealized_net"],
            "equity_calc_usd": payload["equity_now"],
            "reconciliation_delta_usd": 0.0,
            "equity_reconcile_delta": 0.0,
            "challenge_start_ts": payload.get("challenge_start_ts"),
            "day_start_equity": payload["daily_start_equity"],
            "pnl_realized_today": float(payload.get("pnl_realized_today", 0.0)),
            "wins_today": int(payload.get("wins_today", 0)),
            "losses_today": int(payload.get("losses_today", 0)),
            "daily_dd_abs": max(0.0, payload["daily_start_equity"] - payload["equity_now"]),
            "daily_dd_ratio": payload["daily_dd_pct"],
            "daily_dd_pct_percent": payload["daily_dd_pct"] * 100.0,
            "daily_drawdown_usd": max(0.0, payload["daily_start_equity"] - payload["equity_now"]),
            "daily_drawdown_pct": payload["daily_dd_pct"] * 100.0,
            "equity_high_watermark": payload["peak_equity"],
            "global_dd_abs": max(0.0, payload["equity_start"] - payload["equity_now"]),
            "global_dd_ratio": payload["global_dd_pct"],
            "global_dd_pct_percent": payload["global_dd_pct"] * 100.0,
            "global_drawdown_usd": max(0.0, payload["equity_start"] - payload["equity_now"]),
            "global_drawdown_pct": payload["global_dd_pct"] * 100.0,
            "max_global_dd_abs": max(0.0, payload["equity_start"] - payload["equity_now"]),
            "max_global_dd_pct": payload["global_dd_pct"],
            "trades_today_by_symbol": {},
            "realized_pnl_by_symbol": {},
            "fees_by_symbol": {},
        }
    )
    return payload
