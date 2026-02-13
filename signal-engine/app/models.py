from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Status(str, Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"
    RISK_OFF = "RISK_OFF"


class Direction(str, Enum):
    long = "long"
    short = "short"
    none = "none"


class SetupType(str, Enum):
    break_retest = "break_retest"
    sweep_reclaim = "sweep_reclaim"


class Posture(str, Enum):
    RISK_OFF = "RISK_OFF"
    NORMAL = "NORMAL"
    OPPORTUNISTIC = "OPPORTUNISTIC"


class MarketSnapshot(BaseModel):
    funding_rate: float = Field(..., description="Funding rate as a decimal, e.g. 0.01 for 1%")
    oi_change_24h: float = Field(..., description="Open interest change in last 24h as decimal")
    leverage_ratio: float = Field(..., description="Leverage ratio estimate")
    trend_strength: float = Field(0.0, description="0-1 trend strength")


class BiasSignal(BaseModel):
    direction: Direction
    confidence: float = Field(..., ge=0.0, le=1.0)


class TradingViewPayload(BaseModel):
    symbol: str
    direction_hint: Direction
    entry_low: float
    entry_high: float
    sl_hint: float
    setup_type: SetupType
    tf_bias: str = "4h"
    tf_entry: str = "5m"


class ScoreBreakdown(BaseModel):
    regime: int
    bias: int
    structure: int
    bonus: int
    total: int


class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class DecisionRequest(BaseModel):
    tradingview: TradingViewPayload
    market: MarketSnapshot
    bias: BiasSignal
    timestamp: datetime | None = None
    candles: list[Candle] | None = None
    interval: str | None = None


class PostureSnapshot(BaseModel):
    symbol: str
    date: str
    posture: Posture
    reason: str
    computed_at: datetime


class RiskResult(BaseModel):
    risk_pct_used: float
    stop_distance_pct: float
    position_size_usd: float


class TradePlan(BaseModel):
    status: Status
    direction: Direction
    entry_zone: tuple[float, float] | None
    stop_loss: float | None
    take_profit: float | None
    risk_pct_used: float | None
    position_size_usd: float | None
    signal_score: int | None
    posture: Posture
    rationale: list[str]
    raw_input_snapshot: dict[str, Any]


class TradeOutcome(BaseModel):
    symbol: str
    pnl_usd: float
    win: bool
    timestamp: datetime


class EngineState(BaseModel):
    timestamp: datetime
    last_tick_age_seconds: float | None
    running: bool
    balance: float
    equity: float
    unrealized_pnl_usd: float
    realized_pnl_today_usd: float
    trades_today: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_dd_today_pct: float
    daily_loss_remaining_usd: float
    daily_loss_pct: float
    open_positions: list[dict[str, Any]] = Field(default_factory=list)
    recent_trades: list[dict[str, Any]] = Field(default_factory=list)
    cooldown_active: bool
    funding_blackout: bool
    swings_enabled: bool
    current_mode: str
    consecutive_losses: int = 0
    last_decision: str | None = None
    last_skip_reason: str | None = None
    final_entry_gate: str | None = None
    regime_label: str | None = None
    allowed_side: str | None = None
    atr_pct: float | None = None
    ema_fast: float | None = None
    ema_slow: float | None = None
    ema_trend: float | None = None
