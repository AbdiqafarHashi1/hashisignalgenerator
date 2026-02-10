from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..models import Candle, Direction


@dataclass(frozen=True)
class TradeLevels:
    entry: float
    stop_loss: float
    take_profit: float
    entry_zone: tuple[float, float]


def compute_ema(values: list[float], length: int) -> float | None:
    if length <= 0 or len(values) < length:
        return None
    ema = sum(values[:length]) / length
    multiplier = 2 / (length + 1)
    for value in values[length:]:
        ema = (value - ema) * multiplier + ema
    return ema


def _true_range(current: Candle, previous_close: float) -> float:
    return max(
        current.high - current.low,
        abs(current.high - previous_close),
        abs(current.low - previous_close),
    )


def compute_atr_series(candles: list[Candle], period: int) -> list[float]:
    if period <= 0 or len(candles) < period + 1:
        return []
    true_ranges = [_true_range(candles[i], candles[i - 1].close) for i in range(1, len(candles))]
    if len(true_ranges) < period:
        return []
    atr_values: list[float] = []
    atr = sum(true_ranges[:period]) / period
    atr_values.append(atr)
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period
        atr_values.append(atr)
    return atr_values


def trend_direction(candles: list[Candle], ema_length: int) -> tuple[Direction | None, float | None]:
    closes = [c.close for c in candles]
    ema = compute_ema(closes, ema_length)
    if ema is None:
        return None, None
    if closes[-1] > ema:
        return Direction.long, ema
    if closes[-1] < ema:
        return Direction.short, ema
    return None, ema


def _volume_confirmed(candles: list[Candle], cfg: Settings, multiplier: float | None = None) -> bool:
    volumes = [c.volume for c in candles if c.volume is not None]
    if len(volumes) < cfg.volume_sma_period + 1:
        return False
    current_volume = volumes[-1]
    avg = sum(volumes[-(cfg.volume_sma_period + 1):-1]) / cfg.volume_sma_period
    threshold = cfg.volume_confirm_multiplier if multiplier is None else multiplier
    return avg > 0 and current_volume >= avg * threshold


def pullback_continuation_trigger(candles: list[Candle], direction: Direction, ema: float, cfg: Settings) -> bool:
    if len(candles) < 3:
        return False
    prev = candles[-2]
    cur = candles[-1]
    band = ema * cfg.ema_pullback_pct
    if direction == Direction.long:
        pullback = prev.low <= ema + band
        strong_close = cur.close > prev.high and cur.close > cur.open
        engulf = cur.close > prev.open and cur.open <= prev.close
    else:
        pullback = prev.high >= ema - band
        strong_close = cur.close < prev.low and cur.close < cur.open
        engulf = cur.close < prev.open and cur.open >= prev.close
    return pullback and (strong_close or engulf) and _volume_confirmed(candles, cfg)


def breakout_expansion_trigger(candles: list[Candle], direction: Direction, cfg: Settings) -> bool:
    if len(candles) < max(cfg.atr_period + cfg.atr_sma_period, 25):
        return False
    atr_values = compute_atr_series(candles, cfg.atr_period)
    if len(atr_values) < cfg.atr_sma_period + 1:
        return False
    atr_now = atr_values[-1]
    atr_avg = sum(atr_values[-(cfg.atr_sma_period + 1):-1]) / cfg.atr_sma_period
    expansion = atr_avg > 0 and atr_now >= atr_avg * cfg.breakout_atr_multiplier
    if not expansion:
        return False
    prev_high = max(c.high for c in candles[-6:-1])
    prev_low = min(c.low for c in candles[-6:-1])
    cur = candles[-1]
    breakout = cur.close > prev_high if direction == Direction.long else cur.close < prev_low
    return breakout and _volume_confirmed(candles, cfg, multiplier=cfg.breakout_volume_multiplier)


def build_trade_levels(candles: list[Candle], direction: Direction, cfg: Settings, setup: str) -> TradeLevels:
    current = candles[-1]
    previous = candles[-2]
    entry = current.close
    if direction == Direction.long:
        structure_sl = min(current.low, previous.low)
        cap_sl = entry * (1 - cfg.max_stop_pct)
        stop_loss = max(structure_sl, cap_sl)
        tp_mult = 1.2 if setup == "breakout_expansion" else 1.0
        take_profit = entry * (1 + cfg.take_profit_pct * tp_mult)
    else:
        structure_sl = max(current.high, previous.high)
        cap_sl = entry * (1 + cfg.max_stop_pct)
        stop_loss = min(structure_sl, cap_sl)
        tp_mult = 1.2 if setup == "breakout_expansion" else 1.0
        take_profit = entry * (1 - cfg.take_profit_pct * tp_mult)
    entry_zone = (min(current.open, current.close), max(current.open, current.close))
    return TradeLevels(entry=entry, stop_loss=stop_loss, take_profit=take_profit, entry_zone=entry_zone)


def expected_pnl_after_costs(levels: TradeLevels, direction: Direction, cfg: Settings) -> float:
    if direction == Direction.long:
        gross = (levels.take_profit - levels.entry) / levels.entry
    else:
        gross = (levels.entry - levels.take_profit) / levels.entry
    costs = (cfg.fee_rate_bps + cfg.spread_bps + cfg.slippage_bps) / 10_000
    return gross - costs


def momentum_ok(candles: list[Candle], cfg: Settings) -> bool:
    atr_values = compute_atr_series(candles, cfg.atr_period)
    if len(atr_values) < cfg.atr_sma_period:
        return False
    atr = atr_values[-1]
    atr_sma = sum(atr_values[-cfg.atr_sma_period:]) / cfg.atr_sma_period
    return atr > atr_sma


def engulfing_trigger(candles: list[Candle], direction: Direction, ema: float, cfg: Settings) -> bool:
    return pullback_continuation_trigger(candles, direction, ema, cfg)
