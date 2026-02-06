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
    true_ranges = [
        _true_range(candles[i], candles[i - 1].close) for i in range(1, len(candles))
    ]
    if len(true_ranges) < period:
        return []
    atr_values: list[float] = []
    atr = sum(true_ranges[:period]) / period
    atr_values.append(atr)
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period
        atr_values.append(atr)
    return atr_values


def compute_adx(candles: list[Candle], period: int) -> float | None:
    if period <= 0 or len(candles) < period + 1:
        return None

    plus_dm_list: list[float] = []
    minus_dm_list: list[float] = []
    tr_list: list[float] = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
        tr_list.append(_true_range(current, previous.close))

    if len(tr_list) < period:
        return None

    tr_smooth = sum(tr_list[:period])
    plus_dm_smooth = sum(plus_dm_list[:period])
    minus_dm_smooth = sum(minus_dm_list[:period])

    dx_list: list[float] = []

    def _calc_dx(tr_value: float, plus_dm_value: float, minus_dm_value: float) -> float:
        if tr_value == 0:
            return 0.0
        plus_di = 100 * (plus_dm_value / tr_value)
        minus_di = 100 * (minus_dm_value / tr_value)
        denom = plus_di + minus_di
        if denom == 0:
            return 0.0
        return 100 * abs(plus_di - minus_di) / denom

    dx_list.append(_calc_dx(tr_smooth, plus_dm_smooth, minus_dm_smooth))

    for i in range(period, len(tr_list)):
        tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[i]
        plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period) + plus_dm_list[i]
        minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period) + minus_dm_list[i]
        dx_list.append(_calc_dx(tr_smooth, plus_dm_smooth, minus_dm_smooth))

    if len(dx_list) < period:
        return None

    adx = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx


def trend_direction(candles: list[Candle], ema_length: int) -> tuple[Direction | None, float | None]:
    closes = [candle.close for candle in candles]
    ema = compute_ema(closes, ema_length)
    if ema is None:
        return None, None
    last_close = closes[-1]
    if last_close > ema:
        return Direction.long, ema
    if last_close < ema:
        return Direction.short, ema
    return None, ema


def momentum_ok(candles: list[Candle], cfg: Settings) -> bool:
    if cfg.momentum_mode == "adx":
        adx = compute_adx(candles, cfg.adx_period)
        if adx is None:
            return False
        return adx > cfg.adx_threshold
    atr_values = compute_atr_series(candles, cfg.atr_period)
    if len(atr_values) < cfg.atr_sma_period:
        return False
    atr = atr_values[-1]
    atr_sma = sum(atr_values[-cfg.atr_sma_period:]) / cfg.atr_sma_period
    return atr > atr_sma


def _location_ok(candles: list[Candle], ema: float, direction: Direction, cfg: Settings) -> bool:
    if len(candles) < 3:
        return False
    recent = candles[-3:]
    pullback_band = ema * cfg.ema_pullback_pct

    if direction == Direction.long:
        pullback_touch = min(candle.low for candle in recent) <= ema + pullback_band
        break_retest = (
            recent[0].close < ema
            and recent[1].close > ema
            and recent[2].low <= ema + pullback_band
        )
        return pullback_touch or break_retest
    pullback_touch = max(candle.high for candle in recent) >= ema - pullback_band
    break_retest = (
        recent[0].close > ema
        and recent[1].close < ema
        and recent[2].high >= ema - pullback_band
    )
    return pullback_touch or break_retest


def _volume_confirmed(candles: list[Candle], cfg: Settings) -> bool:
    if not cfg.volume_confirm_enabled:
        return True
    volumes = [candle.volume for candle in candles if candle.volume is not None]
    if len(volumes) < 2:
        return True
    current_volume = volumes[-1]
    history = volumes[-(cfg.volume_sma_period + 1):-1]
    if not history:
        return True
    avg_volume = sum(history) / len(history)
    if avg_volume == 0:
        return True
    return current_volume >= avg_volume * cfg.volume_confirm_multiplier


def engulfing_trigger(candles: list[Candle], direction: Direction, ema: float, cfg: Settings) -> bool:
    if len(candles) < 2:
        return False
    previous = candles[-2]
    current = candles[-1]
    prev_body_low = min(previous.open, previous.close)
    prev_body_high = max(previous.open, previous.close)
    curr_body_low = min(current.open, current.close)
    curr_body_high = max(current.open, current.close)

    body = abs(current.close - current.open)
    if body == 0:
        return False

    if direction == Direction.long:
        if current.close <= current.open:
            return False
        if curr_body_low > prev_body_low or curr_body_high < prev_body_high:
            return False
        upper_wick = current.high - max(current.open, current.close)
        if upper_wick > body * cfg.engulfing_wick_ratio:
            return False
    else:
        if current.close >= current.open:
            return False
        if curr_body_low > prev_body_low or curr_body_high < prev_body_high:
            return False
        lower_wick = min(current.open, current.close) - current.low
        if lower_wick > body * cfg.engulfing_wick_ratio:
            return False

    if not _location_ok(candles, ema, direction, cfg):
        return False
    if not _volume_confirmed(candles, cfg):
        return False
    return True


def build_trade_levels(candles: list[Candle], direction: Direction, cfg: Settings) -> TradeLevels:
    current = candles[-1]
    previous = candles[-2]
    entry = current.close
    if direction == Direction.long:
        structure_sl = min(current.low, previous.low)
        cap_sl = entry * (1 - cfg.max_stop_pct)
        stop_loss = max(structure_sl, cap_sl)
        take_profit = entry * (1 + cfg.take_profit_pct)
    else:
        structure_sl = max(current.high, previous.high)
        cap_sl = entry * (1 + cfg.max_stop_pct)
        stop_loss = min(structure_sl, cap_sl)
        take_profit = entry * (1 - cfg.take_profit_pct)
    entry_zone = (min(current.open, current.close), max(current.open, current.close))
    return TradeLevels(entry=entry, stop_loss=stop_loss, take_profit=take_profit, entry_zone=entry_zone)
