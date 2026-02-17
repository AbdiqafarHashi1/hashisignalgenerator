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
    if not cfg.volume_confirm_enabled:
        return True
    volumes = [c.volume for c in candles if c.volume is not None]
    if len(volumes) < cfg.volume_sma_period + 1:
        return True
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
    min_dist = max(0.0, float(cfg.scalp_pullback_min_dist_pct or 0.0))
    dist_from_ema = abs(cur.close - ema) / ema if ema > 0 else 1.0
    if dist_from_ema < min_dist:
        return False
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
    if len(candles) < max(cfg.atr_period + cfg.atr_sma_period, cfg.min_breakout_window):
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
        tp_mult = cfg.breakout_tp_multiplier if setup == "breakout_expansion" else 1.0
        take_profit = entry * (1 + cfg.take_profit_pct * tp_mult)
    else:
        structure_sl = max(current.high, previous.high)
        cap_sl = entry * (1 + cfg.max_stop_pct)
        stop_loss = min(structure_sl, cap_sl)
        tp_mult = cfg.breakout_tp_multiplier if setup == "breakout_expansion" else 1.0
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


def _compute_adx(candles: list[Candle], period: int) -> float | None:
    if period <= 1 or len(candles) < (period * 2) + 1:
        return None
    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, len(candles)):
        cur = candles[i]
        prev = candles[i - 1]
        up_move = cur.high - prev.high
        down_move = prev.low - cur.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    if len(trs) < period:
        return None
    smoothed_tr = sum(trs[:period])
    smoothed_plus_dm = sum(plus_dm[:period])
    smoothed_minus_dm = sum(minus_dm[:period])
    dx_values: list[float] = []
    for i in range(period, len(trs)):
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + trs[i]
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]
        if smoothed_tr <= 0:
            dx_values.append(0.0)
            continue
        plus_di = 100.0 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100.0 * (smoothed_minus_dm / smoothed_tr)
        denom = plus_di + minus_di
        dx_values.append((100.0 * abs(plus_di - minus_di) / denom) if denom > 0 else 0.0)
    if len(dx_values) < period:
        return None
    adx = sum(dx_values[:period]) / period
    for dx in dx_values[period:]:
        adx = ((adx * (period - 1)) + dx) / period
    return adx


def momentum_ok(candles: list[Candle], cfg: Settings) -> bool:
    if cfg.momentum_mode == "adx":
        adx = _compute_adx(candles, cfg.adx_period)
        if adx is None:
            return False
        return adx >= cfg.adx_threshold
    atr_values = compute_atr_series(candles, cfg.atr_period)
    if len(atr_values) < cfg.atr_sma_period:
        return False
    atr = atr_values[-1]
    atr_sma = sum(atr_values[-cfg.atr_sma_period:]) / cfg.atr_sma_period
    return atr > atr_sma


def engulfing_trigger(candles: list[Candle], direction: Direction, ema: float, cfg: Settings) -> bool:
    return pullback_continuation_trigger(candles, direction, ema, cfg)


@dataclass(frozen=True)
class RegimeSignal:
    regime: str
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    signal_score: int
    rationale: list[str]


def compute_rsi(candles: list[Candle], period: int = 14) -> float | None:
    if period <= 0 or len(candles) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(len(candles) - period, len(candles)):
        delta = candles[i].close - candles[i - 1].close
        if delta >= 0:
            gains += delta
        else:
            losses += abs(delta)
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100 - (100 / (1 + rs))


def _rolling_vwap(candles: list[Candle], length: int = 20) -> float | None:
    if not candles:
        return None
    recent = candles[-length:]
    total_vol = sum(max(0.0, c.volume or 0.0) for c in recent)
    if total_vol <= 0:
        return sum(c.close for c in recent) / len(recent)
    total_pv = sum(((c.high + c.low + c.close) / 3.0) * max(0.0, c.volume or 0.0) for c in recent)
    return total_pv / total_vol


def _regime_from_metrics(candles: list[Candle], cfg: Settings) -> tuple[str, float, float, float] | None:
    closes = [c.close for c in candles]
    fast = compute_ema(closes, cfg.scalp_ema_fast)
    slow = compute_ema(closes, cfg.scalp_ema_slow)
    if fast is None or slow is None or slow == 0:
        return None
    atr_values = compute_atr_series(candles, cfg.scalp_atr_period)
    if not atr_values:
        return None
    atr = atr_values[-1]
    atr_pct = atr / candles[-1].close if candles[-1].close > 0 else 0.0
    slope = (fast - slow) / slow
    trend_strength = min(1.0, abs(slope) * cfg.trend_strength_scale)
    regime = "TRENDING" if abs(slope) >= cfg.scalp_trend_slope_min and trend_strength >= cfg.trend_strength_min else "RANGING"
    return regime, atr, atr_pct, slope


def generate_regime_signal(candles: list[Candle], cfg: Settings) -> RegimeSignal | None:
    if len(candles) < max(cfg.scalp_ema_slow + 2, cfg.scalp_atr_period + 5):
        return None
    metrics = _regime_from_metrics(candles, cfg)
    if metrics is None:
        return None
    regime, atr, atr_pct, slope = metrics
    if atr_pct < cfg.scalp_atr_pct_min or atr_pct > cfg.scalp_atr_pct_max:
        return None
    close = candles[-1].close
    prev_close = candles[-2].close
    rsi_now = compute_rsi(candles, cfg.scalp_rsi_period)
    rsi_prev = compute_rsi(candles[:-1], cfg.scalp_rsi_period)
    if rsi_now is None or rsi_prev is None:
        return None
    closes = [c.close for c in candles]
    ema_fast = compute_ema(closes, cfg.scalp_ema_fast)
    ema_slow = compute_ema(closes, cfg.scalp_ema_slow)
    if ema_fast is None or ema_slow is None:
        return None
    vwap = _rolling_vwap(candles)
    score = 50
    score += int(min(cfg.trend_score_max_boost, abs(slope) * cfg.trend_score_slope_scale))
    score += int(min(cfg.atr_score_max_boost, max(0.0, (atr_pct - cfg.scalp_atr_pct_min) * cfg.atr_score_scale)))

    if regime == "TRENDING":
        direction = Direction.long if ema_fast > ema_slow else Direction.short
        pullback_distance = abs(prev_close - ema_fast)
        if pullback_distance > atr * cfg.pullback_atr_mult:
            return None
        if direction == Direction.long:
            confirm = rsi_now >= cfg.trend_rsi_midline and close > prev_close and rsi_now >= rsi_prev
        else:
            confirm = rsi_now <= cfg.trend_rsi_midline and close < prev_close and rsi_now <= rsi_prev
        if not confirm:
            return None
        score += cfg.trend_confirm_score_boost if confirm else 0
        threshold = cfg.min_signal_score_trend
        if score < threshold:
            return None
        sl_mult = cfg.sl_atr_mult
        tp_mult = cfg.tp_atr_mult
        stop = close - atr * sl_mult if direction == Direction.long else close + atr * sl_mult
        take_profit = close + atr * tp_mult if direction == Direction.long else close - atr * tp_mult
        return RegimeSignal(regime=regime, direction=direction, entry=close, stop_loss=stop, take_profit=take_profit, signal_score=min(score, 100), rationale=["trend_pullback_confirm"]) 

    # ranging
    if vwap is None:
        return None
    deviation = close - vwap
    if abs(deviation) < atr * cfg.dev_atr_mult:
        return None
    direction = Direction.long if deviation < 0 else Direction.short
    if direction == Direction.long and rsi_now >= cfg.range_rsi_long_max:
        return None
    if direction == Direction.short and rsi_now <= cfg.range_rsi_short_min:
        return None
    score += cfg.range_base_score_boost
    threshold = cfg.min_signal_score_range
    if score < threshold:
        return None
    stop = close - atr * cfg.range_sl_atr_mult if direction == Direction.long else close + atr * cfg.range_sl_atr_mult
    tp_cap = atr * cfg.range_tp_atr_mult
    target = vwap
    if direction == Direction.long:
        take_profit = min(close + tp_cap, target if target > close else close + tp_cap)
    else:
        take_profit = max(close - tp_cap, target if target < close else close - tp_cap)
    return RegimeSignal(regime=regime, direction=direction, entry=close, stop_loss=stop, take_profit=take_profit, signal_score=min(score, 100), rationale=["range_mean_reversion"])
