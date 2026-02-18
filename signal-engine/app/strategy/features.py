from __future__ import annotations

from dataclasses import dataclass

from ..models import Candle, Direction


@dataclass(frozen=True)
class FeatureSnapshot:
    ema20: float
    ema50: float
    ema200: float
    ema20_slope: float
    ema50_slope: float
    atr: float
    atr_pct: float
    adx: float
    rsi: float
    rsi_prev: float
    macd_hist: float
    macd_hist_prev: float
    swing_high: float
    swing_low: float


@dataclass(frozen=True)
class BiasSnapshot:
    direction: Direction
    strength: float
    ema50: float
    ema200: float
    ema50_slope: float


def ema(values: list[float], length: int) -> float | None:
    if length <= 1 or len(values) < length:
        return None
    value = sum(values[:length]) / length
    alpha = 2.0 / (length + 1)
    for sample in values[length:]:
        value = (sample - value) * alpha + value
    return value


def _ema_series(values: list[float], length: int) -> list[float]:
    if length <= 1 or len(values) < length:
        return []
    series: list[float] = [sum(values[:length]) / length]
    alpha = 2.0 / (length + 1)
    for sample in values[length:]:
        series.append((sample - series[-1]) * alpha + series[-1])
    return series


def atr(candles: list[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(candles)):
        prev_close = candles[i - 1].close
        cur = candles[i]
        trs.append(max(cur.high - cur.low, abs(cur.high - prev_close), abs(cur.low - prev_close)))
    if len(trs) < period:
        return None
    value = sum(trs[:period]) / period
    for tr in trs[period:]:
        value = ((value * (period - 1)) + tr) / period
    return value


def adx(candles: list[Candle], period: int = 14) -> float | None:
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
        dx_values.append(100.0 * abs(plus_di - minus_di) / denom if denom > 0 else 0.0)
    if len(dx_values) < period:
        return None
    value = sum(dx_values[:period]) / period
    for dx in dx_values[period:]:
        value = ((value * (period - 1)) + dx) / period
    return value


def rsi(candles: list[Candle], period: int = 14) -> float | None:
    if len(candles) < period + 1:
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
    return 100.0 - (100.0 / (1 + rs))


def macd_hist(candles: list[Candle], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | None, float | None]:
    closes = [c.close for c in candles]
    fast_series = _ema_series(closes, fast)
    slow_series = _ema_series(closes, slow)
    if not fast_series or not slow_series:
        return None, None
    offset = len(fast_series) - len(slow_series)
    macd_series = [fast_series[i + offset] - slow_series[i] for i in range(len(slow_series))]
    signal_series = _ema_series(macd_series, signal)
    if not signal_series:
        return None, None
    hist = [macd_series[len(macd_series) - len(signal_series) + i] - signal_series[i] for i in range(len(signal_series))]
    if len(hist) < 2:
        return None, None
    return hist[-1], hist[-2]


def structure_swings(candles: list[Candle], lookback: int) -> tuple[float | None, float | None]:
    if len(candles) < 2:
        return None, None
    window = candles[-lookback:] if lookback > 0 else candles
    return max(c.high for c in window), min(c.low for c in window)


def aggregate_to_1h(candles_5m: list[Candle]) -> list[Candle]:
    if len(candles_5m) < 12:
        return []
    blocks = len(candles_5m) // 12
    out: list[Candle] = []
    for i in range(blocks):
        chunk = candles_5m[i * 12:(i + 1) * 12]
        out.append(Candle(
            open=chunk[0].open,
            high=max(c.high for c in chunk),
            low=min(c.low for c in chunk),
            close=chunk[-1].close,
            volume=sum(c.volume or 0.0 for c in chunk),
        ))
    return out


def compute_bias(candles_5m: list[Candle], ema_fast: int = 50, ema_slow: int = 200) -> BiasSnapshot:
    htf = aggregate_to_1h(candles_5m)
    closes = [c.close for c in htf]
    use_htf = len(closes) >= max(ema_fast + 2, 30)
    if not use_htf:
        closes = [c.close for c in candles_5m]
    fast_len = min(ema_fast, max(10, len(closes) // 3)) if len(closes) < ema_fast + 2 else ema_fast
    slow_len = min(ema_slow, max(fast_len + 5, len(closes) - 2)) if len(closes) < ema_slow + 2 else ema_slow
    ema50 = ema(closes, fast_len)
    ema200 = ema(closes, slow_len)
    ema50_prev = ema(closes[:-1], fast_len) if len(closes) > fast_len + 1 else None
    if ema50 is None or ema200 is None or ema50_prev is None:
        return BiasSnapshot(Direction.none, 0.0, ema50 or 0.0, ema200 or 0.0, 0.0)
    slope = (ema50 - ema50_prev) / ema50_prev if ema50_prev else 0.0
    spread = (ema50 - ema200) / ema200 if ema200 else 0.0
    if ema50 > ema200 and slope > 0:
        return BiasSnapshot(Direction.long, min(1.0, abs(spread) * 12), ema50, ema200, slope)
    if ema50 < ema200 and slope < 0:
        return BiasSnapshot(Direction.short, min(1.0, abs(spread) * 12), ema50, ema200, slope)
    return BiasSnapshot(Direction.none, min(1.0, abs(spread) * 8), ema50, ema200, slope)


def compute_features(candles: list[Candle], swing_lookback: int = 30) -> FeatureSnapshot | None:
    if len(candles) < 60:
        return None
    closes = [c.close for c in candles]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200) or ema(closes, 50)
    ema20_prev = ema(closes[:-1], 20)
    ema50_prev = ema(closes[:-1], 50)
    atr_value = atr(candles, 14)
    adx_value = adx(candles, 14)
    rsi_now = rsi(candles, 14)
    rsi_prev = rsi(candles[:-1], 14)
    hist_now, hist_prev = macd_hist(candles)
    swing_high, swing_low = structure_swings(candles[:-1], swing_lookback)
    if None in {ema20, ema50, ema200, ema20_prev, ema50_prev, atr_value, adx_value, rsi_now, rsi_prev, hist_now, hist_prev, swing_high, swing_low}:
        return None
    close = candles[-1].close
    atr_pct = atr_value / close if close > 0 else 0.0
    return FeatureSnapshot(
        ema20=float(ema20),
        ema50=float(ema50),
        ema200=float(ema200),
        ema20_slope=float((ema20 - ema20_prev) / ema20_prev if ema20_prev else 0.0),
        ema50_slope=float((ema50 - ema50_prev) / ema50_prev if ema50_prev else 0.0),
        atr=float(atr_value),
        atr_pct=float(atr_pct),
        adx=float(adx_value),
        rsi=float(rsi_now),
        rsi_prev=float(rsi_prev),
        macd_hist=float(hist_now),
        macd_hist_prev=float(hist_prev),
        swing_high=float(swing_high),
        swing_low=float(swing_low),
    )
