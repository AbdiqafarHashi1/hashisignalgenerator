from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence


@dataclass(frozen=True)
class PivotPoint:
    index: int
    price: float


@dataclass(frozen=True)
class HTFFeatures:
    ema50: float
    ema200: float
    slope: float


@dataclass(frozen=True)
class FeatureSet:
    ema_fast: float
    ema_slow: float
    ema_slope: float
    adx: float
    atr: float
    atr_pct: float
    atr_percentile: float
    rsi: float
    rsi_prev: float
    pivots_high: list[PivotPoint]
    pivots_low: list[PivotPoint]
    htf: HTFFeatures


def _ema(values: Sequence[float], length: int) -> list[float]:
    if len(values) < length:
        return []
    alpha = 2.0 / (length + 1)
    out = [sum(values[:length]) / length]
    for v in values[length:]:
        out.append((v - out[-1]) * alpha + out[-1])
    return out


def _atr(rows: list[dict[str, Any]], period: int = 14) -> list[float]:
    trs: list[float] = []
    for i in range(1, len(rows)):
        cur, prev = rows[i], rows[i - 1]
        tr = max(cur["high"] - cur["low"], abs(cur["high"] - prev["close"]), abs(cur["low"] - prev["close"]))
        trs.append(tr)
    if len(trs) < period:
        return []
    out = [sum(trs[:period]) / period]
    for tr in trs[period:]:
        out.append(((out[-1] * (period - 1)) + tr) / period)
    return out


def _rsi(closes: Sequence[float], period: int = 14) -> list[float]:
    if len(closes) < period + 1:
        return []
    out: list[float] = []
    for idx in range(period, len(closes)):
        gains = 0.0
        losses = 0.0
        for i in range(idx - period + 1, idx + 1):
            delta = closes[i] - closes[i - 1]
            if delta >= 0:
                gains += delta
            else:
                losses += abs(delta)
        if losses == 0:
            out.append(100.0)
        else:
            rs = gains / losses
            out.append(100.0 - (100.0 / (1 + rs)))
    return out


def _adx(rows: list[dict[str, Any]], period: int = 14) -> list[float]:
    if len(rows) < period * 2 + 1:
        return []
    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, len(rows)):
        cur = rows[i]
        prev = rows[i - 1]
        up_move = cur["high"] - prev["high"]
        down_move = prev["low"] - cur["low"]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        trs.append(max(cur["high"] - cur["low"], abs(cur["high"] - prev["close"]), abs(cur["low"] - prev["close"])))

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
        return []
    adx = [sum(dx_values[:period]) / period]
    for dx in dx_values[period:]:
        adx.append(((adx[-1] * (period - 1)) + dx) / period)
    return adx


def _slope(series: Sequence[float], lookback: int, price: float) -> float:
    if len(series) <= lookback or price <= 0:
        return 0.0
    return ((series[-1] - series[-lookback - 1]) / price) / lookback


def _pivot_points(high: Sequence[float], low: Sequence[float], left: int = 2, right: int = 2) -> tuple[list[PivotPoint], list[PivotPoint]]:
    highs: list[PivotPoint] = []
    lows: list[PivotPoint] = []
    for i in range(left, len(high) - right):
        if high[i] == max(high[i - left : i + right + 1]):
            highs.append(PivotPoint(i, float(high[i])))
        if low[i] == min(low[i - left : i + right + 1]):
            lows.append(PivotPoint(i, float(low[i])))
    return highs, lows


def _to_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _resample_1h(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[datetime, list[dict[str, Any]]] = {}
    for row in rows:
        ts = _to_dt(row.get("timestamp", datetime.now(timezone.utc)))
        key = ts.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(buckets):
        chunk = buckets[key]
        out.append({"close": chunk[-1]["close"]})
    return out


def compute_features(rows: list[dict[str, Any]], *, ema_fast: int = 50, ema_slow: int = 200, slope_lookback: int = 20, atr_percentile_window: int = 500, pivot_left: int = 2, pivot_right: int = 2) -> FeatureSet:
    closes = [float(r["close"]) for r in rows]
    highs = [float(r["high"]) for r in rows]
    lows = [float(r["low"]) for r in rows]
    ema_fast_s = _ema(closes, ema_fast)
    ema_slow_s = _ema(closes, ema_slow)
    atr = _atr(rows, 14)
    adx = _adx(rows, 14)
    rsi = _rsi(closes, 14)
    piv_h, piv_l = _pivot_points(highs, lows, pivot_left, pivot_right)

    atr_pct_series = [a / c if c else 0.0 for a, c in zip(atr, closes[-len(atr):])]
    recent_window = atr_pct_series[-atr_percentile_window:] if atr_pct_series else [0.0]
    cur_atr_pct = atr_pct_series[-1] if atr_pct_series else 0.0
    atr_rank = sum(1 for v in recent_window if v <= cur_atr_pct) / max(1, len(recent_window))

    htf = _resample_1h(rows)
    htf_close = [float(v["close"]) for v in htf]
    htf_ema50 = (_ema(htf_close, 50) if htf_close else []) or [htf_close[-1] if htf_close else closes[-1]]
    htf_ema200 = (_ema(htf_close, 200) if htf_close else []) or [htf_close[-1] if htf_close else closes[-1]]

    return FeatureSet(
        ema_fast=float(ema_fast_s[-1]),
        ema_slow=float(ema_slow_s[-1]),
        ema_slope=_slope(ema_slow_s, slope_lookback, closes[-1]),
        adx=float(adx[-1]) if adx else 0.0,
        atr=float(atr[-1]) if atr else 0.0,
        atr_pct=float(cur_atr_pct),
        atr_percentile=float(atr_rank),
        rsi=float(rsi[-1]) if rsi else 50.0,
        rsi_prev=float(rsi[-2]) if len(rsi) > 1 else 50.0,
        pivots_high=piv_h,
        pivots_low=piv_l,
        htf=HTFFeatures(
            ema50=float(htf_ema50[-1]),
            ema200=float(htf_ema200[-1]),
            slope=_slope(htf_ema200, min(10, max(1, len(htf_ema200) - 1)), htf_close[-1] if htf_close else closes[-1]),
        ),
    )
