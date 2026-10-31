"""Trend-pullback signals with a crowd (funding) filter.

A signal on row i is known at the CLOSE of bar i. The backtester enters at the OPEN of bar i+1.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .indicators import rsi
from .regime import UP, DOWN, add_regime


@dataclass
class Params:
    ema_fast: int = 50
    ema_slow: int = 200
    adx_min: float = 20.0
    pullback_atr: float = 1.0   # price must have come within this many ATR of the fast EMA
    lookback: int = 5           # bars in which the pullback must have happened
    rsi_reset: float = 45.0     # RSI dipped below this (long) / rose above 100-this (short)
    stop_atr: float = 1.5
    rr: float = 2.0
    max_hold: int = 48
    fund_z_max: float = 2.0     # skip longs when funding z-score above, shorts when below negative
    allow_long: bool = True
    allow_short: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def funding_zscore(funding: pd.Series | None, index: pd.Index, window: int = 90) -> pd.Series:
    if funding is None or len(funding) < 10:
        return pd.Series(0.0, index=index)
    z = (funding - funding.rolling(window, min_periods=20).mean()) / funding.rolling(window, min_periods=20).std()
    return z.reindex(index.union(z.index)).sort_index().ffill().reindex(index).fillna(0.0)


def build(df: pd.DataFrame, p: Params, funding: pd.Series | None = None) -> pd.DataFrame:
    d = add_regime(df, p.ema_fast, p.ema_slow, p.adx_min)
    d["rsi"] = rsi(d["close"])
    d["fund_z"] = funding_zscore(funding, d.index)

    dist = (d["low"] - d["ema_fast"]) / d["atr"]            # long: how deep the low went toward the EMA
    touched_long = dist.rolling(p.lookback).min() <= p.pullback_atr
    dist_s = (d["high"] - d["ema_fast"]) / d["atr"]
    touched_short = (-dist_s).rolling(p.lookback).min() <= p.pullback_atr

    rsi_dip = d["rsi"].rolling(p.lookback).min() < p.rsi_reset
    rsi_pop = d["rsi"].rolling(p.lookback).max() > 100 - p.rsi_reset

    trig_long = d["close"] > d["high"].shift(1)
    trig_short = d["close"] < d["low"].shift(1)

    long_ok = (d["regime"] == UP) & touched_long & rsi_dip & trig_long & (d["close"] > d["ema_fast"])
    short_ok = (d["regime"] == DOWN) & touched_short & rsi_pop & trig_short & (d["close"] < d["ema_fast"])
    long_ok &= d["fund_z"] <= p.fund_z_max
    short_ok &= d["fund_z"] >= -p.fund_z_max

    d["signal"] = 0
    if p.allow_long:
        d.loc[long_ok, "signal"] = 1
    if p.allow_short:
        d.loc[short_ok, "signal"] = -1
    return d


def live_factors(d: pd.DataFrame, daily: pd.DataFrame | None, funding_last: float | None, oi_change_pct: float | None) -> list[dict]:
    """Human-readable scorecard for the latest closed bar. Mirrors the rules above."""
    r = d.iloc[-1]
    chg = (d["close"].iloc[-1] / d["close"].iloc[-7] - 1) * 100 if len(d) > 7 else 0.0
    f: list[dict] = []

    bias = daily if daily is not None and len(daily) > 210 else d
    from .indicators import ema
    e50, e200, px = ema(bias["close"], 50).iloc[-1], ema(bias["close"], 200).iloc[-1], bias["close"].iloc[-1]
    if px > e50 > e200:
        t, w = 2.0, "Daily: price EMA50 aur EMA200 ke upar. Uptrend."
    elif px < e50 < e200:
        t, w = -2.0, "Daily: price dono EMA ke neeche. Downtrend."
    elif px > e200:
        t, w = 1.0, "Daily: EMA200 ke upar par EMA50 ke neeche. Uptrend mein pullback."
    elif px < e200:
        t, w = -1.0, "Daily: EMA200 ke neeche par EMA50 ke upar. Downtrend mein bounce."
    else:
        t, w = 0.0, "Daily EMA mixed."
    f.append({"name": "Daily trend", "score": t, "note": w})

    reg = r["regime"]
    s = 1.0 if reg == UP else -1.0 if reg == DOWN else 0.0
    f.append({"name": f"Regime ({reg})", "score": s, "note": f"ADX {r['adx']:.0f}, ATR {r['atr_pct']:.1f}% of price."})

    if funding_last is not None:
        fr = funding_last * 100  # to percent
        if fr >= 0.1: c, w = -2.0, "Funding bahut high. Longs crowded, squeeze ka risk."
        elif fr >= 0.05: c, w = -1.0, "Funding high. Longs thode crowded."
        elif fr <= -0.05: c, w = 2.0, "Funding bahut negative. Shorts crowded."
        elif fr <= -0.02: c, w = 1.0, "Funding negative. Shorts thak rahe."
        else: c, w = 0.0, "Funding normal."
        f.append({"name": f"Crowd (funding {fr:+.3f}%)", "score": c, "note": w})
    if oi_change_pct is not None:
        up, dn, ou, od = chg > 0.3, chg < -0.3, oi_change_pct > 2, oi_change_pct < -2
        if up and ou: o, w = 0.5, "Price aur OI dono upar. Naye longs, move mein dum."
        elif up and od: o, w = -0.5, "Price upar, OI neeche. Short covering, kamzor rally."
        elif dn and ou: o, w = -1.0, "Price neeche, OI upar. Naye shorts aa rahe."
        elif dn and od: o, w = 1.0, "Price aur OI neeche. Longs flush hue."
        else: o, w = 0.0, "Price vs OI mein saaf signal nahi."
        f.append({"name": f"Price vs OI ({oi_change_pct:+.1f}%)", "score": o, "note": w})

    hi, lo = d["high"].rolling(30).max().iloc[-1], d["low"].rolling(30).min().iloc[-1]
    du, dd = (hi - r["close"]) / r["atr"], (r["close"] - lo) / r["atr"]
    if dd > 0 and du > 0:
        ratio = dd / du
        if ratio > 1.5: m, w = 1.0, f"30-bar high {du:.1f} ATR door, low {dd:.1f} ATR door. Upar ka room zyada."
        elif ratio < 0.67: m, w = -1.0, f"30-bar low sirf {dd:.1f} ATR door. Neeche ka room zyada."
        else: m, w = 0.0, "High aur low se distance balanced."
    else:
        m, w = 0.0, "Level data kam."
    f.append({"name": "Room to run", "score": m, "note": w})
    return f


def verdict(factors: list[dict], block_reasons: list[str]) -> tuple[str, float]:
    score = float(sum(x["score"] for x in factors))
    trend = factors[0]["score"]
    if block_reasons:
        return "NO TRADE", score
    if score >= 3 and trend > 0:
        return "LONG", score
    if score <= -3 and trend < 0:
        return "SHORT", score
    return "NO TRADE", score
