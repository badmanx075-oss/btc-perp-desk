"""BTC Perp Desk engine v2.

Indicators, regime, causal signals, exits (partial / breakeven / trail), backtest, walk-forward,
Monte Carlo, key levels, volume profile, market structure, trade plan with entry and avoid zones,
setup grading, risk sizing and market data.

Single file on purpose, so it uploads to GitHub from a phone without folders.
Backtest signals use only data known at the bar close. Levels, zones and grade are the live,
discretionary overlay and are NOT part of the backtest (stated in the app).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import requests

UP, DOWN, RANGE = "UPTREND", "DOWNTREND", "RANGE"


# =========================================================================== indicators
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def rolling_rank(s: pd.Series, n: int = 200) -> pd.Series:
    """Percentile rank of the latest value inside the trailing window (0..1)."""
    return s.rolling(n, min_periods=n // 2).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)


def resample_ohlc(df: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    out = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna()


def bar_minutes(index: pd.Index) -> float:
    return (index[1] - index[0]).total_seconds() / 60 if len(index) > 1 else 240.0


# =========================================================================== regime and higher timeframe
def add_regime(df: pd.DataFrame, fast: int = 50, slow: int = 200, adx_min: float = 20.0) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], fast)
    out["ema_slow"] = ema(out["close"], slow)
    out["atr"] = atr(out)
    out["atr_pct"] = out["atr"] / out["close"] * 100
    out["adx"] = adx(out)
    out["vol_rank"] = rolling_rank(out["atr_pct"], 200)
    up = (out["ema_fast"] > out["ema_slow"]) & (out["close"] > out["ema_slow"]) & (out["adx"] >= adx_min)
    dn = (out["ema_fast"] < out["ema_slow"]) & (out["close"] < out["ema_slow"]) & (out["adx"] >= adx_min)
    out["regime"] = RANGE
    out.loc[up, "regime"] = UP
    out.loc[dn, "regime"] = DOWN
    out["high_vol"] = out["vol_rank"] > 0.9
    return out


def describe(row: pd.Series) -> str:
    base = {
        UP: "Trend upar ka hai. Sirf long pullback dhundo.",
        DOWN: "Trend neeche ka hai. Sirf short bounce dhundo.",
        RANGE: "Clear trend nahi (ADX kam ya EMA mixed). Ye trend-pullback strategy ka mausam nahi.",
    }[row["regime"]]
    if bool(row.get("high_vol", False)):
        base += " Volatility ausat se bahut zyada hai, size chhota rakho."
    return base


def htf_bias(d: pd.DataFrame, ema_n: int = 21) -> tuple[pd.Series, bool]:
    """Daily bias for every intraday bar, using only COMPLETED daily bars (shifted one day).
    +1 = close above a rising daily EMA, -1 = below a falling one, 0 = neutral, NaN = not enough history.
    Returns (series, valid). Not valid on daily bars (no higher frame to use)."""
    if bar_minutes(d.index) >= 1440 or len(d) < 50:
        return pd.Series(np.nan, index=d.index), False
    daily = resample_ohlc(d[["open", "high", "low", "close", "volume"]], "1D")
    e = ema(daily["close"], ema_n)
    e = e.where(np.arange(len(e)) >= ema_n)
    slope = e - e.shift(5)
    b = pd.Series(0.0, index=daily.index)
    b[(daily["close"] > e) & (slope > 0)] = 1.0
    b[(daily["close"] < e) & (slope < 0)] = -1.0
    b[slope.isna()] = np.nan
    b = b.shift(1)
    out = b.reindex(d.index.union(b.index)).sort_index().ffill().reindex(d.index)
    return out, True


# =========================================================================== signals
@dataclass
class Params:
    ema_fast: int = 50
    ema_slow: int = 200
    adx_min: float = 20.0
    pullback_atr: float = 1.0     # price came within this many ATR of the fast EMA
    lookback: int = 5             # bars in which the pullback must have happened
    rsi_reset: float = 45.0       # RSI dipped below this (long) / rose above 100-this (short)
    stop_atr: float = 1.5
    rr: float = 2.0               # final target in R (ignored when trailing)
    max_hold: int = 48
    fund_z_max: float = 2.0       # skip longs when funding z above, shorts when below minus
    use_htf: bool = True          # daily trend must agree with the trade
    htf_ema: int = 21
    min_room_atr: float = 1.0     # distance to the recent swing extreme must be at least this many ATR
    room_lookback: int = 60
    min_rvol: float = 0.0         # trigger candle volume vs 20-bar median (0 = off)
    cooldown: int = 3             # no new signal within this many bars of the previous one
    partial_frac: float = 0.5     # share closed at partial_r (0 = off)
    partial_r: float = 1.0
    trail_atr: float = 0.0        # >0: no fixed target, ATR trailing stop after partial_r in profit
    major_score: float = 4.0      # live plan only: a level counts as MAJOR above this strength (and 2+ sources)
    min_room_r: float = 1.0       # live plan only: need this many R of room to the first major level
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
    """Add indicators and the causal signal column. A signal on row i is known at the CLOSE of bar i."""
    d = add_regime(df, p.ema_fast, p.ema_slow, p.adx_min)
    d["rsi"] = rsi(d["close"])
    d["fund_z"] = funding_zscore(funding, d.index)
    d["rvol"] = d["volume"] / d["volume"].rolling(20).median().replace(0, np.nan)
    htf, htf_valid = htf_bias(d, p.htf_ema)
    d["htf"] = htf
    hi20 = d["high"].rolling(20).max().shift(1)
    lo20 = d["low"].rolling(20).min().shift(1)
    d["sweep_bear"] = (d["high"] > hi20) & (d["close"] < hi20)   # stop-run above highs, closed back inside
    d["sweep_bull"] = (d["low"] < lo20) & (d["close"] > lo20)    # stop-run below lows, closed back inside
    d["room_long"] = (d["high"].rolling(p.room_lookback).max() - d["close"]) / d["atr"]
    d["room_short"] = (d["close"] - d["low"].rolling(p.room_lookback).min()) / d["atr"]

    touched_long = ((d["low"] - d["ema_fast"]) / d["atr"]).rolling(p.lookback).min() <= p.pullback_atr
    touched_short = ((d["ema_fast"] - d["high"]) / d["atr"]).rolling(p.lookback).min() <= p.pullback_atr
    rsi_dip = d["rsi"].rolling(p.lookback).min() < p.rsi_reset
    rsi_pop = d["rsi"].rolling(p.lookback).max() > 100 - p.rsi_reset
    trig_long = d["close"] > d["high"].shift(1)
    trig_short = d["close"] < d["low"].shift(1)

    use_htf = p.use_htf and htf_valid
    htf_long = d["htf"].eq(1.0) if use_htf else pd.Series(True, index=d.index)
    htf_short = d["htf"].eq(-1.0) if use_htf else pd.Series(True, index=d.index)
    rvol_ok = d["rvol"].fillna(1.0) >= p.min_rvol

    long_ok = ((d["regime"] == UP) & touched_long & rsi_dip & trig_long & (d["close"] > d["ema_fast"])
               & (d["fund_z"] <= p.fund_z_max) & htf_long & (d["room_long"] >= p.min_room_atr) & rvol_ok)
    short_ok = ((d["regime"] == DOWN) & touched_short & rsi_pop & trig_short & (d["close"] < d["ema_fast"])
                & (d["fund_z"] >= -p.fund_z_max) & htf_short & (d["room_short"] >= p.min_room_atr) & rvol_ok)

    raw = pd.Series(0, index=d.index)
    if p.allow_long:
        raw.loc[long_ok] = 1
    if p.allow_short:
        raw.loc[short_ok] = -1
    sig = raw.copy()
    if p.cooldown > 0:
        recent = (raw != 0).astype(float).shift(1).rolling(p.cooldown, min_periods=1).max().fillna(0) > 0
        sig.loc[recent] = 0
    d["signal_raw"] = raw
    d["signal"] = sig
    return d


def live_factors(d: pd.DataFrame, daily: pd.DataFrame | None, funding_last: float | None,
                 oi_change_pct: float | None) -> list[dict]:
    """Human-readable scorecard for the latest closed bar."""
    r = d.iloc[-1]
    chg = (d["close"].iloc[-1] / d["close"].iloc[-7] - 1) * 100 if len(d) > 7 else 0.0
    f: list[dict] = []

    bias = daily if daily is not None and len(daily) > 210 else d
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
        fr = funding_last * 100
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


# =========================================================================== backtest
@dataclass
class Costs:
    fee_bps: float = 5.0        # per side
    slip_bps: float = 2.0       # per side
    risk_pct: float = 1.0
    max_lev: float = 3.0
    start_equity: float = 1000.0


def simulate_trade(o, h, l, cl, entry_i: int, side: int, sdist: float, p: Params, slip_bps: float = 0.0,
                   fr=None, bar_h: float = 4.0) -> dict:
    """Walk one trade bar by bar. Conservative: the stop is checked first inside every bar, a gap through the
    stop fills at the open, a stop moved to breakeven or trailed only applies from the NEXT bar."""
    n = len(o)
    entry = o[entry_i] * (1 + side * slip_bps / 1e4)
    stop = entry - side * sdist
    trail_on = p.trail_atr > 0
    target = None if trail_on else entry + side * p.rr * sdist
    partial_px = entry + side * p.partial_r * sdist if p.partial_frac > 0 else None
    if partial_px is not None and target is not None and side * (partial_px - target) >= 0:
        partial_px = None
    trail_dist = p.trail_atr * sdist / p.stop_atr if p.stop_atr else 0.0
    frac, legs, fund_acc = 1.0, [], 0.0
    fav = adv = 0.0
    partial_done = activated = False
    last = min(entry_i + p.max_hold, n - 1)
    exit_i = None
    for j in range(entry_i, last + 1):
        if fr is not None:
            fund_acc += frac * fr[j] * bar_h / 8
        adv_px = l[j] if side == 1 else h[j]
        adv = max(adv, -side * (adv_px - entry))
        if (l[j] <= stop) if side == 1 else (h[j] >= stop):
            fill = min(stop, o[j]) if side == 1 else max(stop, o[j])
            rel = side * (stop - entry)
            why = "stop" if rel < -1e-9 * entry else ("breakeven" if abs(rel) <= 1e-9 * entry else "trail")
            legs.append((frac, fill, why))
            frac, exit_i = 0.0, j
            break
        fav_px = h[j] if side == 1 else l[j]
        fav = max(fav, side * (fav_px - entry))
        if partial_px is not None and not partial_done and ((h[j] >= partial_px) if side == 1 else (l[j] <= partial_px)):
            take = min(p.partial_frac, frac)
            legs.append((take, partial_px, "partial"))
            frac -= take
            partial_done = True
            if side * (entry - stop) > 0:
                stop = entry
        if target is not None and ((h[j] >= target) if side == 1 else (l[j] <= target)):
            legs.append((frac, target, "target"))
            frac, exit_i = 0.0, j
            break
        if trail_on:
            if fav >= p.partial_r * sdist:
                activated = True
            if activated:
                tstop = (entry + side * fav) - side * trail_dist
                if side * (tstop - stop) > 0:
                    stop = tstop
    if exit_i is None:
        exit_i = last
        legs.append((frac, cl[last], "time"))
    return {"entry": entry, "stop0": entry - side * sdist, "legs": legs, "exit_i": exit_i, "fund_acc": fund_acc,
            "fav": fav, "adv": adv, "partial": partial_done}


def run(df: pd.DataFrame, p: Params, c: Costs, funding: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = build(df, p, funding)
    idx = d.index
    o, h, l, cl = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    atr_v, sig = d["atr"].values, d["signal"].values
    n = len(d)
    warm = max(p.ema_slow, 220)
    bar_h = bar_minutes(idx) / 60

    fund_rate = None
    if funding is not None and len(funding) > 0:
        fund_rate = funding.reindex(idx.union(funding.index)).sort_index().ffill().reindex(idx).fillna(0.0).values

    eq = c.start_equity
    trades: list[dict] = []
    i = warm
    while i < n - 2:
        if sig[i] == 0 or np.isnan(atr_v[i]):
            i += 1
            continue
        side = int(sig[i])
        entry_i = i + 1
        sdist = p.stop_atr * atr_v[i]
        sim = simulate_trade(o, h, l, cl, entry_i, side, sdist, p, c.slip_bps, fund_rate, bar_h)
        entry = sim["entry"]
        notional = min(eq * c.risk_pct / 100 / (sdist / entry), eq * c.max_lev)
        gross = 0.0
        for f, px, _ in sim["legs"]:
            px_eff = px * (1 - side * c.slip_bps / 1e4)
            gross += f * notional * side * (px_eff - entry) / entry
        fees = notional * 2 * c.fee_bps / 1e4
        fund_cost = notional * side * sim["fund_acc"]
        pnl = gross - fees - fund_cost
        risk_amt = eq * c.risk_pct / 100
        exit_i = sim["exit_i"]
        trades.append({
            "entry_time": idx[entry_i], "exit_time": idx[exit_i], "side": "LONG" if side == 1 else "SHORT",
            "entry": entry, "stop": sim["stop0"], "exit": sim["legs"][-1][1], "reason": sim["legs"][-1][2],
            "partial": sim["partial"], "bars": exit_i - entry_i + 1, "notional": notional, "pnl": pnl,
            "R": pnl / risk_amt, "mfe_R": sim["fav"] / sdist, "mae_R": sim["adv"] / sdist, "equity_before": eq})
        eq += pnl
        i = exit_i + 1
        if eq <= 0:
            break

    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, d
    tr["equity"] = c.start_equity + tr["pnl"].cumsum()
    return tr, d


def rebuild_equity(tr: pd.DataFrame, c: Costs) -> pd.DataFrame:
    """Re-compound a trade list from its R multiples (used for windows and out-of-sample stitching)."""
    tr = tr.copy()
    eq, before, pnl, eqs = c.start_equity, [], [], []
    for r_ in tr["R"].values:
        before.append(eq)
        pn = eq * c.risk_pct / 100 * r_
        eq += pn
        pnl.append(pn)
        eqs.append(eq)
    tr["equity_before"], tr["pnl"], tr["equity"] = before, pnl, eqs
    return tr


def bootstrap_ci(x, n: int = 1000, seed: int = 1) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _max_streak(mask) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def stats(tr: pd.DataFrame, df: pd.DataFrame, c: Costs, p: Params) -> dict:
    if tr is None or tr.empty:
        return {"trades": 0}
    wins, losses = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
    gp, gl = wins["pnl"].sum(), -losses["pnl"].sum()
    avg_w = float(wins["R"].mean()) if len(wins) else 0.0
    avg_l = float(-losses["R"].mean()) if len(losses) else 0.0
    be = avg_l / (avg_w + avg_l) * 100 if (avg_w + avg_l) > 0 else float("nan")
    eqc = pd.concat([pd.Series([c.start_equity]), tr["equity"]], ignore_index=True)
    R = tr["R"].values
    sd = R.std(ddof=1) if len(R) > 1 else 0.0
    ci = bootstrap_ci(R)
    days = max((df.index[-1] - df.index[0]).days, 1)
    out = {
        "trades": len(tr),
        "win_rate": len(wins) / len(tr) * 100,
        "breakeven_win_rate": be,
        "avg_win_R": avg_w, "avg_loss_R": avg_l,
        "avg_R": float(R.mean()),
        "avg_R_ci": ci,
        "sqn": float(np.sqrt(len(R)) * R.mean() / sd) if sd > 0 else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "total_return_pct": (tr["equity"].iloc[-1] / c.start_equity - 1) * 100,
        "max_drawdown_pct": (eqc / eqc.cummax() - 1).min() * 100,
        "avg_bars": float(tr["bars"].mean()) if "bars" in tr else float("nan"),
        "longs": int((tr["side"] == "LONG").sum()),
        "shorts": int((tr["side"] == "SHORT").sum()),
        "buy_hold_pct": (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100,
        "worst_losing_streak": _max_streak((tr["pnl"] <= 0).values),
        "days": days,
        "trades_per_month": len(tr) / days * 30.4,
    }
    if "mfe_R" in tr:
        lo = tr[tr["pnl"] <= 0]
        out["losers_reached_1R_pct"] = float((lo["mfe_R"] >= 1.0).mean() * 100) if len(lo) else 0.0
    return out


def monte_carlo(R, risk_pct: float, n_trades: int | None = None, sims: int = 2000, seed: int = 1) -> dict:
    """Shuffle the trade results many times to see the spread of outcomes and drawdowns."""
    R = np.asarray(R, dtype=float)
    if len(R) < 5:
        return {}
    n = n_trades or len(R)
    rng = np.random.default_rng(seed)
    seq = R[rng.integers(0, len(R), size=(sims, n))] * risk_pct / 100
    eq = np.cumprod(1 + seq, axis=1)
    full = np.concatenate([np.ones((sims, 1)), eq], axis=1)
    peak = np.maximum.accumulate(full, axis=1)
    dd = (full / peak - 1).min(axis=1)
    end = eq[:, -1]
    return {"median_end": float(np.median(end)), "p5_end": float(np.percentile(end, 5)),
            "p95_end": float(np.percentile(end, 95)), "median_dd": float(np.median(dd)),
            "worst5_dd": float(np.percentile(dd, 5)), "prob_loss": float((end < 1).mean()),
            "prob_dd_20": float((dd < -0.2).mean()), "prob_dd_30": float((dd < -0.3).mean()), "sims": sims, "n": n}


GRID = {"adx_min": (15, 20, 25), "pullback_atr": (0.5, 1.0, 1.5), "stop_atr": (1.2, 1.5, 2.0), "rr": (1.5, 2.0, 3.0)}


def optimise(train: pd.DataFrame, base: Params, c: Costs, funding, min_trades: int = 15) -> tuple[pd.DataFrame, Params | None]:
    rows = []
    for adx_min in GRID["adx_min"]:
        for pb in GRID["pullback_atr"]:
            for stop_atr in GRID["stop_atr"]:
                for rr in GRID["rr"]:
                    q = Params(**{**base.as_dict(), "adx_min": adx_min, "pullback_atr": pb, "stop_atr": stop_atr, "rr": rr})
                    tr, _ = run(train, q, c, funding)
                    if tr.empty or len(tr) < min_trades:
                        continue
                    s = stats(tr, train, c, q)
                    score = s["avg_R"] * np.sqrt(min(s["trades"], 60) / 60)   # prefer edges backed by more trades
                    rows.append({"adx_min": adx_min, "pullback_atr": pb, "stop_atr": stop_atr, "rr": rr,
                                 "train_trades": s["trades"], "train_win%": s["win_rate"], "train_avgR": s["avg_R"],
                                 "train_PF": s["profit_factor"], "score": score})
    if not rows:
        return pd.DataFrame(), None
    res = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    b = res.iloc[0]
    q = Params(**{**base.as_dict(), "adx_min": float(b["adx_min"]), "pullback_atr": float(b["pullback_atr"]),
                  "stop_atr": float(b["stop_atr"]), "rr": float(b["rr"])})
    return res, q


def grid_search(df: pd.DataFrame, base: Params, c: Costs, funding, train_frac: float = 0.6, min_trades: int = 15):
    """Single split: choose parameters on the first part, judge them on the unseen last part."""
    cut = int(len(df) * train_frac)
    res, q = optimise(df.iloc[:cut], base, c, funding, min_trades)
    if q is None:
        return pd.DataFrame(), None
    test = df.iloc[max(0, cut - 600):]
    tr_t, _ = run(test, q, c, funding)
    if not tr_t.empty:
        tr_t = rebuild_equity(tr_t[tr_t["entry_time"] >= df.index[cut]], c)
    return res, {"params": q, "test_stats": stats(tr_t, test, c, q), "test_trades": tr_t}


def walk_forward(df: pd.DataFrame, base: Params, c: Costs, funding, folds: int = 3, min_trades: int = 10,
                 warm: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anchored walk-forward: optimise on everything before a window, trade that window blind, repeat."""
    n = len(df)
    start = int(n * 0.4)
    seg = (n - start) // folds
    rows, parts = [], []
    for k in range(folds):
        a = start + k * seg
        b = n if k == folds - 1 else a + seg
        res, q = optimise(df.iloc[:a], base, c, funding, min_trades)
        row = {"fold": k + 1, "test_from": df.index[a], "test_to": df.index[b - 1]}
        if q is None:
            rows.append({**row, "note": "train mein kaafi trades nahi"})
            continue
        tr, _ = run(df.iloc[max(0, a - warm):b], q, c, funding)
        if not tr.empty:
            tr = tr[tr["entry_time"] >= df.index[a]]
        row.update({"adx_min": q.adx_min, "pullback": q.pullback_atr, "stop_atr": q.stop_atr, "rr": q.rr,
                    "train_avgR": float(res.iloc[0]["train_avgR"]), "test_trades": len(tr),
                    "test_avgR": float(tr["R"].mean()) if len(tr) else float("nan"),
                    "test_win%": float((tr["R"] > 0).mean() * 100) if len(tr) else float("nan")})
        rows.append(row)
        if len(tr):
            parts.append(tr)
    oos = pd.DataFrame()
    if parts:
        oos = rebuild_equity(pd.concat(parts).sort_values("entry_time").reset_index(drop=True), c)
    return pd.DataFrame(rows), oos


# =========================================================================== levels, structure, profile
def swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> tuple[pd.Series, pd.Series]:
    """Pivot highs / lows. A pivot at bar i is only confirmed `right` bars later, so the last `right` bars are NaN."""
    h, l = df["high"].values, df["low"].values
    n = len(df)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(left, n - right):
        wh, wl = h[i - left:i + right + 1], l[i - left:i + right + 1]
        if wh.argmax() == left:
            ph[i] = h[i]
        if wl.argmin() == left:
            pl[i] = l[i]
    return pd.Series(ph, index=df.index), pd.Series(pl, index=df.index)


def market_structure(ph: pd.Series, pl: pd.Series) -> tuple[str, str]:
    hs, ls = ph.dropna().tail(3).values, pl.dropna().tail(3).values
    if len(hs) < 2 or len(ls) < 2:
        return "MIXED", "Swing points kam hain."
    hh, hl = hs[-1] > hs[-2], ls[-1] > ls[-2]
    if hh and hl:
        return "BULLISH", "Higher high aur higher low. Buyers control mein."
    if not hh and not hl:
        return "BEARISH", "Lower high aur lower low. Sellers control mein."
    return "MIXED", "Higher high par lower low." if hh else "Lower high par higher low."


def volume_profile(df: pd.DataFrame, bins: int = 48, lookback: int = 300, va: float = 0.7) -> dict:
    """Approximate volume profile: each bar's volume is spread evenly across its high-low range."""
    w = df.tail(lookback)
    lo, hi = float(w["low"].min()), float(w["high"].max())
    if not hi > lo:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "centers": np.array([]), "vols": np.array([])}
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for hh, ll, vv in zip(w["high"].values, w["low"].values, w["volume"].values):
        a = int(np.clip(np.searchsorted(edges, ll, "right") - 1, 0, bins - 1))
        b = int(np.clip(np.searchsorted(edges, hh, "right") - 1, 0, bins - 1))
        vol[a:b + 1] += vv / (b - a + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    k = int(vol.argmax())
    a_, b_, acc, target = k, k, vol[k], va * vol.sum()
    while acc < target and (a_ > 0 or b_ < bins - 1):
        up = vol[b_ + 1] if b_ < bins - 1 else -1.0
        dn = vol[a_ - 1] if a_ > 0 else -1.0
        if up >= dn:
            b_ += 1
            acc += vol[b_]
        else:
            a_ -= 1
            acc += vol[a_]
    return {"poc": float(centers[k]), "vah": float(edges[b_ + 1]), "val": float(edges[a_]),
            "centers": centers, "vols": vol}


def anchored_vwap(df: pd.DataFrame, pos: int) -> pd.Series:
    s = df.iloc[pos:]
    typ = (s["high"] + s["low"] + s["close"]) / 3
    v = s["volume"].where(s["volume"] > 0, 1e-9)
    return (typ * v).cumsum() / v.cumsum()


def build_levels(daily: pd.DataFrame | None, ph: pd.Series, pl: pd.Series, atr_v: float, px: float, vp: dict,
                 avwap_vals: dict) -> pd.DataFrame:
    """Merge swing levels, daily references, volume profile and anchored VWAPs into confluence clusters."""
    items: list[tuple[float, str, str, float]] = []
    for v in ph.dropna().values[-40:]:
        items.append((float(v), "Swing high", "swing", 1.0))
    for v in pl.dropna().values[-40:]:
        items.append((float(v), "Swing low", "swing", 1.0))
    if daily is not None and len(daily) >= 8:
        items += [(float(daily["high"].tail(7).max()), "7D high", "ref", 1.5),
                  (float(daily["low"].tail(7).min()), "7D low", "ref", 1.5),
                  (float(daily["high"].tail(30).max()), "30D high", "ref", 2.0),
                  (float(daily["low"].tail(30).min()), "30D low", "ref", 2.0),
                  (float(daily["high"].iloc[-1]), "Prev day high", "ref", 1.0),
                  (float(daily["low"].iloc[-1]), "Prev day low", "ref", 1.0)]
    for key, label, w in (("poc", "POC", 2.0), ("vah", "VAH", 1.5), ("val", "VAL", 1.5)):
        items.append((vp.get(key, np.nan), label, "profile", w))
    for name, val in avwap_vals.items():
        items.append((float(val), name, "vwap", 1.5))
    items = [it for it in items if np.isfinite(it[0])]
    cols = ["price", "label", "score", "sources", "kind"]
    if not items:
        return pd.DataFrame(columns=cols)
    items.sort(key=lambda t: t[0])
    tol = 0.5 * atr_v
    groups = [[items[0]]]
    for it in items[1:]:
        g = groups[-1]
        mean = sum(x[0] * x[3] for x in g) / sum(x[3] for x in g)
        if it[0] - mean <= tol:
            g.append(it)
        else:
            groups.append([it])
    rows = []
    for g in groups:
        w = sum(x[3] for x in g)
        price = sum(x[0] * x[3] for x in g) / w
        labels, counts = [], {}
        for x in g:
            counts[x[1]] = counts.get(x[1], 0) + 1
            if x[1] not in labels:
                labels.append(x[1])
        labels.sort(key=lambda s: 1 if s in ("Swing high", "Swing low") else 0)
        text = " + ".join(f"{s} x{counts[s]}" if counts[s] > 1 else s for s in labels[:3])
        rows.append({"price": price, "label": text, "score": w, "sources": len({x[2] for x in g}),
                     "kind": "support" if price < px else "resistance"})
    return pd.DataFrame(rows, columns=cols)


# =========================================================================== trade plan
def trade_plan(d: pd.DataFrame, lv: pd.DataFrame, px: float, p: Params, htf_dir: float,
               ph: pd.Series, pl: pd.Series) -> dict:
    """Where to take the trade, where to stay away, and what has to happen first."""
    r = d.iloc[-1]
    atr_v, ef, reg = float(r["atr"]), float(r["ema_fast"]), r["regime"]
    hi30, lo30 = float(d["high"].tail(30).max()), float(d["low"].tail(30).min())
    side = 0
    if reg == UP and (np.isnan(htf_dir) or htf_dir == 1):
        side = 1
    elif reg == DOWN and (np.isnan(htf_dir) or htf_dir == -1):
        side = -1
    plan: dict = {"side": side, "atr": atr_v, "ema": ef, "avoid": [], "targets": [], "watch": [], "zone": None}
    if side == 0:
        span = hi30 - lo30
        plan.update(status="NO BIAS", kind="avoid",
                    headline="Trend aur daily bias match nahi karte. Trade nahi.",
                    avoid=[(lo30 + 0.3 * span, lo30 + 0.7 * span, "Chop zone: range ka beech. Yahan entry mat lo.")],
                    watch=[(hi30, f"Close {hi30:,.0f} ke upar: bullish breakout, tab long setup dhundo."),
                           (lo30, f"Close {lo30:,.0f} ke neeche: bearish breakdown, tab short setup dhundo.")])
        return plan

    sd = side
    near = lv[(lv["price"] <= ef + 0.75 * atr_v) & (lv["price"] >= ef - 1.5 * atr_v)] if sd == 1 else \
        lv[(lv["price"] >= ef - 0.75 * atr_v) & (lv["price"] <= ef + 1.5 * atr_v)]
    conf_score, conf_label = 0.0, f"EMA{p.ema_fast}"
    if len(near):
        c = near.sort_values("score", ascending=False).iloc[0]
        a, b = sorted((float(c["price"]), ef))
        zlo, zhi = a - 0.25 * atr_v, b + 0.25 * atr_v
        conf_score, conf_label = float(c["score"]), f"{c['label']} + EMA{p.ema_fast}"
    else:
        zlo, zhi = ef - 0.5 * atr_v, ef + 0.5 * atr_v

    dist = p.stop_atr * atr_v
    in_zone = zlo <= px <= zhi
    ext = sd * (px - ef) / atr_v
    entry = px if in_zone else (zlo + zhi) / 2
    if sd * (px - entry) > 0 and not in_zone and ext <= 0:      # price already beyond the zone on the pullback side
        entry = px
    stop = entry - sd * dist

    path = lv[lv["price"] > entry + 0.25 * atr_v].sort_values("price") if sd == 1 else \
        lv[lv["price"] < entry - 0.25 * atr_v].sort_values("price", ascending=False)
    allt = [{"price": float(x["price"]), "label": x["label"], "score": float(x["score"]), "sources": int(x["sources"]),
             "R": sd * (float(x["price"]) - entry) / dist} for _, x in path.iterrows()]
    strong = [t for t in allt if t["score"] >= p.major_score and t["sources"] >= 2]
    room_R = strong[0]["R"] if strong else 99.0
    targets = allt[:3] + ([strong[0]] if strong and strong[0] not in allt[:3] else [])

    avoid = [((ef + 2.5 * atr_v, ef + 4.0 * atr_v) if sd == 1 else (ef - 4.0 * atr_v, ef - 2.5 * atr_v))
             + ("Chase zone: price EMA se bahut door. Yahan entry mat lo, pullback ka wait.",)]
    if strong:
        rp = strong[0]["price"]
        avoid.append(((rp - 1.0 * atr_v, rp) if sd == 1 else (rp, rp + 1.0 * atr_v))
                     + (f"{strong[0]['label']} ke bilkul paas. Yahan se room kam.",))

    last_sig = int(d["signal"].iloc[-1])
    last_close = float(d["close"].iloc[-1])
    fresh = last_sig == sd and sd * (px - last_close) / atr_v < 0.6
    inval = zlo - 0.75 * atr_v if sd == 1 else zhi + 0.75 * atr_v
    side_txt = "LONG" if sd == 1 else "SHORT"
    if sd * (px - inval) < 0:
        status, kind, head = "INVALIDATED", "avoid", "Pullback bahut gehra gaya. Setup toot gaya, skip."
    elif room_R < p.min_room_r:
        status, kind = "NO ROOM", "avoid"
        head = f"Aage major level {strong[0]['label']} ({strong[0]['price']:,.0f}) sirf {room_R:.1f}R door hai. Room kam, skip."
    elif fresh:
        status, kind, head = "TRIGGERED", "go", f"Trigger ho chuka hai. {side_txt} entry abhi valid hai."
    elif in_zone:
        status, kind, head = "IN ZONE", "ready", "Price entry zone ke andar hai. Ab sirf trigger candle ka wait."
    elif ext > 2.5:
        status, kind, head = "EXTENDED", "avoid", f"Price EMA se {ext:.1f} ATR extended. Chase mat karo, avoid."
    elif sd * (px - (zhi if sd == 1 else zlo)) > 0:
        gap = abs(px - (zhi if sd == 1 else zlo)) / atr_v
        status, kind = "WAIT PULLBACK", "wait"
        head = f"Price zone se {gap:.1f} ATR door. Pullback ka wait, alert lagao."
    else:
        status, kind = "BELOW ZONE" if sd == 1 else "ABOVE ZONE", "wait"
        head = "Price zone ke us paar chala gaya. Zone reclaim hone ka wait."

    last_pl = pl.dropna(); last_ph = ph.dropna()
    swing_ref = None
    if sd == 1 and len(last_pl[last_pl < entry]):
        swing_ref = float(last_pl[last_pl < entry].iloc[-1]) - 0.2 * atr_v
    elif sd == -1 and len(last_ph[last_ph > entry]):
        swing_ref = float(last_ph[last_ph > entry].iloc[-1]) + 0.2 * atr_v
    plan.update(
        status=status, kind=kind, headline=head, zone=(zlo, zhi), conf_label=conf_label, conf_score=conf_score,
        entry=entry, stop=stop, dist=dist, swing_stop=swing_ref, inval=inval,
        partial_px=entry + sd * p.partial_r * dist if p.partial_frac > 0 else None,
        target_px=entry + sd * p.rr * dist if p.trail_atr <= 0 else None,
        targets=targets, room_R=room_R, avoid=avoid, ext=ext,
        trigger_price=float(d["high"].iloc[-1]) if sd == 1 else float(d["low"].iloc[-1]),
        trigger_text=(f"Candle close {d['high'].iloc[-1]:,.0f} (pichhli candle ka high) ke upar" if sd == 1
                      else f"Candle close {d['low'].iloc[-1]:,.0f} (pichhli candle ka low) ke neeche"))
    return plan


def grade_setup(plan: dict, htf_dir: float, struct: str, adx_v: float, fund_last: float | None,
                factors: list[dict], sweep_recent: bool) -> dict:
    sd = plan["side"]
    items = []

    def add(name: str, mx: float, got: float, note: str):
        items.append({"name": name, "max": mx, "got": got, "note": note})

    if np.isnan(htf_dir):
        add("Daily trend saath hai", 2, 1, "Daily data nahi, aadha credit.")
    else:
        ok = htf_dir == sd
        add("Daily trend saath hai", 2, 2 if ok else 0, "Daily bias trade ke saath hai." if ok else "Daily bias against hai.")
    add("Trend strength (ADX)", 1, 1 if adx_v >= 25 else 0.5 if adx_v >= 20 else 0, f"ADX {adx_v:.0f}")
    want = "BULLISH" if sd == 1 else "BEARISH"
    add("Market structure", 1, 1 if struct == want else 0, f"Structure {struct}")
    cs = plan.get("conf_score", 0.0)
    add("Entry zone par level confluence", 2, 2 if cs >= 3.5 else 1 if cs >= 2.0 else 0,
        f"{plan.get('conf_label', '')} (score {cs:.1f})")
    if fund_last is None:
        add("Crowd crowded nahi", 1, 0.5, "Funding data nahi.")
    else:
        fr = fund_last * 100
        crowded = fr >= 0.05 if sd == 1 else fr <= -0.05
        add("Crowd crowded nahi", 1, 0 if crowded else 1, f"Funding {fr:+.3f}%")
    oi = [f for f in factors if f["name"].startswith("Price vs OI")]
    if oi:
        add("Price vs OI saath hai", 1, 1 if sd * oi[0]["score"] >= 0 else 0, oi[0]["note"])
    else:
        add("Price vs OI saath hai", 1, 0.5, "OI data nahi.")
    rr_ = plan.get("room_R", 99.0)
    add("Aage room (2R+)", 1, 1 if rr_ >= 2 else 0.5 if rr_ >= 1.0 else 0,
        "Aage koi major level nahi." if rr_ >= 99 else f"Pehla major level {rr_:.1f}R door")
    add("Liquidity sweep", 1, 1 if sweep_recent else 0,
        "Haal hi mein stop-run hua aur wapas laut gaya." if sweep_recent else "Recent sweep nahi.")
    total, mx = sum(i["got"] for i in items), sum(i["max"] for i in items)
    letter = "A" if total >= 7 else "B" if total >= 5 else "C"
    return {"items": items, "total": total, "max": mx, "letter": letter, "mult": {"A": 1.0, "B": 0.5, "C": 0.0}[letter]}


def decide(plan: dict, grade: dict, blocks: list[str], reduce_size: bool, high_vol: bool) -> dict:
    side_txt = {1: "LONG", -1: "SHORT", 0: ""}[plan["side"]]
    mult = grade["mult"] * (0.5 if reduce_size else 1.0) * (0.5 if high_vol else 1.0)
    if blocks:
        return {"action": "NO TRADE", "kind": "avoid", "why": " ".join(blocks), "mult": 0.0}
    if plan["kind"] == "avoid":
        return {"action": "NO TRADE", "kind": "avoid", "why": plan["headline"], "mult": 0.0}
    if grade["letter"] == "C":
        return {"action": "NO TRADE", "kind": "avoid",
                "why": f"Setup kamzor hai (grade C, {grade['total']:.1f}/{grade['max']:.0f}). Behtar setup ka wait.", "mult": 0.0}
    if plan["kind"] == "go":
        return {"action": f"ENTER {side_txt}", "kind": "go", "why": plan["headline"], "mult": mult}
    if plan["kind"] == "ready":
        return {"action": f"READY {side_txt}", "kind": "ready", "why": plan["headline"], "mult": mult}
    return {"action": f"WAIT {side_txt}", "kind": "wait", "why": plan["headline"], "mult": mult}


def analyse_live(d: pd.DataFrame, daily: pd.DataFrame | None, px: float, fund_last: float | None,
                 oi_chg: float | None, p: Params) -> dict:
    """Run the whole live analysis on the latest closed bar and return everything the UI needs."""
    r = d.iloc[-1]
    atr_v = float(r["atr"])
    sub = d.tail(600)
    ph, pl = swing_points(sub, 5, 5)
    struct, struct_note = market_structure(ph, pl)
    htf_dir = float(r["htf"]) if not pd.isna(r["htf"]) else float("nan")
    vp = volume_profile(sub)

    avwaps: dict[str, pd.Series] = {}
    avwap_vals: dict[str, float] = {}
    for label, series in (("AVWAP (swing low)", pl), ("AVWAP (swing high)", ph)):
        s = series.dropna()
        if len(s):
            pos = int(sub.index.get_loc(s.index[-1]))
            v = anchored_vwap(sub, pos)
            avwaps[label] = v
            avwap_vals[label] = float(v.iloc[-1])

    lv = build_levels(daily, ph, pl, atr_v, px, vp, avwap_vals)
    factors = live_factors(d, daily, fund_last, oi_chg)
    plan = trade_plan(d, lv, px, p, htf_dir, ph, pl)
    sweep_recent = False
    if plan["side"] == 1:
        sweep_recent = bool(d["sweep_bull"].tail(5).any())
    elif plan["side"] == -1:
        sweep_recent = bool(d["sweep_bear"].tail(5).any())
    grade = grade_setup(plan, htf_dir, struct, float(r["adx"]), fund_last, factors, sweep_recent) if plan["side"] else \
        {"items": [], "total": 0.0, "max": 10.0, "letter": "C", "mult": 0.0}

    bbw = 4 * d["close"].rolling(20).std() / d["close"].rolling(20).mean()
    bb_rank = float((bbw.tail(200) < bbw.iloc[-1]).mean()) if bbw.notna().any() else 0.5

    steps = []
    if np.isnan(htf_dir):
        steps.append(("warn", "Higher timeframe (daily)", "Daily bias nahi nikal paya (data kam ya timeframe daily hai). Filter skip hua."))
    elif htf_dir == 1:
        steps.append(("ok", "Higher timeframe (daily)", f"Daily trend UP: price rising daily EMA{p.htf_ema} ke upar. Long ki taraf jhukao."))
    elif htf_dir == -1:
        steps.append(("ok", "Higher timeframe (daily)", f"Daily trend DOWN: price falling daily EMA{p.htf_ema} ke neeche. Short ki taraf jhukao."))
    else:
        steps.append(("warn", "Higher timeframe (daily)", "Daily trend neutral (price EMA ke paas ya slope flat). Bias kamzor."))
    steps.append(("ok" if r["regime"] != RANGE else "bad", "Regime (trading timeframe)",
                  f"{r['regime']}. EMA{p.ema_fast} {'>' if r['ema_fast'] > r['ema_slow'] else '<'} EMA{p.ema_slow}, "
                  f"ADX {r['adx']:.0f} (min {p.adx_min:g}). " + describe(r)))
    steps.append(("ok" if struct != "MIXED" else "warn", "Market structure", f"{struct}. {struct_note}"))
    sup = lv[lv["price"] < px].sort_values("price", ascending=False).head(1)
    res = lv[lv["price"] > px].sort_values("price").head(1)
    lvl_txt = []
    if len(sup):
        lvl_txt.append(f"Support {sup.iloc[0]['price']:,.0f} ({sup.iloc[0]['label']}, {(px - sup.iloc[0]['price']) / atr_v:.1f} ATR neeche)")
    if len(res):
        lvl_txt.append(f"Resistance {res.iloc[0]['price']:,.0f} ({res.iloc[0]['label']}, {(res.iloc[0]['price'] - px) / atr_v:.1f} ATR upar)")
    steps.append(("ok", "Key levels", ". ".join(lvl_txt) + f". Volume profile: POC {vp['poc']:,.0f}, value area {vp['val']:,.0f} se {vp['vah']:,.0f}."
                  if lvl_txt else "Levels nahi mile."))
    crowd = [f for f in factors if f["name"].startswith("Crowd")]
    oi_f = [f for f in factors if f["name"].startswith("Price vs OI")]
    ctxt = " ".join([crowd[0]["note"] if crowd else "Funding data nahi.", oi_f[0]["note"] if oi_f else "OI data nahi."])
    crowd_bad = bool(crowd) and plan["side"] != 0 and plan["side"] * crowd[0]["score"] < 0
    steps.append(("bad" if crowd_bad else "ok", "Crowd positioning", ctxt))
    vtxt = f"ATR {r['atr_pct']:.2f}% (percentile {r['vol_rank'] * 100:.0f}%)." if not pd.isna(r["vol_rank"]) else f"ATR {r['atr_pct']:.2f}%."
    vtxt += " Bollinger squeeze: breakout aane wala, direction ka wait." if bb_rank < 0.2 else ""
    steps.append(("warn" if bool(r["high_vol"]) or bb_rank < 0.2 else "ok", "Volatility", vtxt))
    steps.append(({"go": "ok", "ready": "ok", "wait": "warn", "avoid": "bad"}[plan["kind"]], "Setup status",
                  f"{plan['status']}: {plan['headline']}"))
    if plan["side"]:
        steps.append(({"A": "ok", "B": "warn", "C": "bad"}[grade["letter"]], "Setup grade",
                      f"Grade {grade['letter']} ({grade['total']:.1f}/{grade['max']:.0f}). A = poora size, B = aadha size, C = skip."))
    return {"plan": plan, "grade": grade, "factors": factors, "levels": lv, "profile": vp, "avwaps": avwaps,
            "struct": (struct, struct_note), "htf": htf_dir, "ph": ph, "pl": pl, "steps": steps, "sweep": sweep_recent,
            "bb_rank": bb_rank}


# =========================================================================== risk sizing
@dataclass
class Ticket:
    side: str
    entry: float
    stop: float
    target1: float
    target2: float
    qty: float
    notional: float
    leverage: float
    risk_amt: float
    liq_price: float | None
    liq_safe: bool
    capped: bool
    note: str


def build_ticket(side: str, price: float, ema_fast: float, atr_value: float, capital: float, risk_pct: float,
                 max_lev: float, stop_atr: float = 1.5, rr: float = 2.0, size_mult: float = 1.0,
                 maint_margin: float = 0.005) -> Ticket:
    sd = 1 if side == "LONG" else -1
    entry, note = price, "Market ke paas limit order."
    ext_atr = (price - ema_fast) / atr_value * sd
    if ext_atr > 2.0:
        entry = ema_fast + sd * 0.25 * atr_value
        note = f"Price EMA se {ext_atr:.1f} ATR extended hai. Chase mat karo, EMA ke paas limit lagao."
    dist = stop_atr * atr_value
    stop = entry - sd * dist
    t1, t2 = entry + sd * rr * dist, entry + sd * (rr + 1) * dist
    risk_amt = capital * risk_pct / 100 * size_mult
    qty = risk_amt / dist
    notional = qty * entry
    lev = notional / capital
    capped = False
    if lev > max_lev:
        notional, lev, capped = capital * max_lev, max_lev, True
        qty = notional / entry
        risk_amt = qty * dist
    liq, safe = None, True
    if lev > 1.0:
        liq = entry * (1 - sd * (1 / lev - maint_margin))
        safe = (liq < stop * 0.97) if sd == 1 else (liq > stop * 1.03)
    return Ticket(side, entry, stop, t1, t2, qty, notional, lev, risk_amt, liq, safe, capped, note)


# =========================================================================== market data
SOURCES = ["Hyperliquid", "OKX", "Bybit", "Binance"]
CCXT_ID = {"OKX": "okx", "Bybit": "bybit", "Binance": "binanceusdm"}
SYMBOL = "BTC/USDT:USDT"
TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
HL_URL = "https://api.hyperliquid.xyz/info"


def _hl(payload: dict):
    r = requests.post(HL_URL, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def hl_ohlcv(timeframe: str, limit: int) -> pd.DataFrame:
    step = TF_MINUTES[timeframe] * 60_000
    now = int(time.time() * 1000)
    rows = _hl({"type": "candleSnapshot",
                "req": {"coin": "BTC", "interval": timeframe, "startTime": now - limit * step, "endTime": now}})
    if not rows:
        raise RuntimeError("Hyperliquid ne candles nahi diye")
    df = pd.DataFrame({
        "ts": [r["t"] for r in rows], "open": [float(r["o"]) for r in rows], "high": [float(r["h"]) for r in rows],
        "low": [float(r["l"]) for r in rows], "close": [float(r["c"]) for r in rows],
        "volume": [float(r["v"]) for r in rows]})
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df = df[~df.index.duplicated()].sort_index()
    return df.iloc[:-1]


def hl_funding(days: int = 60) -> pd.Series | None:
    """Hyperliquid pays funding hourly. Convert to an 8h-equivalent rate so thresholds match other venues."""
    try:
        start = int((time.time() - days * 86400) * 1000)
        out: dict = {}
        for _ in range(5):
            rows = _hl({"type": "fundingHistory", "coin": "BTC", "startTime": start})
            if not rows:
                break
            for r in rows:
                out[pd.to_datetime(r["time"], unit="ms", utc=True)] = float(r["fundingRate"]) * 8
            nxt = rows[-1]["time"] + 1
            if len(rows) < 500 or nxt <= start:
                break
            start = nxt
        if not out:
            return None
        return pd.Series(out).sort_index().resample("8h").mean().dropna()
    except Exception:
        return None


def hl_live() -> dict | None:
    try:
        meta, ctxs = _hl({"type": "metaAndAssetCtxs"})
        i = [u["name"] for u in meta["universe"]].index("BTC")
        c = ctxs[i]
        px = float(c["markPx"])
        return {"price": px, "funding_8h": float(c["funding"]) * 8,
                "oi_usd": float(c["openInterest"]) * px, "prev_day": float(c["prevDayPx"])}
    except Exception:
        return None


def _ex(name: str):
    import ccxt  # lazy: tests and demo mode work without it
    return getattr(ccxt, CCXT_ID[name])({"enableRateLimit": True, "timeout": 15000})


def ccxt_ohlcv(source: str, timeframe: str, limit: int) -> pd.DataFrame:
    ex = _ex(source)
    limit = min(limit, 1500) if source == "OKX" else limit
    step = TF_MINUTES[timeframe] * 60_000
    per_call = 100 if source == "OKX" else 1000
    now = ex.milliseconds()
    since, rows = now - limit * step, []
    while len(rows) < limit:
        chunk = ex.fetch_ohlcv(SYMBOL, timeframe, since=since, limit=per_call)
        if not chunk:
            break
        rows += chunk
        nxt = chunk[-1][0] + step
        if nxt <= since or nxt >= now:
            break
        since = nxt
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    if df.empty:
        raise RuntimeError(f"{source} ne candles nahi diye")
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return df.sort_index().tail(limit).iloc[:-1]


def ccxt_funding(source: str, limit: int = 600) -> pd.Series | None:
    try:
        rows = _ex(source).fetch_funding_rate_history(SYMBOL, limit=limit)
        s = pd.Series({pd.to_datetime(r["timestamp"], unit="ms", utc=True): r["fundingRate"] for r in rows})
        return s.sort_index() if len(s) else None
    except Exception:
        return None


def oi_change_pct(hours: int = 24) -> tuple[float | None, str]:
    for src in ("OKX", "Bybit", "Binance"):
        try:
            rows = _ex(src).fetch_open_interest_history(SYMBOL, "1h", limit=hours + 1)
            vals = [r.get("openInterestValue") or r.get("openInterestAmount") for r in rows]
            vals = [v for v in vals if v]
            if len(vals) >= 2:
                return (vals[-1] / vals[0] - 1) * 100, src
        except Exception:
            continue
    return None, ""


def fetch_live(timeframe: str, candles: int, prefer: str = "Auto") -> dict:
    """Returns dict(df, daily, fund, fund_last, oi, oi_src, price, source, errors). Raises only if every venue fails."""
    order = SOURCES if prefer == "Auto" else [prefer] + [s for s in SOURCES if s != prefer]
    errors: list[str] = []
    for src in order:
        try:
            if src == "Hyperliquid":
                df = hl_ohlcv(timeframe, candles)
                daily = hl_ohlcv("1d", 400) if timeframe != "1d" else df
                fund, live = hl_funding(), hl_live()
            else:
                df = ccxt_ohlcv(src, timeframe, candles)
                daily = ccxt_ohlcv(src, "1d", 400) if timeframe != "1d" else df
                fund, live = ccxt_funding(src), None
            if len(df) < 300:
                raise RuntimeError(f"sirf {len(df)} candles mile")
            fund_last = (live or {}).get("funding_8h")
            if fund_last is None and fund is not None and len(fund):
                fund_last = float(fund.iloc[-1])
            oi, oi_src = oi_change_pct()
            return {"df": df, "daily": daily, "fund": fund, "fund_last": fund_last, "oi": oi, "oi_src": oi_src,
                    "price": (live or {}).get("price", float(df["close"].iloc[-1])), "source": src, "errors": errors}
        except Exception as e:
            errors.append(f"{src}: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError(" | ".join(errors))


def synthetic(timeframe: str = "4h", n: int = 3000, seed: int = 7) -> tuple[pd.DataFrame, pd.Series, float, float]:
    """Regime-switching random walk for demo mode. Not real BTC."""
    rng = np.random.default_rng(seed)
    mins = TF_MINUTES[timeframe]
    drift_by_regime = {0: 0.00035, 1: -0.00035, 2: 0.0}
    vol_by_regime = {0: 0.008, 1: 0.010, 2: 0.006}
    scale = (mins / 240) ** 0.5
    regime, rets = 0, []
    for _ in range(n):
        if rng.random() < 0.01:
            regime = int(rng.choice([0, 1, 2], p=[0.45, 0.25, 0.30]))
        rets.append(rng.normal(drift_by_regime[regime] * (mins / 240), vol_by_regime[regime] * scale))
    close = 30000 * np.exp(np.cumsum(rets))
    open_ = np.r_[30000, close[:-1]]
    spread = np.abs(rng.normal(0, 0.004 * scale, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    idx = pd.date_range(end=pd.Timestamp.now("UTC").floor(f"{mins}min"), periods=n, freq=f"{mins}min")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.uniform(500, 3000, n)}, index=idx)
    ret_trend = pd.Series(rets, index=idx).rolling(18).mean()
    funding = (0.0001 + ret_trend * 0.35).clip(-0.0008, 0.0012).iloc[::int(480 / mins) or 1]
    return df, funding, float(funding.iloc[-1]), float(rng.normal(1.5, 4))
