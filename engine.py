"""BTC Perp Desk engine: indicators, regime, signals, backtest, risk sizing and market data.

Single file on purpose, so it can be uploaded to GitHub from a phone without folders.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import requests

# ======================================================================
# indicators
# ======================================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


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
    out = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna()


# ======================================================================
# regime
# ======================================================================
UP, DOWN, RANGE = "UPTREND", "DOWNTREND", "RANGE"


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


# ======================================================================
# signals
# ======================================================================
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


# ======================================================================
# backtest
# ======================================================================
@dataclass
class Costs:
    fee_bps: float = 5.0        # per side (taker ~4-5 bps on most perp venues)
    slip_bps: float = 2.0       # per side
    risk_pct: float = 1.0
    max_lev: float = 3.0
    start_equity: float = 1000.0


def run(df: pd.DataFrame, p: Params, c: Costs, funding: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = build(df, p, funding)
    idx = d.index
    o, h, l, cl = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    atr_v, sig = d["atr"].values, d["signal"].values
    n = len(d)
    warm = max(p.ema_slow, 220)

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
        entry = o[entry_i] * (1 + side * c.slip_bps / 1e4)
        sdist = p.stop_atr * atr_v[i]
        stop = entry - side * sdist
        target = entry + side * p.rr * sdist
        stop_pct = sdist / entry
        notional = min(eq * c.risk_pct / 100 / stop_pct, eq * c.max_lev)

        exit_px, exit_i, why = None, None, None
        last = min(entry_i + p.max_hold, n - 1)
        for j in range(entry_i, last + 1):
            hit_stop = l[j] <= stop if side == 1 else h[j] >= stop
            hit_tgt = h[j] >= target if side == 1 else l[j] <= target
            if hit_stop:               # stop wins ties
                exit_px, exit_i, why = stop, j, "stop"
                break
            if hit_tgt:
                exit_px, exit_i, why = target, j, "target"
                break
        if exit_px is None:
            exit_i, exit_px, why = last, cl[last], "time"
        exit_px = exit_px * (1 - side * c.slip_bps / 1e4)

        gross = notional * side * (exit_px - entry) / entry
        fees = notional * 2 * c.fee_bps / 1e4
        fund_cost = 0.0
        if fund_rate is not None:
            # funding paid per 8h; approximate by summing per-row rate scaled to bar length
            bars_held = exit_i - entry_i + 1
            bar_hours = max((idx[1] - idx[0]).total_seconds() / 3600, 1e-9)
            avg = float(np.mean(fund_rate[entry_i:exit_i + 1]))
            fund_cost = notional * side * avg * (bars_held * bar_hours / 8)
        pnl = gross - fees - fund_cost
        risk_amt = eq * c.risk_pct / 100
        trades.append({
            "entry_time": idx[entry_i], "exit_time": idx[exit_i], "side": "LONG" if side == 1 else "SHORT",
            "entry": entry, "exit": exit_px, "stop": stop, "target": target, "reason": why,
            "bars": exit_i - entry_i + 1, "notional": notional, "pnl": pnl, "R": pnl / risk_amt,
            "equity_before": eq,
        })
        eq += pnl
        i = exit_i + 1
        if eq <= 0:
            break

    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, d
    tr["equity"] = c.start_equity + tr["pnl"].cumsum()
    return tr, d


def stats(tr: pd.DataFrame, df: pd.DataFrame, c: Costs, p: Params) -> dict:
    if tr is None or tr.empty:
        return {"trades": 0}
    wins, losses = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
    gp, gl = wins["pnl"].sum(), -losses["pnl"].sum()
    eqc = pd.concat([pd.Series([c.start_equity]), tr["equity"]], ignore_index=True)
    dd = (eqc / eqc.cummax() - 1).min() * 100
    first, last = df["close"].iloc[0], df["close"].iloc[-1]
    return {
        "trades": len(tr),
        "win_rate": len(wins) / len(tr) * 100,
        "breakeven_win_rate": 100 / (1 + p.rr),
        "avg_R": tr["R"].mean(),
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "total_return_pct": (tr["equity"].iloc[-1] / c.start_equity - 1) * 100,
        "max_drawdown_pct": dd,
        "avg_bars": tr["bars"].mean(),
        "longs": int((tr["side"] == "LONG").sum()),
        "shorts": int((tr["side"] == "SHORT").sum()),
        "buy_hold_pct": (last / first - 1) * 100,
        "worst_losing_streak": _max_streak(tr["pnl"] <= 0),
    }


def _max_streak(mask: pd.Series) -> int:
    best = cur = 0
    for v in mask.values:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def grid_search(df: pd.DataFrame, base: Params, c: Costs, funding: pd.Series | None, train_frac: float = 0.6,
                min_trades: int = 15) -> tuple[pd.DataFrame, dict | None]:
    """Pick parameters on the first part of the data, then judge them on the unseen last part."""
    cut = int(len(df) * train_frac)
    train, test = df.iloc[:cut], df.iloc[cut - 250:]   # test keeps 250 bars of warm-up history
    rows = []
    for adx_min in (15, 20, 25):
        for pb in (0.5, 1.0, 1.5):
            for stop_atr in (1.2, 1.5, 2.0):
                for rr in (1.5, 2.0, 3.0):
                    q = Params(**{**base.as_dict(), "adx_min": adx_min, "pullback_atr": pb, "stop_atr": stop_atr, "rr": rr})
                    tr, _ = run(train, q, c, funding)
                    s = stats(tr, train, c, q)
                    if s.get("trades", 0) < min_trades:
                        continue
                    rows.append({"adx_min": adx_min, "pullback_atr": pb, "stop_atr": stop_atr, "rr": rr,
                                 "train_trades": s["trades"], "train_win%": s["win_rate"], "train_avgR": s["avg_R"],
                                 "train_PF": s["profit_factor"]})
    if not rows:
        return pd.DataFrame(), None
    res = pd.DataFrame(rows).sort_values("train_avgR", ascending=False).reset_index(drop=True)
    best = res.iloc[0]
    q = Params(**{**base.as_dict(), "adx_min": best["adx_min"], "pullback_atr": best["pullback_atr"],
                  "stop_atr": best["stop_atr"], "rr": best["rr"]})
    tr_t, _ = run(test, q, c, funding)
    tr_t = tr_t[tr_t["entry_time"] >= df.index[cut]] if not tr_t.empty else tr_t
    if not tr_t.empty:
        tr_t = tr_t.copy()
        tr_t["equity"] = c.start_equity + tr_t["pnl"].cumsum()
    out = {"params": q, "test_stats": stats(tr_t, test, c, q), "test_trades": tr_t}
    return res, out


# ======================================================================
# risk
# ======================================================================
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
    liq = None
    safe = True
    if lev > 1.0:
        liq = entry * (1 - sd * (1 / lev - maint_margin))
        safe = (liq < stop * 0.97) if sd == 1 else (liq > stop * 1.03)
    return Ticket(side, entry, stop, t1, t2, qty, notional, lev, risk_amt, liq, safe, capped, note)


# ======================================================================
# data
# ======================================================================
SOURCES = ["Hyperliquid", "OKX", "Bybit", "Binance"]
CCXT_ID = {"OKX": "okx", "Bybit": "bybit", "Binance": "binanceusdm"}
SYMBOL = "BTC/USDT:USDT"
TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
HL_URL = "https://api.hyperliquid.xyz/info"


# ----------------------------------------------------------------- Hyperliquid (plain REST)
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
    return df.iloc[:-1]  # drop the still-forming candle


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
        s = pd.Series(out).sort_index()
        return s.resample("8h").mean().dropna()
    except Exception:
        return None


def hl_live() -> dict | None:
    """Current mark price, funding and open interest straight from the exchange."""
    try:
        meta, ctxs = _hl({"type": "metaAndAssetCtxs"})
        i = [u["name"] for u in meta["universe"]].index("BTC")
        c = ctxs[i]
        px = float(c["markPx"])
        return {"price": px, "funding_8h": float(c["funding"]) * 8,
                "oi_usd": float(c["openInterest"]) * px, "prev_day": float(c["prevDayPx"])}
    except Exception:
        return None


# ----------------------------------------------------------------- ccxt venues
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
    """24h open-interest change. Tries venues in turn; returns (value or None, venue used)."""
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


# ----------------------------------------------------------------- one call for the app
def fetch_live(timeframe: str, candles: int, prefer: str = "Auto") -> dict:
    """Returns dict(df, daily, fund, fund_last, oi, price, source, errors). Raises only if every venue fails."""
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


# ----------------------------------------------------------------- demo
def synthetic(timeframe: str = "4h", n: int = 3000, seed: int = 7) -> tuple[pd.DataFrame, pd.Series, float, float]:
    """Regime-switching random walk for demo mode. Not real BTC, just realistic enough to exercise the tool."""
    rng = np.random.default_rng(seed)
    mins = TF_MINUTES[timeframe]
    drift_by_regime = {0: 0.00035, 1: -0.00035, 2: 0.0}
    vol_by_regime = {0: 0.008, 1: 0.010, 2: 0.006}
    scale = (mins / 240) ** 0.5
    price, regime, rets = 30000.0, 0, []
    for _ in range(n):
        if rng.random() < 0.01:
            regime = int(rng.choice([0, 1, 2], p=[0.45, 0.25, 0.30]))
        r = rng.normal(drift_by_regime[regime] * (mins / 240), vol_by_regime[regime] * scale)
        rets.append(r)
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
