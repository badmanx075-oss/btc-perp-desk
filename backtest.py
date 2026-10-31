"""Bar-by-bar backtester for perpetual futures.

Conservative rules:
- Signal at close of bar i, entry at OPEN of bar i+1 (no lookahead).
- If stop and target are both inside one bar, the STOP is assumed to hit first.
- Fees and slippage are charged on both sides.
- One position at a time. Position size comes from risk %, capped by max leverage.
- Funding paid/received while in the trade is included when a funding series is given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import Params, build


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
