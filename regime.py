"""Market regime classification. The strategy only trades regimes it was built for."""
from __future__ import annotations

import pandas as pd

from .indicators import adx, atr, ema, rolling_rank

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
