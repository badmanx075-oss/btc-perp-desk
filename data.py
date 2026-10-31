"""Market data. Live via ccxt public endpoints (no API key needed) or a synthetic demo series."""
from __future__ import annotations

import numpy as np
import pandas as pd

EXCHANGES = {"Binance Futures": "binanceusdm", "Bybit": "bybit", "OKX": "okx"}
SYMBOL = "BTC/USDT:USDT"
TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def _ex(name: str):
    import ccxt  # imported lazily so tests and demo mode work without it
    return getattr(ccxt, name)({"enableRateLimit": True, "timeout": 15000})


def fetch_ohlcv(exchange: str, timeframe: str, limit: int = 1500) -> pd.DataFrame:
    ex = _ex(exchange)
    step = TF_MINUTES[timeframe] * 60_000
    per_call = 1000 if exchange != "okx" else 100
    now = ex.milliseconds()
    since = now - limit * step
    rows: list = []
    while len(rows) < limit:
        chunk = ex.fetch_ohlcv(SYMBOL, timeframe, since=since, limit=per_call)
        if not chunk:
            break
        rows += chunk
        nxt = chunk[-1][0] + step
        if nxt <= since:
            break
        since = nxt
        if nxt >= now:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df = df.sort_index().tail(limit)
    return df.iloc[:-1]  # drop the still-forming candle


def fetch_funding(exchange: str, limit: int = 600) -> pd.Series | None:
    try:
        ex = _ex(exchange)
        rows = ex.fetch_funding_rate_history(SYMBOL, limit=limit)
        s = pd.Series({pd.to_datetime(r["timestamp"], unit="ms", utc=True): r["fundingRate"] for r in rows})
        return s.sort_index() if len(s) else None
    except Exception:
        return None


def fetch_oi_change(exchange: str, hours: int = 24) -> float | None:
    """Percent change in open interest over the last `hours`. None if the venue does not expose it."""
    try:
        ex = _ex(exchange)
        rows = ex.fetch_open_interest_history(SYMBOL, "1h", limit=hours + 1)
        vals = [r.get("openInterestValue") or r.get("openInterestAmount") for r in rows]
        vals = [v for v in vals if v]
        if len(vals) < 2:
            return None
        return (vals[-1] / vals[0] - 1) * 100
    except Exception:
        return None


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
    idx = pd.date_range(end=pd.Timestamp.now('UTC').floor(f"{mins}min"), periods=n, freq=f"{mins}min")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.uniform(500, 3000, n)}, index=idx)
    ret_trend = pd.Series(rets, index=idx).rolling(18).mean()
    funding = (0.0001 + ret_trend * 0.35).clip(-0.0008, 0.0012).iloc[::int(480 / mins) or 1]
    return df, funding, float(funding.iloc[-1]), float(rng.normal(1.5, 4))
