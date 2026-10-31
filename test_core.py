import numpy as np
import pandas as pd

from desk.backtest import Costs, run, stats, grid_search
from desk.data import synthetic
from desk.risk import build_ticket
from desk.signals import Params, build, live_factors, verdict


def test_no_lookahead_signals_unchanged_when_future_added():
    df, fund, *_ = synthetic("4h", 1500, seed=3)
    p = Params()
    full = build(df, p, fund)["signal"]
    part = build(df.iloc[:1200], p, fund)["signal"]
    assert (full.iloc[:1200].values == part.values).all()


def test_stop_wins_when_both_levels_inside_one_bar():
    idx = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
    close = 100 * 1.005 ** np.arange(400)
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0}, index=idx)
    p = Params(adx_min=0, pullback_atr=50, rsi_reset=101, fund_z_max=99)
    tr, d = run(df, p, Costs(fee_bps=0, slip_bps=0))
    # force a single signal then a huge bar around the entry
    d2 = df.copy()
    sig_i = int(d.index.get_loc(tr.iloc[0]["entry_time"])) - 1
    d2.iloc[sig_i + 1, d2.columns.get_loc("high")] = d2["close"].iloc[sig_i] * 1.5
    d2.iloc[sig_i + 1, d2.columns.get_loc("low")] = d2["close"].iloc[sig_i] * 0.5
    tr2, _ = run(d2, p, Costs(fee_bps=0, slip_bps=0))
    first = tr2.iloc[0]
    assert first["reason"] == "stop" and first["R"] < 0


def test_costs_reduce_pnl():
    df, fund, *_ = synthetic("4h", 3000, seed=5)
    p = Params()
    a, _ = run(df, p, Costs(fee_bps=0, slip_bps=0), fund)
    b, _ = run(df, p, Costs(fee_bps=10, slip_bps=5), fund)
    assert len(a) > 5
    assert b["pnl"].sum() < a["pnl"].sum()


def test_ticket_risk_and_liquidation():
    t = build_ticket("LONG", 100_000, 99_000, 2_000, capital=1000, risk_pct=1, max_lev=3, stop_atr=1.5, rr=2)
    assert abs(t.stop - 97_000) < 1e-6
    assert t.leverage <= 3 + 1e-9
    assert t.liq_safe
    s = build_ticket("SHORT", 100_000, 101_000, 2_000, capital=1000, risk_pct=1, max_lev=3)
    assert s.stop > s.entry and s.target1 < s.entry


def test_stats_and_grid_run():
    df, fund, *_ = synthetic("4h", 3500, seed=11)
    p = Params()
    c = Costs()
    tr, _ = run(df, p, c, fund)
    s = stats(tr, df, c, p)
    assert s["trades"] > 5 and 0 <= s["win_rate"] <= 100
    res, out = grid_search(df, p, c, fund, min_trades=5)
    assert out is None or "test_stats" in out


def test_verdict_blocks():
    f = [{"name": "t", "score": 2.0, "note": ""}, {"name": "x", "score": 2.0, "note": ""}]
    assert verdict(f, [])[0] == "LONG"
    assert verdict(f, ["loss limit"])[0] == "NO TRADE"
    f[0]["score"] = -1.0
    assert verdict(f, [])[0] == "NO TRADE"


def test_live_factors_shape():
    df, fund, fl, oi = synthetic("4h", 1500, seed=2)
    d = build(df, Params(), fund)
    f = live_factors(d, None, fl, oi)
    assert len(f) == 5 and all("score" in x for x in f)
