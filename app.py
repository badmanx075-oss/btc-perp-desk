"""BTC Perp Desk: Streamlit app. Run with:  streamlit run app.py"""
from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from desk import data as D
from desk.backtest import Costs, grid_search, run, stats
from desk.indicators import resample_ohlc
from desk.regime import describe
from desk.risk import build_ticket
from desk.signals import Params, build, live_factors, verdict

st.set_page_config(page_title="BTC Perp Desk", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Live exchange", "Demo (synthetic)"], index=1,
                      help="Demo mode needs no internet. Live mode uses public exchange endpoints, no API key.")
    exchange = st.selectbox("Exchange", list(D.EXCHANGES), disabled=source != "Live exchange")
    tf = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
    candles = st.slider("Candles", 800, 5000, 3000, step=100)

    st.header("Account")
    capital = st.number_input("Capital (USDT)", 100.0, 10_000_000.0, 1000.0, step=100.0)
    risk_pct = st.slider("Risk per trade %", 0.25, 2.0, 1.0, 0.25)
    max_lev = st.slider("Max leverage", 1.0, 5.0, 3.0, 0.5)

    st.header("Strategy")
    p = Params(
        adx_min=st.slider("Min ADX (trend strength)", 10, 35, 20),
        pullback_atr=st.slider("Pullback depth (ATR from EMA50)", 0.25, 2.0, 1.0, 0.25),
        stop_atr=st.slider("Stop distance (ATR)", 0.8, 3.0, 1.5, 0.1),
        rr=st.slider("Reward : Risk", 1.0, 4.0, 2.0, 0.5),
        fund_z_max=st.slider("Funding filter (z-score)", 0.5, 4.0, 2.0, 0.5),
        max_hold=st.slider("Max bars in trade", 12, 120, 48, 6),
        allow_long=st.checkbox("Allow longs", True),
        allow_short=st.checkbox("Allow shorts", True),
    )
    st.header("Costs")
    costs = Costs(
        fee_bps=st.number_input("Fee per side (bps)", 0.0, 20.0, 5.0, 0.5),
        slip_bps=st.number_input("Slippage per side (bps)", 0.0, 20.0, 2.0, 0.5),
        risk_pct=risk_pct, max_lev=max_lev, start_equity=capital,
    )
    st.header("Today")
    losses_today = st.selectbox("Losses so far today", [0, 1, 2, 3])
    news_day = st.checkbox("Big news today (CPI, FOMC)")
    weekend = st.checkbox("Weekend or thin liquidity")
    calm = st.checkbox("I am calm, not revenge trading", True)


# ---------------------------------------------------------------- data
@st.cache_data(ttl=60, show_spinner="Market data aa raha hai...")
def load(source: str, exchange: str, tf: str, candles: int):
    if source.startswith("Demo"):
        df, fund, fl, oi = D.synthetic(tf, max(candles, 3500), seed=7)
        daily = resample_ohlc(df, "1D") if tf != "1d" else df
        return df.tail(candles), daily, fund, fl, oi, None
    ex = D.EXCHANGES[exchange]
    try:
        df = D.fetch_ohlcv(ex, tf, candles)
        daily = D.fetch_ohlcv(ex, "1d", 400) if tf != "1d" else df
        fund = D.fetch_funding(ex)
        oi = D.fetch_oi_change(ex)
        fl = float(fund.iloc[-1]) if fund is not None and len(fund) else None
        return df, daily, fund, fl, oi, None
    except Exception as e:  # network blocked, geo restriction, rate limit
        return None, None, None, None, None, f"{type(e).__name__}: {e}"


df, daily, fund, fund_last, oi_chg, err = load(source, exchange, tf, candles)

st.title("BTC Perp Desk")
if err or df is None:
    st.error(f"Live data nahi mila ({err}). Dusra exchange try karo, ya sidebar mein Demo mode chuno. "
             "Binance kuch regions mein block hota hai, Bybit ya OKX try karo.")
    st.stop()
if source.startswith("Demo"):
    st.warning("Demo mode: ye nakli (synthetic) data hai. Isse real trade decision mat lo. Sirf tool samajhne ke liye.")
if len(df) < 600:
    st.error("Kam candles hain. Candles slider badhao.")
    st.stop()

tab_live, tab_bt, tab_opt, tab_journal, tab_about = st.tabs(
    ["Live decision", "Backtest", "Walk-forward", "Journal", "About"])

d = build(df, p, fund)
last = d.iloc[-1]

# ---------------------------------------------------------------- live decision
with tab_live:
    blocks = []
    if losses_today >= 2: blocks.append("Aaj 2 loss ho chuke. Din band.")
    if not calm: blocks.append("Calm nahi ho. Trade skip karo.")
    factors = live_factors(d, daily, fund_last, oi_chg)
    v, score = verdict(factors, blocks)
    colour = {"LONG": "green", "SHORT": "red", "NO TRADE": "orange"}[v]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC", f"{last['close']:,.0f}", f"{(last['close']/d['close'].iloc[-7]-1)*100:+.1f}% / 6 bars")
    c2.metric("Regime", last["regime"])
    c3.metric("ADX | ATR%", f"{last['adx']:.0f} | {last['atr_pct']:.1f}%")
    c4.metric("Score", f"{score:+.1f}", help="LONG: +3 ya zyada, SHORT: -3 ya kam, aur daily trend saath ho.")

    st.markdown(f"## Verdict: :{colour}[{v}]")
    st.caption(describe(last))
    for b in blocks:
        st.error(b)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Signal tape")
        for f in factors:
            s = f["score"]
            ico = "🟢" if s > 0 else "🔴" if s < 0 else "⚪"
            st.markdown(f"{ico} **{f['name']}** `{s:+.1f}`  \n<small>{f['note']}</small>", unsafe_allow_html=True)
        if fund_last is None:
            st.info("Is exchange se funding data nahi mila, isliye crowd signal skip hua.")
        if oi_chg is None:
            st.info("Open interest history nahi mili, OI signal skip hua.")
        st.caption("Liquidation heatmap ke liye Coinglass dekho. Ye free public data se nahi aata, isliye score mein sirf swing high/low se 'room to run' liya gaya hai.")

    with right:
        st.subheader("Trade ticket")
        if v == "NO TRADE":
            st.info("Bina setup ke trade nahi. Wait karo.")
            if abs(score) >= 3:
                st.caption("Score hai par daily trend ke against hai, isliye skip.")
        else:
            mult = 0.5 if (news_day or weekend) else 1.0
            t = build_ticket(v, float(last["close"]), float(last["ema_fast"]), float(last["atr"]), capital,
                             risk_pct, max_lev, p.stop_atr, p.rr, mult)
            if last["high_vol"]:
                st.warning("Volatility bahut high hai. Size aadha karne ka socho.")
            if mult < 1:
                st.warning("News ya weekend: size aadha kiya gaya.")
            tk = pd.DataFrame({
                "": ["Side", "Entry", "Stop-loss", "Target 1 (aadha book)", "Target 2", "Quantity (BTC)",
                     "Position (USDT)", "Leverage", "Risk (USDT)", "Liquidation (approx)"],
                "Value": [t.side, f"{t.entry:,.0f}", f"{t.stop:,.0f}", f"{t.target1:,.0f}", f"{t.target2:,.0f}",
                          f"{t.qty:.4f}", f"{t.notional:,.0f}", f"{t.leverage:.1f}x isolated", f"{t.risk_amt:,.1f}",
                          f"{t.liq_price:,.0f}" if t.liq_price else "n/a"]})
            st.dataframe(tk, hide_index=True, width="stretch")
            st.caption(t.note)
            if t.capped: st.warning("Size max leverage par cap hui. Asli risk set risk se kam hai.")
            if not t.liq_safe: st.error("Liquidation stop ke bahut paas. Leverage kam karo ya skip karo.")
            st.caption("Stop-loss exchange par actual order lagao. 1R par stop breakeven.")

    st.subheader("Chart")
    view = d.tail(220)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=view.index, open=view["open"], high=view["high"], low=view["low"],
                                 close=view["close"], name="BTC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["ema_fast"], name="EMA 50", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["ema_slow"], name="EMA 200", line=dict(width=1.5)), row=1, col=1)
    sg = view[view["signal"] != 0]
    fig.add_trace(go.Scatter(x=sg.index, y=sg["close"], mode="markers", name="Signal",
                             marker=dict(size=10, symbol=["triangle-up" if s > 0 else "triangle-down" for s in sg["signal"]],
                                         color=["#16a34a" if s > 0 else "#dc2626" for s in sg["signal"]])), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["adx"], name="ADX"), row=2, col=1)
    fig.add_hline(y=p.adx_min, line_dash="dot", row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- backtest
with tab_bt:
    st.subheader("Strategy history par kaisi chali")
    tr, _ = run(df, p, costs, fund)
    s = stats(tr, df, costs, p)
    if s["trades"] == 0:
        st.info("Is data par koi trade nahi bana. Filters halke karo ya candles badhao.")
    else:
        a, b, c, e = st.columns(4)
        a.metric("Trades", s["trades"])
        b.metric("Win rate", f"{s['win_rate']:.0f}%", f"break-even {s['breakeven_win_rate']:.0f}%", delta_color="off")
        c.metric("Avg R / trade", f"{s['avg_R']:+.2f}")
        e.metric("Profit factor", "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}")
        a, b, c, e = st.columns(4)
        a.metric("Strategy return", f"{s['total_return_pct']:+.1f}%")
        b.metric("Buy & hold", f"{s['buy_hold_pct']:+.1f}%")
        c.metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%")
        e.metric("Worst losing streak", s["worst_losing_streak"])
        if s["avg_R"] <= 0:
            st.warning("Average R zero ya negative hai. Ye settings is data par paisa nahi bana rahi.")
        st.caption("Win rate akela kuch nahi batata. 1:2 RR par ~34% win rate break-even hai. Dekhna hai Avg R aur drawdown.")

        fig = go.Figure(go.Scatter(x=tr["exit_time"], y=tr["equity"], mode="lines", name="Equity"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Equity (USDT)")
        st.plotly_chart(fig, width="stretch")

        reg = df.pipe(lambda x: build(x, p, fund))["regime"]
        tr2 = tr.copy()
        tr2["regime"] = reg.reindex(tr2["entry_time"]).values
        by = tr2.groupby("regime").agg(trades=("R", "size"), win_rate=("pnl", lambda x: (x > 0).mean() * 100),
                                       avg_R=("R", "mean")).round(2)
        st.markdown("**Regime ke hisaab se** (entry ke time ka regime)")
        st.dataframe(by, width="stretch")
        st.dataframe(tr.drop(columns=["equity_before"]).round({c: 2 for c in tr.select_dtypes("number").columns}), width="stretch", height=300)
        st.download_button("Trades CSV download", tr.to_csv(index=False).encode(), "backtest_trades.csv", "text/csv")

# ---------------------------------------------------------------- walk-forward
with tab_opt:
    st.subheader("Parameters pehle 60% data par chuno, baaki 40% par test karo")
    st.caption("Best parameters usi data par judge karna dhoka hai (overfitting). Yahan test hissa optimiser ne kabhi dekha nahi.")
    if st.button("Optimiser chalao (kuch second lagenge)"):
        with st.spinner("81 combinations test ho rahe hain..."):
            res, out = grid_search(df, p, costs, fund)
        if out is None:
            st.info("Kisi combination mein kaafi trades nahi bane. Candles badhao.")
        else:
            q, ts = out["params"], out["test_stats"]
            st.markdown(f"**Chune gaye parameters:** ADX≥{q.adx_min:g}, pullback {q.pullback_atr:g} ATR, "
                        f"stop {q.stop_atr:g} ATR, RR {q.rr:g}")
            if ts.get("trades", 0) == 0:
                st.info("Test hisse mein koi trade nahi bana.")
            else:
                a, b, c, e = st.columns(4)
                a.metric("Test trades", ts["trades"])
                b.metric("Test win rate", f"{ts['win_rate']:.0f}%", f"break-even {ts['breakeven_win_rate']:.0f}%", delta_color="off")
                c.metric("Test avg R", f"{ts['avg_R']:+.2f}")
                e.metric("Test max DD", f"{ts['max_drawdown_pct']:.1f}%")
                if ts["trades"] < 20:
                    st.warning(f"Test mein sirf {ts['trades']} trades hain. Itne kam trades se pakka nateeja nahi nikalta.")
                top_train = res.iloc[0]["train_avgR"]
                if ts["avg_R"] <= 0:
                    st.error("Train mein achha dikha par unseen data par kaam nahi kiya. Ye overfitting ka asli example hai.")
                elif ts["avg_R"] < 0.5 * top_train:
                    st.warning("Test result train se kaafi kamzor hai. Bhaarosa kam rakho.")
                else:
                    st.success("Test result train ke kareeb hai. Acha sign, par ye guarantee nahi.")
            st.dataframe(res.head(15).round(2), width="stretch")

# ---------------------------------------------------------------- journal
with tab_journal:
    st.subheader("Trade journal")
    st.caption("Har trade ka reason aur galti likho. CSV download karke GitHub ya Drive mein rakho (Streamlit Cloud pe file save nahi rehti).")
    if "journal" not in st.session_state:
        st.session_state.journal = pd.DataFrame(
            {"date": [pd.Timestamp.now().date()], "side": ["LONG"], "entry": [0.0], "exit": [0.0], "R": [0.0],
             "setup_reason": [""], "mistake": [""]})
    up = st.file_uploader("Purana journal CSV", type="csv")
    if up is not None and st.button("Load"):
        st.session_state.journal = pd.read_csv(up)
    st.session_state.journal = st.data_editor(st.session_state.journal, num_rows="dynamic", width="stretch")
    j = st.session_state.journal
    if len(j) and "R" in j and pd.to_numeric(j["R"], errors="coerce").notna().any():
        r = pd.to_numeric(j["R"], errors="coerce").dropna()
        st.metric("Journal avg R", f"{r.mean():+.2f}", f"{(r>0).mean()*100:.0f}% win rate over {len(r)} trades", delta_color="off")
    buf = io.StringIO(); j.to_csv(buf, index=False)
    st.download_button("Journal CSV download", buf.getvalue().encode(), "journal.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Ye kaise decide karta hai**
1. *Regime:* daily EMA50/200 aur ADX se uptrend, downtrend ya range.
2. *Setup:* sirf trend ke saath pullback. Price EMA50 ke paas aaye, RSI dip kare, phir pichhli candle ka high/low tode.
3. *Crowd filter:* funding z-score bahut high ho to long skip, bahut negative ho to short skip.
4. *Risk:* 1% risk, ATR stop, max leverage cap, liquidation stop se door.

**Honest baatein**
- Win rate koi guarantee nahi. Trend strategies mein 35-45% normal hai, edge 1:2 RR se aata hai. Isliye app Avg R aur drawdown dikhata hai.
- Backtest past hai. Future alag ho sakta hai. Walk-forward tab isliye hai.
- Demo mode ka data nakli hai. Real analysis ke liye Live exchange chuno.
- Ye education ke liye hai, financial advice nahi. Pehle testnet par chalao, phir chhoti capital se.
- India: crypto gains par 30% tax + 1% TDS, loss set-off nahi.
""")
