"""BTC Perp Desk v2: Streamlit app.  Run with:  streamlit run app.py"""
from __future__ import annotations

import io
import math
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402
from engine import Costs, Params  # noqa: E402

st.set_page_config(page_title="BTC Perp Desk", page_icon="📈", layout="wide")

# =============================================================================== sidebar
EXIT_STYLES = {
    "Partial 50% @1R + breakeven (recommended)": (0.5, 0.0),
    "Fixed target (all-in, all-out)": (0.0, 0.0),
    "Partial 50% + ATR trail (winners ko chalne do)": (0.5, 2.5),
}
_qp = st.query_params       # ?demo=1&seed=3&tf=1h pre-selects a demo scenario (handy for sharing and testing)
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Live exchange", "Demo (synthetic)"], index=1 if _qp.get("demo") == "1" else 0,
                      help="Live mode public exchange endpoints use karta hai, API key nahi chahiye. Demo nakli data hai.")
    exchange = st.selectbox("Exchange", ["Auto"] + E.SOURCES, disabled=source != "Live exchange",
                            help="Auto: Hyperliquid, phir OKX, Bybit, Binance. Jo pehle chale wahi use hota hai.")
    demo_seed = st.number_input("Demo scenario (seed)", 1, 999, int(_qp.get("seed", 7)), disabled=source == "Live exchange",
                                help="Demo mode mein alag-alag nakli market situations dekhne ke liye number badlo.")
    tf = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=["1h", "4h", "1d"].index(_qp.get("tf", "4h")) if _qp.get("tf", "4h") in ("1h", "4h", "1d") else 1)
    candles = st.slider("Candles", 800, 5000, 3000, step=100)

    st.header("Account")
    capital = st.number_input("Capital (USDT)", 100.0, 10_000_000.0, 1000.0, step=100.0)
    risk_pct = st.slider("Risk per trade %", 0.25, 2.0, 1.0, 0.25)
    max_lev = st.slider("Max leverage", 1.0, 5.0, 3.0, 0.5)

    st.header("Strategy")
    exit_style = st.selectbox("Exit style", list(EXIT_STYLES), help="Partial exit win rate badhata hai, par edge apne aap nahi badhata.")
    pf, tr_atr = EXIT_STYLES[exit_style]
    with st.expander("Entry filters", expanded=False):
        adx_min = st.slider("Min ADX (trend strength)", 10, 35, 20)
        pullback = st.slider("Pullback depth (ATR from EMA50)", 0.25, 2.0, 1.0, 0.25)
        use_htf = st.checkbox("Daily trend filter", True, help="Trade sirf daily trend ki direction mein.")
        min_room = st.slider("Min room to recent high/low (ATR)", 0.0, 3.0, 1.0, 0.25)
        min_rvol = st.slider("Min relative volume of trigger candle", 0.0, 2.0, 0.0, 0.1, help="0 = off")
        fund_z = st.slider("Funding filter (z-score)", 0.5, 4.0, 2.0, 0.5)
        cooldown = st.slider("Cooldown bars between signals", 0, 10, 3)
    with st.expander("Stops, targets, direction", expanded=False):
        stop_atr = st.slider("Stop distance (ATR)", 0.8, 3.0, 1.5, 0.1)
        rr = st.slider("Final target (R)", 1.0, 4.0, 2.0, 0.5)
        max_hold = st.slider("Max bars in trade", 12, 120, 48, 6)
        allow_long = st.checkbox("Allow longs", True)
        allow_short = st.checkbox("Allow shorts", True)
    with st.expander("Live plan (levels)", expanded=False):
        major_score = st.slider("Major level strength", 2.0, 8.0, 4.0, 0.5, help="Kam = zyada levels 'major' maane jaate hain.")
        min_room_r = st.slider("Min room to next major level (R)", 0.5, 3.0, 1.0, 0.25)
    p = Params(adx_min=adx_min, pullback_atr=pullback, stop_atr=stop_atr, rr=rr, fund_z_max=fund_z, max_hold=max_hold,
               use_htf=use_htf, min_room_atr=min_room, min_rvol=min_rvol, cooldown=cooldown, partial_frac=pf,
               trail_atr=tr_atr, major_score=major_score, min_room_r=min_room_r, allow_long=allow_long,
               allow_short=allow_short)

    st.header("Costs")
    costs = Costs(fee_bps=st.number_input("Fee per side (bps)", 0.0, 20.0, 5.0, 0.5),
                  slip_bps=st.number_input("Slippage per side (bps)", 0.0, 20.0, 2.0, 0.5),
                  risk_pct=risk_pct, max_lev=max_lev, start_equity=capital)

    st.header("Today")
    losses_today = st.selectbox("Losses so far today", [0, 1, 2, 3])
    news_day = st.checkbox("Big news today (CPI, FOMC)")
    weekend = st.checkbox("Weekend or thin liquidity")
    calm = st.checkbox("I am calm, not revenge trading", True)


# =============================================================================== data
@st.cache_data(ttl=60, show_spinner="Market data aa raha hai...")
def load(source: str, exchange: str, tf: str, candles: int, seed: int = 7):
    if source.startswith("Demo"):
        df, fund, fl, oi = E.synthetic(tf, max(candles, 3500), seed=int(seed))
        daily = E.resample_ohlc(df, "1D") if tf != "1d" else df
        return {"df": df.tail(candles), "daily": daily, "fund": fund, "fund_last": fl, "oi": oi, "oi_src": "demo",
                "price": float(df["close"].iloc[-1]), "source": "Demo", "errors": []}, None
    try:
        return E.fetch_live(tf, candles, exchange), None
    except Exception as e:
        return None, str(e)


data, err = load(source, exchange, tf, candles, demo_seed)
st.title("BTC Perp Desk")
if err or data is None:
    st.error("Live data kisi bhi exchange se nahi aaya.")
    st.code(err or "unknown error")
    st.info("Thodi der baad Refresh karo, ya Exchange dropdown mein ek ek karke try karo. Tool samajhne ke liye Demo mode chuno.")
    st.stop()

df, daily, fund = data["df"], data["daily"], data["fund"]
fund_last, oi_chg, live_price = data["fund_last"], data["oi"], data["price"]
if source.startswith("Demo"):
    st.warning("Demo mode: ye nakli (synthetic) data hai. Isse real trade decision mat lo. Sirf tool samajhne ke liye.")
else:
    st.caption(f"Data source: **{data['source']}** (candles + funding), OI: {data['oi_src'] or 'nahi mila'}. Har 60 second mein refresh.")
    if data["errors"]:
        with st.expander("Pehle ke exchanges jo fail hue"):
            st.write("\n".join(data["errors"]))
with st.sidebar.expander("Manual override (agar feed na mile)"):
    mf = st.number_input("Funding % per 8h", -0.5, 0.5, 0.0, 0.001, format="%.3f")
    use_mf = st.checkbox("Funding manual use karo", False)
    mo = st.number_input("OI 24h change %", -50.0, 50.0, 0.0, 0.5)
    use_mo = st.checkbox("OI manual use karo", False)
    if use_mf: fund_last = mf / 100
    if use_mo: oi_chg = mo
if len(df) < 600:
    st.error("Kam candles hain. Candles slider badhao.")
    st.stop()

tab_live, tab_bt, tab_wf, tab_journal, tab_about = st.tabs(["Live decision", "Backtest", "Walk-forward", "Journal", "About"])
d = E.build(df, p, fund)
last = d.iloc[-1]
A = E.analyse_live(d, daily, live_price, fund_last, oi_chg, p)
plan, grade = A["plan"], A["grade"]
SIDE = {1: "LONG", -1: "SHORT", 0: ""}[plan["side"]]
ICON = {"ok": "✅", "warn": "⚠️", "bad": "❌"}
COLOR = {"go": "green", "ready": "blue", "wait": "orange", "avoid": "red"}


def fmt(x: float) -> str:
    return f"{x:,.0f}"


def make_chart(zoom: bool = True, bars: int = 180) -> go.Figure:
    view = d.tail(bars)
    if zoom:        # focus on the tradable band around live price, widened to hold zone, stop and targets
        lo, hi = live_price - 5 * plan["atr"], live_price + 5 * plan["atr"]
        for k in ("stop", "partial_px", "target_px"):
            if plan.get(k):
                lo, hi = min(lo, plan[k]), max(hi, plan[k])
        if plan.get("zone"):
            lo, hi = min(lo, plan["zone"][0]), max(hi, plan["zone"][1])
    else:
        lo, hi = float(view["low"].min()), float(view["high"].max())
        if plan.get("zone"):
            lo, hi = min(lo, plan["zone"][0], plan["stop"]), max(hi, plan["zone"][1], plan["stop"])
        lo, hi = min(lo, live_price), max(hi, live_price)
    pad = 0.03 * (hi - lo)
    y0, y1 = lo - pad, hi + pad
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view.index, open=view["open"], high=view["high"], low=view["low"], close=view["close"], name="BTC"))
    fig.add_trace(go.Scatter(x=view.index, y=view["ema_fast"], name=f"EMA{p.ema_fast}", line=dict(width=1.4, color="#f59e0b")))
    fig.add_trace(go.Scatter(x=view.index, y=view["ema_slow"], name=f"EMA{p.ema_slow}", line=dict(width=1.4, color="#60a5fa")))
    for name, ser in A["avwaps"].items():
        s = ser.reindex(view.index)
        if s.notna().any():
            fig.add_trace(go.Scatter(x=view.index, y=s, name=name, line=dict(width=1.2, dash="dash", color="#a78bfa" if "low" in name else "#f472b6")))
    lv = A["levels"]
    if len(lv):
        show = lv[(lv["price"] > y0) & (lv["price"] < y1) & (lv["score"] >= 2.0)].sort_values("score", ascending=False).head(8)
        for _, r in show.iterrows():
            col = "#16a34a" if r["price"] < live_price else "#dc2626"
            fig.add_hline(y=r["price"], line_dash="dot", line_width=1, line_color=col, opacity=0.6,
                          annotation_text=f"{r['label'][:26]} ({r['score']:.0f})", annotation_position="right", annotation_font_size=10)
    if plan.get("zone"):
        zl, zh = plan["zone"]
        fig.add_hrect(y0=zl, y1=zh, fillcolor="rgba(22,163,74,0.20)", line_width=0, annotation_text="ENTRY ZONE",
                      annotation_position="top left", annotation_font_color="#16a34a", annotation_font_size=10)
        fig.add_hline(y=plan["stop"], line_color="#dc2626", line_dash="dash", line_width=1.2, annotation_text="Stop", annotation_position="top left", annotation_font_color="#dc2626")
        if plan.get("partial_px"):
            fig.add_hline(y=plan["partial_px"], line_color="#22c55e", line_dash="dash", line_width=1, annotation_text="Partial 1R", annotation_position="top left", annotation_font_color="#22c55e")
        if plan.get("target_px"):
            fig.add_hline(y=plan["target_px"], line_color="#22c55e", line_dash="dash", line_width=1, annotation_text="Target", annotation_position="top left", annotation_font_color="#22c55e")
    for a, b, lab in plan.get("avoid", []):
        a2, b2 = max(a, y0), min(b, y1)
        if a2 < b2:
            fig.add_hrect(y0=a2, y1=b2, fillcolor="rgba(220,38,38,0.16)", line_width=0, annotation_text="AVOID",
                          annotation_position="top left", annotation_font_color="#dc2626", annotation_font_size=10)
    fig.add_hline(y=live_price, line_color="#9ca3af", line_dash="dash", line_width=1, annotation_text="Live", annotation_position="right")
    sg = view[view["signal"] != 0]
    if len(sg):
        fig.add_trace(go.Scatter(x=sg.index, y=np.where(sg["signal"] > 0, sg["low"], sg["high"]), mode="markers", name="Past signal",
                                 marker=dict(size=9, symbol=["triangle-up" if s > 0 else "triangle-down" for s in sg["signal"]],
                                             color=["#16a34a" if s > 0 else "#dc2626" for s in sg["signal"]])))
    fig.update_layout(height=640, xaxis_rangeslider_visible=False, margin=dict(l=10, r=170, t=10, b=10),
                      legend=dict(orientation="h", y=1.04), yaxis=dict(range=[y0, y1]))
    return fig


# =============================================================================== live decision
with tab_live:
    blocks = []
    if losses_today >= 2: blocks.append("Aaj 2 loss ho chuke. Din band.")
    if not calm: blocks.append("Calm nahi ho. Trade skip karo.")
    dec = E.decide(plan, grade, blocks, news_day or weekend, bool(last["high_vol"]))

    st.markdown(f"## :{COLOR[dec['kind']]}[{dec['action']}]")
    st.markdown(dec["why"])
    htf_txt = {1.0: "UP", -1.0: "DOWN", 0.0: "NEUTRAL"}.get(A["htf"], "n/a")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BTC live", fmt(live_price), f"{(live_price / d['close'].iloc[-7] - 1) * 100:+.1f}% (6 bars)")
    c2.metric("Daily bias", htf_txt)
    c3.metric("Regime", last["regime"], help=f"ADX {last['adx']:.0f}, ATR {last['atr_pct']:.1f}%")
    c4.metric("Structure", A["struct"][0])
    c5.metric("Setup grade", f"{grade['letter']} ({grade['total']:.1f}/{grade['max']:.0f})" if plan["side"] else "-",
              help="A = poora size, B = aadha, C = skip. Detail neeche expander mein.")

    # ---- where to take / where to avoid
    col_take, col_avoid = st.columns(2)
    with col_take:
        with st.container(border=True):
            st.subheader("✅ Kahan trade lena hai")
            if plan.get("zone"):
                zl, zh = plan["zone"]
                st.markdown(f"**{SIDE} entry zone:** `{fmt(zl)} – {fmt(zh)}`  \n<small>{plan['conf_label']} (confluence {plan['conf_score']:.1f})</small>",
                            unsafe_allow_html=True)
                st.markdown(f"**Trigger:** {plan['trigger_text']}. Bina trigger entry nahi.")
                st.markdown(f"**Stop-loss:** `{fmt(plan['stop'])}` ({p.stop_atr:g} ATR)"
                            + (f"  \n<small>Structure invalidation: {fmt(plan['swing_stop'])} (swing {'low' if plan['side'] == 1 else 'high'} ke bahar)</small>"
                               if plan.get("swing_stop") else ""), unsafe_allow_html=True)
                tg = []
                if plan.get("partial_px"): tg.append(f"**{p.partial_frac * 100:.0f}% profit book** `{fmt(plan['partial_px'])}` (1R), phir stop breakeven")
                if plan.get("target_px"): tg.append(f"**Final target** `{fmt(plan['target_px'])}` ({p.rr:g}R)")
                else: tg.append(f"**Baaki position:** ATR trail ({p.trail_atr:g} ATR), fixed target nahi")
                st.markdown("  \n".join(tg))
                if plan["targets"]:
                    st.caption("Raste ke levels (R = stop distance ke multiple):")
                    st.dataframe(pd.DataFrame([{"Level": fmt(t["price"]),
                                                "Kya hai (★ = major)": t["label"] + (" ★" if (t["score"] >= p.major_score and t["sources"] >= 2) else ""),
                                                "R": round(t["R"], 1)} for t in plan["targets"]]), hide_index=True, width="stretch")
                st.caption(f"Alerts lagao (TradingView): {fmt(zh)} (zone ke upar) aur {fmt(zl)} (zone ke neeche). Order tabhi jab trigger candle close ho.")
            else:
                st.markdown("Abhi koi clear side nahi. Ye levels dekho:")
                for lvl, txt in plan.get("watch", []):
                    st.markdown(f"- {txt}")

    with col_avoid:
        with st.container(border=True):
            st.subheader("⛔ Kahan avoid karna hai")
            for a, b, lab in plan.get("avoid", []):
                st.markdown(f"**`{fmt(a)} – {fmt(b)}`**  \n<small>{lab}</small>", unsafe_allow_html=True)
            extra = []
            if fund_last is not None and plan["side"] and plan["side"] * fund_last * 100 >= 0.05:
                extra.append("Funding tumhari side ki taraf crowded hai. Size aadha ya skip.")
            if news_day: extra.append("Bade news ka din. Size aadha, trade sirf trigger ke baad.")
            if weekend: extra.append("Weekend ya thin liquidity. Wicks bade, size aadha.")
            if bool(last["high_vol"]): extra.append("Volatility top 10% mein. Size aadha.")
            if A["bb_rank"] < 0.2: extra.append("Bollinger squeeze. Breakout se pehle fake moves aate hain.")
            if plan["side"] and plan["kind"] == "avoid":
                extra.append(f"Abhi ki wajah: {plan['headline']}")
            for x in extra: st.markdown(f"- {x}")
            if not plan.get("avoid") and not extra: st.markdown("Koi khaas avoid zone nahi.")

    st.subheader("Chart: hara = entry zone, laal = avoid, dotted = levels")
    zoom = st.checkbox("Trade area par zoom (poora chart dekhne ke liye band karo)", True)
    st.plotly_chart(make_chart(zoom), width="stretch")

    # ---- how the analysis was done
    st.subheader("Analysis kaise hua (step by step)")
    for kind, title, txt in A["steps"]:
        st.markdown(f"{ICON[kind]} **{title}.** {txt}")

    if plan["side"]:
        with st.expander(f"Setup grade {grade['letter']}: {grade['total']:.1f}/{grade['max']:.0f} (A = poora size, B = aadha, C = skip)", expanded=False):
            st.dataframe(pd.DataFrame([{"Check": i["name"], "Score": f"{i['got']:g}/{i['max']:g}", "Detail": i["note"]} for i in grade["items"]]),
                         hide_index=True, width="stretch")

    # ---- ticket
    st.subheader("Trade ticket")
    if dec["mult"] > 0 and plan["side"]:
        t = E.build_ticket(SIDE, plan["entry"], plan["entry"], plan["atr"], capital, risk_pct, max_lev, p.stop_atr, p.rr, dec["mult"])
        if dec["kind"] != "go":
            st.info("Ye plan hai, abhi order mat lagao. Trigger candle close hone par hi.")
        if dec["mult"] < 1:
            st.caption(f"Size {dec['mult']:g}x (grade, news, weekend ya volatility ke hisaab se).")
        a, b, c = st.columns(3)
        a.metric("Quantity (BTC)", f"{t.qty:.4f}", help=f"Position size {fmt(t.notional)} USDT")
        b.metric("Leverage", f"{t.leverage:.1f}x isolated")
        c.metric("Risk", f"{t.risk_amt:,.1f} USDT", help=f"{t.risk_amt / capital * 100:.2f}% of capital")
        if t.capped: st.warning("Size max leverage par cap hui. Asli risk set risk se kam hai.")
        if t.liq_price: st.caption(f"Approx liquidation: {fmt(t.liq_price)}")
        if not t.liq_safe: st.error("Liquidation stop ke bahut paas. Leverage kam karo ya skip karo.")
        st.caption("Stop-loss exchange par actual order lagao. Partial hone par stop breakeven par.")
    else:
        st.info("Abhi trade nahi, isliye ticket nahi.")

    with st.expander("Signal tape (purana score view)"):
        for f in A["factors"]:
            s = f["score"]
            st.markdown(f"{'🟢' if s > 0 else '🔴' if s < 0 else '⚪'} **{f['name']}** `{s:+.1f}`  \n<small>{f['note']}</small>", unsafe_allow_html=True)
        if fund_last is None: st.info("Is exchange se funding nahi mili.")
        if oi_chg is None: st.info("Open interest history nahi mili.")
    with st.expander("Saare levels (confluence clusters)"):
        if len(A["levels"]):
            lvt = A["levels"].copy()
            lvt["distance (ATR)"] = ((lvt["price"] - live_price) / plan["atr"]).round(1)
            lvt["price"] = lvt["price"].round(0)
            lvt["score"] = lvt["score"].round(1)
            st.dataframe(lvt[(lvt["distance (ATR)"].abs() < 10)].sort_values("price", ascending=False), hide_index=True, width="stretch")
        st.caption("Liquidation heatmap free public API se nahi aata. Coinglass par alag dekho.")

# =============================================================================== backtest
with tab_bt:
    st.subheader("Simulated past trades (real trades nahi)")
    win = st.selectbox("Period", ["Poora data", "Last 365 din", "Last 180 din", "Last 90 din"], index=0)
    tr_all, d_bt = E.run(df, p, costs, fund)
    dfw, tr = df, tr_all
    if win != "Poora data" and not tr_all.empty:
        days_n = int(win.split()[1])
        cutoff = df.index[-1] - pd.Timedelta(days=days_n)
        dfw = df[df.index >= cutoff]
        tr = E.rebuild_equity(tr_all[tr_all["entry_time"] >= cutoff], costs) if (tr_all["entry_time"] >= cutoff).any() else tr_all.iloc[0:0]
    s = E.stats(tr, dfw, costs, p)
    st.info(f"Ye sab **simulation** hai: strategy ko {dfw.index[0]:%d %b %Y} se {dfw.index[-1]:%d %b %Y} ({max((dfw.index[-1] - dfw.index[0]).days, 1)} din) "
            "ki history par chalane ka result. Isme aapka koi real ya open trade nahi hai. Abhi ka asli status 'Live decision' tab mein hai.")
    if s["trades"] == 0:
        st.warning("Is period mein koi simulated trade nahi bana. Filters halke karo ya period badhao.")
    else:
        a, b, c, e = st.columns(4)
        a.metric("Simulated trades", s["trades"], help="History par strategy ne itne trades liye hote. Real trades nahi.")
        b.metric("Win rate", f"{s['win_rate']:.0f}%", help="Breakeven trades jo thoda positive nikle, win ginte hain.")
        c.metric("Avg R / trade", f"{s['avg_R']:+.2f}", help="Har trade se average kitna R bana (costs ke baad). Ye asli edge ka number hai.")
        e.metric("Profit factor", "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}")
        st.caption(f"Break-even win rate is payoff par: **{s['breakeven_win_rate']:.0f}%**. "
                   f"Avg R ka 95% range: **{s['avg_R_ci'][0]:+.2f} se {s['avg_R_ci'][1]:+.2f}**. "
                   f"Trades: **{s['trades_per_month']:.1f} per month**, avg holding {s['avg_bars']:.0f} bars.")
        a, b, c, e = st.columns(4)
        a.metric("Strategy return", f"{s['total_return_pct']:+.1f}%")
        b.metric("Buy & hold", f"{s['buy_hold_pct']:+.1f}%")
        c.metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%")
        e.metric("Worst losing streak", s["worst_losing_streak"])
        a, b, c, e = st.columns(4)
        a.metric("Avg win", f"{s['avg_win_R']:+.2f}R")
        b.metric("Avg loss", f"-{s['avg_loss_R']:.2f}R")
        c.metric("SQN", f"{s['sqn']:.2f}", help="<1.6 kamzor, 1.6-2.0 average se neeche, 2.0-2.5 average, 2.5+ acha. Kam trades mein bharosemand nahi.")
        e.metric("Losers jo 1R tak gaye", f"{s.get('losers_reached_1R_pct', 0):.0f}%", help="Agar zyada hai to partial exit madad karta hai.")
        if s["trades"] < 30:
            st.warning(f"Sirf {s['trades']} trades. Itne kam sample se kuch pakka nahi nikalta.")
        if s["avg_R_ci"][0] <= 0 < s["avg_R_ci"][1]:
            st.warning("Avg R ka 95% range zero ko cross karta hai. Matlab edge statistically sabit nahi hai, luck bhi ho sakta hai.")
        elif s["avg_R_ci"][1] <= 0:
            st.error("Avg R negative hai aur range bhi negative. Ye settings is data par paisa nahi bana rahi.")
        st.caption("Win rate akela kuch nahi batata. Partial exit win rate badhata hai (kyunki breakeven trades 'win' ginte hain) par expectancy (Avg R) alag cheez hai.")

        fig = go.Figure(go.Scatter(x=tr["exit_time"], y=tr["equity"], mode="lines", name="Equity"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Equity (USDT)")
        st.plotly_chart(fig, width="stretch")

        reg = d_bt["regime"]
        tr2 = tr.copy()
        tr2["regime"] = reg.reindex(tr2["entry_time"]).values
        grp = lambda k: tr2.groupby(k).agg(trades=("R", "size"), win_rate=("pnl", lambda x: (x > 0).mean() * 100), avg_R=("R", "mean")).round(2)
        l1, l2, l3 = st.columns(3)
        l1.markdown("**Regime ke hisaab se**"); l1.dataframe(grp("regime"), width="stretch")
        l2.markdown("**Side ke hisaab se**"); l2.dataframe(grp("side"), width="stretch")
        l3.markdown("**Exit reason**"); l3.dataframe(grp("reason"), width="stretch")

        mc = E.monte_carlo(tr["R"].values, risk_pct)
        if mc:
            st.markdown(f"**Monte Carlo:** trades ka order {mc['sims']} baar shuffle karke ({mc['n']} trades, {risk_pct:g}% risk)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Median ending equity", f"{mc['median_end']:.2f}x")
            m2.metric("Paisa kam hone ka chance", f"{mc['prob_loss'] * 100:.0f}%")
            m3.metric("Typical worst drawdown", f"{mc['median_dd'] * 100:.0f}%")
            m4.metric("30% se zyada drawdown", f"{mc['prob_dd_30'] * 100:.0f}%")
            st.caption(f"Ending equity ka 5% se 95% range: {mc['p5_end']:.2f}x se {mc['p95_end']:.2f}x. "
                       f"Bure 5% cases mein drawdown {mc['worst5_dd'] * 100:.0f}% ya usse zyada.")
        st.dataframe(tr.drop(columns=["equity_before"]).round({c: 2 for c in tr.select_dtypes("number").columns}), width="stretch", height=300)
        st.download_button("Simulated trades CSV", tr.to_csv(index=False).encode(), "simulated_trades.csv", "text/csv")

# =============================================================================== walk-forward
with tab_wf:
    st.subheader("Walk-forward: parameters pehle chuno, phir anjaan data par test karo")
    st.caption("Data ko hisson mein baanta jaata hai. Har hisse ke liye parameters sirf usse pehle ke data par chune jaate hain, phir wo hissa bina dekhe trade hota hai. "
               "Sirf ye out-of-sample trades judge kiye jaate hain, isliye overfitting nahi chhup sakti.")
    folds_n = st.slider("Folds", 2, 5, 3)
    if st.button("Walk-forward chalao (10-30 second)"):
        with st.spinner("Har fold mein 81 combinations test ho rahe hain..."):
            folds, oos = E.walk_forward(df, p, costs, fund, folds=folds_n)
        st.dataframe(folds.round(2), hide_index=True, width="stretch")
        if oos.empty:
            st.info("Out-of-sample mein koi trade nahi bana. Candles badhao.")
        else:
            so = E.stats(oos, df.loc[oos["entry_time"].iloc[0]:], costs, p)
            a, b, c, e = st.columns(4)
            a.metric("Out-of-sample trades", so["trades"])
            b.metric("Win rate", f"{so['win_rate']:.0f}%")
            c.metric("Avg R", f"{so['avg_R']:+.2f}")
            e.metric("Max drawdown", f"{so['max_drawdown_pct']:.1f}%")
            st.caption(f"Break-even win rate: {so['breakeven_win_rate']:.0f}%. Avg R ka 95% range: {so['avg_R_ci'][0]:+.2f} se {so['avg_R_ci'][1]:+.2f}.")
            valid = int(folds["test_avgR"].notna().sum()) if "test_avgR" in folds else 0
            pos = int((folds["test_avgR"] > 0).sum()) if "test_avgR" in folds else 0
            if so["avg_R"] > 0 and so["avg_R_ci"][0] > 0:
                st.success(f"Robust: out-of-sample edge positive aur statistically alag ({pos}/{valid} folds positive). Phir bhi testnet pehle.")
            elif so["avg_R"] > 0 and valid and pos >= math.ceil(valid * 2 / 3):
                st.warning(f"Promising par pakka nahi ({pos}/{valid} folds positive, range zero cross karti hai). Zyada data ya testnet par dekho.")
            else:
                st.error(f"Sabit nahi hua ({pos}/{valid} folds positive). Anjaan data par edge nahi dikha. Is setup par paisa mat lagao.")
            if so["trades"] < 30:
                st.warning(f"Sirf {so['trades']} out-of-sample trades. Bharosa kam.")
            fig = go.Figure(go.Scatter(x=oos["exit_time"], y=oos["equity"], mode="lines"))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Out-of-sample equity")
            st.plotly_chart(fig, width="stretch")

# =============================================================================== journal
with tab_journal:
    st.subheader("Trade journal")
    st.caption("Apne real trades likho. CSV download karke GitHub ya Drive mein rakho (Streamlit Cloud par file save nahi rehti).")
    if "journal" not in st.session_state:
        st.session_state.journal = pd.DataFrame({"date": [pd.Timestamp.now().date()], "side": ["LONG"], "entry": [0.0], "exit": [0.0],
                                                 "R": [0.0], "grade": ["B"], "setup_reason": [""], "mistake": [""]})
    up = st.file_uploader("Purana journal CSV", type="csv")
    if up is not None and st.button("Load"):
        st.session_state.journal = pd.read_csv(up)
    st.session_state.journal = st.data_editor(st.session_state.journal, num_rows="dynamic", width="stretch")
    j = st.session_state.journal
    if len(j) and "R" in j and pd.to_numeric(j["R"], errors="coerce").notna().any():
        r = pd.to_numeric(j["R"], errors="coerce").dropna()
        st.metric("Journal avg R (aapke real trades)", f"{r.mean():+.2f}", help=f"{(r > 0).mean() * 100:.0f}% win rate, {len(r)} trades")
    buf = io.StringIO(); j.to_csv(buf, index=False)
    st.download_button("Journal CSV download", buf.getvalue().encode(), "journal.csv", "text/csv")

# =============================================================================== about
with tab_about:
    st.markdown("""
### Ye kaise decide karta hai
1. **Daily bias:** daily EMA21 rising ho to long side, falling ho to short side. Sirf poori ho chuki daily candles use hoti hain.
2. **Regime:** trading timeframe par EMA50/200 aur ADX se uptrend, downtrend ya range. Range mein trend strategy trade nahi karti.
3. **Market structure:** swing highs aur lows se higher-high/higher-low (bullish) ya lower-high/lower-low (bearish).
4. **Levels:** swing levels, 7D/30D high-low, volume profile (POC, value area), anchored VWAP. Paas ke levels ek cluster banate hain, jitne zyada sources utna strong.
5. **Entry zone:** EMA50 aur sabse strong paas ka level jahan milte hain. Wahan pullback ka wait, chase nahi.
6. **Avoid zones:** EMA se 2.5 ATR se door (chase), major resistance/support ke bilkul paas, aur range ka beech (chop).
7. **Trigger:** pichhli candle ka high (long) ya low (short) tootna. Bina trigger order nahi.
8. **Crowd:** funding aur price vs OI. Crowded side mein size kam ya skip.
9. **Grade:** 8 checks ka score. A = poora size, B = aadha, C = skip.
10. **Risk:** ATR stop, 1% risk, max leverage cap, liquidation stop se door, partial profit 1R par aur baaki breakeven ya trail ke saath.

### Backtest mein kya hai aur kya nahi
- Backtest mein: daily bias, regime, pullback, RSI, trigger candle, funding filter, room to recent high/low, fees, slippage, funding cost, partial exits.
- Backtest mein NAHI: levels, volume profile, grade, entry/avoid zones. Ye live discretionary overlay hain, inka historical test nahi hua. Isliye unko "helpful guide" samjho, proven edge nahi.
- Stop aur target ek hi candle mein aaye to stop pehle maana jaata hai. Gap par stop open par fill hota hai.

### Sach baatein
- **Win rate guarantee nahi hota.** Partial exit win rate badhata hai par edge nahi badhata. Dekhna hai Avg R, uska 95% range, drawdown aur walk-forward.
- Kam trades ka backtest dhoka de sakta hai. App warning deta hai jab sample chhota ho.
- Demo data nakli hai. Real analysis ke liye Live exchange chuno.
- Liquidation heatmap free API se nahi aata. Coinglass par alag dekho.
- Ye education ke liye hai, financial advice nahi. Pehle testnet par chalao, phir chhoti capital se.
- India: crypto gains par 30% tax + 1% TDS, loss set-off nahi hota.
""")
