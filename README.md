# BTC Perp Desk

Streamlit app jo BTC perpetual ke liye market regime padhta hai, trade setup dhundta hai, risk-based position size nikalta hai, aur strategy ko history par backtest karta hai.

## Kya karta hai

| Tab | Kaam |
|---|---|
| Live decision | Regime, 5 factor ka scorecard (daily trend, regime, funding, price vs OI, room to run), LONG / SHORT / NO TRADE verdict, trade ticket |
| Backtest | Win rate, average R, profit factor, drawdown, regime ke hisaab se result, fees + slippage + funding ke saath |
| Walk-forward | Parameters pehle 60% data par chune jaate hain, phir unseen 40% par test hote hain (overfitting check) |
| Journal | Trade journal, CSV import/export |

## Strategy ek line mein

Trend ke saath pullback: daily EMA50/200 + ADX se trend, price EMA50 ke paas aaye, RSI dip kare, phir pichhli candle ka high/low tode. Funding bahut crowded ho to us taraf trade skip. Stop = ATR based, target = RR based, size = risk % se.

## Local chalao

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Sidebar mein pehle "Demo (synthetic)" se tool samjho, phir "Live exchange" chuno. API key nahi chahiye, sirf public market data use hota hai.

## GitHub par daalo

```bash
git init
git add .
git commit -m "BTC Perp Desk"
git branch -M main
git remote add origin https://github.com/<tumhara-username>/btc-perp-desk.git
git push -u origin main
```

## Streamlit Community Cloud par deploy

1. https://share.streamlit.io par GitHub se login karo.
2. "New app" > apna repo `btc-perp-desk`, branch `main`, main file `app.py`.
3. Deploy dabao. Har `git push` par app khud update hoti hai.

Agar Binance cloud server par block ho (kuch regions mein hota hai), app mein Bybit ya OKX chuno.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests check karte hain: lookahead nahi hai, stop aur target ek hi candle mein aaye to stop pehle maana jaata hai, fees PnL ghatati hain, ticket ka risk aur liquidation sahi hai.

## Backtest ke rules (conservative)

- Signal candle close par, entry agli candle ke open par.
- Stop aur target ek hi candle mein hon to stop pehle.
- Fees (default 5 bps) aur slippage (2 bps) dono taraf, funding bhi.
- Ek time par ek hi position.

## Sach baat

- Koi bhi strategy ka win rate guarantee nahi hota. Trend strategies mein 35-45% win rate normal hai aur edge risk:reward se aata hai. App isliye win rate ke saath break-even win rate aur average R dikhata hai.
- Demo data nakli hai. Us par result random aate hain, aur aana bhi chahiye. Real conclusion sirf live data par, walk-forward ke baad.
- Liquidation heatmap free public API se nahi milta. Coinglass alag se dekho.
- Ye education ke liye hai, financial advice nahi. Pehle testnet par chalao, phir chhoti capital se.
- India: crypto gains par 30% tax + 1% TDS, loss set-off nahi hota.
