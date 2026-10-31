"""Position sizing and trade ticket."""
from __future__ import annotations

from dataclasses import dataclass


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
