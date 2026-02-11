# scripts/simulate_strategy_v2.py
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

def _resolve_path(p):
    p = Path(p)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p

def simulate(path_csv,
             initial_capital=10_000.0,
             trade_pct=0.05,
             fee_pct=0.003,
             allow_short=True,
             min_conf=0.6,
             # 👇 nové volitelné prahy pro „slabší“ vstup:
             min_conf_low=None,
             trade_pct_low=None,
             min_conf_buy=None,
             min_conf_sell=None,
             trades_csv=None,
             equity_csv=None,
             equity_png=None,
             max_hold_bars=None):
    """Simulace obchodování na základě predictions CSV.
    path_csv: predictions_<tf>.csv (columns: date, close, signal, signal_strength)
    fee_pct: 0.003 = 0.3 %
    """

    # ---------- načtení predikcí ----------
    path_csv = _resolve_path(path_csv)
    if not path_csv.exists():
        raise FileNotFoundError(f"Soubor s predikcemi nenalezen: {path_csv}")

    df = pd.read_csv(path_csv, parse_dates=['date'])
    df = df.dropna(subset=['close','signal','signal_strength']).sort_values('date').reset_index(drop=True)

    # ---------- stav účtu / pozice ----------
    cap = float(initial_capital)
    pos_qty = 0.0
    pos_side = None         # 'long' | 'short' | None

    # pro výpočet realizovaného PnL
    entry_price = None
    entry_value = None      # hrubá hodnota při otevření (qty*price)
    entry_fee   = None      # zaplacený fee při otevření (klasicky value*fee_pct)
    entry_cf    = None      # cash flow při otevření (long: -invest; short: +proceeds - fee)
    entry_date  = None
    entry_i     = None
    entry_conf  = None
    entry_sig   = None
    entry_size_pct = None   # aktuální frakce equity v pozici (pro promo/degrad)
    equity_rows = []
    trades_rows = []

    def mark_to_market(price):
        """aktuální equity s případnou otevřenou pozicí"""
        if pos_side == 'long':
            return cap + pos_qty * price
        elif pos_side == 'short':
            return cap - pos_qty * price
        else:
            return cap

    hi_thr = float(min_conf)
    lo_thr = float(min_conf_low) if min_conf_low is not None else None
    hi_pct = float(trade_pct)
    lo_pct = float(trade_pct_low) if trade_pct_low is not None else None
    
    if min_conf_buy is not None:
        df = df[df.get("proba_buy", 0) >= min_conf_buy]
    if min_conf_sell is not None:
        df = df[df.get("proba_sell", 0) >= min_conf_sell]

    # ---------- hlavní smyčka ----------
    for i, row in df.iterrows():
        price = float(row['close'])
        sig   = int(row['signal'])
        conf  = float(row['signal_strength'])

        weak = False
        size_pct = hi_pct

        if conf < hi_thr:
            if lo_thr is not None and lo_pct is not None and conf >= lo_thr:
                weak = True
                size_pct = lo_pct
            else:
                sig = 0
        elif lo_thr is not None and trade_pct_low is not None and conf >= lo_thr:
            # ochrana před dělením nulou (když hi_thr == lo_thr)
            denom = (hi_thr - lo_thr)
            if denom <= 1e-9:
                size_pct = lo_pct
                weak = True
            else:
                slope = (hi_pct - lo_pct) / denom
                size_pct = lo_pct + slope * (conf - lo_thr)
                size_pct = min(hi_pct, max(lo_pct, size_pct))
                weak = True

        # === PROMOCE / DEGRADACE velikosti pozice (dvouprah) ===
        # Cíl: držet expozici ~ size_pct * equity
        if pos_side is not None:
            target_pct = size_pct
            # aktuální expozice vůči equity:
            eq_now = max(1e-9, mark_to_market(price))
            cur_exposure = (pos_qty * price) / eq_now
            # Na začátku pozice se entry_size_pct nastaví níže (při vstupu), tady pracujeme jen s během pozice
            delta_pct = target_pct - cur_exposure
            # prahové šumové pásmo – ať to nepumpuje každou svíčku
            if abs(delta_pct) >= 0.005:  # 0.5 % equity
                if pos_side == 'long':
                    # cílová dodatečná investice
                    delta_cash = delta_pct * eq_now
                    if delta_cash > 0:
                        # promoce – navýšení longu
                        add_qty = delta_cash / price
                        cap -= delta_cash
                        pos_qty += add_qty
                    else:
                        # degradace – částečný exit
                        rem_cash = -delta_cash
                        sell_qty = min(pos_qty, rem_cash / price)
                        proceeds = sell_qty * price * (1.0 - fee_pct)
                        cap += proceeds
                        pos_qty -= sell_qty
                elif pos_side == 'short':
                    # upravujeme short expozici (hrubě stejně: pro zvýšení expozice short "připrodáme", pro snížení část odkoupíme)
                    delta_cash = delta_pct * eq_now
                    if delta_cash > 0:
                        # promoce – přidat short
                        add_qty = delta_cash / price
                        gross = add_qty * price
                        fee   = gross * fee_pct
                        cap  += (gross - fee)
                        pos_qty += add_qty
                    else:
                        # degradace – odkoupit část shortu
                        buy_cash = (-delta_cash)
                        buy_qty  = min(pos_qty, buy_cash / price)
                        cost     = buy_qty * price * (1.0 + fee_pct)
                        cap -= cost
                        pos_qty -= buy_qty
        

        cur_equity = mark_to_market(price)
        equity_rows.append({'date': row['date'], 'equity': cur_equity})

        if pos_side is not None and entry_conf is not None:
            if sig != 0 and sig != entry_sig and conf < entry_conf:
                close_value = pos_qty * price
                if pos_side == 'long':
                    proceeds = close_value * (1.0 - fee_pct)
                    cap += proceeds
                    realized_pnl = proceeds + entry_cf
                    trades_rows.append({
                        'date_open': entry_date, 'date_close': row['date'],
                        'side': 'long', 'qty': pos_qty,
                        'entry_price': entry_price, 'exit_price': price,
                        'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                        'pnl': realized_pnl, 'bars_held': (i - entry_i),
                        'type': 'EXIT REVERSE'
                    })
                elif pos_side == 'short':
                    cap_close = close_value * (1.0 + fee_pct)
                    cap -= cap_close
                    realized_pnl = entry_cf - cap_close
                    trades_rows.append({
                        'date_open': entry_date, 'date_close': row['date'],
                        'side': 'short', 'qty': pos_qty,
                        'entry_price': entry_price, 'exit_price': price,
                        'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                        'pnl': realized_pnl, 'bars_held': (i - entry_i),
                        'type': 'EXIT REVERSE'
                    })
                pos_qty = 0.0; pos_side = None
                entry_price = entry_value = entry_fee = entry_cf = entry_date = entry_i = entry_conf = entry_sig = None
                continue

        # === TIME-STOP ===
        if pos_side is not None and max_hold_bars is not None:
            if (i - entry_i) >= max_hold_bars:
                close_value = pos_qty * price
                if pos_side == 'long':
                    proceeds = close_value * (1.0 - fee_pct)
                    cap += proceeds
                    realized_pnl = proceeds + entry_cf
                    trades_rows.append({
                        'date_open': entry_date, 'date_close': row['date'],
                        'side': 'long', 'qty': pos_qty,
                        'entry_price': entry_price, 'exit_price': price,
                        'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                        'pnl': realized_pnl, 'bars_held': (i - entry_i),
                        'type': 'EXIT TIME'
                    })
                elif pos_side == 'short':
                    cap_close = close_value * (1.0 + fee_pct)
                    cap -= cap_close
                    realized_pnl = entry_cf - cap_close
                    trades_rows.append({
                        'date_open': entry_date, 'date_close': row['date'],
                        'side': 'short', 'qty': pos_qty,
                        'entry_price': entry_price, 'exit_price': price,
                        'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                        'pnl': realized_pnl, 'bars_held': (i - entry_i),
                        'type': 'EXIT TIME'
                    })
                pos_qty = 0.0; pos_side = None
                entry_price = entry_value = entry_fee = entry_cf = entry_date = entry_i = None
                continue

        if sig == 1:  # BUY
            if pos_side == 'short':
                # zavřít short
                # close CF: koupím zpět (cash OUT) + fee
                close_value = pos_qty * price
                cap_close   = close_value * (1.0 + fee_pct)  # výdaj
                cap -= cap_close
                # Realized PnL = entry_cf (cash IN na otevření shortu) + exit_cf (cash OUT záporný)
                realized_pnl = (entry_cf) - cap_close
                trades_rows.append({
                    'date_open': entry_date, 'date_close': row['date'],
                    'side': 'short', 'qty': pos_qty,
                    'entry_price': entry_price, 'exit_price': price,
                    'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                    'pnl': realized_pnl, 'bars_held': (i - entry_i)
                })
                # pozici zavřu
                pos_qty = 0.0; pos_side = None
                entry_price = entry_value = entry_fee = entry_cf = None
                entry_size_pct = size_pct

            if pos_side is None:
                # otevřít long: investujeme část equity
                invest = mark_to_market(price) * size_pct
                qty    = (invest / price)
                fee    = invest * fee_pct
                # cash flow při otevření longu: cash OUT = invest (fee počítáme až při prodeji)
                cap -= invest
                # zapsat vstup
                pos_side   = 'long'
                pos_qty    = qty
                entry_price= price
                entry_value= invest
                entry_fee  = 0.0          # u longu evidujeme fee až při prodeji
                entry_cf   = -invest
                entry_date = row['date']
                entry_i    = i

        elif sig == 2:  # SELL
            if allow_short:
                if pos_side == 'long':
                    # zavřít long
                    close_value = pos_qty * price
                    proceeds    = close_value * (1.0 - fee_pct)  # cash IN
                    cap += proceeds
                    realized_pnl = (proceeds) + (entry_cf)  # entry_cf je záporný
                    trades_rows.append({
                        'date_open': entry_date, 'date_close': row['date'],
                        'side': 'long', 'qty': pos_qty,
                        'entry_price': entry_price, 'exit_price': price,
                        'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                        'pnl': realized_pnl, 'bars_held': (i - entry_i)
                    })
                    pos_qty = 0.0; pos_side = None
                    entry_price = entry_value = entry_fee = entry_cf = None
                    entry_size_pct = size_pct

                if pos_side is None:
                    # otevřít short: prodáme, cash IN hned, platíme fee
                    invest = mark_to_market(price) * size_pct
                    qty    = (invest / price)
                    gross  = qty * price
                    fee    = gross * fee_pct
                    cap += (gross - fee)    # cash flow při otevření shortu
                    pos_side   = 'short'
                    pos_qty    = qty
                    entry_price= price
                    entry_value= gross
                    entry_fee  = fee
                    entry_cf   = (gross - fee)  # cash IN
                    entry_date = row['date']
                    entry_i    = i
            else:
                # bez shortů: SELL = zavři long, jinak nic
                if pos_side == 'long':
                    close_value = pos_qty * price
                    proceeds    = close_value * (1.0 - fee_pct)
                    cap += proceeds
                    realized_pnl = proceeds + entry_cf
                    trades_rows.append({
                        'date_open': entry_date, 'date_close': row['date'],
                        'side': 'long', 'qty': pos_qty,
                        'entry_price': entry_price, 'exit_price': price,
                        'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                        'pnl': realized_pnl, 'bars_held': (i - entry_i)
                    })
                    pos_qty = 0.0; pos_side = None
                    entry_price = entry_value = entry_fee = entry_cf = None
                    entry_size_pct = None

        else:
            # No-Trade -> držíme (nic neděláme)
            pass

    # ---------- uzavři případnou otevřenou pozici na konci ----------
    if pos_side is not None:
        last_price = float(df.iloc[-1]['close'])
        last_date  = df.iloc[-1]['date']
        if pos_side == 'long':
            close_value = pos_qty * last_price
            proceeds    = close_value * (1.0 - fee_pct)
            cap += proceeds
            realized_pnl = proceeds + entry_cf
            trades_rows.append({
                'date_open': entry_date, 'date_close': last_date,
                'side': 'long', 'qty': pos_qty,
                'entry_price': entry_price, 'exit_price': last_price,
                'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                'pnl': realized_pnl, 'bars_held': (len(df) - 1 - entry_i)
            })
        elif pos_side == 'short':
            close_value = pos_qty * last_price
            cap_close   = close_value * (1.0 + fee_pct)
            cap -= cap_close
            realized_pnl = entry_cf - cap_close
            trades_rows.append({
                'date_open': entry_date, 'date_close': last_date,
                'side': 'short', 'qty': pos_qty,
                'entry_price': entry_price, 'exit_price': last_price,
                'entry_fee': entry_fee, 'exit_fee': close_value * fee_pct,
                'pnl': realized_pnl, 'bars_held': (len(df) - 1 - entry_i)
            })
        pos_qty = 0.0; pos_side = None
        entry_price = entry_value = entry_fee = entry_cf = None

        # zapiš finální equity po uzavření
        cur_equity = cap
        equity_rows.append({'date': last_date, 'equity': cur_equity})

    # ---------- metriky ----------
    eq = pd.DataFrame(equity_rows).drop_duplicates(subset=['date']).set_index('date').sort_index()
    rets = eq['equity'].pct_change().fillna(0.0)
    sharpe = (rets.mean()/rets.std())*np.sqrt(252) if rets.std()>0 else 0.0
    dd = (eq['equity'] / eq['equity'].cummax()) - 1.0
    max_dd = dd.min() * 100.0
    final_val = eq['equity'].iloc[-1]
    pnl_pct = (final_val / initial_capital - 1.0) * 100.0

    print("=== SIMULACE v2 ===")
    print(f"allow_short={allow_short}, min_conf={min_conf:.2f}, fee={fee_pct*100:.2f}%")
    print(f"Final Value:      {final_val:,.2f} CZK")
    print(f"Total Return %:   {pnl_pct:.2f}%")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print(f"Max Drawdown %:   {max_dd:.2f}%")
    print(f"Trades:           {len(trades_rows)}")

    # ---------- exporty ----------
    # trades
    trades_df = pd.DataFrame(trades_rows)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=[
            'date_open','date_close','side','qty','entry_price','exit_price',
            'entry_fee','exit_fee','pnl','bars_held'
        ])
    # defaultní cesty
    trades_csv = _resolve_path(trades_csv) if trades_csv else (ROOT_DIR / "results" / "trades_log.csv")
    equity_csv = _resolve_path(equity_csv) if equity_csv else (ROOT_DIR / "results" / "equity_curve.csv")
    equity_png = _resolve_path(equity_png) if equity_png else (ROOT_DIR / "results" / "simulation.png")

    trades_csv.parent.mkdir(parents=True, exist_ok=True)
    equity_csv.parent.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(trades_csv, index=False)
    eq.reset_index().to_csv(equity_csv, index=False)

    # graf equity + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1.plot(eq.index, eq['equity'])
    ax1.set_title('Equity Curve'); ax1.set_ylabel('Equity (CZK)')
    dd_plot = -dd * 100.0
    ax2.plot(eq.index, dd_plot)
    ax2.set_title('Drawdown (%)'); ax2.set_ylabel('DD (%)'); ax2.set_xlabel('Time')
    # anotace metrik
    txt = f"PnL: {pnl_pct:.2f}%  |  Sharpe: {sharpe:.2f}  |  MaxDD: {max_dd:.2f}%  |  Trades: {len(trades_rows)}"
    ax1.text(0.01, 0.95, txt, transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), va='top')
    fig.tight_layout()
    plt.savefig(equity_png, dpi=120)
    plt.close(fig)

    print(f"[OK] Trades CSV:  {trades_csv}")
    print(f"[OK] Equity CSV:  {equity_csv}")
    print(f"[OK] Chart PNG:   {equity_png}")

    return eq, trades_rows


if __name__ == "__main__": 
    ap = argparse.ArgumentParser(description="Simulace strategie na predictions CSV + export trade-logu a grafu.")
    ap.add_argument("--input", required=True, help="např. results/predictions_5m.csv (relativně ke kořeni)")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--trade_pct", type=float, default=0.05)
    ap.add_argument("--fee_pct", type=float, default=0.003, help="0.003 = 0.3 %")
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--min_conf", type=float, default=0.6)
    ap.add_argument("--trades_csv", type=str, default=None, help="cesta k výslednému trades CSV")
    ap.add_argument("--equity_csv", type=str, default=None, help="cesta k výslednému equity CSV")
    ap.add_argument("--equity_png", type=str, default=None, help="cesta k PNG grafu equity")
    ap.add_argument("--min_conf_low", type=float, default=None, help="Nižší práh pro částečný vstup (např. 0.38).")
    ap.add_argument("--trade_pct_low", type=float, default=None, help="Frakce kapitálu pro částečný vstup (např. 0.025).")
    ap.add_argument("--max_hold_bars", type=int, default=None, help="Maximální délka držení pozice v barech (time-stop)")
    ap.add_argument("--min_conf_buy", type=float, default=None, help="Minimální pravděpodobnost BUY pro vstup.")
    ap.add_argument("--min_conf_sell", type=float, default=None, help="Minimální pravděpodobnost SELL pro vstup.")
    args = ap.parse_args()

    simulate(
        path_csv=args.input,
        initial_capital=args.capital,
        trade_pct=args.trade_pct,
        fee_pct=args.fee_pct,
        allow_short=args.allow_short,
        min_conf=args.min_conf,
        trades_csv=args.trades_csv,
        equity_csv=args.equity_csv,
        equity_png=args.equity_png,
        min_conf_low=args.min_conf_low,
        trade_pct_low=args.trade_pct_low,
        max_hold_bars=args.max_hold_bars,
        min_conf_buy=args.min_conf_buy,
        min_conf_sell=args.min_conf_sell
    )
