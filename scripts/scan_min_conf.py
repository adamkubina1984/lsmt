# scripts/scan_min_conf.py
import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# přímý import simulate() ze stejného projektu
import importlib.util, sys
ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = ROOT / "scripts" / "simulate_strategy_v2.py"
spec = importlib.util.spec_from_file_location("simv2", SIM_PATH)
simv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simv2)  # simv2.simulate je k dispozici

def main():
    ap = argparse.ArgumentParser(description="Scan min_conf → metriky PnL/Sharpe/DD/#Trades")
    ap.add_argument("--input", required=True, help="např. results/predictions_5m.csv")
    ap.add_argument("--start", type=float, default=0.45)
    ap.add_argument("--stop",  type=float, default=0.70)
    ap.add_argument("--step",  type=float, default=0.02)
    ap.add_argument("--fee_pct", type=float, default=0.003, help="0.3% = 0.003")
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--out_csv", default=None, help="kam uložit výsledky (default results/scan_min_conf.csv)")
    args = ap.parse_args()

    pred_csv = Path(args.input)
    if not pred_csv.exists():
        print(f"[CHYBA] Nenalezen soubor: {pred_csv}")
        sys.exit(1)

    out_csv = Path(args.out_csv) if args.out_csv else (ROOT / "results" / f"scan_min_conf_{pred_csv.stem}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    rng = np.arange(args.start, args.stop + 1e-9, args.step)
    for thr in rng:
        eq, trades = simv2.simulate(
            path_csv=str(pred_csv),
            initial_capital=10_000.0,
            trade_pct=0.05,
            fee_pct=args.fee_pct,
            allow_short=args.allow_short,
            min_conf=thr
        )
        # --- rozšířené metriky z trades a equity ---
        bars = len(eq)
        bars_per_day = 288.0  # pro TF=5m; pokud TF=1h, změň na 24
        trades_count = len(trades)
        trades_per_day = trades_count / (bars / bars_per_day) if bars > 0 else 0.0

        pnl_list = [float(t.get("pnl", 0.0)) for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        winrate = (len(wins) / trades_count * 100.0) if trades_count else 0.0
        avg_pnl = (sum(pnl_list) / trades_count) if trades_count else 0.0
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_win > 0 else 0.0)

        # spočti metriky z eq (je už v simulate vypsané, ale vezmeme si to znovu)
        final_val = eq["equity"].iloc[-1]
        pnl_pct   = (final_val / 10_000.0 - 1.0) * 100.0
        rets      = eq["equity"].pct_change().fillna(0.0)
        sharpe    = (rets.mean()/rets.std())*np.sqrt(252) if rets.std()>0 else 0.0
        dd        = (eq["equity"]/eq["equity"].cummax() - 1.0).min()*100.0
        rows.append({
            "min_conf": round(float(thr), 3),
            "PnL_%":    round(float(pnl_pct), 3),
            "Sharpe":   round(float(sharpe), 3),
            "MaxDD_%":  round(float(dd), 3),
            "Trades":   trades_count,
            "Trades/Day": round(trades_per_day, 2),
            "WinRate_%":  round(winrate, 2),
            "AvgPnL":     round(avg_pnl, 2),
            "PF":         0 if profit_factor == float("inf") else round(profit_factor, 2),
        })


    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    # ====== ZÁVĚREČNÝ VÝPIS ======
    print("\nTOP podle Sharpe:")
    print(df.sort_values("Sharpe", ascending=False).head(5).to_string(index=False))

    print("\nTOP podle PnL_%:")
    print(df.sort_values("PnL_%", ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    import sys
    main()
