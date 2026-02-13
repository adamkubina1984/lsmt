# scripts/scan_min_conf.py
import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

# přímý import simulate() ze stejného projektu
import importlib.util, sys
ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = ROOT / "scripts" / "simulate_strategy_v2.py"
spec = importlib.util.spec_from_file_location("simv2", SIM_PATH)
simv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simv2)  # simv2.simulate je k dispozici

def _bars_per_day(timeframe: str) -> float:
    tf = (timeframe or "").strip().lower()
    if tf == "1h":
        return 24.0
    return 288.0

def _score_row(row: dict, metric: str) -> float:
    if metric == "pnl":
        return float(row["PnL_%"])
    return float(row["Sharpe"])

def _run_sim(pred_csv_path: Path, thr: float, fee_pct: float, allow_short: bool, bars_per_day: float) -> dict:
    eq, trades = simv2.simulate(
        path_csv=str(pred_csv_path),
        initial_capital=10_000.0,
        trade_pct=0.05,
        fee_pct=fee_pct,
        allow_short=allow_short,
        min_conf=float(thr)
    )

    bars = len(eq)
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

    final_val = eq["equity"].iloc[-1]
    pnl_pct = (final_val / 10_000.0 - 1.0) * 100.0
    rets = eq["equity"].pct_change().fillna(0.0)
    annual_factor = np.sqrt(252.0 * bars_per_day)
    sharpe = (rets.mean()/rets.std())*annual_factor if rets.std()>0 else 0.0
    dd = (eq["equity"]/eq["equity"].cummax() - 1.0).min()*100.0

    return {
        "min_conf": round(float(thr), 3),
        "PnL_%": round(float(pnl_pct), 3),
        "Sharpe": round(float(sharpe), 3),
        "MaxDD_%": round(float(dd), 3),
        "Trades": trades_count,
        "Trades/Day": round(trades_per_day, 2),
        "WinRate_%": round(winrate, 2),
        "AvgPnL": round(avg_pnl, 2),
        "PF": 0 if profit_factor == float("inf") else round(profit_factor, 2),
        "Bars": bars,
    }

def _sim_on_df(df_slice: pd.DataFrame, thr: float, args, bars_per_day: float) -> dict:
    if df_slice.empty:
        return {
            "min_conf": round(float(thr), 3),
            "PnL_%": 0.0,
            "Sharpe": 0.0,
            "MaxDD_%": 0.0,
            "Trades": 0,
            "Trades/Day": 0.0,
            "WinRate_%": 0.0,
            "AvgPnL": 0.0,
            "PF": 0.0,
            "Bars": 0,
        }
    with tempfile.NamedTemporaryFile(prefix="scan_min_conf_", suffix=".csv", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        df_slice.to_csv(tmp_path, index=False)
        return _run_sim(
            pred_csv_path=tmp_path,
            thr=thr,
            fee_pct=args.fee_pct,
            allow_short=args.allow_short,
            bars_per_day=bars_per_day
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser(description="Scan min_conf → metriky PnL/Sharpe/DD/#Trades")
    ap.add_argument("--input", required=True, help="např. results/predictions_5m.csv")
    ap.add_argument("--start", type=float, default=0.45)
    ap.add_argument("--stop",  type=float, default=0.70)
    ap.add_argument("--step",  type=float, default=0.02)
    ap.add_argument("--fee_pct", type=float, default=0.003, help="0.3% = 0.003")
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--timeframe", choices=["5m", "1h"], default="5m", help="Pro výpočet Trades/Day.")
    ap.add_argument("--wf_splits", type=int, default=0, help="Walk-forward split count. 0 = původní in-sample scan.")
    ap.add_argument("--wf_train_frac", type=float, default=0.7, help="Podíl train části uvnitř každého splitu (0-1).")
    ap.add_argument("--select_metric", choices=["sharpe", "pnl"], default="sharpe", help="Metrika pro výběr nejlepšího min_conf na train části.")
    ap.add_argument("--out_csv", default=None, help="kam uložit výsledky (default results/scan_min_conf.csv)")
    args = ap.parse_args()

    pred_csv = Path(args.input)
    if not pred_csv.exists():
        print(f"[CHYBA] Nenalezen soubor: {pred_csv}")
        sys.exit(1)

    out_csv = Path(args.out_csv) if args.out_csv else (ROOT / "results" / f"scan_min_conf_{pred_csv.stem}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    bars_per_day = _bars_per_day(args.timeframe)
    rng = np.arange(args.start, args.stop + 1e-9, args.step)

    if args.wf_splits <= 0:
        rows = []
        for thr in rng:
            rows.append(_run_sim(pred_csv, thr, args.fee_pct, args.allow_short, bars_per_day))
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print("\nTOP podle Sharpe:")
        print(df.sort_values("Sharpe", ascending=False).head(5).to_string(index=False))
        print("\nTOP podle PnL_%:")
        print(df.sort_values("PnL_%", ascending=False).head(5).to_string(index=False))
        return

    # walk-forward režim
    full = pd.read_csv(pred_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    n = len(full)
    if n < 500:
        print(f"[CHYBA] Pro walk-forward je málo řádků: {n}")
        sys.exit(1)
    if not (0.5 <= args.wf_train_frac < 0.95):
        print("[CHYBA] --wf_train_frac musí být v intervalu <0.5, 0.95).")
        sys.exit(1)

    seg = n // args.wf_splits
    if seg < 200:
        print(f"[CHYBA] Příliš mnoho splitů pro délku dat. rows={n}, wf_splits={args.wf_splits}")
        sys.exit(1)

    wf_rows = []
    for s in range(args.wf_splits):
        start = s * seg
        end = n if s == args.wf_splits - 1 else (s + 1) * seg
        chunk = full.iloc[start:end].copy()
        split_i = int(len(chunk) * args.wf_train_frac)
        train_df = chunk.iloc[:split_i].copy()
        test_df = chunk.iloc[split_i:].copy()
        if len(train_df) < 120 or len(test_df) < 80:
            continue

        train_scores = []
        for thr in rng:
            r = _sim_on_df(train_df, thr, args, bars_per_day)
            train_scores.append(r)
        train_best = sorted(train_scores, key=lambda x: _score_row(x, args.select_metric), reverse=True)[0]
        best_thr = float(train_best["min_conf"])

        test_row = _sim_on_df(test_df, best_thr, args, bars_per_day)
        wf_rows.append({
            "split": s + 1,
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "best_min_conf_train": best_thr,
            "train_PnL_%": train_best["PnL_%"],
            "train_Sharpe": train_best["Sharpe"],
            "test_PnL_%": test_row["PnL_%"],
            "test_Sharpe": test_row["Sharpe"],
            "test_MaxDD_%": test_row["MaxDD_%"],
            "test_Trades": test_row["Trades"],
            "test_Trades/Day": test_row["Trades/Day"],
            "test_WinRate_%": test_row["WinRate_%"],
            "test_AvgPnL": test_row["AvgPnL"],
            "test_PF": test_row["PF"],
        })

    if not wf_rows:
        print("[CHYBA] Walk-forward nevrátil žádné validní splity.")
        sys.exit(1)

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(out_csv, index=False)

    summary = {
        "splits": int(len(wf_df)),
        "avg_best_min_conf_train": round(float(wf_df["best_min_conf_train"].mean()), 4),
        "avg_test_PnL_%": round(float(wf_df["test_PnL_%"].mean()), 4),
        "avg_test_Sharpe": round(float(wf_df["test_Sharpe"].mean()), 4),
        "avg_test_MaxDD_%": round(float(wf_df["test_MaxDD_%"].mean()), 4),
        "sum_test_Trades": int(wf_df["test_Trades"].sum()),
        "avg_test_Trades/Day": round(float(wf_df["test_Trades/Day"].mean()), 4),
    }
    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\n=== WALK-FORWARD výsledek ===")
    print(wf_df.to_string(index=False))
    print("\n=== WALK-FORWARD summary ===")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\n[OK] Detail:  {out_csv}")
    print(f"[OK] Summary: {summary_csv}")

if __name__ == "__main__":
    import sys
    main()
