# scripts/select_best_indicators.py
# Evoluční výběr nejlepší kombinace indikátorů na základě výnosnosti

import os
import argparse
import json
import random
import itertools
import subprocess
import tempfile
import sys
from pathlib import Path

# === Konfigurace ===
ALL_INDICATORS = [
    "rsi", "macd", "ema20", "ema50", "ema100", "bb", "keltner",
    "atr", "stoch", "adx", "mfi", "sar", "cci", "roc", "vwap"
]

DEFAULT_MODEL_SCRIPT = "scripts/train_lstm_tradingview.py"
DEFAULT_PREDICT_SCRIPT = "scripts/predict_lstm_tradingview.py"
DEFAULT_SIMULATE_SCRIPT = "scripts/simulate_strategy_v2.py"

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

def run_pipeline(indicators, timeframe, min_trades_per_day, eval_tail_frac=0.2, min_eval_rows=600):
    """
    Natrénuje model, provede predikci a spustí simulaci pro daný set indikátorů.
    Logy běží nebufferovaně (python -u), aby v GUI tekly průběžně.
    PnL a počet obchodů čteme z výstupních CSV (equity_curve.csv, trades_log.csv).
    """
    import pandas as pd

    ind_key = ",".join(sorted(indicators))  # string pro --features
    out_pred = RESULTS_DIR / f"predictions_{timeframe}_tmp.csv"

    # 1) Trénink (STREAM výstupu)
    train_cmd = [sys.executable, "-u", str(DEFAULT_MODEL_SCRIPT),
                 "--timeframe", str(timeframe),
                 "--features", ind_key]
    print("[TRAIN]", " ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, check=True)

    # 2) Predikce (STREAM výstupu)
    pred_cmd = [sys.executable, "-u", str(DEFAULT_PREDICT_SCRIPT),
                "--timeframe", str(timeframe),
                "--output", str(out_pred),
                "--features", ind_key]
    print("[PRED ]", " ".join(pred_cmd), flush=True)
    subprocess.run(pred_cmd, check=True)

    # 3) Simulace (STREAM výstupu)
    #    Hodnotíme out-of-sample tail predikcí (default posledních 20 %),
    #    aby evoluce nepřeučovala čistě na in-sample.
    eval_pred = out_pred
    try:
        df_pred = pd.read_csv(out_pred)
        n_all = len(df_pred)
        n_tail = int(n_all * float(eval_tail_frac))
        if n_tail >= int(min_eval_rows) and n_tail < n_all:
            eval_pred = RESULTS_DIR / f"predictions_{timeframe}_eval_tail_tmp.csv"
            df_pred.tail(n_tail).to_csv(eval_pred, index=False)
            print(f"[EVAL] OOS tail: {n_tail}/{n_all} řádků ({eval_tail_frac:.2f})", flush=True)
        else:
            print(f"[EVAL] OOS tail vynechán (n_all={n_all}, n_tail={n_tail}, min_eval_rows={min_eval_rows})", flush=True)
    except Exception as e:
        print(f"[WARN] Nelze připravit OOS tail, použiju celé predikce: {e}", flush=True)
        eval_pred = out_pred

    sim_cmd = [sys.executable, "-u", str(DEFAULT_SIMULATE_SCRIPT),
               "--input", str(eval_pred),
               "--min_conf", "0.45",
               "--fee_pct", "0.003",
               "--allow_short"]
    print("[SIM  ]", " ".join(sim_cmd), flush=True)
    subprocess.run(sim_cmd, check=True)

    # 4) ČTENÍ výstupních CSV (žádné parsování stdout)
    pnl = 0.0
    trades = 0
    eq_csv = RESULTS_DIR / "equity_curve.csv"
    trades_csv = RESULTS_DIR / "trades_log.csv"

    if eq_csv.exists():
        df_eq = pd.read_csv(eq_csv)
        if not df_eq.empty:
            start = float(df_eq["equity"].iloc[0])
            end   = float(df_eq["equity"].iloc[-1])
            pnl   = (end / start - 1.0) * 100.0

    if trades_csv.exists():
        df_tr = pd.read_csv(trades_csv)
        trades = len(df_tr)

    # Penalizace za nízkou aktivitu (např. týden dat -> 5 dní)
    if trades < (min_trades_per_day * 5):
        pnl = -99.99

    print(f"[EVAL] features=[{ind_key}] -> PnL={pnl:.2f}% | Trades={trades}", flush=True)
    return pnl, indicators

def update_metadata(timeframe, indicators):
    """
    Ulož nejlepší nalezený set do models/features_tv_<tf>.json pod klíč 'features_auto'
    a označ zdroj 'auto'.
    """
    meta_path = MODELS_DIR / f"features_tv_{timeframe}.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # zajisti existenci složky models/
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta["features_auto"] = sorted(indicators)
    meta["selected_features_source"] = "auto"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Nejlepší kombinace uložena do {meta_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evoluční výběr nejlepších indikátorů.")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population_size", type=int, default=6)
    parser.add_argument("--min_trades_per_day", type=int, default=3)
    parser.add_argument("--eval_tail_frac", type=float, default=0.2,
                        help="Jaká část konce predikcí se použije pro OOS hodnocení (0-1).")
    parser.add_argument("--min_eval_rows", type=int, default=600,
                        help="Minimální počet řádků OOS části, jinak se použijí celé predikce.")
    args = parser.parse_args()

    pop_size = args.population_size
    generations = args.generations
    tf = args.timeframe

    population = [random.sample(ALL_INDICATORS, random.randint(5, 10)) for _ in range(pop_size)]
    best_results = []

    for gen in range(generations):
        print(f"\n=== Generace {gen+1}/{generations} ===")
        gen_results = []
        for inds in population:
            pnl, used_inds = run_pipeline(
                inds,
                tf,
                args.min_trades_per_day,
                eval_tail_frac=args.eval_tail_frac,
                min_eval_rows=args.min_eval_rows
            )
            gen_results.append((pnl, used_inds))

        gen_results.sort(reverse=True, key=lambda x: x[0])
        best_results.extend(gen_results)
        best_results.sort(reverse=True, key=lambda x: x[0])
        best_results = best_results[:10]  # top 10 celkově

        print("\nTOP kombinace:")
        for pnl, inds in best_results[:3]:
            print(f"{pnl:.2f}% -> {','.join(sorted(inds))}")

        # Nová populace = elita + mutace
        elite = [inds for _, inds in gen_results[:2]]
        new_pop = elite.copy()
        while len(new_pop) < pop_size:
            base = random.choice(elite)
            modified = base.copy()
            if random.random() < 0.5 and len(modified) > 5:
                modified.remove(random.choice(modified))
            if random.random() < 0.5:
                candidates = [i for i in ALL_INDICATORS if i not in modified]
                if candidates:
                    modified.append(random.choice(candidates))
            new_pop.append(modified)
        population = new_pop

    print("\n=== Nejlepší nalezené kombinace ===")
    for pnl, inds in best_results[:5]:
        print(f"{pnl:.2f}% -> {','.join(sorted(inds))}")

    # Uložit nejlepší kombinaci do metadata
    if best_results:
        update_metadata(tf, best_results[0][1])

if __name__ == "__main__":
    main()
