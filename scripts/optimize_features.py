# scripts/optimize_features.py
# -------------------------------------------------------------
# Evoluční výběr indikátorů pro model LSTM nad TradingView daty
# -------------------------------------------------------------
import os
import subprocess
import json
import time
import tempfile
import random
import shutil
from pathlib import Path
from simulate_strategy_v2 import simulate

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SCRIPTS_DIR = ROOT_DIR / "scripts"

ALL_FEATURES = [
    "rsi", "macd", "ema20", "ema50", "ema100", "bb", "keltner",
    "atr", "stoch", "adx", "mfi", "sar", "cci", "roc", "vwap"
]

DEFAULT_TIMEFRAME = "5m"
MIN_TRADES_PER_DAY = 3
VAL_BARS = 4000
POP_SIZE = 12
GENERATIONS = 8
MUTATION_RATE = 0.3

# --------------------------
def train_model(feature_set, timeframe):
    features_arg = ",".join(feature_set)
    cmd = [
        "python", str(SCRIPTS_DIR / "train_lstm_tradingview.py"),
        "--timeframe", timeframe,
        "--seq_len", "60",
        "--thr_pct", "0.20",
        "--epochs", "20",
        "--batch_size", "32",
        "--cot_shift_days", "10",
        "--features", features_arg
    ]
    print(f"[TRAIN] {features_arg}")
    subprocess.run(cmd, check=True)

# --------------------------
def run_simulation(pred_csv):
    try:
        eq, trades = simulate(
            path_csv=pred_csv,
            allow_short=True,
            min_conf=0.5,
            fee_pct=0.001
        )
        final = eq['equity'].iloc[-1]
        trades_per_day = len(trades) / (len(eq) / (24*12))  # assuming 5m bars
        return final, trades_per_day
    except Exception as e:
        print("[SIM ERR]", e)
        return 0.0, 0.0

# --------------------------
def evaluate_feature_set(features, timeframe):
    train_model(features, timeframe)
    pred_path = RESULTS_DIR / f"predictions_gold_{timeframe}.csv"
    pred_cmd = [
        "python", str(SCRIPTS_DIR / "predict_lstm_tradingview.py"),
        "--timeframe", timeframe,
        "--output", str(pred_path),
        "--min_conf", "0.0",
        "--features", ",".join(features)
    ]
    subprocess.run(pred_cmd, check=True)
    pnl, trades_per_day = run_simulation(pred_path)
    print(f"[EVAL] Features={features} | PnL={pnl:.2f} | Trades/Day={trades_per_day:.2f}")
    return pnl if trades_per_day >= MIN_TRADES_PER_DAY else 0.0

# --------------------------
def mutate(features):
    out = features[:]
    if random.random() < 0.5 and len(out) > 2:
        out.remove(random.choice(out))
    else:
        new = random.choice([f for f in ALL_FEATURES if f not in out])
        out.append(new)
    return list(set(out))

# --------------------------
def run_evolution():
    pop = [random.sample(ALL_FEATURES, k=random.randint(4, 8)) for _ in range(POP_SIZE)]
    scores = {}
    for gen in range(GENERATIONS):
        print(f"\n=== Generace {gen+1}/{GENERATIONS} ===")
        scored = []
        for fset in pop:
            key = ",".join(sorted(fset))
            if key not in scores:
                score = evaluate_feature_set(fset, DEFAULT_TIMEFRAME)
                scores[key] = score
            scored.append((scores[key], fset))

        scored.sort(reverse=True)
        top_half = [f for _, f in scored[:POP_SIZE//2]]
        pop = top_half[:]
        while len(pop) < POP_SIZE:
            base = random.choice(top_half)
            child = mutate(base)
            pop.append(child)

    best = max(scores.items(), key=lambda x: x[1])
    print("\n=== NEJLEPŠÍ SET ===")
    print("Indikátory:", best[0])
    print("Skóre (zisk):", best[1])

if __name__ == "__main__":
    run_evolution()
