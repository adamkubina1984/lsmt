# scripts/select_indicators_evo.py
import random
import itertools
import pandas as pd
from predict_lstm_tradingview import simulate_strategy_with_indicators

# --- dostupné indikátory ---
ALL_INDICATORS = [
    "rsi", "macd", "ema20", "ema50", "ema100", "bb", "keltner", "atr",
    "stoch", "adx", "mfi", "sar", "cci", "roc", "vwap"
]

# --- parametry ---
TIMEFRAME = "5m"
OUTPUT_CSV = "results/predictions_gold_5m.csv"
MIN_CONF = 0.55
FEE_PCT = 0.001
ALLOW_SHORT = True
MIN_TRADES_PER_DAY = 10
BAR_PER_DAY = 288  # 5m timeframe

POP_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.3
SEED = 32

random.seed(SEED)

def fitness(indicators):
    metrics = simulate_strategy_with_indicators(
        indicators,
        timeframe=TIMEFRAME,
        out_csv=OUTPUT_CSV,
        min_conf=MIN_CONF,
        fee_pct=FEE_PCT,
        allow_short=ALLOW_SHORT
    )
    if not metrics:
        return -float("inf"), 0, 0, 0

    trades_per_day = metrics["Trades"] / (metrics["Bars"] / BAR_PER_DAY)
    penalty = -100 if trades_per_day < MIN_TRADES_PER_DAY else 0
    score = metrics["PnL_%"] + penalty
    return score, metrics["PnL_%"], metrics["Sharpe"], metrics["Trades"]

def mutate(combo):
    combo = set(combo)
    if random.random() < 0.5 and len(combo) > 1:
        combo.remove(random.choice(list(combo)))
    else:
        available = [i for i in ALL_INDICATORS if i not in combo]
        if available:
            combo.add(random.choice(available))
    return list(combo)

def crossover(p1, p2):
    return list(set(p1[:len(p1)//2] + p2[len(p2)//2:]))

def evolve():
    population = [random.sample(ALL_INDICATORS, k=random.randint(4,8)) for _ in range(POP_SIZE)]
    history = []

    for gen in range(GENERATIONS):
        scored = []
        print(f"\n=== Generace {gen+1}/{GENERATIONS} ===")
        for combo in population:
            score, pnl, sharpe, trades = fitness(combo)
            print(f"{combo} -> PnL: {pnl:.2f}%, Sharpe: {sharpe:.2f}, Trades: {trades}")
            scored.append((score, combo, pnl, sharpe, trades))
            history.append({"gen": gen+1, "combo": ",".join(combo), "pnl": pnl, "sharpe": sharpe, "trades": trades})

        scored.sort(reverse=True)
        top_half = [combo for _, combo, _, _, _ in scored[:POP_SIZE // 2]]
        new_population = top_half[:]
        while len(new_population) < POP_SIZE:
            if random.random() < MUTATION_RATE:
                parent = random.choice(top_half)
                child = mutate(parent)
            else:
                p1, p2 = random.sample(top_half, 2)
                child = crossover(p1, p2)
            new_population.append(child)
        population = new_population

    df = pd.DataFrame(history)
    df.to_csv("results/evo_indicator_selection.csv", index=False)
    print("\n[OK] Výsledky uloženy do results/evo_indicator_selection.csv")

if __name__ == "__main__":
    evolve()
