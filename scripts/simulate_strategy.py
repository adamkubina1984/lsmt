# scripts/simulate_strategy.py

import argparse
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def simulate(data_path, initial_capital=10000, trade_pct=0.05, fee_pct=0.003):
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)

    df = df.dropna(subset=['close', 'signal'])
    capital = initial_capital
    position = 0
    history = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        signal = row['signal']
        price = row['close']
        strength = row['signal_strength']

        if strength < 0.5:
            signal = 2  # No-trade

        if signal == 0 and position == 0:
            trade_value = capital * trade_pct
            position = trade_value * (1 - fee_pct) / price
            capital -= trade_value

        elif signal == 1 and position > 0:
            proceeds = position * price * (1 - fee_pct)
            capital += proceeds
            position = 0

        net_value = capital + position * price
        history.append({
            'date': row.name,
            'capital': capital,
            'position_value': position * price,
            'total_value': net_value
        })

    result_df = pd.DataFrame(history)
    result_df.set_index('date', inplace=True)

    result_df['returns'] = result_df['total_value'].pct_change().fillna(0)
    sharpe = np.mean(result_df['returns']) / np.std(result_df['returns']) * np.sqrt(252) if np.std(result_df['returns']) > 0 else 0
    drawdown = (result_df['total_value'] / result_df['total_value'].cummax()) - 1
    max_drawdown = drawdown.min()

    summary = {
        'Final Value': result_df['total_value'].iloc[-1],
        'Total Return %': (result_df['total_value'].iloc[-1] / initial_capital - 1) * 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown %': max_drawdown * 100
    }

    print("Souhrn výsledků simulace:")
    for k, v in summary.items():
        print(f"{k}: {v:.2f}")

    return result_df, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Relativní cesta ke vstupnímu CSV se signály")
    args = parser.parse_args()

    input_path = os.path.join(BASE_DIR, "..", args.input)
    simulate(input_path)
