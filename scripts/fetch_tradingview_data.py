# scripts/fetch_tradingview_data.py
# ---------------------------------------
# Stahování dat z TradingView pro GOLD, VIX, DXY a COT
# ----------------------------------------------------
# Autor: ChatGPT (na základě specifikace "Analýza AI modelu")
# Popis: Tento skript stáhne historická data pro:
#   - GOLD (TVC:GC=F) – timeframe 5min a 1h
#   - VIX (CBOE Volatility Index)
#   - DXY (US Dollar Index)
#   - COT (Commitment of Traders - sentiment)
# Data se uloží do data/raw/ ve formátu CSV.
# ---------------------------------------

import os
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval
import pandas as pd

# --- Základní cesty projektu ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Přihlášení do TradingView ---
# Pokud máš účet na TradingView, doplň své přihlašovací údaje:
# Bez loginu funguje "nologin" režim (omezený počet svíček)
tv = TvDatafeed()  # nebo TvDatafeed('tv_username', 'tv_password')


def fetch_and_save(symbol: str, exchange: str, interval: Interval, n_bars: int, filename: str):
    """
    Obecná funkce pro stažení a uložení dat z TradingView.
    """
    print(f"[INFO] Stahuji {symbol} ({interval.name}) ...")
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)

    if df is None or df.empty:
        print(f"[VAROVÁNÍ] Data pro {symbol} se nepodařilo stáhnout.")
        return

    df.reset_index(inplace=True)
    df.rename(columns={"datetime": "date"}, inplace=True)
    out_path = os.path.join(DATA_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"[OK] {symbol} uloženo jako {filename}")


def main():
    # --- GOLD ---
    fetch_and_save("GOLD", "TVC", Interval.in_5_minute, 5000, "gold_5m.csv")
    fetch_and_save("GOLD", "TVC", Interval.in_1_hour, 5000, "gold_1h.csv")

    # --- VIX ---
    fetch_and_save("VIX", "CBOE", Interval.in_daily, 5000, "vix.csv")

    # --- DXY ---
    fetch_and_save("DXY", "TVC", Interval.in_daily, 5000, "dxy.csv")

    # --- COT (Commitment of Traders) ---
    # COT data TradingView poskytuje jako futures sentiment (1x týdně)
    fetch_and_save("CFTC:091741_F", "CFTC", Interval.in_weekly, 2000, "cot.csv")

    print("\n[OK] Stahování dokončeno. Data jsou uložena v data/raw/")

if __name__ == "__main__":
    main()
