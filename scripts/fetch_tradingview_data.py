# scripts/fetch_tradingview_data.py
# ---------------------------------------
# Stahovani dat z TradingView pro GOLD/VIX a volitelne DXY/COT.
# Data se ukladaji do data/raw/ ve formatu CSV.
# ---------------------------------------

import os
import argparse
from tvDatafeed import TvDatafeed, Interval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Bez loginu funguje nologin rezim (omezeny pocet svicek)
tv = TvDatafeed()  # nebo TvDatafeed('tv_username', 'tv_password')


def fetch_and_save(symbol: str, exchange: str, interval: Interval, n_bars: int, filename: str):
    print(f"[INFO] Stahuji {symbol} ({interval.name}) ...")
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)

    if df is None or df.empty:
        print(f"[VAROVANI] Data pro {symbol} se nepodarilo stahnout.")
        return

    df.reset_index(inplace=True)
    df.rename(columns={"datetime": "date"}, inplace=True)
    out_path = os.path.join(DATA_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"[OK] {symbol} ulozeno jako {filename}")


def main(use_dxy: bool = False, use_cot: bool = False):
    fetch_and_save("GOLD", "TVC", Interval.in_5_minute, 5000, "gold_5m.csv")
    fetch_and_save("GOLD", "TVC", Interval.in_1_hour, 5000, "gold_1h.csv")
    fetch_and_save("VIX", "CBOE", Interval.in_daily, 5000, "vix.csv")

    if use_dxy:
        fetch_and_save("DXY", "TVC", Interval.in_daily, 5000, "dxy.csv")
    else:
        print("[INFO] DXY fetch je vypnuty (--use_dxy pro zapnuti).")

    if use_cot:
        fetch_and_save("CFTC:091741_F", "CFTC", Interval.in_weekly, 2000, "cot.csv")
    else:
        print("[INFO] COT fetch je vypnuty (--use_cot pro zapnuti).")

    print("\n[OK] Stahovani dokonceno. Data jsou ulozena v data/raw/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stazeni dat z TradingView (GOLD/VIX + volitelne DXY/COT).")
    ap.add_argument("--use_dxy", action="store_true", help="Zapnout stahovani DXY.")
    ap.add_argument("--use_cot", action="store_true", help="Zapnout stahovani COT.")
    args = ap.parse_args()
    main(use_dxy=args.use_dxy, use_cot=args.use_cot)
