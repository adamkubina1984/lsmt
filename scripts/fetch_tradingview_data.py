# scripts/fetch_tradingview_data.py
# ---------------------------------------
# Stahovani dat z TradingView pro GOLD/VIX a volitelne DXY/COT.
# Data se ukladaji do data/raw/ ve formatu CSV.
#
# Důležité: TradingView typicky vrací max ~5000 barů na jeden dotaz.
# Proto je defaultní režim APPEND: při každém spuštění vezme posledních N barů
# a doplní jen nově příchozí svíčky do existujícího CSV. Tím si časem vybuduješ
# libovolně dlouhou historii bez "hacků".
# ---------------------------------------

import os
import argparse
from pathlib import Path
import pandas as pd

# tvDatafeed musí být nainstalovaný v prostředí uživatele
from tvDatafeed import TvDatafeed, Interval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Bez loginu funguje nologin rezim (omezeny pocet svicek)
tv = TvDatafeed()  # nebo TvDatafeed('tv_username', 'tv_password')


def _normalize_tv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje DF z tvDatafeed: date + lowercase sloupce + sort."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy().reset_index()
    # tvDatafeed často vrací datetime index
    if "datetime" in df.columns and "date" not in df.columns:
        df.rename(columns={"datetime": "date"}, inplace=True)
    if "date" not in df.columns:
        # fallback: první sloupec
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # odstraň úplné duplicity
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def _append_or_overwrite(out_path: Path, df_new: pd.DataFrame, mode: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "overwrite" or (not out_path.exists()):
        df_new.to_csv(out_path, index=False)
        return

    # append: načti starý, spoj, dedup podle date
    try:
        df_old = pd.read_csv(out_path)
        df_old.columns = [str(c).lower() for c in df_old.columns]
        if "date" not in df_old.columns:
            df_old.rename(columns={df_old.columns[0]: "date"}, inplace=True)
        df_old["date"] = pd.to_datetime(df_old["date"], errors="coerce")
        df_old = df_old.dropna(subset=["date"]).sort_values("date")
    except Exception:
        df_old = pd.DataFrame()

    if df_old.empty:
        df_new.to_csv(out_path, index=False)
        return

    # sjednoť sloupce (union) – chybějící doplň NaN
    all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
    df_old = df_old.reindex(columns=all_cols)
    df_new = df_new.reindex(columns=all_cols)

    df = pd.concat([df_old, df_new], ignore_index=True)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)


def fetch_and_save(symbol: str, exchange: str, interval: Interval, n_bars: int, filename: str, mode: str = "append"):
    print(f"[INFO] Stahuji {symbol} ({interval.name}) ...")
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=int(n_bars))
    df = _normalize_tv_df(df)

    if df.empty:
        print(f"[VAROVANI] Data pro {symbol} se nepodarilo stahnout.")
        return

    out_path = Path(DATA_DIR) / filename
    before = None
    if out_path.exists() and mode == "append":
        try:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                before = sum(1 for _ in f) - 1
        except Exception:
            before = None

    _append_or_overwrite(out_path, df, mode=mode)

    after = None
    try:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            after = sum(1 for _ in f) - 1
    except Exception:
        after = None

    if before is not None and after is not None:
        print(f"[OK] {symbol} ulozeno jako {filename} | rows: {before} -> {after}")
    else:
        print(f"[OK] {symbol} ulozeno jako {filename}")


def main(use_dxy: bool = False, use_cot: bool = False, n_bars: int = 5000, mode: str = "append"):
    fetch_and_save("GOLD", "TVC", Interval.in_5_minute, n_bars, "gold_5m.csv", mode=mode)
    fetch_and_save("GOLD", "TVC", Interval.in_1_hour,   n_bars, "gold_1h.csv", mode=mode)
    fetch_and_save("VIX",  "CBOE", Interval.in_daily,   n_bars, "vix.csv",     mode=mode)

    if use_dxy:
        fetch_and_save("DXY", "TVC", Interval.in_daily, n_bars, "dxy.csv", mode=mode)
    else:
        print("[INFO] DXY fetch je vypnuty (--use_dxy pro zapnuti).")

    if use_cot:
        # COT je weekly, typicky stačí míň barů
        fetch_and_save("CFTC:091741_F", "CFTC", Interval.in_weekly, min(2000, int(n_bars)), "cot.csv", mode=mode)
    else:
        print("[INFO] COT fetch je vypnuty (--use_cot pro zapnuti).")

    print("\n[OK] Stahovani dokonceno. Data jsou ulozena v data/raw/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stazeni dat z TradingView (GOLD/VIX + volitelne DXY/COT).")
    ap.add_argument("--use_dxy", action="store_true", help="Zapnout stahovani DXY.")
    ap.add_argument("--use_cot", action="store_true", help="Zapnout stahovani COT.")
    ap.add_argument("--bars", type=int, default=5000, help="Kolik poslednich barů stáhnout (TV typicky max ~5000).")
    ap.add_argument("--mode", choices=["append", "overwrite"], default="append", help="append = doplní nová data do existujícího CSV")
    args = ap.parse_args()

    main(use_dxy=args.use_dxy, use_cot=args.use_cot, n_bars=args.bars, mode=args.mode)
