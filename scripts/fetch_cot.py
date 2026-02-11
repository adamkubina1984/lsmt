# scripts/fetch_cot.py
# ------------------------------------------------------------
# COT (Commitments of Traders) pro zlato -> data/raw/cot.csv (sloupce: date,cot)
# Primárně z TradingView (tvDatafeed), fallback z lokálního CSV.
# Přihlášení k TV přes env: TV_USERNAME / TV_PASSWORD (doporučeno).
# ------------------------------------------------------------

import os, sys, time, argparse
import pandas as pd
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR  = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

def tv_client():
    try:
        from tvDatafeed import TvDatafeed
    except Exception as e:
        print(f"[WARN] tvDatafeed nelze importovat: {e}")
        return None
    user = os.environ.get("TV_USERNAME")
    pwd  = os.environ.get("TV_PASSWORD")
    try:
        if user and pwd:
            print("[INFO] Přihlašuji se do TradingView (user/pass z env)...")
            return TvDatafeed(user, pwd)
        print("[WARN] TV bez loginu – rozsah může být omezen, CFTC feed často timeoutuje.")
        return TvDatafeed()  # nologin
    except Exception as e:
        print(f"[WARN] tvDatafeed init selhal: {e}")
        return None

def fetch_from_tv(symbol: str, exchange: str, interval: str = "W", n_bars: int = 2000) -> pd.DataFrame:
    from tvDatafeed import Interval
    tv = tv_client()
    if tv is None:
        return pd.DataFrame()

    # správné mapování intervalů
    i = interval.strip().lower()
    if i in ("w","1w","week","weekly"):
        tv_interval = Interval.in_weekly
    elif i in ("d","1d","day","daily"):
        tv_interval = Interval.in_daily
    else:
        tv_interval = Interval.in_weekly

    # retry loop (TV CFTC feed bývá náladový)
    for attempt in range(1, 4):
        try:
            print(f"[INFO] TV fetch {exchange}:{symbol} {tv_interval} (pokusu {attempt}/3)")
            df = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=n_bars)
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={"datetime":"date"})
                val_col = "close" if "close" in df.columns else (df.columns[-1] if len(df.columns)>1 else None)
                if not val_col:
                    print("[WARN] Nenalezen datový sloupec v odpovědi TV.")
                    return pd.DataFrame()
                out = df[["date", val_col]].copy()
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out = out.dropna(subset=["date"]).rename(columns={val_col:"cot"})
                print(f"[OK] TV COT: {len(out)} řádků")
                return out
            else:
                print("[WARN] TV vrátil prázdná data.")
        except Exception as e:
            print(f"[WARN] TV chyba: {e}")
        time.sleep(2*attempt)  # krátká prodleva a znovu
    return pd.DataFrame()

def read_and_normalize_local_cot(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Soubor nenalezen: {path}")
    df = pd.read_csv(path)
    # najdi sloupec s datem
    date_cols = [c for c in df.columns if str(c).lower() in
                 ("date","datetime","report_date","report_date_as_yyyy-mm-dd","report_date_as_mm_dd_yyyy","report_date_as_yyyymmdd")]
    if not date_cols:
        date_cols = [df.columns[0]]
    df = df.rename(columns={date_cols[0]:"date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    low = {c.lower(): c for c in df.columns}

    if "cot" in low:
        out = df[["date", low["cot"]]].rename(columns={low["cot"]:"cot"})
        out["cot"] = pd.to_numeric(out["cot"], errors="coerce")
        return out.dropna(subset=["cot"])

    # pokus o net = (non-commercial long − non-commercial short)
    long_cols  = [c for c in df.columns if "long"  in c.lower() and ("non" in c.lower() or "spec" in c.lower())]
    short_cols = [c for c in df.columns if "short" in c.lower() and ("non" in c.lower() or "spec" in c.lower())]
    if long_cols and short_cols:
        lc, sc = long_cols[0], short_cols[0]
        tmp = df[["date", lc, sc]].copy()
        tmp["cot"] = pd.to_numeric(tmp[lc], errors="coerce") - pd.to_numeric(tmp[sc], errors="coerce")
        return tmp[["date","cot"]].dropna()

    # fallback: jakýkoli sloupec s 'net'
    net_cols = [c for c in df.columns if "net" in c.lower()]
    if net_cols:
        nc = net_cols[0]
        out = df[["date", nc]].rename(columns={nc:"cot"})
        out["cot"] = pd.to_numeric(out["cot"], errors="coerce")
        return out.dropna(subset=["cot"])

    raise ValueError("Nepodařilo se najít sloupce pro výpočet COT (hledám 'COT' nebo Non-commercial Long/Short).")

def main():
    p = argparse.ArgumentParser(description="Stažení/normalizace COT pro zlato (TV / CSV).")
    p.add_argument("--source", choices=["tv","file","auto"], required=True)
    p.add_argument("--symbol", help="TV symbol, např. CFTC:091741_F (Gold).")
    p.add_argument("--exchange", default="CFTC")
    p.add_argument("--interval", default="W")
    p.add_argument("--nbars", type=int, default=2000)
    p.add_argument("--file", help="Cesta k lokálnímu CSV (fallback).")
    args = p.parse_args()

    out_path = os.path.join(DATA_DIR, "cot.csv")

    if args.source in ("tv","auto"):
        # známé kandidáty pro zlato (mohou se lišit mezi konty/zdroji na TV)
        candidates = []
        if args.symbol:
            candidates = [(args.exchange, args.symbol)]
        else:
            candidates = [
                ("CFTC","091741_F"),   # Gold (Managed Money Net) – časté
                ("CFTC","088691_F"),   # Gold (Legacy)
                ("CFTC","099741_F"),   # někdy "Non-Commercial Net"
            ]
        for exch, sym in candidates:
            df = fetch_from_tv(symbol=sym, exchange=exch, interval=args.interval, n_bars=args.nbars)
            if not df.empty:
                df = df[["date","cot"]].copy()
                df["cot"] = pd.to_numeric(df["cot"], errors="coerce")
                df = df.dropna(subset=["cot"])
                df.to_csv(out_path, index=False)
                print(f"[✓] TV COT uloženo: {out_path} ({len(df)} řádků) z {exch}:{sym}")
                return
        if args.source == "tv":
            print("[ERROR] TV nevrátil COT. Zkus přidat --symbol a přihlašovací údaje (TV_USERNAME/TV_PASSWORD), nebo použij --source file.")
            sys.exit(1)
        else:
            print("[WARN] TV nevrátil COT, zkouším fallback file...")

    if args.source in ("file","auto"):
        if not args.file:
            print("[ERROR] Pro --source file/auto uveď --file <cesta_k_csv> s COT exportem.")
            sys.exit(1)
        df = read_and_normalize_local_cot(args.file)
        if df.empty:
            print("[ERROR] Fallback CSV je prázdné nebo neobsahuje COT.")
            sys.exit(1)
        df.to_csv(out_path, index=False)
        print(f"[✓] COT z CSV uloženo: {out_path} ({len(df)} řádků)")
        return

if __name__ == "__main__":
    main()
