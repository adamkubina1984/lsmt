# scripts/live_alerts.py
import os, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from features import add_indicators

# Windows zvuk / toast (toast volitelně)
WIN = (os.name == 'nt')
if WIN:
    import winsound
    try:
        from win10toast import ToastNotifier
        TOASTER = ToastNotifier()
    except Exception:
        TOASTER = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

def _read_csv(path: Path):
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize date column
    dc = [c for c in df.columns if c.lower() in ("date","datetime","time")]
    if dc:
        df.rename(columns={dc[0]: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, src_col: str, out_col: str):
    if right.empty:
        left[out_col] = np.nan
        return left
    r = right[["date", src_col]].rename(columns={src_col: out_col}).sort_values("date")
    l = left.sort_values("date").copy()
    return pd.merge_asof(l, r, on="date", direction="backward")

def load_meta(timeframe="5m"):
    meta_path = MODELS / f"features_tv_{timeframe}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Chybí metadata: {meta_path}")
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = load_model(ROOT / meta["model_path"])
    scaler = joblib.load(ROOT / meta["scaler_path"])
    feats = meta["feature_cols"]
    seq_len = int(meta["seq_len"])
    indicators = [f.strip().lower() for f in meta.get("features", [])]
    cot_shift_days = int(meta.get("cot_shift_days", 0))
    use_dxy = bool(meta.get("use_dxy", False))
    use_cot = bool(meta.get("use_cot", False))
    return model, scaler, feats, seq_len, indicators, cot_shift_days, use_dxy, use_cot

def last_closed_bar_5m(ts: pd.Timestamp) -> bool:
    # jednoduchá detekce uzavřené 5m svíčky v lokálním čase souboru:
    # spoléháme na to, že gold_5m.csv je generováno fetch skriptem po uzavření baru
    return True

def predict_last(model, scaler, feats, seq_len, timeframe="5m", indicators=None, cot_shift_days=0, use_dxy=False, use_cot=False):
    # načti GOLD
    gold = _read_csv(DATA / f"gold_{'5m' if timeframe=='5m' else '1h'}.csv")
    if gold.empty or len(gold) < seq_len+1:
        return None
    # zajisti "average"
    if "average" not in gold.columns and "high" in gold.columns and "low" in gold.columns:
        gold["average"] = (gold["high"] + gold["low"]) / 2.0

    # VIX/DXY/COT merge
    def safe_merge(name, col):
        nonlocal gold
        df = _read_csv(DATA / f"{name}.csv")
        if df.empty:
            gold[col] = np.nan
        else:
            if name == "cot" and cot_shift_days != 0 and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce") + pd.Timedelta(days=cot_shift_days)
            src = col if col in df.columns else ("close" if "close" in df.columns else df.columns[-1])
            gold = _merge_asof(gold, df, src, col)
    safe_merge("vix","vix")
    if use_dxy:
        safe_merge("dxy","dxy")
    else:
        gold["dxy"] = 0.0
    if use_cot:
        safe_merge("cot","cot")
    else:
        gold["cot"] = 0.0
    for c in ("vix","dxy","cot"):
        gold[c] = gold[c].ffill().bfill()

    gold = add_indicators(gold, indicators or [])

    # vyber jen požadované featury (chybějící doplň 0)
    X_df = gold.copy()
    for c in feats:
        if c not in X_df.columns: X_df[c] = 0.0
    X_df = X_df[feats].ffill().bfill()
    X_all = scaler.transform(X_df.values.astype(float))
    X_seq = X_all[-seq_len:]  # poslední sekvence
    X_seq = np.expand_dims(X_seq, axis=0)  # (1, seq_len, n_feat)

    probs = model.predict(X_seq, verbose=0)[0]  # (3,)
    cls = int(np.argmax(probs))
    strength = float(np.max(probs))
    # vrať i aktuální cenu a datum posledního baru
    last = gold.iloc[-1]
    return {
        "date": pd.to_datetime(last["date"]),
        "price": float(last["close"]),
        "signal": cls,
        "strength": strength,
        "proba_no_trade": float(probs[0]),
        "proba_buy": float(probs[1]),
        "proba_sell": float(probs[2])
    }

def beep(signal:int):
    if not WIN: 
        return
    # odlišné tóny: Buy vyšší, Sell nižší
    if signal == 1:
        winsound.Beep(1200, 300)
        winsound.Beep(1500, 300)
    elif signal == 2:
        winsound.Beep(600, 300)
        winsound.Beep(400, 300)

def toast(title, text):
    if WIN and TOASTER:
        try:
            TOASTER.show_toast(title, text, duration=5, threaded=True)
        except Exception:
            pass

def live_loop(timeframe="5m", min_conf=0.47, allow_short=True, poll_sec=10, out_csv=None):
    print("[INFO] Start Live Alerts...")
    out_csv = Path(out_csv) if out_csv else (RESULTS / "live_alerts.csv")
    # připrav výstupní soubor
    if not out_csv.exists():
        pd.DataFrame(columns=["date", "signal", "strength", "price", "proba_no_trade", "proba_buy", "proba_sell"]).to_csv(out_csv, index=False)

    model, scaler, feats, seq_len, indicators, cot_shift_days, use_dxy, use_cot = load_meta(timeframe=timeframe)
    last_alert_time = None

    while True:
        try:
            # poslední uzavřený bar
            res = predict_last(
                model, scaler, feats, seq_len,
                timeframe=timeframe,
                indicators=indicators,
                cot_shift_days=cot_shift_days,
                use_dxy=use_dxy,
                use_cot=use_cot
            )
            if res is None:
                time.sleep(poll_sec); continue

            ts = res["date"]
            price = res["price"]
            sig = res["signal"]
            conf= res["strength"]

            # upozorňuj jen jednou na nový bar (a jen silné signály)
            # záznam je nový, pokud timestamp změněn
            is_new = (last_alert_time is None) or (ts != last_alert_time)
            good  = (conf >= min_conf) and (sig in (1,2)) and (allow_short or sig==1)

            if is_new and good:
                # zvuk + toast + log do CSV
                beep(sig)
                title = "LSTM ALERT"
                txt   = f"{'BUY' if sig==1 else 'SELL'} @ {price:.2f} | conf={conf:.2f}"
                print(f"[ALERT] {ts}  {txt}")
                toast(title, txt)
                # append row
                row = pd.DataFrame([{
                    "date": ts,
                    "signal": sig,
                    "strength": conf,
                    "price": price,
                    "proba_no_trade": res["proba_no_trade"],
                    "proba_buy": res["proba_buy"],
                    "proba_sell": res["proba_sell"]
                }])
                row.to_csv(out_csv, mode="a", header=False, index=False)
                last_alert_time = ts

        except KeyboardInterrupt:
            print("[INFO] Stop Live Alerts (CTRL+C).")
            break
        except Exception as e:
            print("[WARN] Live loop error:", e)

        time.sleep(poll_sec)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Live Alerts: akustická (a toast) upozornění na BUY/SELL signály.")
    ap.add_argument("--timeframe", choices=["5m","1h"], default="5m")
    ap.add_argument("--min_conf", type=float, default=0.47)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--poll_sec", type=int, default=10, help="jak často kontrolovat (s)")
    ap.add_argument("--out_csv", type=str, default=None, help="kam logovat alerty (CSV)")
    args = ap.parse_args()

    live_loop(timeframe=args.timeframe,
              min_conf=args.min_conf,
              allow_short=args.allow_short,
              poll_sec=args.poll_sec,
              out_csv=args.out_csv)
