# scripts/live_alerts.py
import os, time, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from keras.models import load_model
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
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_len = int(meta["seq_len"])
    indicators = [f.strip().lower() for f in meta.get("features", [])]
    cot_shift_days = int(meta.get("cot_shift_days", 0))
    use_dxy = bool(meta.get("use_dxy", False))
    use_cot = bool(meta.get("use_cot", False))

    ts = (meta.get("two_stage") or {})
    ts_enabled = bool(ts.get("enabled"))
    req_trade = ts.get("model_path_trade")
    req_dir = ts.get("model_path_dir")
    req_scaler_trade = ts.get("scaler_path_trade")
    req_scaler_dir = ts.get("scaler_path_dir")
    ts_ready = ts_enabled and all([req_trade, req_dir, req_scaler_trade, req_scaler_dir])

    if ts_ready:
        cfg = {
            "timeframe": timeframe,
            "mode": "two_stage",
            "seq_len": seq_len,
            "indicators": indicators,
            "cot_shift_days": cot_shift_days,
            "use_dxy": use_dxy,
            "use_cot": use_cot,
            "model_trade": load_model(ROOT / req_trade),
            "model_dir": load_model(ROOT / req_dir),
            "scaler_trade": joblib.load(ROOT / req_scaler_trade),
            "scaler_dir": joblib.load(ROOT / req_scaler_dir),
            "feature_cols_trade": ts.get("feature_cols_trade") or meta.get("feature_cols_trade") or [],
            "feature_cols_dir": ts.get("feature_cols_dir") or meta.get("feature_cols_dir") or [],
            "features_trade": [f.strip().lower() for f in (ts.get("features_trade") or indicators)],
            "features_dir": [f.strip().lower() for f in (ts.get("features_dir") or indicators)],
            "min_conf_trade": float(ts.get("min_conf_trade", meta.get("min_conf_trade", 0.48))),
            "min_conf_dir": float(ts.get("min_conf_dir", meta.get("min_conf_dir", 0.55))),
            "min_margin_dir": float(ts.get("min_margin_dir", meta.get("min_margin_dir", 0.05))),
            "vol_filter": meta.get("vol_filter", {}),
        }
        if not cfg["feature_cols_trade"] or not cfg["feature_cols_dir"]:
            raise ValueError("Two-stage metadata jsou nekompletní: chybí feature_cols_trade/feature_cols_dir.")
        return cfg

    # fallback na single-stage
    return {
        "timeframe": timeframe,
        "mode": "single",
        "seq_len": seq_len,
        "indicators": indicators,
        "cot_shift_days": cot_shift_days,
        "use_dxy": use_dxy,
        "use_cot": use_cot,
        "model": load_model(ROOT / meta["model_path"]),
        "scaler": joblib.load(ROOT / meta["scaler_path"]),
        "feature_cols": meta["feature_cols"],
    }

def last_closed_bar_5m(ts: pd.Timestamp) -> bool:
    # jednoduchá detekce uzavřené 5m svíčky v lokálním čase souboru:
    # spoléháme na to, že gold_5m.csv je generováno fetch skriptem po uzavření baru
    return True

def predict_last(cfg):
    # načti GOLD
    timeframe = cfg["timeframe"]
    seq_len = int(cfg["seq_len"])
    indicators = cfg.get("indicators", [])
    cot_shift_days = int(cfg.get("cot_shift_days", 0))
    use_dxy = bool(cfg.get("use_dxy", False))
    use_cot = bool(cfg.get("use_cot", False))

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

    if cfg["mode"] == "two_stage":
        union_feats = []
        for f in (cfg.get("features_trade", []) + cfg.get("features_dir", [])):
            f = (f or "").strip().lower()
            if f and f not in union_feats:
                union_feats.append(f)
        for f in ("atr", "bb"):
            if f not in union_feats:
                union_feats.append(f)
        gold = add_indicators(gold, union_feats)

        fcols_trade = cfg["feature_cols_trade"]
        fcols_dir = cfg["feature_cols_dir"]
        for c in fcols_trade:
            if c not in gold.columns:
                gold[c] = 0.0
        for c in fcols_dir:
            if c not in gold.columns:
                gold[c] = 0.0

        X_tr = gold[fcols_trade].ffill().bfill().values.astype(float)
        X_di = gold[fcols_dir].ffill().bfill().values.astype(float)
        X_tr = cfg["scaler_trade"].transform(X_tr)
        X_di = cfg["scaler_dir"].transform(X_di)
        X_tr = np.expand_dims(X_tr[-seq_len:], axis=0)
        X_di = np.expand_dims(X_di[-seq_len:], axis=0)

        p_trade = float(cfg["model_trade"].predict(X_tr, verbose=0)[0][0])
        p_buy = float(cfg["model_dir"].predict(X_di, verbose=0)[0][0])
        p_sell = 1.0 - p_buy
        p_no = 1.0 - p_trade

        cls = 0
        if p_trade >= cfg["min_conf_trade"]:
            dir_conf = max(p_buy, p_sell)
            margin = abs(p_buy - p_sell)
            if dir_conf >= cfg["min_conf_dir"] and margin >= cfg["min_margin_dir"]:
                cls = 1 if p_buy > p_sell else 2

        # volatility gate
        vol_cfg = cfg.get("vol_filter", {}) or {}
        if bool(vol_cfg.get("enabled")) and ("atr_norm" in gold.columns) and ("bb_width" in gold.columns):
            thr_atr = float(vol_cfg.get("min_atr_norm", gold["atr_norm"].quantile(0.05)))
            thr_bb = float(vol_cfg.get("min_bb_width", gold["bb_width"].quantile(0.05)))
            last = gold.iloc[-1]
            if (float(last["atr_norm"]) < thr_atr) and (float(last["bb_width"]) < thr_bb):
                cls = 0
                p_trade = 0.0
                p_no = 1.0
                p_buy = 0.0
                p_sell = 0.0

        strength = float(p_buy if cls == 1 else (p_sell if cls == 2 else p_no))
    else:
        gold = add_indicators(gold, indicators or [])
        feats = cfg["feature_cols"]
        X_df = gold.copy()
        for c in feats:
            if c not in X_df.columns:
                X_df[c] = 0.0
        X_df = X_df[feats].ffill().bfill()
        X_all = cfg["scaler"].transform(X_df.values.astype(float))
        X_seq = X_all[-seq_len:]
        X_seq = np.expand_dims(X_seq, axis=0)
        probs = cfg["model"].predict(X_seq, verbose=0)[0]
        cls = int(np.argmax(probs))
        strength = float(np.max(probs))
        p_no = float(probs[0])
        p_buy = float(probs[1])
        p_sell = float(probs[2])

    # vrať i aktuální cenu a datum posledního baru
    last = gold.iloc[-1]
    return {
        "date": pd.to_datetime(last["date"]),
        "price": float(last["close"]),
        "signal": cls,
        "strength": strength,
        "proba_no_trade": p_no,
        "proba_buy": p_buy,
        "proba_sell": p_sell
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

    cfg = load_meta(timeframe=timeframe)
    print(f"[INFO] Live inference mode: {cfg['mode']}", flush=True)
    last_alert_time = None

    while True:
        try:
            # poslední uzavřený bar
            res = predict_last(cfg)
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
