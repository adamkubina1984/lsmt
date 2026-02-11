# scripts/train_lstm_tradingview.py
# ------------------------------------------------------------
# Trénink LSTM nad daty z TradingView (GOLD + VIX/DXY/COT – podmíněně)
# - Všechny cesty RELATIVNÍ k projektu
# - Vstupy se očekávají v data/raw/
# - Výstupy (model/scaler/metadata) do models/
# - Klasifikace: Buy / Sell / No-Trade dle budoucí změny (horizon, thr_pct)
# ------------------------------------------------------------

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from pathlib import Path
from features import add_indicators
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _read_csv_safe(path: str, date_col: str = "date") -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[VAROVÁNÍ] Soubor neexistuje: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # najdi časový sloupec
    date_candidates = [c for c in df.columns if c.lower() in ("date", "datetime", "time")]
    if not date_candidates:
        # fallback: první sloupec jako date
        date_candidates = [df.columns[0]]
    df.rename(columns={date_candidates[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # standardizuj názvy
    df.columns = [c.lower() for c in df.columns]
    return df

def _load_json_robust(path: str) -> dict:
    with open(path, "rb") as fb:
        raw = fb.read()
    txt = raw.decode("utf-8", errors="ignore").lstrip("\ufeff").strip()
    import json as _json
    try:
        return _json.loads(txt)
    except _json.JSONDecodeError:
        start = txt.find("{")
        if start != -1:
            depth = 0; end = -1
            for i,ch in enumerate(txt[start:], start=start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i+1; break
            if end != -1:
                return _json.loads(txt[start:end])
        raise



def _merge_asof(left_df: pd.DataFrame, right_df: pd.DataFrame, right_col: str, out_col: str) -> pd.DataFrame:
    if right_df.empty:
        left_df[out_col] = np.nan
        print(f"[INFO] Zdroj '{out_col}' chybí → nastavím NaN (nebude zahrnut do feature).")
        return left_df
    if "date" not in right_df.columns:
        raise ValueError("Pravý DataFrame nemá sloupec 'date'.")
    right = right_df[["date", right_col]].rename(columns={right_col: out_col}).sort_values("date")
    left = left_df.sort_values("date").copy()
    merged = pd.merge_asof(left, right, on="date", direction="backward")
    return merged


def _make_labels(close: pd.Series, horizon: int, thr_pct: float) -> pd.Series:
    ret_fwd = (close.shift(-horizon) - close) / close * 100.0
    labels = pd.Series(0, index=close.index)  # 0 = No-Trade
    labels[ret_fwd > thr_pct] = 1           # 1 = Buy
    labels[ret_fwd < -thr_pct] = 2          # 2 = Sell
    return labels


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, split_index: int):
    X_seqs, y_seqs, end_idx = [], [], []
    for end in range(seq_len - 1, len(X)):
        start = end - (seq_len - 1)
        X_seqs.append(X[start : end + 1])
        y_seqs.append(int(y[end]))
        end_idx.append(end)
    X_seqs = np.array(X_seqs)
    y_seqs = np.array(y_seqs)
    end_idx = np.array(end_idx)
    # časový split
    train_mask = end_idx < split_index
    X_train, y_train = X_seqs[train_mask], y_seqs[train_mask]
    X_val, y_val = X_seqs[~train_mask], y_seqs[~train_mask]
    return X_train, y_train, X_val, y_val

def train_lstm(
    timeframe: str = "5m",
    seq_len: int = 60,
    horizon: int = 3,
    thr_pct: float = 0.2,
    epochs: int = 20,
    batch_size: int = 32,
    model_name: str = None,
    cot_shift_days: int = 3,
    features: str = "rsi,macd,ema20,bb,atr",
    output_dir: str = "models",
    mode: str = "multi"                   # << přidán parametr
):
    # --- výstupní složka a identifikátor běhu ---
    OUT_DIR = os.path.join(ROOT_DIR, output_dir)  # respektuje parametr --output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    run_id = f"lstm_{timeframe}_{seq_len}lb_{datetime.now():%Y%m%d_%H%M%S}"

    # --- Načtení a merge dat (stejně jako máš, zkráceno) ---
    gold_file = f"gold_{'5m' if timeframe == '5m' else '1h'}.csv"
    gold_path = os.path.join(DATA_DIR, gold_file)
    gold = _read_csv_safe(gold_path)
    if gold.empty:
        raise FileNotFoundError(f"Chybí GOLD data: {gold_path}")
    for c in ["open", "high", "low", "close"]:
        if c not in gold.columns:
            raise ValueError(f"V GOLD datech chybí sloupec '{c}'.")
    if "volume" not in gold.columns:
        gold["volume"] = 0.0
    gold["average"] = (gold["high"] + gold["low"]) / 2.0

    # --- VIX ---
    vix = _read_csv_safe(os.path.join(DATA_DIR, "vix.csv"))
    if not vix.empty:
        vix_col = "vix" if "vix" in vix.columns else ("close" if "close" in vix.columns else vix.columns[-1])
        gold = _merge_asof(gold, vix, vix_col, "vix")
    else:
        gold["vix"] = np.nan

    # --- DXY ---
    dxy = _read_csv_safe(os.path.join(DATA_DIR, "dxy.csv"))
    if not dxy.empty:
        dxy_col = "dxy" if "dxy" in dxy.columns else ("close" if "close" in dxy.columns else dxy.columns[-1])
        gold = _merge_asof(gold, dxy, dxy_col, "dxy")
    else:
        gold["dxy"] = np.nan

    # --- COT (weekly, může chybět) ---
    cot_path = os.path.join(DATA_DIR, "cot.csv")
    cot = _read_csv_safe(cot_path)
    if not cot.empty:
        # posun (report je k úterý, publikace v pátek)
        if cot_shift_days != 0:
            cot["date"] = cot["date"] + pd.Timedelta(days=cot_shift_days)
        # najdi sloupec s hodnotou
        cot_val_col = None
        for cand in ("cot", "net", "value", "close"):
            if cand in cot.columns:
                cot_val_col = cand; break
        if cot_val_col is None:
            cot_val_col = cot.columns[-1]
        gold = _merge_asof(gold, cot[["date", cot_val_col]].rename(columns={cot_val_col:"cot"}), "cot", "cot")
    else:
        gold["cot"] = np.nan
    
    # --- zvolené indikátory (CSV -> list) ---
    wanted_feats = [s.strip().lower() for s in features.split(",") if s.strip()]

    # přidání indikátorů a sestavení feature_cols
    _before_cols = set(gold.columns)
    gold = add_indicators(gold, wanted_feats)
    indicator_cols = [c for c in gold.columns if c not in _before_cols and c != "date"]
    feature_cols = ["open", "high", "low", "close", "volume", "average"]
    for c in ("vix", "dxy", "cot"):
        if gold[c].notna().any():
            feature_cols.append(c)
    feature_cols += indicator_cols
    print(f"[INFO] Přidané indikátorové sloupce: {indicator_cols}")

    # --- Labely a střih do validního rozpětí ---
    labels = _make_labels(gold["close"], horizon=horizon, thr_pct=thr_pct)
    valid_len = len(gold) - horizon
    gold = gold.iloc[:valid_len].copy()
    labels = labels.iloc[:valid_len].copy()

    # --- škálování + sekvence ---
    split_index = int(len(gold) * 0.8)
    if split_index <= seq_len:
        raise ValueError(f"Train split ({split_index}) je příliš malý pro seq_len {seq_len}. Přidej data nebo sniž seq_len.")
    scaler = StandardScaler()
    X_all = gold[feature_cols].values.astype(float)
    scaler.fit(X_all[:split_index])
    X_scaled = scaler.transform(X_all)

    X_train, y_train, X_val, y_val = _build_sequences(X_scaled, labels.values, seq_len, split_index - 1)
    print(f"[INFO] Train sekvence: {X_train.shape}, Val sekvence: {X_val.shape}")
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError("Po vytvoření sekvencí nemám žádná trénovací/validační data.")

    # --- větvení podle režimu ---
    if mode == "multi":
        last = Dense(3, activation='softmax')
        loss = "sparse_categorical_crossentropy"
        y_tr, y_va = y_train, y_val

    elif mode == "trade":
        # binární: obchod vs no-trade
        y_tr = (y_train != 0).astype(int)
        y_va = (y_val   != 0).astype(int)
        last = Dense(1, activation='sigmoid')
        loss = "binary_crossentropy"

    elif mode == "direction":
        # binární: BUY(1) vs SELL(0) – filtruj jen bary s trade
        m_tr = (y_train != 0)
        m_va = (y_val   != 0)
        X_train = X_train[m_tr]
        X_val   = X_val[m_va]
        y_tr    = (y_train[m_tr] == 1).astype(int)
        y_va    = (y_val[m_va]   == 1).astype(int)
        last = Dense(1, activation='sigmoid')
        loss = "binary_crossentropy"
    else:
        raise ValueError(f"Neznámý mód: {mode}")

    # --- model ---
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        last
    ])
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # class_weight – spočítat na y_tr (správná proměnná pro zvolený mód)
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_tr)
    if classes.size >= 2:
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
        class_weight = {int(k): float(v) for k, v in zip(classes, cw)}
        print("[INFO] Použité class_weight:", class_weight)
    else:
        print("[WARN] V trénovacích datech je jen jedna třída -> class_weight vypínám.")
        class_weight = None


    model.fit(X_train, y_tr, validation_data=(X_val, y_va),
              epochs=epochs, batch_size=batch_size,
              callbacks=[es], verbose=1,
              class_weight=class_weight)

    # --- uložení ---
    if mode == "trade":
        model_name = f"lstm_trade_{timeframe}.h5"
        scaler_name = f"scaler_trade_{timeframe}.pkl"
    elif mode == "direction":
        model_name = f"lstm_dir_{timeframe}.h5"
        scaler_name = f"scaler_dir_{timeframe}.pkl"
    else:
        model_name = f"lstm_tv_{timeframe}.h5"
        scaler_name = f"scaler_tv_{timeframe}.pkl"

    model_path   = os.path.join(OUT_DIR, model_name)
    scaler_path  = os.path.join(OUT_DIR, scaler_name)
    meta_path    = os.path.join(OUT_DIR, f"features_tv_{timeframe}.json")   # kanonická meta
    archive_path = os.path.join(OUT_DIR, f"{run_id}_meta.json")             # archiv běhu


    model.save(model_path)
    joblib.dump(scaler, scaler_path)


    # načti/aktualizuj metadata konzistentně
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = _load_json_robust(meta_path)
        except Exception as e:
            print(f"[WARN] Neplatný JSON v {meta_path}: {e} -> přepisuji čistým metadatem.")
            meta = {}

    meta["timeframe"] = timeframe
    meta["seq_len"] = seq_len
    meta["threshold_pct"] = thr_pct
    meta["feature_cols"] = feature_cols
    meta["features"] = wanted_feats  # aktuální sada indikátorů (ruční/auto)

    # odkazy pro multi i two-stage
    if mode == "multi":
        meta["model_path"]  = os.path.relpath(model_path, ROOT_DIR)
        meta["scaler_path"] = os.path.relpath(scaler_path, ROOT_DIR)
    else:
        ts = meta.get("two_stage", {})
        if mode == "trade":
            ts["model_path_trade"]  = os.path.relpath(model_path, ROOT_DIR)
            ts["scaler_path_trade"] = os.path.relpath(scaler_path, ROOT_DIR)
            ts["features_trade"]    = wanted_feats
        elif mode == "direction":
            ts["model_path_dir"]  = os.path.relpath(model_path, ROOT_DIR)
            ts["scaler_path_dir"] = os.path.relpath(scaler_path, ROOT_DIR)
            ts["features_dir"]    = wanted_feats

        ts["enabled"] = bool(ts.get("model_path_trade")) and bool(ts.get("model_path_dir"))
        ts.setdefault("min_conf_trade", 0.48)
        ts.setdefault("min_conf_dir",   0.55)
        ts.setdefault("min_margin_dir", 0.05)
        meta["two_stage"] = ts


    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Model uložen:   {model_path}")
    print(f"[OK] Scaler uložen:  {scaler_path}")
    print(f"[OK] Metadata uložena: {meta_path}")
    print(f"[OK] Archiv meta:      {archive_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trénink LSTM nad daty z TradingView (GOLD + volitelné VIX/DXY/COT).")
    parser.add_argument("--output_dir", type=str, default="models", help="Složka pro uložení modelu/scaler/meta")
    parser.add_argument("--timeframe", choices=["5m", "1h"], default="5m")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--thr_pct", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--cot_shift_days", type=int, default=3, help="Posun COT dat (dny, + posouvá kupředu; default 3)")
    parser.add_argument("--features", type=str, default="rsi,macd,ema20,bb,atr", help="Seznam indikátorů oddělených čárkou (např. rsi,macd,ema20,bb,atr)")
    parser.add_argument("--mode", choices=["multi", "trade", "direction"], default="multi", help="multi = původní 3-class; trade = binární Trade/NoTrade; direction = binární Buy/Sell (jen na samplech, kde nebyl No-Trade)")

    args = parser.parse_args()

    train_lstm(
        timeframe=args.timeframe,
        seq_len=args.seq_len,
        horizon=args.horizon,
        thr_pct=args.thr_pct,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        cot_shift_days=args.cot_shift_days,
        features=args.features,
        output_dir=args.output_dir,
        mode=args.mode              # << přidáno
    )
