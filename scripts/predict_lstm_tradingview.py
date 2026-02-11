# scripts/predict_lstm_tradingview.py
# ------------------------------------------------------------
# Predikce signálů LSTM nad daty z TradingView (GOLD + VIX/DXY/COT)
# - Čte stejné features z metadata JSON
# - Použije správný scaler a model
# - Uloží CSV s: date, close, signal, signal_strength
# ------------------------------------------------------------
import glob
import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import subprocess
import uuid
import sys
from pathlib import Path
from tensorflow.keras.models import load_model
from features import add_indicators
from simulate_strategy_v2 import simulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[VAROVÁNÍ] Soubor neexistuje: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    date_candidates = [c for c in df.columns if c.lower() in ("date", "datetime", "time")]
    if not date_candidates:
        date_candidates = [df.columns[0]]
    df.rename(columns={date_candidates[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def _merge_asof(left_df: pd.DataFrame, right_df: pd.DataFrame, right_col: str, out_col: str) -> pd.DataFrame:
    if right_df.empty:
        left_df[out_col] = np.nan
        return left_df
    right = right_df[["date", right_col]].rename(columns={right_col: out_col}).sort_values("date")
    left = left_df.sort_values("date").copy()
    merged = pd.merge_asof(left, right, on="date", direction="backward")
    return merged

def _find_meta_for_tf(models_dir: str, timeframe: str) -> str:
    """
    Vrátí cestu k metadatům pro daný timeframe.
    1) Preferuje staré 'features_tv_<tf>.json'
    2) Jinak vezme nejnovější '*_<tf>_*_meta.json' nebo 'lstm_<tf>_*_meta.json'
    """
    cand1 = os.path.join(models_dir, f"features_tv_{timeframe}.json")
    if os.path.exists(cand1):
        return cand1
    pats = [
        os.path.join(models_dir, f"*_{timeframe}_*_meta.json"),
        os.path.join(models_dir, f"lstm_{timeframe}_*_meta.json"),
        os.path.join(models_dir, f"*{timeframe}*_meta.json"),
    ]
    metas = []
    for p in pats:
        metas.extend(glob.glob(p))
    if not metas:
        raise FileNotFoundError(f"Nenalezena metadata pro '{timeframe}'. Hledal jsem: {pats}")
    metas.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return metas[0]

def _model_input_seq_nfeat(model_path: str) -> tuple[int,int]:
    """
    Zjistí (seq_len, n_features) z uloženého Keras modelu.
    Očekává tvar vstupu (None, seq_len, n_features).
    """
    m = load_model(model_path)
    ish = m.input_shape
    if isinstance(ish, list):  # některé modely mají víc vstupů
        ish = ish[0]
    return int(ish[1]), int(ish[2])

def _load_json_robust(path: str) -> dict:
    """Načte JSON i když je ve file BOM, komentář nebo přilepené další objekty."""
    with open(path, "rb") as fb:
        raw = fb.read()
    # odstranění BOM, převod na str
    txt = raw.decode("utf-8", errors="ignore").lstrip("\ufeff").strip()
    # 1) standardní cesta
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # 2) pokus: vytáhni první vyvážený objekt {...}
        start = txt.find("{")
        if start == -1:
            raise
        depth = 0
        end = -1
        for i, ch in enumerate(txt[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            candidate = txt[start:end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        # 3) konec: vyhoď chybu se snippetem
        preview = txt[:160].replace("\n", "\\n")
        raise json.JSONDecodeError(f"Soubor není platný JSON (path={path}, preview='{preview}...')", txt, 0)


def _build_sequences(X: np.ndarray, seq_len: int):
    X_seqs, end_idx = [], []
    for end in range(seq_len - 1, len(X)):
        start = end - (seq_len - 1)
        X_seqs.append(X[start: end + 1])
        end_idx.append(end)
    return np.array(X_seqs), np.array(end_idx)


def _log_signal_distribution(out_df: pd.DataFrame):
    if "signal" in out_df.columns:
        print("[INFO] Rozložení signálů:", out_df["signal"].value_counts().to_dict())


def _maybe_override_feature_cols_from_companion(scaler_path: str, fcols: list, label: str) -> list:
    """
    Zkusí načíst featury z 'companion' JSON souboru vedle scaleru:
    - <scaler>_features.json (klíč 'feature_cols' nebo 'feature_cols_<label>')
    - <scaler>_meta.json (dtto)
    Pokud nalezený seznam má stejnou délku jako scaler.n_features_in_, přepíše fcols.
    """
    try:
        sp = Path(scaler_path)
        cand1 = sp.with_suffix("").as_posix() + "_features.json"
        cand2 = sp.with_suffix("").as_posix() + "_meta.json"
        for cand in (cand1, cand2):
            if os.path.exists(cand):
                with open(cand, "r", encoding="utf-8") as f:
                    jf = json.load(f)
                key_specific = f"feature_cols_{label}"
                feats = jf.get("feature_cols") or jf.get(key_specific)
                return feats if feats else fcols
    except Exception as e:
        print(f"[WARN] {_short(scaler_path)}: companion features load failed → {e}", flush=True)
    return fcols

def _infer_feature_cols_from_features(gold_df: pd.DataFrame, used_feats: list[str]) -> list[str]:
    """
    Zkonstruuje pořadí sloupců pro single-stage predikci, pokud meta nemá 'feature_cols'.
    Vychází z běžné konvence našich tréninků.
    """
    base = ["open","high","low","close","volume","average","vix","dxy","cot"]

    # mapování indikátor -> očekávané sloupce (po add_indicators)
    mapping = {
        "macd":  ["macd","macd_signal","macd_hist"],
        "rsi":   ["rsi14"],
        "atr":   ["atr14","atr_norm"],
        "mfi":   ["mfi14"],
        "ema20": ["ema20","ema20_dist"],
        "ema50": ["ema50","ema50_dist"],
        "ema100":["ema100","ema100_dist"],
        "bb":    ["bb_ma","bb_up","bb_lo","bb_width","bb_pos"],
        "roc":   ["roc","roc14","roc_14"],  # vezmeme, co reálně v df existuje
    }

    out = list(base)
    for f in used_feats or []:
        f = f.strip().lower()
        cols = mapping.get(f, [f])
        for c in cols:
            if c in gold_df.columns and c not in out:
                out.append(c)

    # odstranění duplicit pro jistotu, zachovat pořadí
    seen = set(); out = [x for x in out if not (x in seen or seen.add(x))]
    return out


def _short(path: str) -> str:
    try:
        return os.path.relpath(path, ROOT_DIR)
    except Exception:
        return path


def simulate_strategy_with_indicators(features: list[str],
                                      timeframe: str = "5m",
                                      out_csv: str = None,
                                      min_conf: float = 0.55,
                                      fee_pct: float = 0.001,
                                      allow_short: bool = True) -> dict:
    """
    Spustí predikci pro dané indikátory a vyhodnotí simulací.
    Vrací metriky ve formátu očekávaném evolucí: {"PnL_%", "Sharpe", "Trades", "Bars"}.
    """
    # 1) spustíme predikci do dočasného nebo zadaného out_csv
    tmp_csv = out_csv or os.path.join(RESULTS_DIR, f"_tmp_{uuid.uuid4().hex[:8]}.csv")
    pred_cmd = [sys.executable, "-u", os.path.join("scripts", "predict_lstm_tradingview.py"),
                "--timeframe", timeframe, "--output", tmp_csv, "--features", ",".join(features)]
    print("[RUN]", " ".join(pred_cmd), flush=True)
    subprocess.run(pred_cmd, check=True)

    # 2) simulace
    eq, trades = simulate(path_csv=tmp_csv, min_conf=min_conf, allow_short=allow_short, fee_pct=fee_pct)

    # 3) derivované metriky
    bars = len(eq)
    if bars <= 1:
        return {"PnL_%": 0.0, "Sharpe": 0.0, "Trades": 0, "Bars": bars}
    pnl_pct_series = eq["equity"].pct_change().fillna(0.0)
    sharpe = (pnl_pct_series.mean()/pnl_pct_series.std())*np.sqrt(252) if pnl_pct_series.std()>0 else 0.0
    pnl_total = (eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0) * 100.0

    return {"PnL_%": round(pnl_total, 2), "Sharpe": round(sharpe, 2), "Trades": len(trades), "Bars": bars}



def predict(timeframe: str,
            out_csv: str,
            min_conf: float = 0.0,
            features: list[str] = None,
            use_two_stage: bool = False):

    # === META ===
    meta_path = _find_meta_for_tf(MODELS_DIR, timeframe)
    meta = _load_json_robust(meta_path)
    print(f"[INFO] Načtena metadata: {os.path.relpath(meta_path, ROOT_DIR)}")

    # Paths (relativní vůči ROOT_DIR)
    model_path = os.path.join(ROOT_DIR, meta.get("model_path", f"models/lstm_tv_{timeframe}.h5"))
    scaler_path = os.path.join(ROOT_DIR, meta.get("scaler_path", f"models/scaler_tv_{timeframe}.pkl"))

    # seq_len – pokud chybí v meta, načti z modelu
    seq_len = meta.get("seq_len")
    if seq_len is None:
        try:
            seq_len, nfeat_from_model = _model_input_seq_nfeat(model_path)
            print(f"[INFO] seq_len odvozen z modelu: {seq_len} (n_features={nfeat_from_model})", flush=True)
        except Exception as e:
            raise KeyError(f"Metadata neobsahují 'seq_len' a nepodařilo se ho odvodit z modelu: {e}")
    else:
        seq_len = int(seq_len)

    # pokud byly featury předány z CLI, mají přednost
    wanted_feats = [f.strip().lower() for f in (features if features else meta.get("features", ["rsi","macd","ema20","bb","atr"]))]
    ts = meta.get("two_stage", {})

    # Rozhodnutí o two-stage: jen pokud jsou k dispozici oba modely
    req_trade = ts.get("model_path_trade")
    req_dir   = ts.get("model_path_dir")

    if (ts.get("enabled") or use_two_stage):
        if not req_trade or not req_dir:
            print("[WARN] Two-stage vyžádáno, ale v metadata.two_stage chybí model_path_trade/dir -> padám na single-stage.", flush=True)
            use_two_stage = False
        else:
            use_two_stage = True
    else:
        use_two_stage = False

    min_conf_trade = float(ts.get("min_conf_trade", 0.48))
    min_conf_dir = float(ts.get("min_conf_dir", 0.55))
    min_margin_dir = float(ts.get("min_margin_dir", 0.05))
    feats_trade = ts.get("features_trade") or wanted_feats
    feats_dir = ts.get("features_dir") or wanted_feats

    # === DATA ===
    gold_path = os.path.join(DATA_DIR, f"gold_{'5m' if timeframe == '5m' else '1h'}.csv")
    gold = _read_csv_safe(gold_path)
    if gold.empty:
        raise FileNotFoundError(f"Chybí GOLD data: {gold_path}")
    if "average" not in gold.columns:
        if "high" in gold.columns and "low" in gold.columns:
            gold["average"] = (gold["high"] + gold["low"]) / 2.0
        else:
            raise ValueError("Chybí high/low pro výpočet 'average'.")

    # makra
    vix = _read_csv_safe(os.path.join(DATA_DIR, "vix.csv"))
    if not vix.empty:
        vix_col = "vix" if "vix" in vix.columns else ("close" if "close" in vix.columns else vix.columns[-1])
        gold = _merge_asof(gold, vix, vix_col, "vix")
    else:
        gold["vix"] = np.nan
    dxy = _read_csv_safe(os.path.join(DATA_DIR, "dxy.csv"))
    if not dxy.empty:
        dxy_col = "dxy" if "dxy" in dxy.columns else ("close" if "close" in dxy.columns else dxy.columns[-1])
        gold = _merge_asof(gold, dxy, dxy_col, "dxy")
    else:
        gold["dxy"] = np.nan
    cot = _read_csv_safe(os.path.join(DATA_DIR, "cot.csv"))
    if not cot.empty:
        cot_shift_days = int(meta.get("cot_shift_days", 0))
        if cot_shift_days != 0:
            cot["date"] = cot["date"] + pd.Timedelta(days=cot_shift_days)
        cot_col = next((c for c in ("cot", "net", "value", "close") if c in cot.columns), cot.columns[-1])
        gold = _merge_asof(gold, cot, cot_col, "cot")
    else:
        gold["cot"] = np.nan
    for c in ("vix", "dxy", "cot"):
        gold[c] = gold[c].ffill().bfill()

    # === SCALER (společný pro single-stage) ===
    scaler = joblib.load(scaler_path)

    # --- feature_cols: z meta nebo bezpečný dopočet z features + zarovnání na scaler ---
    feature_cols = meta.get("feature_cols")
    if not feature_cols:
        gold_tmp = add_indicators(gold.copy(), wanted_feats)
        feature_cols = _infer_feature_cols_from_features(gold_tmp, wanted_feats)
        try:
            sc_tmp = joblib.load(scaler_path)
            nfi = getattr(sc_tmp, "n_features_in_", None)
            if nfi is not None:
                candidates = ["bb_pos","bb_width","bb_ma","bb_up","bb_lo","rsi14","ema100","ema100_dist","adx","stoch_k","stoch_d","vwap","sar"]
                for c in candidates:
                    if len(feature_cols) >= nfi:
                        break
                    if c in gold_tmp.columns and c not in feature_cols:
                        feature_cols.append(c)
                if len(feature_cols) > nfi:
                    feature_cols = feature_cols[:nfi]
            print(f"[INFO] feature_cols inferováno ({len(feature_cols)} sloupců) z features={wanted_feats}", flush=True)
        except Exception as e:
            print("[WARN] Nepodařilo se zarovnat na scaler:", e, flush=True)

    # --- Two-stage výchozí seznamy (až TEĎ už existuje feature_cols) ---
    fcols_trade = ts.get("feature_cols_trade") or feature_cols
    fcols_dir   = ts.get("feature_cols_dir")   or feature_cols

    scaler_path_trade = os.path.join(ROOT_DIR, ts.get("scaler_path_trade") or meta.get("scaler_path", f"models/scaler_tv_{timeframe}.pkl"))
    scaler_path_dir   = os.path.join(ROOT_DIR, ts.get("scaler_path_dir")   or meta.get("scaler_path", f"models/scaler_tv_{timeframe}.pkl"))


    # === MULTI-CLASS (původní) — jen pokud není two-stage ===
    if not use_two_stage:
        used_feats = features if features is not None else wanted_feats
        _before_cols = set(gold.columns)
        gold_mc = add_indicators(gold.copy(), used_feats)
        ind_cols = [c for c in gold_mc.columns if c not in _before_cols and c != "date"]
        if ind_cols:
            print(f"[INFO] Přidané indikátory v predikci: {ind_cols}")

        # doplň feature_cols
        for col in feature_cols:
            if col not in gold_mc.columns:
                gold_mc[col] = 0.0

        X_df = gold_mc[feature_cols].copy().ffill().bfill()
        print("[DBG] SINGLE scaler:", _short(scaler_path), "| n_features_in_ =", getattr(scaler, "n_features_in_", None),
              "| X_df.shape[1] =", X_df.shape[1], flush=True)
        X_scaled = scaler.transform(X_df.values.astype(float))
        X_seqs, _ = _build_sequences(X_scaled, seq_len)
        if X_seqs.shape[0] == 0:
            raise ValueError("Příliš málo dat pro sekvenci – zvýš počet barů nebo sniž seq_len.")

        model = load_model(model_path)
        probs = model.predict(X_seqs, verbose=0)
        labels = np.argmax(probs, axis=1)
        strength = np.max(probs, axis=1)

        out = gold_mc.iloc[seq_len - 1:].copy().reset_index(drop=True).iloc[:len(labels)]
        out["signal"] = labels
        out["signal_strength"] = strength
        out["proba_no_trade"] = probs[:, 0]
        out["proba_buy"] = probs[:, 1]
        out["proba_sell"] = probs[:, 2]

        if min_conf > 0:
            mask_weak = out["signal_strength"] < min_conf
            out.loc[mask_weak, "signal"] = 0

        out = out[["date", "close", "signal", "signal_strength", "proba_no_trade", "proba_buy", "proba_sell"]]
        out.to_csv(out_csv, index=False)
        print(f"[OK] Predikce (multi) uložena: {out_csv}")
        return  # konec single-stage

    # === TWO-STAGE (Trade → Direction) ===

    # 0) Případné přepsání fcols podle companion JSON vedle scaleru
    scaler_trade = joblib.load(scaler_path_trade)
    scaler_dir = joblib.load(scaler_path_dir)
    nfi_trade = getattr(scaler_trade, "n_features_in_", None)
    nfi_dir = getattr(scaler_dir, "n_features_in_", None)
    fcols_trade = _maybe_override_feature_cols_from_companion(scaler_path_trade, fcols_trade, "trade")
    fcols_dir = _maybe_override_feature_cols_from_companion(scaler_path_dir, fcols_dir, "dir")

    # 1) TRADE / NO-TRADE
    gold_trade = add_indicators(gold.copy(), feats_trade)
    for col in fcols_trade:
        if col not in gold_trade.columns:
            gold_trade[col] = gold[col] if col in gold.columns else 0.0

    missing = [c for c in fcols_trade if c not in gold_trade.columns]
    extra = [c for c in gold_trade.columns if c not in fcols_trade]
    if missing:
        raise RuntimeError(f"[TRADE] Chybí featury: {missing}. Doplň je ve features.add_indicators/merge makro dat.")
    if extra:
        print("[DBG] TRADE extra cols (ignorovány):", extra[:10], flush=True)

    X_tr = gold_trade[fcols_trade].copy().ffill().bfill().values.astype(float)

    print("[DBG] scaler_path_trade =", _short(scaler_path_trade), flush=True)
    print("[DBG] scaler_trade.n_features_in_ =", nfi_trade, "| fcols_trade len =", len(fcols_trade), flush=True)
    if nfi_trade is not None and nfi_trade != len(fcols_trade):
        raise ValueError(f"[TRADE] Nesoulad: scaler očekává {nfi_trade} featur, ale fcols_trade má {len(fcols_trade)}. "
                         f"Oprav meta JSON nebo použij odpovídající scaler.")

    print("[DBG] TRADE fcols len =", len(fcols_trade), "| X_tr shape =", X_tr.shape, flush=True)
    if nfi_trade is not None and nfi_trade != X_tr.shape[1]:
        raise ValueError(f"[TRADE] Počet featur nesedí: scaler očekává {nfi_trade}, ale X_tr má {X_tr.shape[1]}. "
                         f"Zkontroluj feature_cols_trade v meta a výpočet indikátorů.")

    X_tr_scaled = scaler_trade.transform(X_tr)
    X_tr_seqs, _ = _build_sequences(X_tr_scaled, seq_len)
    model_path_trade = ts.get("model_path_trade")
    if not model_path_trade:
        raise KeyError("V metadata.two_stage chybí 'model_path_trade'.")
    model_trade = load_model(os.path.join(ROOT_DIR, model_path_trade))
    p_trade = model_trade.predict(X_tr_seqs, verbose=0).reshape(-1)
    mask_trade = (p_trade >= min_conf_trade)

    # 2) DIRECTION (BUY vs SELL) – jen tam, kde trade prošel
    gold_dir = add_indicators(gold.copy(), feats_dir)
    for col in fcols_dir:
        if col not in gold_dir.columns:
            gold_dir[col] = gold[col] if col in gold.columns else 0.0

    missing_dir = [c for c in fcols_dir if c not in gold_dir.columns]
    extra_dir = [c for c in gold_dir.columns if c not in fcols_dir]
    if missing_dir:
        raise RuntimeError(f"[DIR] Chybí featury: {missing_dir}. Doplň je ve features.add_indicators/merge makro dat.")
    if extra_dir:
        print("[DBG] DIR   extra cols (ignorovány):", extra_dir[:10], flush=True)

    X_dir = gold_dir[fcols_dir].copy().ffill().bfill().values.astype(float)

    print("[DBG] scaler_path_dir   =", _short(scaler_path_dir), flush=True)
    print("[DBG] scaler_dir.n_features_in_   =", nfi_dir, "| fcols_dir   len =", len(fcols_dir), flush=True)
    if nfi_dir is not None and nfi_dir != len(fcols_dir):
        raise ValueError(f"[DIR] Nesoulad: scaler očekává {nfi_dir} featur, ale fcols_dir má {len(fcols_dir)}. "
                         f"Oprav meta JSON nebo použij odpovídající scaler.")

    print("[DBG] DIR   fcols len =", len(fcols_dir), "| X_dir shape =", X_dir.shape, flush=True)
    if nfi_dir is not None and nfi_dir != X_dir.shape[1]:
        raise ValueError(f"[DIR] Počet featur nesedí: scaler očekává {nfi_dir}, ale X_dir má {X_dir.shape[1]}. "
                         f"Zkontroluj feature_cols_dir v meta a výpočet indikátorů.")

    X_dir_scaled = scaler_dir.transform(X_dir)
    X_dir_seqs, _ = _build_sequences(X_dir_scaled, seq_len)
    model_path_dir = ts.get("model_path_dir")
    if not model_path_dir:
        raise KeyError("V metadata.two_stage chybí 'model_path_dir'.")
    model_dir = load_model(os.path.join(ROOT_DIR, model_path_dir))
    p_buy = model_dir.predict(X_dir_seqs, verbose=0).reshape(-1)
    p_sell = 1.0 - p_buy

    # 3) Rekonstrukce výstupu
    out = gold.iloc[seq_len - 1:].copy().reset_index(drop=True)
    n = min(len(out), len(p_trade), len(p_buy))
    out = out.iloc[:n]

    out["proba_no_trade"] = 1.0 - p_trade[:n]
    out["proba_buy"] = 0.0
    out["proba_sell"] = 0.0
    out["signal"] = 0
    out["signal_strength"] = 0.0

    idx = np.where(mask_trade[:n])[0]
    if idx.size > 0:
        # směrové konfidence
        out.loc[idx, "proba_buy"] = p_buy[idx]
        out.loc[idx, "proba_sell"] = p_sell[idx]
        dir_conf = np.maximum(p_buy[idx], p_sell[idx])
        margin = np.abs(p_buy[idx] - p_sell[idx])

        ok_dir = (dir_conf >= min_conf_dir) & (margin >= min_margin_dir)
        ok_buy = ok_dir & (p_buy[idx] > p_sell[idx])
        ok_sell = ok_dir & (p_sell[idx] > p_buy[idx])

        out.loc[idx[ok_buy], "signal"] = 1
        out.loc[idx[ok_buy], "signal_strength"] = p_buy[idx][ok_buy]
        out.loc[idx[ok_sell], "signal"] = 2
        out.loc[idx[ok_sell], "signal_strength"] = p_sell[idx][ok_sell]

    # globální min_conf filtr (volitelný)
    if min_conf > 0:
        mask_weak = out["signal_strength"] < min_conf
        out.loc[mask_weak, "signal"] = 0

    out = out[["date", "close", "signal", "signal_strength", "proba_no_trade", "proba_buy", "proba_sell"]]
    out.to_csv(out_csv, index=False)
    print(f"[OK] Predikce (two-stage) uložena: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predikce LSTM nad TV daty (GOLD + makra).")
    parser.add_argument("--timeframe", choices=["5m", "1h"], default="5m")
    parser.add_argument("--output", type=str, default=None, help="Výstupní CSV (default results/predictions_<tf>.csv)")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Filtr síly signálu (0–1)")
    parser.add_argument("--features", type=str, default=None, help="Seznam indikátorů oddělených čárkou (např. rsi,macd,ema20)")
    parser.add_argument("--use_two_stage", action="store_true", help="Použít dvoustupňovou inferenci (Trade→Direction).")

    args = parser.parse_args()
    out_csv = args.output or os.path.join(RESULTS_DIR, f"predictions_{args.timeframe}.csv")
    features_list = [f.strip() for f in args.features.split(",")] if args.features else None
    predict(timeframe=args.timeframe,
            out_csv=out_csv,
            min_conf=args.min_conf,
            features=features_list,
            use_two_stage=args.use_two_stage)
