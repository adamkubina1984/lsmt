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
from keras.models import load_model
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
            use_two_stage: bool = False,
            force_single: bool = False,
            use_dxy: bool = False,
            use_cot: bool = False):

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
    ts = meta.get("two_stage", {}) or {}

    # --- Two-stage: CLI má přednost, jinak auto dle metadata.two_stage.enabled ---
    req_trade = ts.get("model_path_trade") or meta.get("model_path_trade")
    req_dir   = ts.get("model_path_dir")   or meta.get("model_path_dir")
    req_scaler_trade = ts.get("scaler_path_trade") or meta.get("scaler_path_trade")
    req_scaler_dir   = ts.get("scaler_path_dir")   or meta.get("scaler_path_dir")

    auto_two_stage = bool(ts.get("enabled")) and bool(req_trade) and bool(req_dir) and bool(req_scaler_trade) and bool(req_scaler_dir)

    cli_two_stage = bool(use_two_stage)
    if force_single:
        use_two_stage = False
        print("[INFO] Single-stage FORCED (--force_single).", flush=True)
    elif (not cli_two_stage) and auto_two_stage:
        use_two_stage = True
        print("[INFO] Two-stage AUTO: metadata.two_stage.enabled = true", flush=True)
    else:
        use_two_stage = cli_two_stage

    if use_two_stage and not (req_trade and req_dir and req_scaler_trade and req_scaler_dir):
        raise ValueError("Two-stage režim je aktivní, ale chybí model/scaler cesty v metadatech (two_stage.*).")

    min_conf_trade = float(ts.get("min_conf_trade", meta.get("min_conf_trade", 0.48)))
    min_conf_dir   = float(ts.get("min_conf_dir",   meta.get("min_conf_dir",   0.55)))
    min_margin_dir = float(ts.get("min_margin_dir", meta.get("min_margin_dir", 0.05)))

    feats_trade = [f.strip().lower() for f in (ts.get("features_trade") or meta.get("features_trade") or wanted_feats)]
    feats_dir   = [f.strip().lower() for f in (ts.get("features_dir")   or meta.get("features_dir")   or wanted_feats)]

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
    use_dxy = bool(use_dxy or meta.get("use_dxy", False))
    use_cot = bool(use_cot or meta.get("use_cot", False))

    if use_dxy:
        dxy = _read_csv_safe(os.path.join(DATA_DIR, "dxy.csv"))
        if not dxy.empty:
            dxy_col = "dxy" if "dxy" in dxy.columns else ("close" if "close" in dxy.columns else dxy.columns[-1])
            gold = _merge_asof(gold, dxy, dxy_col, "dxy")
        else:
            gold["dxy"] = np.nan
    else:
        gold["dxy"] = 0.0

    if use_cot:
        cot = _read_csv_safe(os.path.join(DATA_DIR, "cot.csv"))
        if not cot.empty:
            cot_shift_days = int(meta.get("cot_shift_days", 0))
            if cot_shift_days != 0:
                cot["date"] = cot["date"] + pd.Timedelta(days=cot_shift_days)
            cot_col = next((c for c in ("cot", "net", "value", "close") if c in cot.columns), cot.columns[-1])
            gold = _merge_asof(gold, cot, cot_col, "cot")
        else:
            gold["cot"] = np.nan
    else:
        gold["cot"] = 0.0
    for c in ("vix", "dxy", "cot"):
        gold[c] = gold[c].ffill().bfill()
    # === SCALER/feature_cols (pouze pro single-stage) ===
    feature_cols = meta.get("feature_cols")
    scaler = None
    if not use_two_stage:
        scaler = joblib.load(scaler_path)
        if not feature_cols:
            raise ValueError(
                "Metadata neobsahují 'feature_cols'. "
                "Predikce je zastavena kvůli ochraně konzistence feature pořadí. "
                "Spusť znovu trénink a ulož aktuální metadata."
            )

    # === Two-stage scaler paths ===
    scaler_path_trade = os.path.join(ROOT_DIR, req_scaler_trade) if req_scaler_trade else None
    scaler_path_dir   = os.path.join(ROOT_DIR, req_scaler_dir) if req_scaler_dir else None


    # === MULTI-CLASS (původní) — jen pokud není two-stage ===
    if not use_two_stage:
        used_feats = features if features is not None else wanted_feats
        _before_cols = set(gold.columns)
        gold_mc = add_indicators(gold.copy(), used_feats)
        ind_cols = [c for c in gold_mc.columns if c not in _before_cols and c != "date"]
        if ind_cols:
            print(f"[INFO] Přidané indikátory v predikci: {ind_cols}")

        nfi = getattr(scaler, "n_features_in_", None)
        if nfi is not None and int(nfi) != int(len(feature_cols)):
            raise ValueError(
                f"Nesoulad scaleru a metadata: scaler očekává {nfi} featur, "
                f"ale metadata mají {len(feature_cols)}."
            )

        missing_cols = [col for col in feature_cols if col not in gold_mc.columns]
        if missing_cols:
            raise ValueError(
                "V datech/příznacích chybí sloupce požadované metadaty: "
                f"{missing_cols[:12]}{' ...' if len(missing_cols) > 12 else ''}. "
                "Predikce zastavena (bez tichého doplňování nul)."
            )

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
        _log_signal_distribution(out)
        if (out["signal"] != 0).sum() == 0:
            print("[WARN] Predikce obsahuje jen NO-TRADE signály.", flush=True)
        out.to_csv(out_csv, index=False)
        print(f"[OK] Predikce (multi) uložena: {out_csv}")
        return  # konec single-stage
    # === TWO-STAGE (Trade → Direction) ===

    # 0) Načti two-stage scalery
    scaler_trade = joblib.load(scaler_path_trade)
    scaler_dir   = joblib.load(scaler_path_dir)
    nfi_trade = getattr(scaler_trade, "n_features_in_", None)
    nfi_dir   = getattr(scaler_dir,   "n_features_in_", None)

    # --- feature_cols_trade/dir: beru z meta.two_stage nebo fallback z top-level meta ---
    fcols_trade = ts.get("feature_cols_trade") or meta.get("feature_cols_trade")
    fcols_dir   = ts.get("feature_cols_dir")   or meta.get("feature_cols_dir")
    if not fcols_trade or not fcols_dir:
        raise ValueError("Two-stage režim vyžaduje feature_cols_trade a feature_cols_dir (v meta.two_stage nebo na top-level metadatech).")

    # 1) Společný základ featur – indikátory počítáme JEDNOU (bez časového rozjetí)
    union_feats = []
    for f in (feats_trade + feats_dir):
        f = (f or "").strip().lower()
        if f and f not in union_feats:
            union_feats.append(f)

    # Volatility gate potřebuje atr + bb (když chybí v union, doplním)
    for f in ("atr", "bb"):
        if f not in union_feats:
            union_feats.append(f)

    gold_feat = add_indicators(gold.copy(), union_feats)

    # 2) Volatility gate: opravdu mrtvý trh -> NO-TRADE
    vol_cfg = (meta.get("vol_filter") or ts.get("vol_filter") or {})
    q_dead = float(vol_cfg.get("quantile_dead", 0.05))
    thr_atr = vol_cfg.get("min_atr_norm")
    thr_bb  = vol_cfg.get("min_bb_width")
    if ("atr_norm" in gold_feat.columns) and ("bb_width" in gold_feat.columns):
        thr_atr = float(thr_atr) if thr_atr is not None else float(gold_feat["atr_norm"].quantile(q_dead))
        thr_bb  = float(thr_bb)  if thr_bb  is not None else float(gold_feat["bb_width"].quantile(q_dead))
        gold_feat["vol_dead"] = ((gold_feat["atr_norm"] < thr_atr) & (gold_feat["bb_width"] < thr_bb)).astype(int)
    else:
        gold_feat["vol_dead"] = 0

    # 3) Připrav featury pro trade a direction (bez tichého doplňování)
    miss_trade = [c for c in fcols_trade if c not in gold_feat.columns]
    miss_dir = [c for c in fcols_dir if c not in gold_feat.columns]
    if miss_trade:
        raise ValueError(
            "Two-stage TRADE: chybí požadované feature sloupce: "
            f"{miss_trade[:12]}{' ...' if len(miss_trade) > 12 else ''}."
        )
    if miss_dir:
        raise ValueError(
            "Two-stage DIR: chybí požadované feature sloupce: "
            f"{miss_dir[:12]}{' ...' if len(miss_dir) > 12 else ''}."
        )

    if nfi_trade is not None and nfi_trade != len(fcols_trade):
        raise ValueError(f"[TRADE] Nesoulad: scaler očekává {nfi_trade} featur, ale feature_cols_trade má {len(fcols_trade)}.")
    if nfi_dir is not None and nfi_dir != len(fcols_dir):
        raise ValueError(f"[DIR] Nesoulad: scaler očekává {nfi_dir} featur, ale feature_cols_dir má {len(fcols_dir)}.")

    X_tr = gold_feat[fcols_trade].copy().ffill().bfill().values.astype(float)
    X_dir = gold_feat[fcols_dir].copy().ffill().bfill().values.astype(float)

    X_tr_scaled = scaler_trade.transform(X_tr)
    X_dir_scaled = scaler_dir.transform(X_dir)

    X_tr_seqs, _ = _build_sequences(X_tr_scaled, seq_len)
    X_dir_seqs, _ = _build_sequences(X_dir_scaled, seq_len)
    if X_tr_seqs.shape[0] == 0 or X_dir_seqs.shape[0] == 0:
        raise ValueError("Příliš málo dat pro sekvenci – zvýš počet barů nebo sniž seq_len.")

    # 4) Predikce
    model_trade = load_model(os.path.join(ROOT_DIR, req_trade))
    model_dir   = load_model(os.path.join(ROOT_DIR, req_dir))

    p_trade = model_trade.predict(X_tr_seqs, verbose=0).reshape(-1)
    p_buy   = model_dir.predict(X_dir_seqs, verbose=0).reshape(-1)
    p_sell  = 1.0 - p_buy

    # 5) Rekonstrukce výstupu – 1 řádek na uzavřený bar (end of seq)
    out = gold_feat.iloc[seq_len - 1:].copy().reset_index(drop=True)
    n = min(len(out), len(p_trade), len(p_buy))
    out = out.iloc[:n]

    out["proba_trade"] = p_trade[:n]
    out["proba_no_trade"] = 1.0 - p_trade[:n]
    out["proba_buy"] = 0.0
    out["proba_sell"] = 0.0
    out["signal"] = 0

    mask_trade = (p_trade[:n] >= min_conf_trade)
    idx = np.where(mask_trade)[0]
    if idx.size > 0:
        out.loc[idx, "proba_buy"]  = p_buy[idx]
        out.loc[idx, "proba_sell"] = p_sell[idx]

        dir_conf = np.maximum(p_buy[idx], p_sell[idx])
        margin = np.abs(p_buy[idx] - p_sell[idx])
        ok_dir = (dir_conf >= min_conf_dir) & (margin >= min_margin_dir)

        ok_buy  = ok_dir & (p_buy[idx] > p_sell[idx])
        ok_sell = ok_dir & (p_sell[idx] > p_buy[idx])

        out.loc[idx[ok_buy],  "signal"] = 1
        out.loc[idx[ok_sell], "signal"] = 2

    # 6) Volatility gate: mrtvý trh -> NO-TRADE
    if "vol_dead" in out.columns:
        dead = out["vol_dead"].astype(bool).values
        if dead.any():
            out.loc[dead, "signal"] = 0
            out.loc[dead, "proba_trade"] = 0.0
            out.loc[dead, "proba_no_trade"] = 1.0
            out.loc[dead, "proba_buy"] = 0.0
            out.loc[dead, "proba_sell"] = 0.0

    # 7) signal_strength = konfidence vybraného rozhodnutí
    out["signal_strength"] = np.where(
        out["signal"] == 1, out["proba_buy"],
        np.where(out["signal"] == 2, out["proba_sell"], out["proba_no_trade"])
    )

    # globální min_conf filtr (jen na BUY/SELL)
    if min_conf > 0:
        mask_weak = out["signal"].isin([1,2]) & (out["signal_strength"] < min_conf)
        out.loc[mask_weak, "signal"] = 0
        out.loc[mask_weak, "signal_strength"] = out.loc[mask_weak, "proba_no_trade"]

    # výstupní sloupce
    cols = ["date", "close", "signal", "signal_strength", "proba_no_trade", "proba_trade", "proba_buy", "proba_sell", "vol_dead"]
    for c in ("atr_norm", "bb_width"):
        if c in out.columns and c not in cols:
            cols.append(c)

    out = out[[c for c in cols if c in out.columns]]
    _log_signal_distribution(out)
    if (out["signal"] != 0).sum() == 0:
        print("[WARN] Predikce obsahuje jen NO-TRADE signály.", flush=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Predikce (two-stage) uložena: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predikce LSTM nad TV daty (GOLD + makra).")
    parser.add_argument("--timeframe", choices=["5m", "1h"], default="5m")
    parser.add_argument("--output", type=str, default=None, help="Výstupní CSV (default results/predictions_<tf>.csv)")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Filtr síly signálu (0–1)")
    parser.add_argument("--features", type=str, default=None, help="Seznam indikátorů oddělených čárkou (např. rsi,macd,ema20)")
    parser.add_argument("--use_two_stage", action="store_true", help="Použít dvoustupňovou inferenci (Trade→Direction).")
    parser.add_argument("--force_single", action="store_true", help="Vynutit single-stage inferenci (ignoruje auto two-stage z metadata).")
    parser.add_argument("--use_dxy", action="store_true", help="Zapnout DXY feature i bez metadata flagu.")
    parser.add_argument("--use_cot", action="store_true", help="Zapnout COT feature i bez metadata flagu.")

    args = parser.parse_args()
    out_csv = args.output or os.path.join(RESULTS_DIR, f"predictions_{args.timeframe}.csv")
    features_list = [f.strip() for f in args.features.split(",")] if args.features else None
    predict(timeframe=args.timeframe,
            out_csv=out_csv,
            min_conf=args.min_conf,
            features=features_list,
            use_two_stage=args.use_two_stage,
            force_single=args.force_single,
            use_dxy=args.use_dxy,
            use_cot=args.use_cot)
