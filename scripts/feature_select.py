# scripts/feature_select.py
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from features import add_indicators

def make_labels(df, thr_pct=0.3):
    # jednoduchý label: procentní změna budoucí close vs. aktuální close
    pct = (df["close"].shift(-1) - df["close"]) / df["close"] * 100.0
    y = np.where(pct >  thr_pct, 1, np.where(pct < -thr_pct, 2, 0))  # 1=BUY, 2=SELL, 0=NoTrade
    return pd.Series(y, index=df.index).astype(int)

ap = argparse.ArgumentParser()
ap.add_argument("--timeframe", choices=["5m","1h"], default="5m")
ap.add_argument("--features", type=str, default="rsi,macd,ema20,bb,atr")
ap.add_argument("--topk", type=int, default=12)
args = ap.parse_args()

wanted = [s.strip() for s in args.features.split(",") if s.strip()]

df = pd.read_csv(Path("data/raw")/f"gold_{args.timeframe}.csv")
df.columns = [c.lower() for c in df.columns]
df = add_indicators(df, wanted)
df["y"] = make_labels(df)

base_cols = ["open","high","low","close","volume"]
feat_cols = [c for c in df.columns if c not in base_cols + ["date","y"]]

X = df[base_cols + feat_cols].values
y = df["y"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, shuffle=False)

rf = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
rf.fit(X_tr, y_tr)

importances = rf.feature_importances_
ranks = sorted(zip(base_cols + feat_cols, importances), key=lambda x: x[1], reverse=True)
top = ranks[:args.topk]

Path("results").mkdir(parents=True, exist_ok=True)
out = Path("results")/f"feature_ranking_{args.timeframe}.csv"
pd.DataFrame(top, columns=["feature","importance"]).to_csv(out, index=False)
print(f"[OK] Uloženo {out}")
