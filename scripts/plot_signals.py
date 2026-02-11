# scripts/plot_signals.py
# Graf posledních N svíček s BUY/SELL signály a confidence panelem.
# Vyžaduje: mplfinance (pip install mplfinance)

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalizace časového sloupce
    dcols = [c for c in df.columns if c.lower() in ("date", "datetime", "time")]
    if dcols:
        df.rename(columns={dcols[0]: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, src_col: str, out_col: str) -> pd.DataFrame:
    if right.empty:
        left[out_col] = np.nan
        return left
    r = right[["date", src_col]].rename(columns={src_col: out_col}).sort_values("date")
    l = left.sort_values("date").copy()
    return pd.merge_asof(l, r, on="date", direction="backward")

def load_ohlc_and_predictions(timeframe: str, pred_csv: Path | None) -> pd.DataFrame:
    gold_path = DATA / f"gold_{'5m' if timeframe == '5m' else '1h'}.csv"
    gold = _read_csv(gold_path)
    if gold.empty:
        raise FileNotFoundError(f"Chybí OHLC: {gold_path}")
    # OHLC sanity
    for col in ("open","high","low","close"):
        if col not in gold.columns:
            raise RuntimeError(f"V {gold_path.name} chybí sloupec '{col}'.")
    # average pro případné použití (není nutné pro vykreslení)
    if "average" not in gold.columns:
        gold["average"] = (gold["high"] + gold["low"]) / 2.0

    preds_path = pred_csv if pred_csv else (ROOT / "results" / f"predictions_{timeframe}.csv")
    preds = _read_csv(preds_path)
    if preds.empty or not {"signal","signal_strength"}.issubset(preds.columns):
        raise FileNotFoundError(f"Chybí predikce: {preds_path} (nebo sloupce 'signal','signal_strength').")

    # Merge predikcí k time-close barům (backward asof)
    df = pd.merge_asof(
        gold.sort_values("date"),
        preds.sort_values("date")[["date","signal","signal_strength"]],
        on="date",
        direction="backward"
    )
    return df

def add_overlays(df: pd.DataFrame, use_vix: bool, use_dxy: bool) -> pd.DataFrame:
    out = df.copy()
    if use_vix:
        vix = _read_csv(DATA / "vix.csv")
        if not vix.empty:
            src = "vix" if "vix" in vix.columns else ("close" if "close" in vix.columns else vix.columns[-1])
            out = _merge_asof(out, vix, src, "vix")
    if use_dxy:
        dxy = _read_csv(DATA / "dxy.csv")
        if not dxy.empty:
            src = "dxy" if "dxy" in dxy.columns else ("close" if "close" in dxy.columns else dxy.columns[-1])
            out = _merge_asof(out, dxy, src, "dxy")
    # fill makra, ať se neztrácí linie
    for c in ("vix","dxy"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").ffill().bfill()
    return out

def plot_signals(
    timeframe: str = "5m",
    window: int = 100,
    min_conf: float = 0.47,
    show_vix: bool = False,
    show_dxy: bool = False,
    out_png: Path | None = None,
) -> Path:
    out_png = Path(out_png) if out_png else (RESULTS / f"signals_{timeframe}.png")
    df_all = load_ohlc_and_predictions(timeframe, pred_csv=None)
    df_all = add_overlays(df_all, show_vix, show_dxy)

    if len(df_all) < window + 5:
        raise RuntimeError(f"Málo dat pro zobrazení (rows={len(df_all)}). Sniž --window nebo aktualizuj data.")

    dff = df_all.iloc[-window:].copy()
    dff["signal"] = pd.to_numeric(dff["signal"], errors="coerce").fillna(0).astype(int)
    dff["signal_strength"] = pd.to_numeric(dff["signal_strength"], errors="coerce").fillna(0.0).astype(float)
    dff.set_index("date", inplace=True)

    # OHLC pro mplfinance
    ohlc = dff[["open","high","low","close"]]

    # Výběr signálů
    conf = dff["signal_strength"]
    sig  = dff["signal"]
    risk = 1.0 - conf
    long_mask  = (sig == 1) & (conf >= min_conf)
    short_mask = (sig == 2) & (conf >= min_conf)
    long_pts   = dff.loc[long_mask, "close"]
    short_pts  = dff.loc[short_mask, "close"]

    # Mini-statistika pro titulek
    n_long = int(long_mask.sum())
    n_short= int(short_mask.sum())
    n_above= int((conf >= min_conf).sum())
    pct_above = 100.0 * n_above / len(dff)
    avg_conf = float(conf.mean()) if len(conf) else 0.0

    # Addplots
    apds = []
    # confidence (panel 1)
    apds.append(mpf.make_addplot(conf.values, panel=1, ylabel="Confidence", color="tab:blue"))
    apds.append(mpf.make_addplot(np.full_like(conf.values, min_conf), panel=1, color="gray", linestyle="--"))
    # risk jako sekundární osa
    apds.append(mpf.make_addplot(risk.values, panel=1, color="tab:red", secondary_y=True))
    # markery
    if not long_pts.empty:
        apds.append(mpf.make_addplot(long_pts.values, type="scatter", panel=0, marker="^", markersize=80, color="green"))
    if not short_pts.empty:
        apds.append(mpf.make_addplot(short_pts.values, type="scatter", panel=0, marker="v", markersize=80, color="red"))
    # volitelné overlay VIX/DXY (normalizované 0..1) do panelu 1
    if show_vix and "vix" in dff.columns:
        v = pd.to_numeric(dff["vix"], errors="coerce")
        if v.notna().sum() > 0:
            v_norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
            apds.append(mpf.make_addplot(v_norm.values, panel=1, color="tab:orange", linestyle=":"))
    if show_dxy and "dxy" in dff.columns:
        d = pd.to_numeric(dff["dxy"], errors="coerce")
        if d.notna().sum() > 0:
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
            apds.append(mpf.make_addplot(d_norm.values, panel=1, color="tab:purple", linestyle=":"))

    # === vykreslení ===
    # styl
    mc = mpf.make_marketcolors(up="g", down="r", edge="i", wick="i", volume="in")
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle="-", gridcolor="#e0e0e0")

    # Titulek s mini-statistikou
    title = (
        f"Signals – {timeframe} | window={len(dff)} | min_conf={min_conf:.2f} | "
        f"BUY={n_long} SELL={n_short} | >=thr={pct_above:.1f}% | avg_conf={avg_conf:.2f}"
    )

    # vykreslení přímo pomocí mpf.plot() bez vlastní figure
    mpf.plot(
        ohlc,
        type="candle",
        addplot=apds,
        style=s,
        title=title,
        ylabel="Price",
        figratio=(12,7),
        figscale=1.2,
        xrotation=0,
        savefig=dict(fname=out_png, dpi=120, bbox_inches="tight"),
    )

    print(f"[OK] Graf uložen: {out_png}")
    return out_png


    # Legenda markerů
    handles, labels = [], []
    if not long_pts.empty:
        handles += [plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="green", markersize=10, linestyle="None")]
        labels  += ["BUY (conf ≥ min_conf)"]
    if not short_pts.empty:
        handles += [plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="red", markersize=10, linestyle="None")]
        labels  += ["SELL (conf ≥ min_conf)"]
    if handles:
        ax_price.legend(handles, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    print(f"[OK] Graf uložen: {out_png}")
    return out_png

def main():
    ap = argparse.ArgumentParser(description="Vykreslí posledních N svíček s BUY/SELL signály a confidence.")
    ap.add_argument("--timeframe", choices=["5m", "1h"], default="5m")
    ap.add_argument("--window", type=int, default=100, help="Kolik posledních svíček zobrazit")
    ap.add_argument("--min_conf", type=float, default=0.47, help="Prahová síla signálu pro markery")
    ap.add_argument("--show_vix", action="store_true", help="Zobrazit normalizovaný VIX v panelu confidence")
    ap.add_argument("--show_dxy", action="store_true", help="Zobrazit normalizovaný DXY v panelu confidence")
    ap.add_argument("--out_png", type=str, default=None, help="Kam uložit PNG (default results/signals_<tf>.png)")
    args = ap.parse_args()

    plot_signals(
        timeframe=args.timeframe,
        window=args.window,
        min_conf=args.min_conf,
        show_vix=args.show_vix,
        show_dxy=args.show_dxy,
        out_png=args.out_png,
    )

if __name__ == "__main__":
    main()
