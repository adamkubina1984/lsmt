# scripts/live_runner.py
# Live runner: periodicky (podle TF) volitelně stáhne data (fetch), počká buffer,
# spustí predikci (predict_lstm_tradingview.py), a při silném signálu pípne.
# Loguje alerty do results/live_signals_<tf>.csv

import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime, timezone
import argparse
import pandas as pd
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA    = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FETCH_SCRIPT   = SCRIPTS / "fetch_tradingview_data.py"      # ← uprav, pokud máš jiný fetch
PREDICT_SCRIPT = SCRIPTS / "predict_lstm_tradingview.py"    # používá natrénovaný model

IS_WINDOWS = (os.name == "nt")

def py():
    return os.environ.get("PYTHON", sys.executable)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_fetch():
    if not FETCH_SCRIPT.exists():
        log(f"[WARN] Nenalezen fetch skript: {FETCH_SCRIPT.name}")
        return
    try:
        log(f"Spouštím fetch: {FETCH_SCRIPT.name}")
        subprocess.run([py(), str(FETCH_SCRIPT)], cwd=str(ROOT), check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        log(f"[WARN] Fetch selhal: {e}")

def latest_bar_timestamp(tf: str):
    f = DATA / f"gold_{'5m' if tf=='5m' else '1h'}.csv"
    if not f.exists():
        return None, None
    try:
        df = pd.read_csv(f, usecols=['date'])
        if df.empty:
            return None, None
        ts = pd.to_datetime(df.iloc[-1, 0], errors='coerce')
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        return ts, mtime
    except Exception:
        return None, None

def run_predict(timeframe: str, out_csv: Path):
    cmd = [py(), str(PREDICT_SCRIPT), "--timeframe", timeframe, "--output", str(out_csv)]
    log(">>> " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.stdout: log(r.stdout.strip())
    if r.stderr: log("[STDERR] " + r.stderr.strip())
    if r.returncode != 0:
        raise RuntimeError(f"predict_lstm_tradingview.py failed (code {r.returncode})")

    if not out_csv.exists():
        raise FileNotFoundError(f"Chybí výstup predikce: {out_csv}")
    df = pd.read_csv(out_csv)
    if df.empty:
        raise RuntimeError("Predikce je prázdná.")
    last = df.iloc[-1]
    ts  = pd.to_datetime(last['date'], errors='coerce')
    sig = int(last['signal'])
    conf = float(last['signal_strength'])
    close = float(last['close']) if 'close' in last else float('nan')
    proba_buy = float(last['proba_buy']) if 'proba_buy' in last else float('nan')
    proba_sell = float(last['proba_sell']) if 'proba_sell' in last else float('nan')
    proba_no_trade = float(last['proba_no_trade']) if 'proba_no_trade' in last else float('nan')
    return ts, sig, conf, close, proba_buy, proba_sell, proba_no_trade

def beep(signal_id: int):
    try:
        if IS_WINDOWS:
            import winsound
            if signal_id == 1:
                winsound.Beep(1200, 250); winsound.Beep(1500, 250)
            elif signal_id == 2:
                winsound.Beep(700, 250); winsound.Beep(500, 250)
        else:
            print("\a", end="", flush=True)
    except Exception as e:
        log(f"[WARN] Nelze přehrát zvuk: {e}")

def sleep_to_next_bar(timeframe: str, buffer_sec: int):
    now = datetime.utcnow()
    now_ts = int(now.timestamp())
    period = 300 if timeframe == "5m" else 3600
    rem = period - (now_ts % period)
    sleep_s = rem + buffer_sec
    if sleep_s < buffer_sec:
        sleep_s = buffer_sec
    log(f"Čekám {sleep_s}s do další kontroly (TF={timeframe}, buffer={buffer_sec}s)…")
    time.sleep(sleep_s)

def live_loop(timeframe="5m",
              min_conf=0.47,
              allow_short=True,
              buffer_sec=10,
              auto_fetch=False,
              out_csv=None,
              update_plot=False,
              plot_window=100,
              plot_overlay_vix=False,
              plot_overlay_dxy=False):
    log("Start Live Runner")
    out_pred = RESULTS / f"predictions_{timeframe}.csv"
    live_csv = Path(out_csv) if out_csv else (RESULTS / f"live_signals_{timeframe}.csv")
    if not live_csv.exists():
        with open(live_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([
                "date", "close", "signal", "signal_strength", "beeped",
                "proba_buy", "proba_sell", "proba_no_trade", "time_utc"
            ])

    last_emitted_ts = None

    while True:
        try:
            sleep_to_next_bar(timeframe, buffer_sec)

            if auto_fetch:
                run_fetch()

            ts, mtime = latest_bar_timestamp(timeframe)
            if ts is None:
                log("[WARN] gold CSV chybí nebo je prázdné, zkusím fetch a pokračuji…")
                run_fetch()
                time.sleep(max(3, buffer_sec))
                ts, mtime = latest_bar_timestamp(timeframe)
                if ts is None:
                    continue

            if mtime:
                elapsed = (datetime.now(timezone.utc) - mtime).total_seconds()
                if elapsed < buffer_sec:
                    extra = buffer_sec - elapsed + 0.5
                    log(f"Soubor se právě změnil, čekám ještě {extra:.1f}s…")
                    time.sleep(extra)

            pred_ts, sig, conf, close, proba_buy, proba_sell, proba_no_trade = run_predict(timeframe, out_pred)

            if last_emitted_ts is not None and pred_ts <= last_emitted_ts:
                log(f"Žádná nová predikce (poslední {last_emitted_ts}, teď {pred_ts}).")
                continue

            strong = (conf >= min_conf) and (sig in (1,2)) and (allow_short or sig == 1)
            acted = False
            if strong:
                beep(sig)
                acted = True
                human = "BUY" if sig == 1 else "SELL"
                log(f"[ALERT] {pred_ts}  {human} @ {close:.2f}  |  conf={conf:.2f}")

            with open(live_csv, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([
                    pred_ts.isoformat(), close, sig, f"{conf:.4f}", int(acted),
                    f"{proba_buy:.4f}", f"{proba_sell:.4f}", f"{proba_no_trade:.4f}",
                    datetime.utcnow().isoformat()
                ])

            if update_plot:
                try:
                    plot_cmd = [
                        py(), str(SCRIPTS / "plot_signals.py"),
                        "--input", str(out_pred),
                        "--window", str(int(plot_window)),
                    ]
                    if plot_overlay_vix:
                        plot_cmd.append("--overlay_vix")
                    if plot_overlay_dxy:
                        plot_cmd.append("--overlay_dxy")
                    subprocess.run(plot_cmd, cwd=str(ROOT), check=False, capture_output=True, text=True)
                except Exception as e:
                    log(f"[WARN] Plot update selhal: {e}")

            last_emitted_ts = pred_ts

        except KeyboardInterrupt:
            log("Stop Live Runner (CTRL+C).")
            break
        except Exception as e:
            log(f"[ERR] {e}")
            time.sleep(5)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live Runner – auto-fetch + buffer + predikce + pípnutí + online graf")
    ap.add_argument("--timeframe", choices=["5m","1h"], default="5m")
    ap.add_argument("--min_conf", type=float, default=0.47)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--buffer_sec", type=int, default=10, help="čekací doba po uzavření baru (s)")
    ap.add_argument("--auto_fetch", action="store_true", help="spustí fetch před každou iterací")
    ap.add_argument("--out_csv", type=str, default=None, help="kam ukládat live signály (CSV)")
    ap.add_argument("--update_plot", action="store_true", help="po každé nové svíčce přegeneruje graf signálů")
    ap.add_argument("--plot_window", type=int, default=100)
    ap.add_argument("--plot_overlay_vix", action="store_true")
    ap.add_argument("--plot_overlay_dxy", action="store_true")
    args = ap.parse_args()

    live_loop(
        timeframe=args.timeframe,
        min_conf=args.min_conf,
        allow_short=args.allow_short,
        buffer_sec=args.buffer_sec,
        auto_fetch=args.auto_fetch,
        out_csv=args.out_csv,
        update_plot=args.update_plot,
        plot_window=args.plot_window,
        plot_overlay_vix=args.plot_overlay_vix,
        plot_overlay_dxy=args.plot_overlay_dxy
    )
