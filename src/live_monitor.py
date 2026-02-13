# src/live_monitor.py
# Rychlý, neblokující Live Monitor s jasným doporučením BUY/SELL/NO TRADE,
# oddělené vlákno pro fetch/predict, UI průběžně renderuje bez zamrznutí.

import os, sys, time, threading, queue, subprocess, csv
from pathlib import Path
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import messagebox

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA    = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

IS_WIN = (os.name == "nt")
try:
    import winsound
except Exception:
    winsound = None

def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    for cand in ("date", "datetime", "time"):
        if cand in df.columns:
            df.rename(columns={cand: "date"}, inplace=True)
            break
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def run_script(cmd: list, cwd: Path):
    try:
        p = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        out = (p.stdout or "").strip()
        return p.returncode, out
    except Exception as e:
        return 1, str(e)

class LiveWorker(threading.Thread):
    """Background worker: does fetch/predict/read, posts results to queue."""
    def __init__(self, cfg, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = out_queue
        self._stop_evt = threading.Event()
        self._last_bar = None

    def stop(self):
        self._stop_evt.set()

    def compute_next_target(self, last_dt: datetime) -> datetime:
        if self.cfg['timeframe'] == '1h':
            base = last_dt.replace(minute=0, second=0, microsecond=0)
            return base + timedelta(hours=1, minutes=self.cfg['buffer'])
        else:
            # 5 minute bars
            minute = (last_dt.minute // 5) * 5
            base = last_dt.replace(minute=minute, second=0, microsecond=0)
            return base + timedelta(minutes=5) + timedelta(seconds=self.cfg['buffer'])

    def run(self):
        while not self._stop_evt.is_set():
            try:
                if self.cfg.get("auto_fetch"):
                    rc, out = run_script([
                        self.cfg["python"], str(self.cfg["fetch"])
                    ], Path(self.cfg["root"]))
                    if rc != 0:
                        self.q.put({"msg": f"[FETCH ERR] rc={rc} | {out[-240:] if out else 'bez vystupu'}"})
                        time.sleep(self.cfg["refresh"])
                        continue

                predict_cmd = [
                    self.cfg["python"], str(self.cfg["predict"]),
                    "--timeframe", self.cfg["timeframe"],
                    "--output", str(self.cfg["pred_csv"]),
                ]
                rc, out = run_script(predict_cmd, Path(self.cfg["root"]))
                if rc != 0:
                    self.q.put({"msg": f"[PREDICT ERR] rc={rc} | {out[-240:] if out else 'bez vystupu'}"})
                    time.sleep(self.cfg["refresh"])
                    continue

                gold = read_csv_any(Path(self.cfg["gold_csv"]))
                preds = read_csv_any(Path(self.cfg["pred_csv"]))
                if gold.empty or preds.empty:
                    self.q.put({"msg": f"[WAIT] prazdna data: gold={gold.empty} preds={preds.empty}"})
                    time.sleep(self.cfg["refresh"])
                    continue

                df = pd.merge_asof(
                    gold.sort_values("date"),
                    preds.sort_values("date")[[c for c in ["date","signal","signal_strength","proba_no_trade","proba_trade","proba_buy","proba_sell","vol_dead","atr_norm","bb_width"] if c in preds.columns]],
                    on="date",
                    direction="backward"
                ).iloc[-180:]

                # === Rozhodnutí beru přímo z predikčního CSV (signal + signal_strength) ===
                last_row = df.iloc[-1]
                last_dt = last_row.get("date")
                try:
                    sig = int(last_row.get("signal", 0))
                except Exception:
                    sig = 0
                try:
                    conf = float(last_row.get("signal_strength", 0.0))
                except Exception:
                    conf = 0.0

                # respektuj volbu short
                if sig == 2 and not self.cfg.get("allow_short", True):
                    sig = 0

                # detekce nové svíčky (pípnutí / log jen jednou)
                new_bar = (self._last_bar is None) or (last_dt != self._last_bar)
                if new_bar:
                    self._last_bar = last_dt

                payload = {
                    "df": df,
                    "new_bar": bool(new_bar),
                    "last_dt": last_dt,
                    "sig": sig,
                    "conf": conf,
                    "close": float(last_row.get("close", float('nan'))),
                    "proba_no_trade": float(last_row.get("proba_no_trade", float('nan'))),
                    "proba_trade": float(last_row.get("proba_trade", float('nan'))),
                    "proba_buy": float(last_row.get("proba_buy", float('nan'))),
                    "proba_sell": float(last_row.get("proba_sell", float('nan'))),
                    "vol_dead": int(last_row.get("vol_dead", 0)) if str(last_row.get("vol_dead", 0)).strip() != '' else 0,
                    "atr_norm": float(last_row.get("atr_norm", float('nan'))),
                    "bb_width": float(last_row.get("bb_width", float('nan'))),
                    "msg": f"Updated {last_dt} | conf={conf:.2f} | sig={sig}"
                }
                self.q.put(payload)


            except Exception as e:
                self.q.put({"msg": f"Worker error: {e}"})

            time.sleep(self.cfg["refresh"])


class LiveMonitor(tk.Tk):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.title(f"Live Monitor – TF={cfg['timeframe']} | min_conf={cfg['min_conf']:.2f} | short={'ON' if cfg['allow_short'] else 'OFF'}")
        self.geometry("1200x780")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.data_q = queue.Queue()
        self.worker = LiveWorker(cfg, self.data_q)

        # UI
        top = tk.Frame(self)
        top.pack(fill='x', padx=10, pady=6)
        self.status_lbl = tk.Label(top, text="Starting…")
        self.status_lbl.pack(side='left')

        self.action_lbl = tk.Label(top, text="WAIT", bg="#95a5a6", fg="white", font=("Segoe UI", 16, "bold"), padx=12, pady=4)
        self.action_lbl.pack(side='right')

        self.fig = matplotlib.figure.Figure(figsize=(12,7), dpi=100)
        self.ax_price = self.fig.add_subplot(2,1,1)
        self.ax_conf  = self.fig.add_subplot(2,1,2, sharex=self.ax_price)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=6)

        self._last_alert_dt = None
        self.after(500, self.poll_updates)
        self.worker.start()


    def _beep_strong(self, signal_id: int):
        if not self.cfg.get("beep", True):
            return
        try:
            if IS_WIN and winsound is not None:
                if signal_id == 1:  # BUY
                    winsound.Beep(1200, 180); winsound.Beep(1500, 180)
                elif signal_id == 2:  # SELL
                    winsound.Beep(700, 180); winsound.Beep(500, 180)
            else:
                print("", end="", flush=True)
        except Exception:
            pass

    def _append_alert_csv(self, payload: dict):
        if not self.cfg.get("log_csv", True):
            return
        out_path = Path(self.cfg.get("alerts_csv") or (RESULTS / f"live_alerts_{self.cfg['timeframe']}.csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = [
            "date", "timeframe", "signal", "signal_strength", "close",
            "proba_no_trade", "proba_trade", "proba_buy", "proba_sell", "vol_dead", "logged_utc"
        ]

        write_header = (not out_path.exists())
        try:
            with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(header)
                sig = int(payload.get("sig", 0) or 0)
                human = "BUY" if sig == 1 else ("SELL" if sig == 2 else "NO_TRADE")
                dt = payload.get("last_dt")
                w.writerow([
                    getattr(dt, "isoformat", lambda: str(dt))(),
                    self.cfg.get("timeframe"),
                    human,
                    f"{float(payload.get('conf', 0.0)):.4f}",
                    f"{float(payload.get('close', float('nan'))):.4f}",
                    f"{float(payload.get('proba_no_trade', float('nan'))):.6f}",
                    f"{float(payload.get('proba_trade', float('nan'))):.6f}",
                    f"{float(payload.get('proba_buy', float('nan'))):.6f}",
                    f"{float(payload.get('proba_sell', float('nan'))):.6f}",
                    int(payload.get('vol_dead', 0) or 0),
                    datetime.utcnow().isoformat()
                ])
        except PermissionError:
            print(f"[WARN] Nelze zapsat alert CSV (soubor je zřejmě otevřen): {out_path}")
        except Exception as e:
            print(f"[WARN] Zápis alert CSV selhal: {e}")

    def poll_updates(self):
        try:
            while True:
                payload = self.data_q.get_nowait()
                self.update_from_payload(payload)
        except queue.Empty:
            pass
        if self.worker.is_alive():
            self.after(200, self.poll_updates)
        else:
            self.status_lbl.config(text="Worker finished.")
            self.action_lbl.config(text="STOPPED", bg="#7f8c8d")

    def update_from_payload(self, payload):
        # payload keys: df, new_bar(bool), last_dt, sig, conf, next_eta_s, msg
        df       = payload.get('df')
        new_bar  = payload.get('new_bar', False)
        last_dt  = payload.get('last_dt')
        sig      = payload.get('sig', 0)
        conf     = float(payload.get('conf', 0.0))
        eta_s    = payload.get('eta_s', None)
        msg      = payload.get('msg', '')

        # status line
        if eta_s is not None:
            mm = int(eta_s) // 60; ss = int(eta_s) % 60
            self.status_lbl.config(text=f"{msg} | Next bar in {mm:02d}:{ss:02d}")
        else:
            self.status_lbl.config(text=msg)

        # action banner
        lo_thr = self.cfg.get('min_conf_low')
        strong = (conf >= self.cfg['min_conf']) and (sig in (1,2)) and (self.cfg['allow_short'] or sig == 1)
        weak   = (lo_thr is not None) and (conf >= lo_thr) and (not strong) and (sig in (1,2)) and (self.cfg['allow_short'] or sig == 1)

        # ALERT: jen u STRONG a jen při novém baru
        if new_bar and strong:
            if (self._last_alert_dt is None) or (last_dt != self._last_alert_dt):
                self._last_alert_dt = last_dt
                self._beep_strong(sig)
                self._append_alert_csv(payload)
        if new_bar:
            try:
                human = "BUY" if sig == 1 else ("SELL" if sig == 2 else "NO_TRADE")
                print(f"[LIVE] {last_dt} | {human} | conf={conf:.3f}", flush=True)
            except Exception:
                pass

        if sig==1 and strong:
            self.action_lbl.config(text=f"BUY STRONG ({conf:.2f}) | size≈{self.cfg['trade_pct']*100:.1f}%", bg="#27ae60")
        elif sig==2 and strong and self.cfg['allow_short']:
            self.action_lbl.config(text=f"SELL STRONG ({conf:.2f}) | size≈{self.cfg['trade_pct']*100:.1f}%", bg="#c0392b")
        elif sig==1 and weak:
            self.action_lbl.config(text=f"BUY WEAK ({conf:.2f}) | size≈{self.cfg['trade_pct_low']*100:.1f}%", bg="#58d68d")
        elif sig==2 and weak and self.cfg['allow_short']:
            self.action_lbl.config(text=f"SELL WEAK ({conf:.2f}) | size≈{self.cfg['trade_pct_low']*100:.1f}%", bg="#e67e73")
        else:
            self.action_lbl.config(text=f"NO TRADE", bg="#f39c12")

        # plot
        if df is None or df.empty:
            return

        t = np.arange(len(df))
        o = df['open'].values; h=df['high'].values; l=df['low'].values; c=df['close'].values
        sig_a = df['signal'].fillna(0).astype(int).values
        conf_a= df['signal_strength'].fillna(0.0).astype(float).values
        risk  = 1.0 - conf_a

        # redraw (lightweight)
        self.ax_price.clear()
        self.ax_conf.clear()

        # vstupní pole
        N = len(df)
        x = np.arange(N)
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        sig_a  = df['signal'].fillna(0).astype(int).values
        conf_a = df['signal_strength'].fillna(0.0).astype(float).values
        risk   = 1.0 - conf_a

        # 1) PODBARVENÍ ROZHODNUTÍ pod každou svíčkou ...
        xaxis_tx = self.ax_price.get_xaxis_transform()
        lo_thr = self.cfg.get('min_conf_low')
        for i in range(N):
            strong = (conf_a[i] >= self.cfg['min_conf'])
            weak   = (lo_thr is not None) and (conf_a[i] >= lo_thr) and (not strong)
            decision = 'hold'
            if strong and sig_a[i] == 1:
                decision = 'buy_strong'
            elif strong and sig_a[i] == 2 and self.cfg['allow_short']:
                decision = 'sell_strong'
            elif weak and sig_a[i] == 1:
                decision = 'buy_weak'
            elif weak and sig_a[i] == 2 and self.cfg['allow_short']:
                decision = 'sell_weak'

            color_map = {
                'buy_strong':'#2ecc71', 'sell_strong':'#e74c3c',
                'buy_weak':'#7ee2a5',  'sell_weak':'#f29b8f',
                'hold':'#f39c12'
            }
            alpha_map = {'buy_strong':0.18,'sell_strong':0.18,'buy_weak':0.12,'sell_weak':0.12,'hold':0.10}
            color = color_map[decision]
            alpha = alpha_map[decision]
            self.ax_price.add_patch(
                Rectangle((x[i]-0.5, 0.0), 1.0, 1.0,
                          transform=xaxis_tx, facecolor=color, edgecolor='none',
                          alpha=alpha, zorder=0)
            )


        # 2) Svíčky (knoty + těla)
        for i in range(N):
            ccol = 'green' if c[i] >= o[i] else 'red'
            # knot
            self.ax_price.plot([x[i], x[i]], [l[i], h[i]], color=ccol, linewidth=0.8, zorder=2)
            # tělo
            y0, y1 = min(o[i], c[i]), max(o[i], c[i])
            self.ax_price.add_patch(
                Rectangle((x[i] - 0.35, y0), 0.7, max(1e-9, y1 - y0),
                          facecolor=ccol, edgecolor=ccol, alpha=0.85, zorder=3)
            )

        # 3) Markery signálů (jen nad prahem)
        buy_idx  = (sig_a == 1) & (conf_a >= self.cfg['min_conf'])
        sell_idx = (sig_a == 2) & (conf_a >= self.cfg['min_conf'])
        h_buy = self.ax_price.scatter(x[buy_idx],  c[buy_idx],  marker='^', s=80, c='green', label='BUY', zorder=4)
        h_sell = None
        if self.cfg['allow_short']:
            h_sell = self.ax_price.scatter(x[sell_idx], c[sell_idx], marker='v', s=80, c='red',   label='SELL', zorder=4)

        # 4) Legenda pro pásy + markery
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor='#2ecc71', alpha=0.25, label='BUY zone'),
            Patch(facecolor='#e74c3c', alpha=0.25, label='SELL zone' if self.cfg['allow_short'] else '—'),
            Patch(facecolor='#f39c12', alpha=0.25, label='NO TRADE zone'),
        ]
        if h_buy is not None:
            legend_handles.append(h_buy)
        if h_sell is not None:
            legend_handles.append(h_sell)
        self.ax_price.legend(legend_handles, [h.get_label() for h in legend_handles],
                             loc='upper left', fontsize=8, framealpha=0.3)

        # 5) Spodní panel: proba_no_trade, proba_buy, proba_sell + threshold + risk
        try:
            has_probs = all(col in df.columns for col in ('proba_no_trade', 'proba_buy', 'proba_sell'))
            if has_probs:
                y_no_trade = df['proba_no_trade'].fillna(0).clip(0, 1).values
                y_buy      = df['proba_buy'].fillna(0).clip(0, 1).values
                y_sell     = df['proba_sell'].fillna(0).clip(0, 1).values
                self.ax_conf.plot(x, y_no_trade, linestyle='-', alpha=0.6, label='proba_no_trade', color='blue')
                self.ax_conf.plot(x, y_buy,      linestyle='-', alpha=0.6, label='proba_buy',      color='green')
                self.ax_conf.plot(x, y_sell,     linestyle='-', alpha=0.6, label='proba_sell',     color='red')
        except Exception as e:
            print("[VAROVÁNÍ] Problém při vykreslení pravděpodobností:", e)

        self.ax_conf.plot(x, np.full(N, self.cfg['min_conf']), color='gray', linestyle='--', label=f"min_conf {self.cfg['min_conf']:.2f}")
        self.ax_conf.plot(x, risk,   color='tab:red',  alpha=0.6, label='risk = 1 - conf')

        if self.cfg.get("min_conf_low") is not None:
            self.ax_conf.plot(x, np.full(N, self.cfg["min_conf_low"]),
                              color='gray', linestyle='--', alpha=0.5, label=f"min_conf_low {self.cfg['min_conf_low']:.2f}")

        # Titulek a vzhled
        self.ax_price.set_title(f"TF={self.cfg['timeframe']} | window={N} | last={last_dt} | conf={conf:.2f}")
        self.ax_price.grid(True, linestyle='--', alpha=0.3)
        self.ax_conf.legend(loc='upper left', fontsize=8)
        self.ax_conf.grid(True, linestyle='--', alpha=0.3)

        # X osa – řidší popisky
        step = max(1, N // 10)
        self.ax_conf.set_xticks(x[::step])
        self.ax_conf.set_xticklabels([dt.strftime('%m-%d %H:%M') for dt in df['date'].iloc[::step]], rotation=0)

        self.canvas.draw()

        # if IS_WIN and winsound and new_bar:
        #     if strong:
        #         winsound.Beep(800, 300)  # vyšší tón
        #     elif weak and self.cfg['beep_weak']:
        #         winsound.Beep(600, 200)  # nižší tón
                


    def on_close(self):
        self.worker.stop()
        self.destroy()

def main():
    import argparse

    ap = argparse.ArgumentParser(description='Live Monitor (fast, threaded)')
    ap.add_argument('--base', type=str, default='gold')
    ap.add_argument('--timeframe', choices=['5m','1h'], default='5m')
    ap.add_argument('--min-conf', dest='min_conf', type=float, default=0.55)
    ap.add_argument('--allow-short', action='store_true')
    ap.add_argument('--refresh-sec', type=float, default=5.0)
    ap.add_argument('--buffer-sec', type=int, default=10)
    ap.add_argument('--auto-fetch', action='store_true', help='Run fetch_tradingview_data.py before each bar')
    ap.add_argument("--min-conf-low", dest="min_conf_low", type=float, default=None, help="Nižší práh pro 'slabé' signály (např. 0.38).")
    ap.add_argument("--trade-pct", type=float, default=0.05)
    ap.add_argument("--trade-pct-low", type=float, default=0.025)
    ap.add_argument("--beep-weak", action="store_true", help="Pípnout i na slabé signály (krátké).")
    ap.add_argument("--no-beep", action="store_true", help="Vypnout pípání.")
    ap.add_argument("--no-log", action="store_true", help="Vypnout logování STRONG alertů do CSV.")
    ap.add_argument("--alerts-csv", type=str, default=None, help="Cesta k CSV pro STRONG alerty (default results/live_alerts_<tf>.csv)")
    args = ap.parse_args()

    base = (args.base or "gold").strip().lower()
    gold_csv = ROOT / "data" / "raw" / f"{base}_{args.timeframe}.csv"
    pred_csv = ROOT / "results" / f"predictions_{base}_{args.timeframe}.csv"

    cfg = {
        'base'       : base,
        'timeframe'  : args.timeframe,
        'min_conf'   : args.min_conf,
        'allow_short': bool(args.allow_short),
        'refresh'    : float(args.refresh_sec),
        'buffer'     : int(args.buffer_sec),
        'auto_fetch' : bool(args.auto_fetch),
        'python'     : sys.executable,
        'root'       : str(ROOT),
        'fetch'      : SCRIPTS / 'fetch_tradingview_data.py',
        'predict'    : SCRIPTS / 'predict_lstm_tradingview.py',
        'gold_csv'   : str(gold_csv),
        'pred_csv'   : str(pred_csv),
        'min_conf_low': args.min_conf_low,
        'trade_pct'  : args.trade_pct,
        'trade_pct_low': args.trade_pct_low,
        'beep_weak'  : bool(args.beep_weak),
        'beep'       : (not bool(args.no_beep)),
        'log_csv'    : (not bool(args.no_log)),
        'alerts_csv' : args.alerts_csv,
    }

    print(f"[LIVE] base={base} tf={args.timeframe} gold_csv={gold_csv} pred_csv={pred_csv}", flush=True)
    app = LiveMonitor(cfg)
    app.mainloop()

if __name__ == '__main__':
    main()
