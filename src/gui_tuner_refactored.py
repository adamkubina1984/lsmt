# src/gui_tuner.py
# Rozšířené GUI pro náš LSTM obchodní systém (Windows-friendly).
# Všechny cesty jsou relativně k rootu projektu (složka výš nad /src).

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import subprocess
import threading
import os
import signal
from pathlib import Path
import csv

# --- cesty relativně k rootu projektu ---
HERE    = Path(__file__).resolve().parent
ROOT    = HERE.parent
SCRIPTS = ROOT / "scripts"
DATA    = ROOT / "data" / "raw"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"

TRAIN_SCRIPT    = SCRIPTS / "train_lstm_tradingview.py"
PREDICT_SCRIPT  = SCRIPTS / "predict_lstm_tradingview.py"
SIM_SCRIPT      = SCRIPTS / "simulate_strategy_v2.py"
FETCH_SCRIPT    = SCRIPTS / "fetch_tradingview_data.py"
SCAN_SCRIPT     = SCRIPTS / "scan_min_conf.py"
LIVE_SCRIPT     = SCRIPTS / "live_alerts.py"

# ---------------------- Tooltip helper ----------------------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tw = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _=None):
        if self.tw: return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("Segoe UI", 9)
        )
        label.pack(ipadx=6, ipady=4)

    def _hide(self, _=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None

def create_tooltip(widget, text: str):
    ToolTip(widget, text)

# ---------------------- Spouštění příkazů -------------------
def run_cmd(cmd: list, log_func, cwd: Path = None):
    """Spustí příkaz v jiném vlákně a streamuje výstup do GUI logu (one-shot)."""
    def _runner():
        log_func(f">>> {' '.join(str(x) for x in cmd)}")
        try:
            with subprocess.Popen(
                [str(c) for c in cmd],
                cwd=str(cwd) if cwd else str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) as p:
                for line in p.stdout:
                    if line is not None:
                        log_func(line.rstrip())
                ret = p.wait()
                if ret == 0:
                    log_func("[OK] Hotovo.")
                else:
                    log_func(f"[CHYBA] Proces skončil s návratovým kódem {ret}.")
        except Exception as e:
            log_func(f"[VÝJIMKA] {e}")
    threading.Thread(target=_runner, daemon=True).start()

# ---------------------------- GUI ---------------------------
class TradingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSTM obchodní bot – GUI tuner")
        self.geometry("1060x820")

        # Live alerts (spouštíme jako samostatný proces)
        self._live_proc = None
        self._build_ui()
    
    def _build_ui(self):
        # --- horní panel s parametry ---
        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        # řádek 1
        tk.Label(top, text="Timeframe:").grid(row=0, column=0, sticky="w")
        self.var_tf = tk.StringVar(value="5m")
        opt_tf = tk.OptionMenu(top, self.var_tf, "5m", "1h")
        opt_tf.grid(row=0, column=1, padx=(5, 18))
        create_tooltip(opt_tf,
            "Výběr časového rámce dat a modelu.\n"
            "5m = intradenní, 1h = swing/kratší poziční."
        )

        tk.Label(top, text="seq_len").grid(row=0, column=2, sticky="e")
        self.var_seq = tk.IntVar(value=30)
        ent_seq = tk.Entry(top, textvariable=self.var_seq, width=6)
        ent_seq.grid(row=0, column=3, padx=5)
        create_tooltip(
            ent_seq,
            "Délka sekvence (počet posledních svíček pro LSTM).\n"
            "Doporučení: 5m → 20–60 (výchozí 30), 1h → 30–80.\n"
            "Delší sekvence = více kontextu, pomalejší trénink."
        )

        tk.Label(top, text="thr_pct %").grid(row=0, column=4, sticky="e")
        self.var_thr = tk.DoubleVar(value=0.30)
        ent_thr = tk.Entry(top, textvariable=self.var_thr, width=6)
        ent_thr.grid(row=0, column=5, padx=5)
        create_tooltip(
            ent_thr,
            "Threshold pro labely (%). Pokud budoucí změna > +thr → BUY, < −thr → SELL, jinak No-Trade.\n"
            "Doporučení: 5m → 0.20–0.50, 1h → 0.30–0.70."
        )

        tk.Label(top, text="epochs").grid(row=0, column=6, sticky="e")
        self.var_epochs = tk.IntVar(value=20)
        ent_ep = tk.Entry(top, textvariable=self.var_epochs, width=6)
        ent_ep.grid(row=0, column=7, padx=5)
        create_tooltip(
            ent_ep,
            "Počet epoch tréninku. 5m → 15–40, 1h → 20–50.\n"
            "Sleduj val_loss – když přestane klesat, nemá smysl navyšovat."
        )

        tk.Label(top, text="batch").grid(row=0, column=8, sticky="e")
        self.var_batch = tk.IntVar(value=32)
        ent_batch = tk.Entry(top, textvariable=self.var_batch, width=6)
        ent_batch.grid(row=0, column=9, padx=5)
        create_tooltip(
            ent_batch,
            "Velikost batch.\nDoporučení: 32–128 (podle RAM/CPU)."
        )

        # řádek 2
        tk.Label(top, text="cot_shift").grid(row=1, column=0, sticky="w")
        self.var_cot = tk.IntVar(value=3)
        ent_cot = tk.Entry(top, textvariable=self.var_cot, width=6)
        ent_cot.grid(row=1, column=1, padx=5)
        create_tooltip(
            ent_cot,
            "Posun COT dat v dnech (COT je k úterý, publikace pátek).\n"
            "Výchozí 3 = posunout dopředu, aby platil od publikace dál."
        )

        tk.Label(top, text="min_conf").grid(row=1, column=2, sticky="e")
        self.var_minc = tk.DoubleVar(value=0.55)
        ent_minc = tk.Entry(top, textvariable=self.var_minc, width=6)
        ent_minc.grid(row=1, column=3, padx=5)
        create_tooltip(
            ent_minc,
            "Minimální síla signálu (pravděpodobnost vítězné třídy 0–1).\n"
            "Doporučení: 5m → 0.50–0.65 (často 0.55), 1h → 0.45–0.60."
        )

        tk.Label(top, text="fee %").grid(row=1, column=4, sticky="e")
        self.var_fee = tk.DoubleVar(value=0.30)
        ent_fee = tk.Entry(top, textvariable=self.var_fee, width=6)
        ent_fee.grid(row=1, column=5, padx=5)
        create_tooltip(
            ent_fee,
            "Poplatek v procentech (za transakci). 0.30 % = 0.003 ve skriptu.\n"
            "Doporučení: 0.10–0.50 % (podle brokera a trhu)."
        )

        self.var_allow_short = tk.BooleanVar(value=True)
        chk_short = tk.Checkbutton(top, text="Povolit short (short selling)", variable=self.var_allow_short)
        chk_short.grid(row=1, column=6, padx=10, sticky="w")
        create_tooltip(
            chk_short,
            "Pokud zaškrtnuto, strategie může otevírat i short pozice (SELL signály).\n"
            "Doporučeno zapnout."
        )

        # --- řádek 3: symbol/exchange/base ---
        tk.Label(top, text="TV symbol:").grid(row=2, column=0, sticky="e")
        self.var_symbol = tk.StringVar(value="GOLD")
        tk.Entry(top, textvariable=self.var_symbol, width=10).grid(row=2, column=1, padx=5)

        tk.Label(top, text="Exchange:").grid(row=2, column=2, sticky="e")
        self.var_exchange = tk.StringVar(value="TVC")
        tk.Entry(top, textvariable=self.var_exchange, width=10).grid(row=2, column=3, padx=5)

        tk.Label(top, text="Base prefix:").grid(row=2, column=4, sticky="e")
        self.var_base = tk.StringVar(value="gold")
        tk.Entry(top, textvariable=self.var_base, width=10).grid(row=2, column=5, padx=5)

        # --- tlačítka ---
        btns = tk.Frame(self)
        btns.pack(fill="x", padx=10, pady=5)

        btn_fetch = tk.Button(btns, text="📥 Stáhnout data", width=16, command=self.on_fetch)
        btn_train = tk.Button(btns, text="🧠 Trénovat",     width=16, command=self.on_train)
        btn_pred  = tk.Button(btns, text="🔮 Predikovat",   width=16, command=self.on_predict)
        btn_sim   = tk.Button(btns, text="📊 Simulace",     width=16, command=self.on_simulate)
        btn_scan  = tk.Button(btns, text="📈 Scan min_conf",width=16, command=self.on_scan)
        btn_live_on = tk.Button(btns, text="📡 Live monitor…", width=18, command=self.on_open_live_monitor)
        btn_open_trades = tk.Button(btns, text="📄 Otevřít trade-log", width=18, command=self.on_open_trades_file)
        btn_open_png    = tk.Button(btns, text="🖼 Otevřít graf",      width=14, command=self.on_open_chart_file)
        btn_show_trades = tk.Button(btns, text="🔎 Poslední obchody",  width=16, command=self.on_show_trades)

        btn_fetch.grid(row=0, column=0, padx=5)
        btn_train.grid(row=0, column=1, padx=5)
        btn_pred.grid(row=0, column=2, padx=5)
        btn_sim.grid(row=0, column=3, padx=5)
        btn_scan.grid(row=0, column=4, padx=5)
        btn_live_on.grid(row=0, column=5, padx=5)
        btn_open_trades.grid(row=0, column=6, padx=5)
        btn_open_png.grid(row=0, column=7, padx=5)
        btn_show_trades.grid(row=0, column=8, padx=5)

        # tooltips tlačítek
        create_tooltip(btn_fetch,
            "Stáhne (nebo aktualizuje) CSV do data/raw/:\n"
            " - gold_5m/1h (z TV/Yahoo), VIX, DXY (COT z CSV).\n"
            "Spusť před tréninkem, pokud nemáš data."
        )
        create_tooltip(btn_train,
            "Natrénuje LSTM model podle parametrů výše.\n"
            "Výstup: models/lstm_tv_<tf>.h5 + scaler + features.json"
        )
        create_tooltip(btn_pred,
            "Použije natrénovaný model k predikci signálů.\n"
            "Výstup: results/predictions_<tf>.csv"
        )
        create_tooltip(btn_sim,
            "Spustí backtest nad predictions_<tf>.csv.\n"
            "Použije min_conf a fee, volitelně short.\n"
            "Může exportovat trade-log a graf."
        )
        create_tooltip(btn_scan,
            "Otestuje více hodnot min_conf a uloží CSV s metrikami (PnL, Sharpe, DD, Trades)."
        )
        create_tooltip(btn_live_on,  "Spustí akustická upozornění BUY/SELL (scripts/live_alerts.py).")
        #create_tooltip(btn_live_off, "Ukončí běžící Live Alerts proces.")
        create_tooltip(btn_open_trades, "Otevře aktuální trade-log CSV ve výchozím programu (Excel).")
        create_tooltip(btn_open_png,    "Otevře poslední PNG graf equity.")
        create_tooltip(btn_show_trades, "Zobrazí posledních N obchodů v okně.")

        # --- export panel ---
        exp = tk.LabelFrame(self, text="Export při simulaci")
        exp.pack(fill="x", padx=10, pady=6)

        self.var_export = tk.BooleanVar(value=True)
        chk_export = tk.Checkbutton(exp, text="Exportovat trade-log a graf", variable=self.var_export)
        chk_export.grid(row=0, column=0, padx=5, pady=4, sticky="w")
        create_tooltip(chk_export, "Pokud je zapnuto, simulace uloží CSV s obchody a PNG graf equity.")

        tk.Label(exp, text="trades_csv:").grid(row=0, column=1, sticky="e")
        self.var_trades_path = tk.StringVar(value=str(RESULTS / "trades_log.csv"))
        ent_trades = tk.Entry(exp, textvariable=self.var_trades_path, width=36)
        ent_trades.grid(row=0, column=2, padx=5, pady=4)
        create_tooltip(ent_trades, "Cesta k CSV s obchody (výchozí results/trades_log.csv).")

        tk.Label(exp, text="equity_png:").grid(row=0, column=3, sticky="e")
        self.var_chart_path = tk.StringVar(value=str(RESULTS / "simulation.png"))
        ent_chart = tk.Entry(exp, textvariable=self.var_chart_path, width=36)
        ent_chart.grid(row=0, column=4, padx=5, pady=4)
        create_tooltip(ent_chart, "Cesta k PNG grafu (výchozí results/simulation.png).")

        # --- dvouprahový vstup (slabé signály menší velikostí) ---
        thr2 = tk.LabelFrame(self, text="Dvouprahový vstup (volitelné)")
        thr2.pack(fill="x", padx=10, pady=6)

        self.var_use_two_thr = tk.BooleanVar(value=False)
        chk_two = tk.Checkbutton(thr2, text="Použít i slabší signály (nižší práh) s menší velikostí pozice",
                         variable=self.var_use_two_thr)
        chk_two.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=4)

        tk.Label(thr2, text="min_conf_low:").grid(row=1, column=0, sticky="e")
        self.var_min_conf_low = tk.DoubleVar(value=0.38)   # doporučený start
        tk.Entry(thr2, textvariable=self.var_min_conf_low, width=6).grid(row=1, column=1, padx=5)

        tk.Label(thr2, text="trade_pct_low:").grid(row=1, column=2, sticky="e")
        self.var_trade_pct_low = tk.DoubleVar(value=0.025) # 2.5 % kapitálu pro slabé
        tk.Entry(thr2, textvariable=self.var_trade_pct_low, width=6).grid(row=1, column=3, padx=5)

        # tooltips (volitelné, pokud už máš helper create_tooltip)
        create_tooltip(chk_two, "Zapne dvouprahovou logiku: silné signály (≥ min_conf) jedou plnou velikostí, slabé (≥ min_conf_low) jedou menší.")
        create_tooltip(self.nametowidget(thr2.grid_slaves(row=1, column=1)[0]),
                       "Nižší práh síly signálu (např. 0.38–0.40) pro částečné vstupy.")
        create_tooltip(self.nametowidget(thr2.grid_slaves(row=1, column=3)[0]),
                       "Frakce kapitálu pro částečný vstup (např. 0.02–0.03).")


        # --- scan panel (rozsahy) ---
        scan = tk.LabelFrame(self, text="Scan min_conf – rozsah")
        scan.pack(fill="x", padx=10, pady=6)

        tk.Label(scan, text="start").grid(row=0, column=0, sticky="e")
        self.var_scan_start = tk.DoubleVar(value=0.45)
        ent_s = tk.Entry(scan, textvariable=self.var_scan_start, width=6)
        ent_s.grid(row=0, column=1, padx=5)

        tk.Label(scan, text="stop").grid(row=0, column=2, sticky="e")
        self.var_scan_stop = tk.DoubleVar(value=0.70)
        ent_t = tk.Entry(scan, textvariable=self.var_scan_stop, width=6)
        ent_t.grid(row=0, column=3, padx=5)

        tk.Label(scan, text="step").grid(row=0, column=4, sticky="e")
        self.var_scan_step = tk.DoubleVar(value=0.01)
        ent_p = tk.Entry(scan, textvariable=self.var_scan_step, width=6)
        ent_p.grid(row=0, column=5, padx=5)

        create_tooltip(ent_s, "Počáteční min_conf (např. 0.45).")
        create_tooltip(ent_t, "Konečná min_conf (např. 0.70).")
        create_tooltip(ent_p, "Krok min_conf (např. 0.01).")

        # --- log okno ---
        self.txt = tk.Text(self, height=18)
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

        # jistota: vytvoř složky
        for p in [DATA, MODELS, RESULTS]:
            p.mkdir(parents=True, exist_ok=True)

    # ------------------- log helper -------------------
    def log(self, msg: str):
        msg = msg.replace("\u2713", "[OK]")  # CP1250 safe
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    # ------------------- actions ----------------------
    def on_fetch(self):
        if not FETCH_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {FETCH_SCRIPT}")
            return
        cmd = [self.py(), str(FETCH_SCRIPT),
               "--symbol", self.var_symbol.get(),
               "--exchange", self.var_exchange.get(),
               "--base", self.var_base.get()]
        run_cmd(cmd, self.log, ROOT)

    def on_train(self):
        if not TRAIN_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {TRAIN_SCRIPT}")
            return
        cmd = [
            self.py(), str(TRAIN_SCRIPT),
            "--timeframe",        self.var_tf.get(),
            "--seq_len",         f"{self.var_seq.get()}",
            "--thr_pct",         f"{self.var_thr.get():.2f}",
            "--epochs",          f"{self.var_epochs.get()}",
            "--batch_size",      f"{self.var_batch.get()}",
            "--cot_shift_days",  f"{self.var_cot.get()}",
        ]
        run_cmd(cmd, self.log, ROOT)

    def on_predict(self):
        if not PREDICT_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {PREDICT_SCRIPT}")
            return
        out_csv = RESULTS / f"predictions_{self.var_base.get()}_{self.var_tf.get()}.csv"
        cmd = [self.py(), str(PREDICT_SCRIPT),
               "--timeframe", self.var_tf.get(),
               "--output", str(out_csv)]
        run_cmd(cmd, self.log, ROOT)

    def on_simulate(self):
        if not SIM_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Chybí {SIM_SCRIPT}")
            return
        pred_csv = Path("results") / f"predictions_{self.var_base.get()}_{self.var_tf.get()}.csv" \
                   if hasattr(self, "var_base") else Path("results") / f"predictions_{self.var_tf.get()}.csv"

        cmd = [
            self.py(), str(SIM_SCRIPT),
            "--input",     str(pred_csv),
            "--min_conf",  f"{self.var_minc.get():.2f}",
            "--fee_pct",   f"{self.var_fee.get()/100.0:.6f}",
        ]
        if self.var_allow_short.get():
            cmd.append("--allow_short")

        # exporty (pokud jsou zapnuté)
        if self.var_export.get():
            cmd += ["--trades_csv", self.var_trades_path.get(),
                    "--equity_png",  self.var_chart_path.get()]

        # dvouprahová logika – přidej jen když je zaškrtnuto
        if getattr(self, "var_use_two_thr", None) and self.var_use_two_thr.get():
            cmd += ["--min_conf_low",  f"{self.var_min_conf_low.get():.3f}",
                    "--trade_pct_low", f"{self.var_trade_pct_low.get():.3f}"]

        run_cmd(cmd, self.log, ROOT)


    def on_scan(self):
        if not SCAN_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {SCAN_SCRIPT}")
            return
        pred_csv = Path("results") / f"predictions_{self.var_base.get()}_{self.var_tf.get()}.csv"
        if not (ROOT / pred_csv).exists():
            messagebox.showwarning("Upozornění", f"Nejprve vygeneruj predikce ({pred_csv}).")
            return
        cmd = [self.py(), str(SCAN_SCRIPT),
               "--input", str(pred_csv),
               "--allow_short",
               "--fee_pct", f"{self.var_fee.get()/100.0:.6f}",
               "--start",   f"{self.var_scan_start.get():.3f}",
               "--stop",    f"{self.var_scan_stop.get():.3f}",
               "--step",    f"{self.var_scan_step.get():.3f}"]
        run_cmd(cmd, self.log, ROOT)

    def on_open_trades_file(self):
        path = Path(self.var_trades_path.get())
        if not path.is_absolute(): path = ROOT / path
        if path.exists():
            try: os.startfile(str(path))
            except Exception as e: messagebox.showerror("Chyba", str(e))
        else:
            messagebox.showwarning("Upozornění", f"Soubor {path} neexistuje. Spusť nejprve simulaci s exportem.")

    def on_open_chart_file(self):
        path = Path(self.var_chart_path.get())
        if not path.is_absolute(): path = ROOT / path
        if path.exists():
            try: os.startfile(str(path))
            except Exception as e: messagebox.showerror("Chyba", str(e))
        else:
            messagebox.showwarning("Upozornění", f"Soubor {path} neexistuje. Spusť nejprve simulaci s exportem.")

    def on_show_trades(self, last_n:int = 20):
        path = Path(self.var_trades_path.get())
        if not path.is_absolute(): path = ROOT / path
        if not path.exists():
            messagebox.showwarning("Upozornění", f"Soubor {path} neexistuje. Spusť nejprve simulaci s exportem.")
            return

        rows = []
        try:
            with open(path, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            with open(path, newline='', encoding="cp1250") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        if not rows:
            messagebox.showinfo("Info", "Trade-log je prázdný.")
            return

        try: rows.sort(key=lambda r: r.get("date_close",""))
        except Exception: pass
        rows = rows[-last_n:]

        win = tk.Toplevel(self)
        win.title(f"Poslední obchody (N={len(rows)})")
        win.geometry("960x420")
        cols = ["date_open","date_close","side","qty","entry_price","exit_price","pnl"]
        tree = ttk.Treeview(win, columns=cols, show="headings", height=16)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120 if c not in ("qty","pnl") else 90, anchor="center")
        for r in rows:
            tree.insert("", "end", values=[r.get(c,"") for c in cols])
        tree.pack(fill="both", expand=True, padx=8, pady=8)

    # ---------- Live Alerts ----------
    def on_live_start(self):
        if not LIVE_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {LIVE_SCRIPT}")
            return
        if self._live_proc and self._live_proc.poll() is None:
            self.log("[INFO] Live Alerts už běží.")
            return
        cmd = [
            self.py(), str(LIVE_SCRIPT),
            "--timeframe", self.var_tf.get(),
            "--min_conf",  f"{self.var_minc.get():.2f}",
            "--buffer_sec","10",
            "--auto_fetch",
            "--update_plot",
            "--plot_window","100"
        ]
        if self.var_allow_short.get():
            cmd.append("--allow_short")
        cmd += ["--poll_sec", "10"]
        try:
            self._live_proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            threading.Thread(target=self._pump_live_output, daemon=True).start()
            self.log("[INFO] Live Alerts spuštěny.")
        except Exception as e:
            messagebox.showerror("Chyba", str(e))

    def _pump_live_output(self):
        try:
            for line in self._live_proc.stdout:
                if line is None: break
                self.log(line.rstrip())
        except Exception as e:
            self.log(f"[VÝJIMKA Live] {e}")

    def on_live_stop(self):
        if self._live_proc and self._live_proc.poll() is None:
            try:
                # na Windows stačí terminate(); na *nix SIGTERM
                self._live_proc.terminate()
                self._live_proc = None
                self.log("[INFO] Live Alerts ukončeny.")
            except Exception as e:
                messagebox.showerror("Chyba", str(e))
        else:
            self.log("[INFO] Live Alerts neběží.")

    def py(self):
        return os.environ.get("PYTHON", "python")

    def on_open_live_monitor(self):
        # spuštění samostatného okna/ procesu
        live_py = Path(ROOT) / "src" / "live_monitor.py"
        if not live_py.exists():
            messagebox.showerror("Chyba", f"Soubor {live_py} nebyl nalezen.")
            return
        cmd = [
            self.py(), str(live_py),
            "--timeframe", self.var_tf.get(),
            "--min-conf",  str(self.var_minc.get()),
        ]
        if self.var_allow_short.get():
            cmd.append("--allow-short")
        # volitelně auto fetch každým cyklem (Yahoo; TV se ti v nologin módu hází timeouty)
        cmd.append("--auto-fetch")
        # refresh intervalu můžeš přidat takhle (např. 5 s):
        # cmd += ["--refresh-sec", "5", "--buffer-sec", "10"]

        subprocess.Popen(cmd, cwd=str(ROOT))
        self.log("▶ Spuštěn Live monitor v novém okně. Zavřením okna se alerty vypnou.")

if __name__ == "__main__":
    app = TradingGUI()
    app.mainloop()
