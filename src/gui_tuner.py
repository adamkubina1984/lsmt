# Refaktoring původního GUI do záložek (Notebook) – rozložení přesně dle dohody
# Mění se pouze vizuální uspořádání. Všechny callbacky a vazby na skripty zůstávají.
import json
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import subprocess
import threading
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
from pathlib import Path

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
AUTOSEL_SCRIPT  = SCRIPTS / "select_best_indicators.py"

# --- Indikátory pro GUI (labels) a jejich tooltipy ---
INDICATOR_LIST = [
    "RSI", "MACD", "EMA20", "EMA50", "EMA100", "Bollinger", "ATR",
    "Stochastic", "ADX", "MFI", "Parabolic SAR", "CCI", "Keltner", "ROC", "VWAP"
]

INDICATOR_TOOLTIPS = {
    "RSI": "Relative Strength Index – překoupenost/přeprodanost (0–100).",
    "MACD": "Moving Average Convergence Divergence – momentum/trend.",
    "EMA20": "Exponenciální klouzavý průměr (20) – rychlý trend.",
    "EMA50": "Exponenciální klouzavý průměr (50) – střední trend.",
    "EMA100": "Exponenciální klouzavý průměr (100) – pomalý/hladký trend.",
    "Bollinger": "Pásma 2×σ kolem MA – volatilita a extrémy.",
    "ATR": "Average True Range – aktuální volatilita.",
    "Stochastic": "%K/%D – oscilátor rozsahu (překoupeno/přeprodáno).",
    "ADX": "Síla trendu (bez směru).",
    "MFI": "Money Flow Index – objemově vážený RSI.",
    "Parabolic SAR": "Stop And Reverse – body potenciálního obratu.",
    "CCI": "Odchylka ceny od typické hodnoty (mean deviation).",
    "Keltner": "Kanály podle EMA a ATR – alternativa k BB.",
    "ROC": "Rate of Change – procentní změna ceny.",
    "VWAP": "Volume Weighted Average Price – objemově vážená cena."
}

ROOT_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

def _meta_path(tf: str) -> Path:
    return MODELS_DIR / f"features_tv_{tf}.json"

def _load_json_robust(path: Path) -> dict:
    raw = path.read_bytes()
    txt = raw.decode("utf-8", errors="ignore").lstrip("\ufeff").strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        start = txt.find("{")
        if start == -1:
            raise
        depth = 0; end = -1
        for i,ch in enumerate(txt[start:], start=start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i+1; break
        if end != -1:
            return json.loads(txt[start:end])
        raise

def _load_meta(tf: str) -> dict:
    p = _meta_path(tf)
    if not p.exists():
        raise FileNotFoundError(f"Chybí metadata: {p}")
    return _load_json_robust(p)

def _save_meta(tf: str, meta: dict) -> None:
    p = _meta_path(tf)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _decide_two_stage(tf: str, force_mode: str = "auto") -> tuple[bool, str|None]:
    """
    force_mode: 'auto' | 'two' | 'single'
    Vrací (use_two_stage, error_message).
    """
    meta = _load_meta(tf)
    ts = meta.get("two_stage", {})
    if force_mode == "single":
        return False, None
    if force_mode == "two":
        missing = []
        for k in ("model_path_trade", "model_path_dir"):
            v = ts.get(k)
            if not v: missing.append(k)
            else:
                if not (ROOT_DIR / v).exists():
                    missing.append(f"{k} (soubor nenalezen: {v})")
        if missing:
            return False, "Two-stage není připraveno:\n- " + "\n- ".join(missing)
        return True, None
    # auto
    use = bool(ts.get("enabled") and ts.get("model_path_trade") and ts.get("model_path_dir"))
    return use, None


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
        self.geometry("1120x860")

        # Live alerts (spouštíme jako samostatný proces)
        self._live_proc = None
        self._build_ui()

    def _load_two_stage_thresholds_from_meta(self):
        try:
            tf = self.var_tf.get()
            meta_path = ROOT / "models" / f"features_tv_{tf}.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ts = meta.get("two_stage", {})
            self.var_use_two_stage.set(bool(ts.get("enabled", False)))
            self.var_ts_mct.set(float(ts.get("min_conf_trade", 0.48)))
            self.var_ts_mcd.set(float(ts.get("min_conf_dir",   0.55)))
            self.var_ts_mmd.set(float(ts.get("min_margin_dir", 0.05)))
            self.log(f"[INFO] Two-stage prahy načteny z {meta_path.name}")
        except FileNotFoundError:
            self.log("[VAROVÁNÍ] Soubor s metadaty nebyl nalezen.")
        except Exception as e:
            self.log(f"[CHYBA] Načtení two-stage parametrů selhalo: {e}")

    def _save_two_stage_settings_to_meta(self):
        try:
            tf = self.var_tf.get()
            meta_path = ROOT / "models" / f"features_tv_{tf}.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            ts = meta.get("two_stage", {})
            ts["enabled"]        = bool(self.var_use_two_stage.get())
            ts["min_conf_trade"] = float(self.var_ts_mct.get())
            ts["min_conf_dir"]   = float(self.var_ts_mcd.get())
            ts["min_margin_dir"] = float(self.var_ts_mmd.get())
            meta["two_stage"] = ts
            # zdroj indikátorů (pro přehled do meta)
            meta["selected_features_source"] = self.var_feat_source.get()
            # volitelně ulož i ruční sadu
            if self.var_feat_source.get() == "manual":
                manual = [k for k,v in self.indicator_vars.items() if v.get()]
                meta["features_manual"] = manual
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self.log(f"[OK] Two-stage prahy uloženy do {meta_path.name}")
        except Exception as e:
            self.log(f"[CHYBA] Uložení two-stage parametrů selhalo: {e}")

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)

        # -------------------- ZÁLOŽKA 1: PŘÍPRAVA A ANALÝZA --------------------
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text='Příprava a analýza')

        # Grid layout – 3 sloupce nahoře + log přes celou šířku dole
        tab1.grid_columnconfigure(0, weight=1, uniform="col")
        tab1.grid_columnconfigure(1, weight=2, uniform="col")
        tab1.grid_columnconfigure(2, weight=1, uniform="col")
        tab1.grid_rowconfigure(0, weight=1)

        # 1) Sběr dat (vlevo)
        frame_data = tk.LabelFrame(tab1, text="Sběr dat", padx=10, pady=10)
        frame_data.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        btn_tv = tk.Button(frame_data, text="📥 Stáhnout data (GOLD + VIX)", width=26, command=self.on_fetch)
        btn_tv.pack(fill='x', pady=6)
        create_tooltip(btn_tv, "Stáhne/aktualizuje GOLD a VIX. DXY/COT jsou výchozí režim OFF.")

        row3 = tk.Frame(frame_data)
        row3.pack(fill='x', pady=(6,2))
        tk.Label(row3, text="Base prefix:").pack(side='left')
        self.var_base = tk.StringVar(value="gold")
        tk.Entry(row3, textvariable=self.var_base, width=10).pack(side='left', padx=6)

        # 2) Trénink a predikce (uprostřed)
        frame_train = tk.LabelFrame(tab1, text="Trénink a predikce", padx=12, pady=10)
        frame_train.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        frame_train.grid_columnconfigure(0, weight=1)

        # ────────────────────────────────────────────────────────────
        # Společné: Timeframe + přepínač zdroje indikátorů
        tf_row = tk.Frame(frame_train)
        tf_row.pack(fill='x', pady=(0, 6))
        tk.Label(tf_row, text="Timeframe:").pack(side='left')
        self.var_tf = tk.StringVar(value="5m")
        cb_tf = ttk.Combobox(tf_row, textvariable=self.var_tf, values=["5m", "1h"], state='readonly', width=8)
        cb_tf.pack(side='left', padx=6)
        create_tooltip(cb_tf, "Časový rámec dat a modelu.\n5m = intradenní, 1h = hodinový.")

        self.var_feat_source = tk.StringVar(value="manual")
        mode_row = tk.Frame(frame_train)
        mode_row.pack(fill='x')
        tk.Label(mode_row, text="Výběr indikátorů:").pack(side='left')
        tk.Radiobutton(mode_row, text="Ruční", variable=self.var_feat_source, value="manual",
                       command=self.update_feature_checkboxes).pack(side='left', padx=(4, 0))
        tk.Radiobutton(mode_row, text="Automatický", variable=self.var_feat_source, value="auto",
                       command=self.update_feature_checkboxes).pack(side='left', padx=(4, 0))
        create_tooltip(mode_row, "Ruční: checkboxy přejdou do --features.\nAutomatický: použije se sada z models/features_tv_<TF>.json.")

        # ────────────────────────────────────────────────────────────
        # A) TRÉNINK – pouze prvky, které ovlivňují train_lstm_tradingview.py
        train_box = tk.LabelFrame(frame_train, text="Trénink modelu (train_lstm_tradingview.py)")
        train_box.pack(fill='x', pady=(8, 6))

        row1 = tk.Frame(train_box); row1.pack(fill='x', pady=2)
        tk.Label(row1, text="seq_len").pack(side='left')
        self.var_seq = tk.IntVar(value=30)
        e_seq = tk.Entry(row1, textvariable=self.var_seq, width=6); e_seq.pack(side='left', padx=(4, 12))
        create_tooltip(e_seq, "Délka sekvence (počet posledních svíček pro LSTM).")

        tk.Label(row1, text="thr_pct %").pack(side='left')
        self.var_thr = tk.DoubleVar(value=0.30)
        e_thr = tk.Entry(row1, textvariable=self.var_thr, width=6); e_thr.pack(side='left', padx=(4, 12))
        create_tooltip(e_thr, "Tvorba labelů: > +thr ⇒ BUY, < −thr ⇒ SELL, jinak No-Trade.")

        tk.Label(row1, text="epochs").pack(side='left')
        self.var_epochs = tk.IntVar(value=20)
        e_ep = tk.Entry(row1, textvariable=self.var_epochs, width=6); e_ep.pack(side='left', padx=(4, 12))
        create_tooltip(e_ep, "Počet epoch. Sleduj val_loss – když nepadají, nemá smysl přidávat.")

        tk.Label(row1, text="batch").pack(side='left')
        self.var_batch = tk.IntVar(value=32)
        e_ba = tk.Entry(row1, textvariable=self.var_batch, width=6); e_ba.pack(side='left', padx=(4, 12))
        create_tooltip(e_ba, "Velikost batch. Obvykle 32–128 (podle RAM/CPU).")

        # COT je v default pipeline vypnutý; cot_shift proto v GUI schován.

        # Indikátory (ruční/auto) – sdílený panel, 3 sloupce
        tk.Label(train_box, text="Indikátory:").pack(anchor='w', pady=(8, 2))
        self.indicator_frame = tk.Frame(train_box); self.indicator_frame.pack(fill='x')
        for c in range(3): self.indicator_frame.grid_columnconfigure(c, weight=1)

        self.indicator_vars = {}
        self.indicator_checkbuttons = []
        for i, name in enumerate(INDICATOR_LIST):
            var = tk.BooleanVar(value=(name in ["RSI", "MACD", "EMA20"]))
            key = name.lower().replace(" ", "_")
            self.indicator_vars[key] = var
            cb = tk.Checkbutton(self.indicator_frame, text=name, variable=var, anchor="w")
            cb.grid(row=i//3, column=i%3, sticky='w', padx=6, pady=2)
            self.indicator_checkbuttons.append(cb)
            create_tooltip(cb, INDICATOR_TOOLTIPS.get(name, ""))
        create_tooltip(self.indicator_frame, "Zaškrtni, které indikátory zahrnout do tréninku (--features).\nV režimu Automatický se --features neposílá (vezme se z meta).")

        # Tlačítko tréninku
        btns_train = tk.Frame(train_box); btns_train.pack(fill='x', pady=(6, 2))
        tk.Button(btns_train, text="🧠 Trénovat model", width=20, command=self.on_train).pack(anchor='w', pady=2)

        # ────────────────────────────────────────────────────────────
        # B) PREDIKCE – pouze prvky, které ovlivňují predict_lstm_tradingview.py
        pred_box = tk.LabelFrame(frame_train, text="Predikce (predict_lstm_tradingview.py)")
        pred_box.pack(fill='x', pady=(8, 0))

        # Two-stage inference + prahy (pracují s metadaty)
        ts_row = tk.Frame(pred_box); ts_row.pack(fill='x', pady=(2, 2))
        self.var_use_two_stage = tk.BooleanVar(value=False)
        tk.Checkbutton(ts_row, text="Použít Two-stage inference", variable=self.var_use_two_stage).pack(side='left', padx=(0,12))
        create_tooltip(ts_row, "Dvoustupňová inference: 1) Trade/No-Trade  2) BUY/SELL.\nPři zapnutí se prahy čtou z meta.")

        # Práhy (GUI ↔ meta)
        thr_row = tk.Frame(pred_box); thr_row.pack(fill='x', pady=(2, 2))
        tk.Label(thr_row, text="min_conf_trade").pack(side='left')
        self.var_ts_mct = tk.DoubleVar(value=0.48)
        tk.Entry(thr_row, textvariable=self.var_ts_mct, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(thr_row, "Krok 1: práh pro Trade/No-Trade (sigmoid).")

        tk.Label(thr_row, text="min_conf_dir").pack(side='left')
        self.var_ts_mcd = tk.DoubleVar(value=0.55)
        tk.Entry(thr_row, textvariable=self.var_ts_mcd, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(thr_row, "Krok 2: max(p_buy, p_sell) ≥ min_conf_dir.")

        tk.Label(thr_row, text="min_margin_dir").pack(side='left')
        self.var_ts_mmd = tk.DoubleVar(value=0.05)
        tk.Entry(thr_row, textvariable=self.var_ts_mmd, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(thr_row, "Krok 2: |p_buy − p_sell| ≥ min_margin_dir (potlačí „šedé“ signály).")

        # Načíst/Uložit do meta (two-stage část)
        io_row = tk.Frame(pred_box); io_row.pack(fill='x', pady=(2, 4))
        tk.Button(io_row, text="Načíst z meta",  command=self._load_two_stage_thresholds_from_meta).pack(side='left', padx=(0, 6))
        tk.Button(io_row, text="Uložit do meta", command=self._save_two_stage_settings_to_meta).pack(side='left')
        create_tooltip(io_row, "Načte/uloží two-stage prahy a nastavení do models/features_tv_<TF>.json.")

        # Globální min_conf filtr (predikce) – nezávislý na two-stage
        mc_row = tk.Frame(pred_box); mc_row.pack(fill='x', pady=(2, 6))
        tk.Label(mc_row, text="min_conf (globální filtr)").pack(side='left')
        self.var_minc = tk.DoubleVar(value=0.55)
        tk.Entry(mc_row, textvariable=self.var_minc, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(mc_row, "Globální filtr na signal_strength\n(pod prahem se signál přepíše na No-Trade).")

        # Tlačítko predikce
        btns_pred = tk.Frame(pred_box); btns_pred.pack(fill='x', pady=(2, 0))
        tk.Button(btns_pred, text="🔮 Predikovat", width=20, command=self.on_predict).pack(anchor='w', pady=2)

        # Po sestavení UI zamkni checkboxy v AUTO módu
        self.update_feature_checkboxes()


        # --- Automatický výběr indikátorů (evoluční)
        auto_box = tk.LabelFrame(frame_train, text="Automatický výběr indikátorů (evoluce)")
        auto_box.pack(fill='x', padx=0, pady=(10, 8))

        row_auto = tk.Frame(auto_box)
        row_auto.pack(fill='x', pady=4)

        # layout do jedné řádky (grid), tlačítko před "Min obchodů/den:"
        row_auto.grid_columnconfigure(0, weight=0)
        row_auto.grid_columnconfigure(1, weight=0)
        row_auto.grid_columnconfigure(2, weight=0)
        row_auto.grid_columnconfigure(3, weight=0)
        row_auto.grid_columnconfigure(4, weight=0) 
        row_auto.grid_columnconfigure(5, weight=0)
        row_auto.grid_columnconfigure(6, weight=0)

        tk.Label(row_auto, text="Generace:").grid(row=0, column=0, sticky="w")
        self.evo_generations = tk.IntVar(value=5)
        tk.Entry(row_auto, textvariable=self.evo_generations, width=5).grid(row=0, column=1, padx=(4, 12), sticky="w")

        tk.Label(row_auto, text="Populace:").grid(row=0, column=2, sticky="w")
        self.evo_population = tk.IntVar(value=6)
        tk.Entry(row_auto, textvariable=self.evo_population, width=4).grid(row=0, column=3, padx=(4, 12), sticky="w")

        tk.Label(row_auto, text="Min obchodů/den:").grid(row=0, column=4, sticky="w")
        self.evo_min_trades = tk.IntVar(value=3)
        tk.Entry(row_auto, textvariable=self.evo_min_trades, width=5).grid(row=0, column=5, padx=(4, 12), sticky="w")

        btn_evo = tk.Button(row_auto, text="🎯  Spustit výběr", command=self.on_find_best_indicators)
        btn_evo.grid(row=0, column=6, padx=(6, 12), sticky="w")

        create_tooltip(auto_box,
            "Spustí skript select_best_indicators.py se zadanými parametry a timeframe.\n"
            "Výsledek (nejlepší sada) se uloží do models/features_tv_<TF>.json jako 'features_auto'."
            "Po doběhu přepni 'Výběr indikátorů' na Automatický a trénuj/predikuj.")


        # 3) Analýza modelu (vpravo)
        frame_eval = tk.LabelFrame(tab1, text="Analýza modelu", padx=10, pady=10)
        frame_eval.grid(row=0, column=2, sticky='nsew', padx=10, pady=10)
        for label, cmd in [
            ("📈 Scan min_conf",     self.on_scan),
            ("📄 Otevřít trade-log", self.on_open_trades_file),
            ("🖼 Otevřít graf",      self.on_open_chart_file),
            ("🔎 Poslední obchody",  self.on_show_trades),
        ]:
            tk.Button(frame_eval, text=label, width=22, command=cmd).pack(fill='x', pady=6)

        # --- scan panel (rozsahy) ---
        row4 = tk.Frame(frame_eval)
        row4.pack(fill='x', pady=(6,0))

        tk.Label(row4, text="start").grid(row=0, column=0, sticky="e")
        self.var_scan_start = tk.DoubleVar(value=0.45)
        ent_s = tk.Entry(row4, textvariable=self.var_scan_start, width=6)
        ent_s.grid(row=0, column=1, padx=5)

        tk.Label(row4, text="stop").grid(row=0, column=2, sticky="e")
        self.var_scan_stop = tk.DoubleVar(value=0.70)
        ent_t = tk.Entry(row4, textvariable=self.var_scan_stop, width=6)
        ent_t.grid(row=0, column=3, padx=5)

        tk.Label(row4, text="step").grid(row=0, column=4, sticky="e")
        self.var_scan_step = tk.DoubleVar(value=0.01)
        ent_p = tk.Entry(row4, textvariable=self.var_scan_step, width=6)
        ent_p.grid(row=0, column=5, padx=5)

        create_tooltip(ent_s, "Počáteční min_conf (např. 0.45).")
        create_tooltip(ent_t, "Konečná min_conf (např. 0.70).")
        create_tooltip(ent_p, "Krok min_conf (např. 0.01).")
       

        # LOG – přes celou šířku
        log_frame = tk.LabelFrame(tab1, text="Log")
        log_frame.grid(row=1, column=0, columnspan=3, sticky='nsew', padx=10, pady=(0,10))
        tab1.grid_rowconfigure(1, weight=1)

        # horní lišta logu: Auto-scroll + tlačítka
        log_toolbar = tk.Frame(log_frame)
        log_toolbar.grid(row=0, column=0, columnspan=3, sticky='ew', padx=6, pady=(6,2))
        log_toolbar.grid_columnconfigure(0, weight=1)
        self.var_autoscroll = tk.BooleanVar(value=True)
        tk.Checkbutton(log_toolbar, text="Auto-scroll", variable=self.var_autoscroll).pack(side='left')

        tk.Button(log_toolbar, text="Vyčistit", command=self._clear_log).pack(side='right')
        tk.Button(log_toolbar, text="Uložit…", command=self._save_log).pack(side='right', padx=(0,6))

        # Text + scrollbary
        txt_wrap = tk.Frame(log_frame)
        txt_wrap.grid(row=1, column=0, sticky='nsew', padx=6, pady=(0,6))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.txt = tk.Text(txt_wrap, height=12, wrap='none')
        self.txt.grid(row=0, column=0, sticky='nsew')

        yscroll = ttk.Scrollbar(txt_wrap, orient='vertical', command=self.txt.yview)
        yscroll.grid(row=0, column=1, sticky='ns')
        xscroll = ttk.Scrollbar(txt_wrap, orient='horizontal', command=self.txt.xview)
        xscroll.grid(row=1, column=0, sticky='ew')

        txt_wrap.grid_rowconfigure(0, weight=1)
        txt_wrap.grid_columnconfigure(0, weight=1)

        self.txt.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        # -------------------- ZÁLOŽKA 2: SIMULACE A LIVE REŽIM -----------------
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text='Simulace a Live režim')

        tab2.grid_columnconfigure(0, weight=1)
        tab2.grid_rowconfigure(0, weight=1)

        live_frame = tk.LabelFrame(tab2, text="Simulace a Live monitoring", padx=12, pady=10)
        live_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Tlačítka – vlevo nahoře
        row_btns = tk.Frame(live_frame)
        row_btns.pack(fill='x')
        tk.Button(row_btns, text="📊 Spustit simulaci",  width=20, command=self.on_simulate).pack(side='left')
        tk.Button(row_btns, text="📡 Spustit Live režim", width=20, command=self.on_open_live_monitor).pack(side='left', padx=8)

        # Riziko na obchod – vizuální, logiku neměníme
        rk = tk.Frame(live_frame)
        rk.pack(fill='x', pady=(12,4))
        tk.Label(rk, text="Riziko na obchod (%):").pack(side='left')
        self.var_risk_pct = tk.DoubleVar(value=5.0)
        tk.Entry(rk, textvariable=self.var_risk_pct, width=8).pack(side='left', padx=6)

        # === NOVÉ: ladicí parametry simulace ===
        tk.Label(live_frame, text="Pokročilé parametry simulace:").pack(anchor='w', pady=(8,2))
        adv = tk.Frame(live_frame)
        adv.pack(fill='x', pady=(0,6))

        tk.Label(adv, text="min_conf_low").pack(side='left')
        self.var_mclo = tk.DoubleVar(value=-1)
        ent_mclo = tk.Entry(adv, textvariable=self.var_mclo, width=6)
        ent_mclo.pack(side='left', padx=(4,12))

        tk.Label(adv, text="trade_pct_low").pack(side='left')
        self.var_tplo = tk.DoubleVar(value=0.0)
        ent_tplo = tk.Entry(adv, textvariable=self.var_tplo, width=6)
        ent_tplo.pack(side='left', padx=(4,12))

        tk.Label(adv, text="max_hold").pack(side='left')
        self.var_max_hold = tk.IntVar(value=0)
        ent_maxh = tk.Entry(adv, textvariable=self.var_max_hold, width=6)
        ent_maxh.pack(side='left', padx=(4,12))

        create_tooltip(adv, "Ladicí simulace:\n- min_conf_low: práh pro slabší signál\n- trade_pct_low: velikost slabší pozice\n- max_hold: časový limit držení (v barech).\nNastav -1 / 0 pro deaktivaci.")

        # Výstup Live predikcí – textové okno
        tk.Label(live_frame, text="Výstup predikcí (Live):").pack(anchor='w', pady=(10,2))
        self.live_output = tk.Text(live_frame, height=14, state='disabled')
        self.live_output.pack(fill='both', expand=True)

        # jistota: vytvoř složky
        for p in [DATA, MODELS, RESULTS]:
            p.mkdir(parents=True, exist_ok=True)

        self.update_feature_checkboxes()

    def on_find_best_indicators(self):
        """Spustí evoluční skript pro výběr indikátorů a streamuje log do GUI."""
        evo_script = SCRIPTS / "select_best_indicators.py"
        if not evo_script.exists():
            messagebox.showerror("Chyba", f"Soubor {evo_script} neexistuje.")
            return

        tf   = self.var_tf.get()
        gens = max(1, int(self.evo_generations.get()))
        pop  = max(2, int(self.evo_population.get()))
        min_tr = max(0, int(self.evo_min_trades.get()))
    
        cmd = [
            self.py(), "-u", str(evo_script),
            "--timeframe", tf,
            "--generations", str(gens),
            "--population_size", str(pop),
            "--min_trades_per_day", str(min_tr),
        ]
        self.log(f"[INFO] Spouštím evoluční výběr: TF={tf} | gen={gens} | pop={pop} | min_trades/den={min_tr}")
        run_cmd(cmd, self.log_async, ROOT)


    def log_current_auto_features(self):
        """Vypíše do logu aktuální auto výběr z models/features_tv_<tf>.json."""
        tf = self.var_tf.get()
        meta_path = ROOT / "models" / f"features_tv_{tf}.json"
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            auto_feats = meta.get("features_auto") or meta.get("features")
            if auto_feats:
                self.log(f"[INFO] Auto sada ({tf}): {', '.join(auto_feats)}")
            else:
                self.log(f"[INFO] V metadatech zatím není auto sada (features_auto/features).")
        except Exception as e:
            self.log(f"[VAROVÁNÍ] Nelze načíst metadata: {meta_path} | {e}")


    def update_feature_checkboxes(self):
        """Enable/disable checkboxy podle přepínače Ruční/Automatický."""
        is_manual = (self.var_feat_source.get() == "manual")
        state = tk.NORMAL if is_manual else tk.DISABLED
        for cb in getattr(self, "indicator_checkbuttons", []):
            try:
                cb.configure(state=state)
            except Exception:
                pass

    def get_selected_indicators(self):
        """
        Vrátí seznam indikátorů pro skripty.
        - 'auto' => načti z models/features_tv_<tf>.json -> features_auto (fallback: features)
        - 'manual' => z checkboxů; mapuje labely z GUI na slugs pro --features
        """
        # mapování názvů z GUI na slugs pro skripty (features.py)
        name_to_slug = {
            "rsi": "rsi",
            "macd": "macd",
            "ema20": "ema20",
            "ema50": "ema50",
            "ema100": "ema100",
            "bollinger": "bb",
            "atr": "atr",
            "stochastic": "stoch",
            "adx": "adx",
            "mfi": "mfi",
            "parabolic_sar": "sar",
            "cci": "cci",
            "keltner": "keltner",
            "roc": "roc",
            "vwap": "vwap",
        }

        if self.var_feat_source.get() == "auto":
            # načti z metadat
            tf = self.var_tf.get()
            meta_path = ROOT / "models" / f"features_tv_{tf}.json"
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                auto = meta.get("features_auto")  # preferuj auto
                if not auto:
                    auto = meta.get("features")  # fallback: features
                return auto or []
            except Exception as e:
                print(f"[VAROVÁNÍ] Nelze načíst {meta_path}: {e}")
                return []
        else:
            # ruční – checkboxy
            selected = []
            for gui_key, var in self.indicator_vars.items():
                if var.get():
                    slug = name_to_slug.get(gui_key, gui_key)
                    selected.append(slug)
            return selected


    # ------------------- log helper -------------------
    def log_async(self, msg: str):
        """Jediné správné asynchronní logování – žádné dvojité vkládání."""
        self.after(0, self._append_log, msg)

    def _append_log(self, msg: str):
        """Bezpečný zápis do Text v UI threadu + podmíněný autoscroll."""
        if msg is None:
            return
        if isinstance(msg, bytes):
            try:
                msg = msg.decode("utf-8", "ignore")
            except Exception:
                msg = str(msg)

        # pokud je Text zamčený, dočasně povolíme
        orig_state = self.txt.cget("state")
        if orig_state == "disabled":
            self.txt.configure(state="normal")

        # vložení řádku
        if not msg.endswith("\n"):
            msg = msg + "\n"
        self.txt.insert("end", msg)

        # podmíněný autoscroll
        if getattr(self, "var_autoscroll", None) is None or self.var_autoscroll.get():
            self.txt.see("end")          # POSUNOUT NA KONEC

        # vrátit původní stav
        if orig_state == "disabled":
            self.txt.configure(state="disabled")

    def log(self, msg: str):
        # necháme jako sync API; uvnitř používejme jednotně _append_log
        self._append_log(msg)

    # ------------------- helpery pro log ----------------------
    def _clear_log(self):
        self.txt.delete('1.0', tk.END)

    def _save_log(self):
        try:
            path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text','*.txt'),('Všechny','*.*')], initialfile='gui_log.txt')
            if not path:
                return
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.txt.get('1.0', tk.END))
        except Exception as e:
            messagebox.showerror('Chyba při ukládání logu', str(e))

    # ------------------- actions (zachováno beze změn) ----------------------
    def on_fetch(self):
        if not FETCH_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {FETCH_SCRIPT}")
            return
        # fetch_tradingview_data.py aktuálně běží bez CLI argumentů
        cmd = [self.py(), str(FETCH_SCRIPT)]
        run_cmd(cmd, self.log_async, ROOT)

    def on_train(self):
        if not TRAIN_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {TRAIN_SCRIPT}")
            return

        # Načtení hodnot z GUI
        seq   = self.var_seq.get()
        thr   = self.var_thr.get()
        ep    = self.var_epochs.get()
        batch = self.var_batch.get()
        cotsh = getattr(self, "var_cot", tk.IntVar(value=0)).get()

        # Zdroj indikátorů: auto = None (vezme se z metadat), ruční = checkboxy
        use_auto = (self.var_feat_source.get() == "auto")
        features = None if use_auto else self.get_selected_indicators()

        # Sestavení příkazu — --features posílat jen když je ruční výběr
        cmd = [
            self.py(), str(TRAIN_SCRIPT),
            "--timeframe",        self.var_tf.get(),
            "--seq_len",          str(seq),
            "--thr_pct",          f"{thr:.2f}",
            "--epochs",           str(ep),
            "--batch_size",       str(batch),
            "--cot_shift_days",   str(cotsh),
        ]
        if features:
            cmd += ["--features", ",".join(features)]

        # Spustit jako subprocess kvůli neblokujícímu GUI a streamování logu
        run_cmd(cmd, self.log_async, ROOT)


    def on_predict(self):
        if not PREDICT_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {PREDICT_SCRIPT}")
            return

        tf = self.var_tf.get()
        base = getattr(self, 'var_base', tk.StringVar(value='gold')).get()
        out_csv = RESULTS / f"predictions_{base}_{tf}.csv"

        # zda GUI chce two-stage
        use_ts = bool(getattr(self, "var_use_two_stage", None) and self.var_use_two_stage.get())

        # pokud je two-stage zapnuto, ulož prahy a ověř kompletnost modelů v metadatech
        if use_ts:
            try:
                self._save_two_stage_settings_to_meta()  # ať predikce čte aktuální prahy
            except Exception as e:
                self.log(f"[VAROVÁNÍ] Uložení two-stage prahů před predikcí selhalo: {e}")

            try:
                meta_path = ROOT / "models" / f"features_tv_{tf}.json"
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                messagebox.showerror("Two-stage", f"Nelze načíst metadata {meta_path.name}:\n{e}")
                return

            ts = meta.get("two_stage", {})
            missing = []
            for key in ("model_path_trade", "model_path_dir"):
                v = ts.get(key)
                if not v:
                    missing.append(f"{key} (není v JSONu)")
                else:
                    p = (ROOT / v)
                    if not p.exists():
                        missing.append(f"{key} (soubor nenalezen: {v})")

            if missing:
                messagebox.showerror(
                    "Two-stage není připraveno",
                    "Doplň prosím trénink two-stage modelů (Trade i Direction):\n- " + "\n- ".join(missing)
                )
                return

        # sestavení příkazu
        cmd = [
            self.py(), str(PREDICT_SCRIPT),
            "--timeframe", tf,
            "--output", str(out_csv)
        ]

        # --features: posílej jen když je zdroj indikátorů „ruční“
        features = None if (self.var_feat_source.get() == "auto") else self.get_selected_indicators()
        if features:
            cmd += ["--features", ",".join(features)]
        if use_ts:
            cmd.append("--use_two_stage")

        run_cmd(cmd, self.log_async, ROOT)



    def on_simulate(self):
        if not SIM_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Chybí {SIM_SCRIPT}")
            return
        base = getattr(self, 'var_base', tk.StringVar(value='gold')).get()
        pred_csv = Path("results") / f"predictions_{base}_{self.var_tf.get()}.csv"
        minc = getattr(self, 'var_minc', tk.DoubleVar(value=0.55)).get()
        fee = getattr(self, 'var_fee', tk.DoubleVar(value=0.30)).get()
        allow = getattr(self, 'var_allow_short', tk.BooleanVar(value=True)).get()
        mclo = getattr(self, 'var_mclo', tk.DoubleVar(value=-1)).get()
        tplo = getattr(self, 'var_tplo', tk.DoubleVar(value=0.0)).get()
        maxh = getattr(self, 'var_max_hold', tk.IntVar(value=0)).get()


        cmd = [
            self.py(), str(SIM_SCRIPT),
            "--input", str(pred_csv),
            "--min_conf", f"{minc:.2f}",
            "--fee_pct", f"{fee/100.0:.6f}"
        ]
        if mclo > 0 and tplo > 0:
            cmd += [
            "--min_conf_low", f"{mclo:.2f}",
            "--trade_pct_low", f"{tplo:.4f}"
            ]    
        if maxh > 0:
            cmd += ["--max_hold_bars", f"{maxh}"]
        if allow:
            cmd.append("--allow_short")
        run_cmd(cmd, self.log_async, ROOT)

    def on_scan(self):
        if not SCAN_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {SCAN_SCRIPT}")
            return
        base = getattr(self, 'var_base', tk.StringVar(value='gold')).get()
        pred_csv = Path("results") / f"predictions_{base}_{self.var_tf.get()}.csv"
        if not (ROOT / pred_csv).exists():
            messagebox.showwarning("Upozornění", f"Nejprve vygeneruj predikce ({pred_csv}).")
            return
        start = getattr(self, 'var_scan_start', tk.DoubleVar(value=0.45)).get()
        stop  = getattr(self, 'var_scan_stop',  tk.DoubleVar(value=0.70)).get()
        step  = getattr(self, 'var_scan_step',  tk.DoubleVar(value=0.01)).get()
        fee   = getattr(self, 'var_fee',        tk.DoubleVar(value=0.30)).get()
        out_csv = Path("results") / f"scan_min_conf_{pred_csv.stem}.csv"
        cmd = [self.py(), str(SCAN_SCRIPT),
               "--input", str(pred_csv),
               "--allow_short",
               "--fee_pct", f"{fee/100.0:.6f}",
               "--start",   f"{start:.3f}",
               "--stop",    f"{stop:.3f}",
               "--step",    f"{step:.3f}",
               "--out_csv", str(out_csv)]
        self._run_scan_with_summary(cmd, ROOT / out_csv)

    def _run_scan_with_summary(self, cmd: list, out_csv: Path):
        def _runner():
            self.log_async(f">>> {' '.join(str(x) for x in cmd)}")
            try:
                with subprocess.Popen(
                    [str(c) for c in cmd],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                ) as p:
                    for line in p.stdout:
                        if line is not None:
                            self.log_async(line.rstrip())
                    ret = p.wait()
                    if ret != 0:
                        self.log_async(f"[CHYBA] Scan skoncil s navratovym kodem {ret}.")
                        return
                self.log_async(f"[OK] Scan CSV: {out_csv}")
                self._log_scan_summary(out_csv)
            except Exception as e:
                self.log_async(f"[VYJIMKA] {e}")
        threading.Thread(target=_runner, daemon=True).start()

    def _log_scan_summary(self, out_csv: Path):
        if not out_csv.exists():
            self.log_async(f"[VAROVANI] Vysledny soubor nenalezen: {out_csv}")
            return
        try:
            with open(out_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            with open(out_csv, newline="", encoding="cp1250") as f:
                rows = list(csv.DictReader(f))

        if not rows:
            self.log_async("[INFO] Scan probehl, ale CSV je prazdne.")
            return

        def _to_float(v, default=-1e18):
            try:
                return float(v)
            except Exception:
                return default

        top_sharpe = sorted(rows, key=lambda r: _to_float(r.get("Sharpe")), reverse=True)[:5]
        top_pnl = sorted(rows, key=lambda r: _to_float(r.get("PnL_%")), reverse=True)[:5]

        self.log_async("[SCAN] TOP podle Sharpe:")
        for r in top_sharpe:
            self.log_async(
                f"min_conf={r.get('min_conf')} | PnL={r.get('PnL_%')}% | "
                f"Sharpe={r.get('Sharpe')} | MaxDD={r.get('MaxDD_%')}% | Trades={r.get('Trades')}"
            )

        self.log_async("[SCAN] TOP podle PnL_%:")
        for r in top_pnl:
            self.log_async(
                f"min_conf={r.get('min_conf')} | PnL={r.get('PnL_%')}% | "
                f"Sharpe={r.get('Sharpe')} | MaxDD={r.get('MaxDD_%')}% | Trades={r.get('Trades')}"
            )

    def on_open_trades_file(self):
        path = Path(getattr(self, 'var_trades_path', tk.StringVar(value=str(RESULTS / 'trades_log.csv'))).get())
        if not path.is_absolute(): path = ROOT / path
        if path.exists():
            try: os.startfile(str(path))
            except Exception as e: messagebox.showerror("Chyba", str(e))
        else:
            messagebox.showwarning("Upozornění", f"Soubor {path} neexistuje. Spusť nejprve simulaci.")

    def on_open_chart_file(self):
        path = Path(getattr(self, 'var_chart_path', tk.StringVar(value=str(RESULTS / 'simulation.png'))).get())
        if not path.is_absolute(): path = ROOT / path
        if path.exists():
            try: os.startfile(str(path))
            except Exception as e: messagebox.showerror("Chyba", str(e))
        else:
            messagebox.showwarning("Upozornění", f"Soubor {path} neexistuje. Spusť nejprve simulaci.")

    def on_show_trades(self, last_n:int = 20):
        path = Path(getattr(self, 'var_trades_path', tk.StringVar(value=str(RESULTS / 'trades_log.csv'))).get())
        if not path.is_absolute(): path = ROOT / path
        if not path.exists():
            messagebox.showwarning("Upozornění", f"Soubor {path} neexistuje. Spusť nejprve simulaci.")
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
        # Ponecháno z původní verze – v tomto layoutu se nepoužívá (máme on_open_live_monitor)
        if not LIVE_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {LIVE_SCRIPT}")
            return
        if self._live_proc and self._live_proc.poll() is None:
            self.log("[INFO] Live Alerts už běží.")
            return
        cmd = [
            self.py(), str(LIVE_SCRIPT),
            "--timeframe", self.var_tf.get(),
            "--min_conf",  f"{getattr(self,'var_minc', tk.DoubleVar(value=0.55)).get():.2f}",
            "--poll_sec", "10"
        ]
        if getattr(self,'var_allow_short', tk.BooleanVar(value=True)).get():
            cmd.append("--allow_short")
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
                self.log_async(line.rstrip())
        except Exception as e:
            self.log_async(f"[VÝJIMKA Live] {e}")

    def on_live_stop(self):
        if self._live_proc and self._live_proc.poll() is None:
            try:
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
        live_py = Path(ROOT) / "src" / "live_monitor.py"
        if not live_py.exists():
            messagebox.showerror("Chyba", f"Soubor {live_py} nebyl nalezen.")
            return
        cmd = [
            self.py(), str(live_py),
            "--timeframe", self.var_tf.get(),
            "--min-conf",  str(getattr(self,'var_minc', tk.DoubleVar(value=0.55)).get()),
        ]
        if getattr(self,'var_allow_short', tk.BooleanVar(value=True)).get():
            cmd.append("--allow-short")
        cmd.append("--auto-fetch")
        subprocess.Popen(cmd, cwd=str(ROOT))
        self.log("▶ Spuštěn Live monitor v novém okně. Zavřením okna se alerty vypnou.")

if __name__ == "__main__":
    app = TradingGUI()
    app.mainloop()
