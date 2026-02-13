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

def run_cmd_sequence(cmds: list[list], log_func, cwd: Path = None):
    """Spustí více příkazů postupně v 1 vlákně a streamuje výstup do GUI logu."""
    def _runner():
        for idx, cmd in enumerate(cmds, start=1):
            log_func(f">>> [{idx}/{len(cmds)}] " + " ".join(str(x) for x in cmd))
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
                        break
            except Exception as e:
                log_func(f"[VÝJIMKA] {e}")
                break
    threading.Thread(target=_runner, daemon=True).start()

# ---------------------------- GUI ---------------------------
class TradingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSTM obchodní bot – GUI tuner")
        self.geometry("1120x860")

        # Live alerts (spouštíme jako samostatný proces)
        self._live_proc = None
        self._help_win = None
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
        # Sdílené proměnné mezi záložkami (propisují se automaticky)
        self.var_minc = tk.DoubleVar(value=0.50)
        self.var_mclo = tk.DoubleVar(value=0.45)
        self.var_max_hold = tk.IntVar(value=0)
        self.var_allow_short = tk.BooleanVar(value=True)
        self.var_fee = tk.DoubleVar(value=0.30)

        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)

        # -------------------- ZÁLOŽKA 1: PŘÍPRAVA A ANALÝZA --------------------
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text='Příprava a analýza')

        # Grid layout – 3 sloupce nahoře + pomocné panely + log dole
        tab1.grid_columnconfigure(0, weight=1, uniform="col")
        tab1.grid_columnconfigure(1, weight=2, uniform="col")
        tab1.grid_columnconfigure(2, weight=1, uniform="col")
        tab1.grid_rowconfigure(0, weight=1)

        # 1) Levý sloupec: samostatné framy Sběr dat + Automatický výběr
        frame_left = tk.Frame(tab1)
        frame_left.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        frame_left.grid_columnconfigure(0, weight=1)

        frame_data = tk.LabelFrame(frame_left, text="Sběr dat", padx=10, pady=10)
        frame_data.grid(row=0, column=0, sticky='nsew')
        btn_tv = tk.Button(frame_data, text="📥 Stáhnout data (GOLD + VIX)", width=26, command=self.on_fetch)
        btn_tv.pack(fill='x', pady=6)
        create_tooltip(btn_tv, "Stáhne/aktualizuje GOLD a VIX. DXY/COT jsou výchozí režim OFF.")


        # Parametry fetch (append/overwrite + bars + volitelné DXY/COT)
        fetch_opts = tk.Frame(frame_data)
        fetch_opts.pack(fill='x', pady=(4, 2))

        tk.Label(fetch_opts, text="Fetch mode:").pack(side='left')
        self.var_fetch_mode = tk.StringVar(value="append")
        cb_fm = ttk.Combobox(fetch_opts, textvariable=self.var_fetch_mode, values=["append", "overwrite"], state="readonly", width=10)
        cb_fm.pack(side='left', padx=(6, 12))
        create_tooltip(cb_fm, "append = doplňuje jen nové svíčky (doporučeno)\noverwrite = přepíše CSV posledními N bary")

        tk.Label(fetch_opts, text="bars:").pack(side='left')
        self.var_fetch_bars = tk.IntVar(value=5000)
        e_fb = tk.Entry(fetch_opts, textvariable=self.var_fetch_bars, width=7)
        e_fb.pack(side='left', padx=(6, 12))
        create_tooltip(e_fb, "Kolik posledních barů stáhnout (TV typicky max ~5000 na dotaz).")

        flags_row = tk.Frame(frame_data)
        flags_row.pack(fill='x', pady=(2, 2))
        self.var_fetch_dxy = tk.BooleanVar(value=False)
        self.var_fetch_cot = tk.BooleanVar(value=False)
        tk.Checkbutton(flags_row, text="Stahovat DXY", variable=self.var_fetch_dxy).pack(side='left')
        tk.Checkbutton(flags_row, text="Stahovat COT", variable=self.var_fetch_cot).pack(side='left', padx=12)
        create_tooltip(flags_row, "Zapne stahování doplňkových dat (volitelné).\nDXY = denní\nCOT = týdenní")

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
        btn_help = tk.Button(tf_row, text="❓ Nápověda", command=self.on_open_help_window)
        btn_help.pack(side='right')
        create_tooltip(btn_help, "Otevře podrobnou nápovědu ke všem sekcím GUI.")

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
        create_tooltip(
            e_seq,
            "Počet posledních svíček, které model vidí najednou.\n"
            "Prakticky: 5m TF obvykle 20–80 (start 30), 1h TF obvykle 12–48.\n"
            "Vyšší hodnota = víc kontextu, ale pomalejší trénink a větší riziko přeučení."
        )

        tk.Label(row1, text="thr_pct %").pack(side='left')
        self.var_thr = tk.DoubleVar(value=0.30)
        e_thr = tk.Entry(row1, textvariable=self.var_thr, width=6); e_thr.pack(side='left', padx=(4, 12))
        create_tooltip(e_thr, "Tvorba labelů: > +thr ⇒ BUY, < −thr ⇒ SELL, jinak No-Trade.")

        tk.Label(row1, text="epochs").pack(side='left')
        self.var_epochs = tk.IntVar(value=20)
        e_ep = tk.Entry(row1, textvariable=self.var_epochs, width=6); e_ep.pack(side='left', padx=(4, 12))
        create_tooltip(e_ep, "Počet epoch. Doporučený start 15–40. Když val_loss stagnuje, další epochy už nepomáhají.")

        tk.Label(row1, text="batch").pack(side='left')
        self.var_batch = tk.IntVar(value=24)
        e_ba = tk.Entry(row1, textvariable=self.var_batch, width=6); e_ba.pack(side='left', padx=(4, 12))
        create_tooltip(
            e_ba,
            "Počet vzorků na jeden update gradientu.\n"
            "Doporučený rozsah 16–128 (start 24).\n"
            "Menší batch = pomalejší, ale někdy stabilnější; větší batch = rychlejší, ale může zhoršit generalizaci."
        )

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


        # Režim tréninku: multi vs two-stage (trade+direction)
        tm_row = tk.Frame(train_box); tm_row.pack(fill='x', pady=(6, 2))
        tk.Label(tm_row, text="Režim tréninku:").pack(side='left')
        self.var_train_mode = tk.StringVar(value="two_stage")
        rb_m = tk.Radiobutton(tm_row, text="Multi (3-class)", variable=self.var_train_mode, value="multi")
        rb_m.pack(side='left', padx=8)
        rb_ts = tk.Radiobutton(tm_row, text="Two-stage (Trade + Direction)", variable=self.var_train_mode, value="two_stage")
        rb_ts.pack(side='left', padx=8)
        create_tooltip(
            tm_row,
            "Multi: jeden 3-class model (BUY/SELL/NO-TRADE).\n"
            "Two-stage: 2 modely (Trade/NoTrade -> Buy/Sell), obvykle stabilnější v live."
        )

        # Tlačítko tréninku
        btns_train = tk.Frame(train_box); btns_train.pack(fill='x', pady=(6, 2))
        btn_train = tk.Button(btns_train, text="🧠 Trénovat model", width=20, command=self.on_train)
        btn_train.pack(anchor='w', pady=2)
        create_tooltip(
            btn_train,
            "Spustí trénink pro aktuální timeframe.\n"
            "Výstup: model + scaler + metadata v models/.\n"
            "Live režim používá modely z těchto metadat."
        )

        # ────────────────────────────────────────────────────────────
        # B) PREDIKCE – pouze prvky, které ovlivňují predict_lstm_tradingview.py
        pred_box = tk.LabelFrame(frame_train, text="Predikce (predict_lstm_tradingview.py)")
        pred_box.pack(fill='x', pady=(8, 0))

        # Two-stage inference + prahy (pracují s metadaty)
        ts_row = tk.Frame(pred_box); ts_row.pack(fill='x', pady=(2, 2))
        self.var_use_two_stage = tk.BooleanVar(value=False)
        tk.Checkbutton(ts_row, text="Použít Two-stage inference", variable=self.var_use_two_stage).pack(side='left', padx=(0,12))
        self.var_force_single = tk.BooleanVar(value=False)
        tk.Checkbutton(ts_row, text="Vynutit Single-stage", variable=self.var_force_single).pack(side='left', padx=12)
        create_tooltip(
            ts_row,
            "Použít Two-stage inference: vynutí dvoustupňovou predikci.\n"
            "Vynutit Single-stage: dočasně ignoruje metadata a jede 3-class model.\n"
            "Bez obou voleb běží AUTO podle metadata.two_stage.enabled."
        )

        # Práhy (GUI ↔ meta)
        thr_row = tk.Frame(pred_box); thr_row.pack(fill='x', pady=(2, 2))
        tk.Label(thr_row, text="min_conf_trade").pack(side='left')
        self.var_ts_mct = tk.DoubleVar(value=0.48)
        tk.Entry(thr_row, textvariable=self.var_ts_mct, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(thr_row, "Krok 1 (Trade/No-Trade): obvykle 0.45–0.60. Vyšší = méně obchodů.")

        tk.Label(thr_row, text="min_conf_dir").pack(side='left')
        self.var_ts_mcd = tk.DoubleVar(value=0.55)
        tk.Entry(thr_row, textvariable=self.var_ts_mcd, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(thr_row, "Krok 2 (směr): max(p_buy, p_sell) >= min_conf_dir. Obvykle 0.50–0.65.")

        tk.Label(thr_row, text="min_margin_dir").pack(side='left')
        self.var_ts_mmd = tk.DoubleVar(value=0.05)
        tk.Entry(thr_row, textvariable=self.var_ts_mmd, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(thr_row, "Krok 2 (jistota směru): |p_buy - p_sell| >= min_margin_dir. Obvykle 0.03–0.12.")

        # Načíst/Uložit do meta (two-stage část)
        io_row = tk.Frame(pred_box); io_row.pack(fill='x', pady=(2, 4))
        tk.Button(io_row, text="Načíst z meta",  command=self._load_two_stage_thresholds_from_meta).pack(side='left', padx=(0, 6))
        tk.Button(io_row, text="Uložit do meta", command=self._save_two_stage_settings_to_meta).pack(side='left')
        create_tooltip(io_row, "Načte/uloží two-stage prahy a nastavení do models/features_tv_<TF>.json.")

        # Globální min_conf filtr (predikce) – nezávislý na two-stage
        mc_row = tk.Frame(pred_box); mc_row.pack(fill='x', pady=(2, 6))
        tk.Label(mc_row, text="min_conf (globální filtr)").pack(side='left')
        tk.Entry(mc_row, textvariable=self.var_minc, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(
            mc_row,
            "Globální filtr výsledné síly signálu.\n"
            "Prakticky: 0.45–0.60. Vyšší hodnota výrazně sníží počet obchodů."
        )

        # Sdílené pokročilé parametry (propisují se do záložky Simulace a Live režim)
        adv_pred = tk.Frame(pred_box); adv_pred.pack(fill='x', pady=(0, 6))
        tk.Label(adv_pred, text="min_conf_low").pack(side='left')
        tk.Entry(adv_pred, textvariable=self.var_mclo, width=6).pack(side='left', padx=(4, 12))
        tk.Label(adv_pred, text="trade_pct_low").pack(side='left')
        self.var_trade_pct_low_info = tk.StringVar(value="auto")
        tk.Label(adv_pred, textvariable=self.var_trade_pct_low_info, width=14, anchor='w').pack(side='left', padx=(4, 12))
        tk.Label(adv_pred, text="max_hold").pack(side='left')
        tk.Entry(adv_pred, textvariable=self.var_max_hold, width=6).pack(side='left', padx=(4, 12))
        create_tooltip(
            adv_pred,
            "Pokročilé parametry simulace/live (sdílené mezi záložkami).\n"
            "Výchozí start:\n"
            "min_conf=0.50, min_conf_low=0.45, max_hold=0 (vypnuto).\n"
            "trade_pct_low se počítá automaticky z Riziko na obchod (%).\n"
            "Pravidlo: min_conf_low musí být menší než min_conf."
        )

        # Tlačítko predikce
        btns_pred = tk.Frame(pred_box); btns_pred.pack(fill='x', pady=(2, 0))
        btn_predict = tk.Button(btns_pred, text="🔮 Predikovat", width=20, command=self.on_predict)
        btn_predict.pack(anchor='w', pady=2)
        create_tooltip(
            btn_predict,
            "Vytvoří predictions_<base>_<TF>.csv podle aktuálního nastavení režimu (auto/two/single).\n"
            "Stejný predikční skript používá i Live monitor."
        )

        # Po sestavení UI zamkni checkboxy v AUTO módu
        self.update_feature_checkboxes()


        # --- Automatický výběr indikátorů (evoluční) - samostatný frame v levém sloupci
        frame_auto = tk.LabelFrame(frame_left, text="Automatický výběr indikátorů (evoluce)", padx=10, pady=8)
        frame_auto.grid(row=1, column=0, sticky='ew', pady=(10, 0))

        row_auto = tk.Frame(frame_auto)
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
        btn_evo.grid(row=1, column=0, columnspan=7, padx=(0, 0), pady=(8, 0), sticky="w")

        create_tooltip(
            frame_auto,
            "Spustí evoluční výběr indikátorů pro zvolený timeframe.\n"
            "Výsledek se uloží do models/features_tv_<TF>.json jako features_auto.\n"
            "Pak přepni v Tréninku volbu na 'Automatický' a spusť trénink/predikci."
        )


        # 3) Analýza modelu (vpravo)
        frame_eval = tk.LabelFrame(tab1, text="Analyza modelu", padx=10, pady=10)
        frame_eval.grid(row=0, column=2, sticky='nsew', padx=10, pady=10)
        for label, cmd in [
            ("Scan min_conf",      self.on_scan),
            ("Otevrit trade-log",  self.on_open_trades_file),
            ("Otevrit graf",       self.on_open_chart_file),
            ("Posledni obchody",   self.on_show_trades),
        ]:
            b = tk.Button(frame_eval, text=label, width=22, command=cmd)
            b.pack(fill='x', pady=6)
            if "Scan" in label:
                create_tooltip(
                    b,
                    "Pouzije stejny model/predikce jako tlacitko Predikovat (stejny timeframe/base).\n"
                    "Projede rozsah min_conf a hleda kompromis mezi poctem obchodu a PnL."
                )
            elif "trade-log" in label:
                create_tooltip(b, "Otevre CSV obchodu vytvorene simulaci.")
            elif "graf" in label:
                create_tooltip(b, "Otevre posledni PNG equity/simulace.")
            elif "Posledni" in label:
                create_tooltip(b, "Zobrazi posledni obchody z aktualniho trade-logu.")

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

        create_tooltip(ent_s, "Pocatecni min_conf (napr. 0.45).")
        create_tooltip(ent_t, "Konecna min_conf (napr. 0.70).")
        create_tooltip(ent_p, "Krok min_conf (napr. 0.01).")

        # Sdílené parametry backtestu (Scan + Simulace)
        row5 = tk.Frame(frame_eval)
        row5.pack(fill='x', pady=(8, 0))
        tk.Label(row5, text="min_conf").grid(row=0, column=0, sticky="e")
        tk.Entry(row5, textvariable=self.var_minc, width=6).grid(row=0, column=1, padx=5)
        tk.Label(row5, text="fee %").grid(row=0, column=2, sticky="e")
        tk.Entry(row5, textvariable=self.var_fee, width=6).grid(row=0, column=3, padx=5)
        tk.Checkbutton(row5, text="allow_short", variable=self.var_allow_short).grid(row=0, column=4, padx=(8, 0), sticky="w")
        create_tooltip(row5, "Sdilene parametry pro Scan i Simulaci. Hodnoty se propisuji i do zalozky Simulace a Live rezim.")
       

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
        notebook.add(tab2, text='Simulace a Live rezim')

        tab2.grid_columnconfigure(0, weight=1)
        tab2.grid_rowconfigure(0, weight=1)

        live_frame = tk.LabelFrame(tab2, text="Simulace a Live monitoring", padx=12, pady=10)
        live_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Tlačítka – vlevo nahoře
        row_btns = tk.Frame(live_frame)
        row_btns.pack(fill='x')
        tk.Button(row_btns, text="Spustit simulaci",  width=20, command=self.on_simulate).pack(side='left')
        tk.Button(row_btns, text="Spustit Live rezim", width=20, command=self.on_open_live_monitor).pack(side='left', padx=8)
        tk.Button(row_btns, text="Stop Live rezim", width=18, command=self.on_stop_live_monitor).pack(side='left', padx=4)
        self.var_live_status = tk.StringVar(value="Live: zastaveno")
        tk.Label(row_btns, textvariable=self.var_live_status, fg="#2c3e50").pack(side='right')


        # Nastavení Live monitoru (graf + pípání + CSV log STRONG alertů)
        lm_box = tk.LabelFrame(live_frame, text="Nastaveni Live monitoru", padx=10, pady=6)
        lm_box.pack(fill='x', pady=(10, 6))

        lm_row1 = tk.Frame(lm_box); lm_row1.pack(fill='x', pady=(2, 2))
        self.var_live_auto_fetch = tk.BooleanVar(value=True)
        tk.Checkbutton(lm_row1, text="Auto-fetch (append)", variable=self.var_live_auto_fetch).pack(side='left')
        self.var_live_beep_weak = tk.BooleanVar(value=False)
        tk.Checkbutton(lm_row1, text="Pípat i slabé", variable=self.var_live_beep_weak).pack(side='left', padx=10)
        self.var_live_no_beep = tk.BooleanVar(value=False)
        tk.Checkbutton(lm_row1, text="Vypnout pípání", variable=self.var_live_no_beep).pack(side='left', padx=10)
        self.var_live_no_log = tk.BooleanVar(value=False)
        tk.Checkbutton(lm_row1, text="Vypnout CSV log", variable=self.var_live_no_log).pack(side='left', padx=10)
        create_tooltip(lm_row1, "Vychozi: pipa jen STRONG signaly a zapisuje STRONG alerty do CSV.")

        lm_row2 = tk.Frame(lm_box); lm_row2.pack(fill='x', pady=(2, 2))
        tk.Label(lm_row2, text="CSV pro STRONG alerty:").pack(side='left')
        self.var_alerts_csv = tk.StringVar(value="")
        tk.Entry(lm_row2, textvariable=self.var_alerts_csv, width=60).pack(side='left', padx=(6, 6))
        tk.Button(lm_row2, text="…", width=3, command=self._pick_alerts_csv).pack(side='left')
        create_tooltip(lm_row2, "Nech prazdne = default results/live_alerts_<tf>.csv")

        # Riziko na obchod – řídí trade_pct pro simulaci i live
        rk = tk.Frame(live_frame)
        rk.pack(fill='x', pady=(12,4))
        tk.Label(rk, text="Riziko na obchod (%):").pack(side='left')
        self.var_risk_pct = tk.DoubleVar(value=5.0)
        ent_risk = tk.Entry(rk, textvariable=self.var_risk_pct, width=8)
        ent_risk.pack(side='left', padx=6)
        create_tooltip(ent_risk, "Pouzije se jako trade_pct. Priklad: 5.0 = 5 % kapitalu na STRONG vstup.")

        # === NOVÉ: ladicí parametry simulace ===
        tk.Label(live_frame, text="Pokročilé parametry simulace:").pack(anchor='w', pady=(8,2))
        adv = tk.Frame(live_frame)
        adv.pack(fill='x', pady=(0,6))

        tk.Label(adv, text="min_conf").pack(side='left')
        ent_minc = tk.Entry(adv, textvariable=self.var_minc, width=6)
        ent_minc.pack(side='left', padx=(4,12))

        tk.Label(adv, text="fee %").pack(side='left')
        ent_fee = tk.Entry(adv, textvariable=self.var_fee, width=6)
        ent_fee.pack(side='left', padx=(4,12))
        cb_allow = tk.Checkbutton(adv, text="allow_short", variable=self.var_allow_short)
        cb_allow.pack(side='left', padx=(4,12))

        tk.Label(adv, text="min_conf_low").pack(side='left')
        ent_mclo = tk.Entry(adv, textvariable=self.var_mclo, width=6)
        ent_mclo.pack(side='left', padx=(4,12))

        tk.Label(adv, text="trade_pct_low").pack(side='left')
        tk.Label(adv, textvariable=self.var_trade_pct_low_info, width=14, anchor='w').pack(side='left', padx=(4,12))

        tk.Label(adv, text="max_hold").pack(side='left')
        ent_maxh = tk.Entry(adv, textvariable=self.var_max_hold, width=6)
        ent_maxh.pack(side='left', padx=(4,12))

        create_tooltip(adv, "Ladici simulace: min_conf_low je prah pro slabsi signal, trade_pct_low se odvodi z rizika (40 % STRONG), max_hold je casovy limit drzeni (v barech). Nastav -1/0 pro deaktivaci.")
        create_tooltip(ent_minc, "Vychozi: 0.50. Doporuceny rozsah cca 0.47-0.56 (vice obchodu 0.47, prisnejsi filtr 0.56).")
        create_tooltip(ent_fee, "Vychozi: 0.30 (%). Simulace i scan pouzivaji fee_pct = fee/100.")
        create_tooltip(cb_allow, "Povoli short obchody v simulaci a scanu.")
        create_tooltip(ent_mclo, "Vychozi: 0.45. Musi byt mensi nez min_conf.")
        create_tooltip(ent_maxh, "Vychozi: 0 (vypnuto). Napr. 24 nebo 48 pro casovy stop.")
        self.var_risk_pct.trace_add("write", self._on_risk_changed)
        self._refresh_risk_derived_labels()

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
        base = getattr(self, 'var_base', tk.StringVar(value='gold')).get().strip().lower() or "gold"
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

    def _append_live_output(self, msg: str):
        if not hasattr(self, "live_output"):
            return
        if msg is None:
            return
        if isinstance(msg, bytes):
            try:
                msg = msg.decode("utf-8", "ignore")
            except Exception:
                msg = str(msg)
        orig_state = self.live_output.cget("state")
        if orig_state == "disabled":
            self.live_output.configure(state="normal")
        if not msg.endswith("\n"):
            msg = msg + "\n"
        self.live_output.insert("end", msg)
        self.live_output.see("end")
        if orig_state == "disabled":
            self.live_output.configure(state="disabled")

    def log_live_async(self, msg: str):
        self.after(0, self._append_live_output, msg)

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

    def _build_help_text(self) -> str:
        return (
            "LSTM GUI Tuner - Napoveda\n"
            "=========================\n\n"
            "1) Sber dat\n"
            "- Fetch mode:\n"
            "  append = doplnuje nove svicky,\n"
            "  overwrite = prepise soubor poslednimi N bary.\n"
            "- bars: kolik poslednich baru stahnout (typicky 1000-5000).\n"
            "- DXY/COT: volitelna makro data, zapinej jen pokud je opravdu pouzivas.\n\n"
            "2) Automaticky vyber indikatoru (evoluce)\n"
            "- Generace/Populace urcuji delku a sirku hledani.\n"
            "- Min obchodu/den brani tomu, aby vyhrala strategie s malym poctem obchodu.\n"
            "- Vysledek se uklada do models/features_tv_<TF>.json jako features_auto.\n\n"
            "3) Trenink modelu\n"
            "- seq_len: kolik svicek model vidi najednou.\n"
            "  5m: obvykle 20-80 (start 30), 1h: obvykle 12-48.\n"
            "- thr_pct: jak citlive se tvori BUY/SELL/NO-TRADE labely.\n"
            "- epochs: obvykle 15-40; vyssi dava smysl jen pokud val_loss stale klesa.\n"
            "- batch: obvykle 16-128 (start 24).\n"
            "- Rezim treninku:\n"
            "  Multi = 1 model (BUY/SELL/NO-TRADE),\n"
            "  Two-stage = 2 modely (Trade/NoTrade -> Buy/Sell).\n\n"
            "4) Predikce\n"
            "- AUTO: rezim se vezme z metadata.two_stage.enabled.\n"
            "- Pouzit Two-stage inference: vynuti two-stage.\n"
            "- Vynutit Single-stage: docasne ignoruje two-stage metadata.\n"
            "- min_conf_trade/min_conf_dir/min_margin_dir: prahy two-stage.\n"
            "  Prakticky:\n"
            "  min_conf_trade 0.45-0.60,\n"
            "  min_conf_dir   0.50-0.65,\n"
            "  min_margin_dir 0.03-0.12.\n"
            "- min_conf (globalni): obvykle 0.47-0.56 (vychozi 0.50).\n\n"
            "5) Analyza modelu\n"
            "- Scan min_conf prochazi rozsah hodnot a ukaze nejlepsi kompromis.\n"
            "- Trade-log/graf/posledni obchody ctou vysledky posledni simulace.\n"
            "- Analyza pouziva stejna predikcni data jako Predikce (a tim i stejny model/rezim).\n\n"
            "6) Simulace a Live rezim\n"
            "- Simulace testuje pravidla na CSV predikcich a vytvori equity/trade log.\n"
            "- Live monitor pouziva stejny predikcni skript a metadata.\n"
            "- Pokud se nedari zapis CSV, zkontroluj, ze soubor neni otevreny v Excelu.\n\n"
            "Kompletni sada parametru (sdilena mezi zalozkami)\n"
            "- min_conf:\n"
            "  Globalni filtr sily signalu (0-1). Vyssi hodnota = mene obchodu.\n"
            "  Pouziva Predikce, Scan min_conf i Simulace.\n"
            "  Doporuceni: 0.47-0.56 (vychozi 0.50).\n"
            "  Prakticky: 0.47 = vice obchodu, 0.56 = prisnejsi filtr.\n"
            "- fee %:\n"
            "  Poplatek za obchod v procentech (0.30 = 0.3 %).\n"
            "  Pouziva Scan i Simulace (neovlivnuje vypocet samotne predikce).\n"
            "- allow_short:\n"
            "  Povoli short obchody. Kdyz je vypnuto, bere se jen long.\n"
            "  Pouziva Scan i Simulace.\n"
            "- min_conf_low:\n"
            "  Druhy (nizsi) prah pro slabsi vstupy. Musi byt mensi nez min_conf.\n"
            "  Pouziva Simulace (a Live monitor, pokud je parametr predan).\n"
            "- trade_pct_low:\n"
            "  Velikost slabsi pozice jako cast kapitalu.\n"
            "  POCITA se automaticky z rizika: trade_pct_low = 0.40 * trade_pct.\n"
            "  Pouziva Simulace (a Live monitor, pokud je parametr predan).\n"
            "- max_hold:\n"
            "  Time-stop v poctu baru. 0 = vypnuto.\n"
            "  Priklad: 24 na TF 5m je cca 2 hodiny drzeni.\n"
            "  Pouziva Simulace (a Live monitor, pokud je parametr predan).\n\n"
            "Poznamka k predikci vs simulaci\n"
            "- Predikce vytvari signaly (a jejich confidence).\n"
            "- Simulace teprve rozhodne, ktere signaly obchodovat podle min_conf,\n"
            "  allow_short, fee a pokrocilych parametru.\n"
            "- Proto je normalni, ze pri stejne predikci mohou ruzna nastaveni\n"
            "  simulace dat odlisny pocet obchodu a odlisne PnL.\n\n"
            "Vychozi pokrocile parametry\n"
            "- min_conf = 0.50\n"
            "- min_conf_low = 0.45\n"
            "- riziko na obchod = 5 % (trade_pct=0.05)\n"
            "- trade_pct_low(auto) = 2 %\n"
            "- max_hold = 0 (vypnuto)\n\n"
            "Doporuceny postup prace\n"
            "1. Fetch dat\n"
            "2. (Volitelne) Evoluce indikatoru\n"
            "3. Trenink\n"
            "4. Predikce\n"
            "5. Scan min_conf\n"
            "6. Simulace\n"
            "7. Live monitor\n"
        )

    def on_open_help_window(self):
        if self._help_win is not None and self._help_win.winfo_exists():
            self._help_win.lift()
            self._help_win.focus_force()
            return

        win = tk.Toplevel(self)
        win.title("Nápověda")
        win.geometry("920x700")
        self._help_win = win

        wrapper = tk.Frame(win)
        wrapper.pack(fill='both', expand=True, padx=10, pady=10)
        wrapper.grid_rowconfigure(0, weight=1)
        wrapper.grid_columnconfigure(0, weight=1)

        txt = tk.Text(wrapper, wrap='word')
        txt.grid(row=0, column=0, sticky='nsew')
        ysb = ttk.Scrollbar(wrapper, orient='vertical', command=txt.yview)
        ysb.grid(row=0, column=1, sticky='ns')
        txt.configure(yscrollcommand=ysb.set)

        txt.insert('1.0', self._build_help_text())
        txt.configure(state='disabled')

    # ------------------- actions (zachováno beze změn) ----------------------
    def on_fetch(self):

        if not FETCH_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {FETCH_SCRIPT}")
            return

        bars = int(getattr(self, "var_fetch_bars", tk.IntVar(value=5000)).get())
        mode = getattr(self, "var_fetch_mode", tk.StringVar(value="append")).get().strip() or "append"
        use_dxy = bool(getattr(self, "var_fetch_dxy", tk.BooleanVar(value=False)).get())
        use_cot = bool(getattr(self, "var_fetch_cot", tk.BooleanVar(value=False)).get())

        cmd = [self.py(), str(FETCH_SCRIPT), "--bars", str(bars), "--mode", mode]
        if use_dxy:
            cmd.append("--use_dxy")
        if use_cot:
            cmd.append("--use_cot")

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

        base_cmd = [
            self.py(), str(TRAIN_SCRIPT),
            "--timeframe",        self.var_tf.get(),
            "--seq_len",          str(seq),
            "--thr_pct",          f"{thr:.2f}",
            "--epochs",           str(ep),
            "--batch_size",       str(batch),
            "--cot_shift_days",   str(cotsh),
        ]
        if features:
            base_cmd += ["--features", ",".join(features)]

        mode = getattr(self, "var_train_mode", tk.StringVar(value="two_stage")).get()

        if mode == "multi":
            cmd = base_cmd + ["--mode", "multi"]
            run_cmd(cmd, self.log_async, ROOT)
            return

        # Two-stage = spustit dva tréninky postupně: trade a direction
        cmds = [
            base_cmd + ["--mode", "trade"],
            base_cmd + ["--mode", "direction"],
        ]
        run_cmd_sequence(cmds, self.log_async, ROOT)
    def on_predict(self):

        if not PREDICT_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Nenalezen {PREDICT_SCRIPT}")
            return

        tf = self.var_tf.get()
        base = getattr(self, 'var_base', tk.StringVar(value='gold')).get()
        out_csv = RESULTS / f"predictions_{base}_{tf}.csv"

        # režimy predikce: auto / two-stage / single
        use_ts = bool(getattr(self, "var_use_two_stage", None) and self.var_use_two_stage.get())
        force_single = bool(getattr(self, "var_force_single", None) and self.var_force_single.get())

        # pokud je two-stage zapnuto, ulož prahy do metadat a ověř kompletnost modelů
        if use_ts and (not force_single):
            try:
                self._save_two_stage_settings_to_meta()
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

        cmd = [
            self.py(), str(PREDICT_SCRIPT),
            "--timeframe", tf,
            "--output", str(out_csv),
            "--min_conf", f"{getattr(self,'var_minc', tk.DoubleVar(value=0.50)).get():.4f}"
        ]

        # --features: posílej jen když je zdroj indikátorů „ruční“
        features = None if (self.var_feat_source.get() == "auto") else self.get_selected_indicators()
        if features:
            cmd += ["--features", ",".join(features)]

        if force_single:
            cmd.append("--force_single")
        elif use_ts:
            cmd.append("--use_two_stage")
        # else: auto (predict si rozhodne podle meta)

        run_cmd(cmd, self.log_async, ROOT)

    def _safe_risk_pct(self) -> float:
        try:
            v = float(getattr(self, "var_risk_pct", tk.DoubleVar(value=5.0)).get())
        except Exception:
            v = 5.0
        if v < 0:
            v = 0.0
        if v > 100:
            v = 100.0
        return v

    def _trade_pct_strong(self) -> float:
        return self._safe_risk_pct() / 100.0

    def _trade_pct_low(self) -> float:
        # Slabý vstup = 40 % STRONG vstupu.
        return self._trade_pct_strong() * 0.40

    def _refresh_risk_derived_labels(self):
        low = self._trade_pct_low()
        if hasattr(self, "var_trade_pct_low_info"):
            self.var_trade_pct_low_info.set(f"{low:.4f} ({low*100:.2f}%)")

    def _on_risk_changed(self, *_):
        self._refresh_risk_derived_labels()

    def on_simulate(self):
        if not SIM_SCRIPT.exists():
            messagebox.showerror("Chyba", f"Chybí {SIM_SCRIPT}")
            return
        base = getattr(self, 'var_base', tk.StringVar(value='gold')).get()
        pred_csv = Path("results") / f"predictions_{base}_{self.var_tf.get()}.csv"
        pred_abs = ROOT / pred_csv
        if not pred_abs.exists():
            messagebox.showwarning("Upozornění", f"Nejprve vygeneruj predikce ({pred_csv}).")
            self.log_live_async(f"[WARN] Simulace nespustena: chybi {pred_csv}")
            return
        minc = getattr(self, 'var_minc', tk.DoubleVar(value=0.50)).get()
        fee = getattr(self, 'var_fee', tk.DoubleVar(value=0.30)).get()
        allow = getattr(self, 'var_allow_short', tk.BooleanVar(value=True)).get()
        mclo = getattr(self, 'var_mclo', tk.DoubleVar(value=-1)).get()
        maxh = getattr(self, 'var_max_hold', tk.IntVar(value=0)).get()
        tps = self._trade_pct_strong()
        tplo_eff = self._trade_pct_low()


        cmd = [
            self.py(), str(SIM_SCRIPT),
            "--input", str(pred_csv),
            "--min_conf", f"{minc:.2f}",
            "--fee_pct", f"{fee/100.0:.6f}",
            "--trade_pct", f"{tps:.4f}",
        ]
        adv_notes = [f"trade_pct={tps:.4f} ({tps*100:.2f}%)", f"trade_pct_low(auto)={tplo_eff:.4f} ({tplo_eff*100:.2f}%)"]
        if mclo > 0 and tplo_eff > 0:
            cmd += [
                "--min_conf_low", f"{mclo:.2f}",
                "--trade_pct_low", f"{tplo_eff:.4f}"
            ]
            adv_notes.append(f"min_conf_low={mclo:.2f}, trade_pct_low={tplo_eff:.4f}")
            if mclo >= minc:
                adv_notes.append("POZOR: min_conf_low >= min_conf, slabé vstupy se prakticky nepoužijí")
        elif mclo > 0 and tplo_eff <= 0:
            adv_notes.append("POZOR: riziko na obchod je 0 %, slabé vstupy se nepoužijí")
        else:
            adv_notes.append("Slabé vstupy vypnuté (min_conf_low <= 0)")
        if maxh > 0:
            cmd += ["--max_hold_bars", f"{maxh}"]
            adv_notes.append(f"max_hold_bars={maxh}")
        if allow:
            cmd.append("--allow_short")
            adv_notes.append("allow_short=True")
        else:
            adv_notes.append("allow_short=False")

        self.log_live_async("[SIM] Parametry: " + " | ".join(adv_notes))
        self.log_async("[SIM] Parametry: " + " | ".join(adv_notes))

        # Simulace se loguje jak do hlavního Logu, tak do panelu Live výstupu.
        def _runner():
            self.log_async(f">>> {' '.join(str(x) for x in cmd)}")
            self.log_live_async(f">>> {' '.join(str(x) for x in cmd)}")
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
                            line = line.rstrip()
                            self.log_async(line)
                            self.log_live_async(line)
                    ret = p.wait()
                    if ret == 0:
                        self.log_async("[OK] Simulace hotova.")
                        self.log_live_async("[OK] Simulace hotova.")
                    else:
                        self.log_async(f"[CHYBA] Simulace skoncila s navratovym kodem {ret}.")
                        self.log_live_async(f"[CHYBA] Simulace skoncila s navratovym kodem {ret}.")
            except Exception as e:
                self.log_async(f"[VYJIMKA] {e}")
                self.log_live_async(f"[VYJIMKA] {e}")
        threading.Thread(target=_runner, daemon=True).start()

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
        allow = getattr(self, 'var_allow_short', tk.BooleanVar(value=True)).get()
        out_csv = Path("results") / f"scan_min_conf_{pred_csv.stem}.csv"
        cmd = [self.py(), str(SCAN_SCRIPT),
               "--input", str(pred_csv),
               "--fee_pct", f"{fee/100.0:.6f}",
               "--start",   f"{start:.3f}",
               "--stop",    f"{stop:.3f}",
               "--step",    f"{step:.3f}",
               "--out_csv", str(out_csv)]
        if allow:
            cmd.append("--allow_short")
        self.log_async(f"[SCAN] Parametry: min_conf_range={start:.2f}-{stop:.2f}/{step:.2f} | fee={fee:.2f}% | allow_short={allow}")
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
        best_pnl = sorted(rows, key=lambda r: _to_float(r.get("PnL_%")), reverse=True)[0]

        self.log_async("[SCAN] TOP podle Sharpe:")
        for r in top_sharpe:
            self.log_async(
                f"min_conf={r.get('min_conf')} | PnL={r.get('PnL_%')}% | "
                f"Sharpe={r.get('Sharpe')} | MaxDD={r.get('MaxDD_%')}% | Trades={r.get('Trades')}"
            )
        self.log_async(
            "[SCAN] Nejvyšší PnL: "
            f"min_conf={best_pnl.get('min_conf')} | PnL={best_pnl.get('PnL_%')}% | "
            f"Sharpe={best_pnl.get('Sharpe')} | MaxDD={best_pnl.get('MaxDD_%')}% | Trades={best_pnl.get('Trades')}"
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
            "--min_conf",  f"{getattr(self,'var_minc', tk.DoubleVar(value=0.50)).get():.2f}",
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
            self._set_live_status("bezi")
            self.log("[INFO] Live Alerts spuštěny.")
        except Exception as e:
            messagebox.showerror("Chyba", str(e))

    def _pump_live_output(self):
        try:
            for line in self._live_proc.stdout:
                if line is None:
                    break
                line = line.rstrip()
                l = line.lower()
                if "[fetch err]" in l or "[predict err]" in l or "worker error" in l:
                    self._set_live_status("chyba")
                elif "[wait]" in l:
                    self._set_live_status("ceka")
                elif line.startswith("[LIVE]") or "updated" in l:
                    self._set_live_status("bezi")
                self.log_live_async(line)
        except Exception as e:
            self._set_live_status("chyba")
            self.log_async(f"[VYJIMKA Live] {e}")
            self.log_live_async(f"[VYJIMKA Live] {e}")
        finally:
            try:
                if self._live_proc is None or self._live_proc.poll() is not None:
                    self._set_live_status("zastaveno")
            except Exception:
                pass

    def on_stop_live_monitor(self):
        if self._live_proc and self._live_proc.poll() is None:
            try:
                self._live_proc.terminate()
                self._live_proc = None
                self._set_live_status("zastaveno")
                self.log_live_async("[LIVE] monitor zastaven uzivatelem.")
            except Exception as e:
                self._set_live_status("chyba")
                messagebox.showerror("Chyba", str(e))
        else:
            self._set_live_status("zastaveno")
            self.log_live_async("[LIVE] monitor uz nebezi.")

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


    def _pick_alerts_csv(self):
        try:
            p = filedialog.asksaveasfilename(
                title="Uložit STRONG alerty do CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
            )
            if p:
                self.var_alerts_csv.set(p)
        except Exception as e:
            messagebox.showerror("Chyba", str(e))

    def _set_live_status(self, text: str):
        try:
            self.var_live_status.set(f"Live: {text}")
        except Exception:
            pass

    def on_open_live_monitor(self):

        self._set_live_status("spoustim")
        live_py = Path(ROOT) / "src" / "live_monitor.py"
        if not live_py.exists():
            self._set_live_status("chyba")
            messagebox.showerror("Chyba", f"Soubor {live_py} nebyl nalezen.")
            return

        tf = self.var_tf.get()
        base = getattr(self, 'var_base', tk.StringVar(value='gold')).get().strip().lower() or "gold"
        meta_path = ROOT / "models" / f"features_tv_{tf}.json"
        mode_info = "single-stage"
        model_info = ""
        try:
            if meta_path.exists():
                meta = _load_json_robust(meta_path)
                ts = meta.get("two_stage", {}) or {}
                if bool(ts.get("enabled")) and (ts.get("model_path_trade") or meta.get("model_path_trade")) and (ts.get("model_path_dir") or meta.get("model_path_dir")):
                    mode_info = "two-stage (AUTO z metadata)"
                    model_info = f"trade={ts.get('model_path_trade') or meta.get('model_path_trade')} | dir={ts.get('model_path_dir') or meta.get('model_path_dir')}"
                else:
                    mode_info = "single-stage (AUTO z metadata)"
                    model_info = f"model={meta.get('model_path', f'models/lstm_tv_{tf}.h5')}"
        except Exception as e:
            self.log_async(f"[WARN] Nelze nacist metadata pro Live info: {e}")

        cmd = [
            self.py(), str(live_py),
            "--base", base,
            "--timeframe", tf,
            "--min-conf",  str(getattr(self,'var_minc', tk.DoubleVar(value=0.50)).get()),
            "--trade-pct", f"{self._trade_pct_strong():.4f}",
            "--trade-pct-low", f"{self._trade_pct_low():.4f}",
        ]
        mclo = getattr(self, 'var_mclo', tk.DoubleVar(value=-1)).get()
        if mclo > 0:
            cmd += ["--min-conf-low", f"{mclo:.2f}"]

        if getattr(self,'var_allow_short', tk.BooleanVar(value=True)).get():
            cmd.append("--allow-short")

        # Live monitor nastavení
        if getattr(self, "var_live_auto_fetch", tk.BooleanVar(value=True)).get():
            cmd.append("--auto-fetch")

        if getattr(self, "var_live_beep_weak", tk.BooleanVar(value=False)).get():
            cmd.append("--beep-weak")

        if getattr(self, "var_live_no_beep", tk.BooleanVar(value=False)).get():
            cmd.append("--no-beep")

        if getattr(self, "var_live_no_log", tk.BooleanVar(value=False)).get():
            cmd.append("--no-log")

        alerts_csv = getattr(self, "var_alerts_csv", tk.StringVar(value="")).get().strip()
        if alerts_csv:
            cmd += ["--alerts-csv", alerts_csv]

        if self._live_proc and self._live_proc.poll() is None:
            try:
                self._live_proc.terminate()
            except Exception:
                pass
            self._live_proc = None

        self.log_live_async(f"[LIVE] Rezim: {mode_info}")
        if model_info:
            self.log_live_async(f"[LIVE] Modely: {model_info}")
        self.log_live_async(f"[LIVE] Data: base={base} | pred_csv=results/predictions_{base}_{tf}.csv")
        self.log_live_async(
            f"[LIVE] Risk: strong={self._trade_pct_strong()*100:.2f}% | "
            f"weak(auto)={self._trade_pct_low()*100:.2f}%"
        )
        self.log_live_async(f">>> {' '.join(str(x) for x in cmd)}")

        self._live_proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        threading.Thread(target=self._pump_live_output, daemon=True).start()
        self.log("▶ Spuštěn Live monitor v novém okně. Zavřením okna se alerty vypnou.")
if __name__ == "__main__":
    app = TradingGUI()
    app.mainloop()
