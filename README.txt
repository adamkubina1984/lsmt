# LSTM Obchodni Bot

Tento projekt je urcen k predikci obchodnich signalu (Buy / Sell / No-Trade) pomoci LSTM site nad daty GOLD, VIX, DXY.

---

## Struktura slozek

- `scripts/` - trenink, predikce, simulace, stahovani dat
- `src/` - graficke rozhrani (GUI)
- `models/` - ulozene modely a skalovace
- `data/raw/` - vstupni CSV data
- `results/` - predikce a simulace

---

## Instalace

1. Nainstaluj pozadovane knihovny:
```bash
pip install -r requirements.txt
```

---

## Aktualizace GUI/Live (2026-02)

- `src/gui_tuner.py`
  - Live vystup se zapisuje do panelu `Vystup predikci (Live)` a nezahlcuje hlavni `Log`.
  - Parametr `Riziko na obchod (%)` je napojen na simulaci i live:
    - `trade_pct = risk/100`
    - `trade_pct_low` se odvozuje automaticky jako `0.40 * trade_pct`.
  - Pri spusteni Live se loguje pouzity datovy zdroj:
    - `base`
    - `results/predictions_<base>_<tf>.csv`

- `src/live_monitor.py`
  - Novy CLI parametr `--base` pro sjednoceni se zbytkem GUI.
  - Live worker pouziva predikcni vystup `results/predictions_<base>_<tf>.csv`.
  - Chyby `fetch/predict` se neignoruji potichu:
    - stavove zpravy (`[FETCH ERR]`, `[PREDICT ERR]`, `[WAIT]`).
  - Pri nove svicce se vypise kratky live radek:
    - `[LIVE] <datetime> | BUY/SELL/NO_TRADE | conf=<...>`
