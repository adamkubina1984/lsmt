# LSTM Obchodní Bot

Tento projekt je určen k predikci obchodních signálů (Buy / Sell / No-Trade) pomocí LSTM neuronové sítě nad daty GOLD, VIX, DXY. Cílem je efektivní generování signálů a simulace obchodní strategie.

---

## 📂 Struktura složek

- `scripts/` – trénink, predikce, simulace, stahování dat
- `src/` – grafické rozhraní (GUI)
- `models/` – uložené modely a škálovače
- `data/raw/` – vstupní CSV data
- `results/` – predikce a simulace

---

## 🔧 Instalace

1. Nainstaluj požadované knihovny:
```bash
pip install -r requirements.txt
