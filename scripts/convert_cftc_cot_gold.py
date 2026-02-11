import pandas as pd
import os

# === Cesty ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Cesta ke tvému souboru f_year.xls (uprav podle umístění)
SOURCE_FILE = r"C:\Users\adamk\Můj disk\Trader\lstm\data\raw\f_year.xls"
OUTPUT_FILE = os.path.join(DATA_DIR, "cot.csv")

# === 1) Načtení tabulky ===
print(f"[INFO] Načítám {SOURCE_FILE} ...")
df = pd.read_excel(SOURCE_FILE)

# === 2) Vyfiltruj GOLD – COMMODITY EXCHANGE INC. ===
mask = df["Market_and_Exchange_Names"].str.contains("GOLD - COMMODITY EXCHANGE INC.", case=False, na=False)
gold_df = df[mask].copy()
print(f"[INFO] Nalezeno {len(gold_df)} řádků pro GOLD COMEX.")

if gold_df.empty:
    raise ValueError("V souboru nejsou data pro 'GOLD - COMMODITY EXCHANGE INC.'")

# === 3) Vytvoř COT = Managed Money Long - Managed Money Short ===
long_col = "M_Money_Positions_Long_ALL"
short_col = "M_Money_Positions_Short_ALL"
date_col = "Report_Date_as_MM_DD_YYYY"

gold_df["date"] = pd.to_datetime(gold_df[date_col], errors="coerce")
gold_df["cot"] = pd.to_numeric(gold_df[long_col], errors="coerce") - pd.to_numeric(gold_df[short_col], errors="coerce")

# === 4) Vyčisti a připrav výstup ===
cot = gold_df[["date", "cot"]].dropna().copy()
# pokud je stejné datum vícekrát, vem ten s největší absolutní hodnotou (net pozice)
cot = cot.sort_values(["date", "cot"], key=lambda s: s.abs()).drop_duplicates(subset=["date"], keep="last")
cot = cot.sort_values("date").reset_index(drop=True)

cot.to_csv(OUTPUT_FILE, index=False)
print(f"[✓] Hotovo! Uloženo {len(cot)} řádků do {OUTPUT_FILE}")
print(cot.tail())
