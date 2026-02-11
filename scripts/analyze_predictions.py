import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print("Počet řádků:", len(df))
print("Počet signálů podle třídy:\n", df['signal'].value_counts(dropna=False))
print("Průměrná síla:", df['signal_strength'].mean())
print("90. percentil síly:", df['signal_strength'].quantile(0.90))
print("95. percentil síly:", df['signal_strength'].quantile(0.95))
print("99. percentil síly:", df['signal_strength'].quantile(0.99))
