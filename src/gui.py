# src/gui.py

import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Obchodní LSTM Bot")

        # Timeframe
        tk.Label(root, text="Timeframe:").grid(row=0, column=0, sticky="w")
        self.timeframe_var = tk.StringVar(value="5m")
        tk.OptionMenu(root, self.timeframe_var, "5m", "1h").grid(row=0, column=1)

        # Výběr CSV
        tk.Button(root, text="Vybrat CSV", command=self.vyber_csv).grid(row=1, column=0, pady=5)
        self.csv_label = tk.Label(root, text="(žádný soubor)")
        self.csv_label.grid(row=1, column=1, sticky="w")
        self.csv_path = None

        # Tlačítka
        tk.Button(root, text="📥 Stáhnout data", command=self.stahni_data).grid(row=2, column=0, pady=5)
        tk.Button(root, text="🧠 Trénovat model", command=self.trenuj_model).grid(row=3, column=0, pady=5)
        tk.Button(root, text="🔮 Predikovat", command=self.predikuj).grid(row=4, column=0, pady=5)
        tk.Button(root, text="📊 Simulace", command=self.simuluj).grid(row=5, column=0, pady=5)

        self.output = tk.Text(root, height=15, width=70)
        self.output.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    def log(self, text):
        self.output.insert(tk.END, f"{text}\n")
        self.output.see(tk.END)

    def vyber_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path = path
            self.csv_label.config(text=os.path.basename(path))

    def spust(self, prikaz):
        self.log(f">>> {prikaz}")
        try:
            output = subprocess.check_output(prikaz, stderr=subprocess.STDOUT, shell=True, text=True)
            self.log(output)
        except subprocess.CalledProcessError as e:
            self.log(f"[CHYBA] {e.output}")

    def stahni_data(self):
        skript = os.path.join(BASE_DIR, "..", "scripts", "fetch_data.py")
        self.spust(f"python \"{skript}\"")

    def trenuj_model(self):
        if not self.csv_path:
            messagebox.showerror("Chyba", "Vyber CSV soubor pro trénink.")
            return
        skript = os.path.join(BASE_DIR, "..", "scripts", "train_lstm.py")
        self.spust(f"python \"{skript}\" --data \"{os.path.relpath(self.csv_path, start=os.path.join(BASE_DIR, '..'))}\" --output_dir models")

    def predikuj(self):
        if not self.csv_path:
            messagebox.showerror("Chyba", "Vyber CSV soubor pro predikci.")
            return
        skript = os.path.join(BASE_DIR, "..", "scripts", "predict_lstm.py")
        self.spust(
            f"python \"{skript}\" "
            f"--data \"{os.path.relpath(self.csv_path, start=os.path.join(BASE_DIR, '..'))}\" "
            f"--model models/lstm_model.h5 "
            f"--scaler models/scaler.pkl "
            f"--output results/predikce.csv"
        )

    def simuluj(self):
        skript = os.path.join(BASE_DIR, "..", "scripts", "simulate_strategy.py")
        self.spust(f"python \"{skript}\" --input results/predikce.csv")

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()
