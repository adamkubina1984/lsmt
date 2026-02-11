import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def roc(series: pd.Series, period: int = 12) -> pd.Series:
    return ((series - series.shift(period)) / series.shift(period)) * 100

def vwap(df: pd.DataFrame) -> pd.Series:
    return (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()


def add_indicators(df: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    out = df.copy()

    if "rsi" in wanted:
        out["rsi14"] = rsi(out["close"], 14)

    if "macd" in wanted:
        macd_line, signal_line, hist = macd(out["close"], 12, 26, 9)
        out["macd"], out["macd_signal"], out["macd_hist"] = macd_line, signal_line, hist

    if "ema20" in wanted:
        out["ema20"] = ema(out["close"], 20)
        out["ema20_dist"] = (out["close"] - out["ema20"]) / out["ema20"]

    if "ema50" in wanted:
        out["ema50"] = ema(out["close"], 50)
        out["ema50_dist"] = (out["close"] - out["ema50"]) / out["ema50"]

    if "ema100" in wanted:
        out["ema100"] = ema(out["close"], 100)
        out["ema100_dist"] = (out["close"] - out["ema100"]) / out["ema100"]

    if "roc" in wanted:
        out["roc12"] = roc(out["close"], 12)

    if "vwap" in wanted and "volume" in out.columns:
        out["vwap"] = vwap(out)
        out["vwap_dist"] = (out["close"] - out["vwap"]) / out["vwap"]

    if "bb" in wanted:
        ma = out["close"].rolling(20).mean()
        sd = out["close"].rolling(20).std(ddof=0)
        up = ma + 2 * sd
        lo = ma - 2 * sd
        out["bb_ma"], out["bb_up"], out["bb_lo"] = ma, up, lo
        out["bb_width"] = (up - lo) / ma
        out["bb_pos"] = (out["close"] - ma) / (up - lo).replace(0, np.nan)

    if "atr" in wanted:
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs()
        ], axis=1).max(axis=1)
        out["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()
        out["atr_norm"] = out["atr14"] / out["close"]

    if "stoch" in wanted:
        low_min = out["low"].rolling(14).min()
        high_max = out["high"].rolling(14).max()
        out["stoch_k"] = 100 * (out["close"] - low_min) / (high_max - low_min + 1e-9)
        out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    if "adx" in wanted:
        plus_dm = (out["high"].diff() > out["low"].diff()).astype(float) * out["high"].diff()
        minus_dm = (out["low"].diff() > out["high"].diff()).astype(float) * out["low"].diff()
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        out["adx"] = dx.rolling(14).mean()

    if "mfi" in wanted and "volume" in out.columns:
        tp = (out["high"] + out["low"] + out["close"]) / 3
        mf = tp * out["volume"]
        pos = mf.where(tp > tp.shift(), 0.0)
        neg = mf.where(tp < tp.shift(), 0.0)
        mfi = 100 - (100 / (1 + pos.rolling(14).sum() / (neg.rolling(14).sum() + 1e-9)))
        out["mfi14"] = mfi

    if "sar" in wanted:
        af = 0.02
        max_af = 0.2
        trend = 1
        ep = out["high"].iloc[0]
        sar = out["low"].iloc[0]
        sar_list = []
        for i in range(1, len(out)):
            prev_sar = sar
            if trend == 1:
                sar = prev_sar + af * (ep - prev_sar)
                if out["low"].iloc[i] < sar:
                    trend = -1
                    sar = ep
                    ep = out["low"].iloc[i]
                    af = 0.02
                elif out["high"].iloc[i] > ep:
                    ep = out["high"].iloc[i]
                    af = min(af + 0.02, max_af)
            else:
                sar = prev_sar + af * (ep - prev_sar)
                if out["high"].iloc[i] > sar:
                    trend = 1
                    sar = ep
                    ep = out["high"].iloc[i]
                    af = 0.02
                elif out["low"].iloc[i] < ep:
                    ep = out["low"].iloc[i]
                    af = min(af + 0.02, max_af)
            sar_list.append(sar)
        out["sar"] = pd.Series([np.nan] + sar_list)
        out["sar_bin"] = (out["close"] > out["sar"]).astype(int)

    if "cci" in wanted:
        tp = (out["high"] + out["low"] + out["close"]) / 3
        ma = tp.rolling(20).mean()
        md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        out["cci"] = (tp - ma) / (0.015 * md + 1e-9)

    if "keltner" in wanted:
        typical_price = (out["high"] + out["low"] + out["close"]) / 3
        ema_tp = typical_price.ewm(span=20, adjust=False).mean()
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=10, adjust=False).mean()
        out["kc_mid"] = ema_tp
        out["kc_up"] = ema_tp + 2 * atr
        out["kc_low"] = ema_tp - 2 * atr

    out = out.ffill().dropna().reset_index(drop=True)
    return out
