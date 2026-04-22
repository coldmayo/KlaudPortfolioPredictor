import numpy as np
import pandas as pd

def RSI(s, window=14):
    delta = s.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))

def sto_osc(s, window=14):
    L = s.rolling(window).min()
    H = s.rolling(window).max()

    denom = (H - L).replace(0, np.nan)

    return 100 * (s - L) / denom

def will_R(s, window=14):
    L = s.rolling(window).min()
    H = s.rolling(window).max()

    denom = (H - L).replace(0, np.nan)

    return -100 * (H - s) / denom

def MACD(s, fast=12, slow=26, signal=9):
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()

    hist = macd - signal_line

    return macd, signal_line, hist

def OBV(close, volume):
    direction = np.sign(close.diff())
    direction.iloc[0] = 0

    obv = (direction * volume).cumsum()
    return obv
