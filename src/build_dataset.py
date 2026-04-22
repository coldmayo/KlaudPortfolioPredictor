import pandas as pd
import yfinance as y

from features import RSI, sto_osc, will_R, OBV

# Dataset info: Classification
# Goal: Predict if the stock price goes up or down
# | Date | features | future return |
# features: momentum, volativity, trend, volume behavior

def main():

	stocks = ["AAPL", "AMD", "GOOGL", "AMZN", "BA", "CAT", "CELH", "XOM", "GTLB", "HAS", "JNJ", "MSGS", "MSFT", "NVDA", "ORCL", "PFE", "RDDT", "HOOD", "^GSPC", "^VIX"]   # Technically stocks and ETFs

	dfs = []

	for s in stocks:
		df = y.download(s, period="10y", interval="1d", auto_adjust=True)
		df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

		df["ticker"] = s
        # Create features

        # Returns
		df["ret_1d"] = df["Close"].pct_change(1)   # 1 day return
		df["ret_5d"] = df["Close"].pct_change(5)   # 5 day return
		df["ret_10d"] = df["Close"].pct_change(10)   # 10 day return

		# Moving Averages
		df["ma_10"] = df["Close"].rolling(10).mean()
		df["ma_50"] = df["Close"].rolling(50).mean()
		df["ma_ratio"] = df["ma_10"] / df["ma_50"]

		# Volatility
		df["volatility_10"] = df["ret_1d"].rolling(10).std()

		# Volume Features
		df["vol_change"] = df["Volume"].pct_change()
		df["vol_ma_10"] = df["Volume"].rolling(10).mean()

		# Relative Str Index
		df["RSI"] = RSI(df["Close"], 14)
		df["RSI_overb"] = (df["RSI"] > 70).astype(int)
		df["RSI_overs"] = (df["RSI"] < 30).astype(int)
		df["RSI_change"] = df["RSI"].diff()

		# Stochastic Oscillator
		df["Sto_Osc"] = sto_osc(df["Close"], 14)
		df["WillR"] = will_R(df["Close"], 14)

		# OBV
		df["OBV"] = OBV(df["Close"], df["Volume"])
		df["obv_change"] = df["OBV"].diff()
		df["obv_norm"] = (df["OBV"] - df["OBV"].rolling(50).mean()) / df["OBV"].rolling(50).std()
		df["obv_ma_10"] = df["OBV"].rolling(10).mean()
		df["obv_ratio"] = df["OBV"] / df["obv_ma_10"]

		# Target
		df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

		df = df.dropna()
		dfs.append(df)
		
	full_df = pd.concat(dfs)
	full_df = pd.get_dummies(full_df, columns=["ticker"], prefix="stock")
	full_df.to_csv("dataset.csv", index=False)

if "__main__" == __name__:
    main()
