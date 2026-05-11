import pandas as pd
import yfinance as y
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

from features import RSI, sto_osc, will_R, OBV

# Dataset info: Classification
# Goal: Predict if the stock price goes up or down
# | Date | features | future return |
# features: momentum, volativity, trend, volume behavior

def build_sequ(df, window=60):
    X, y, fwd_rets, prices = [], [], [], []
    scaler = StandardScaler()

    df["fwd_ret"] = df["Close"].pct_change(20).shift(-20)
    df["target"] = 0
    df.loc[df["fwd_ret"] > 0.01, "target"] = 1
    df.loc[df["fwd_ret"] < -0.01, "target"] = -1

    df = df.dropna(subset=["fwd_ret"])

    feature_cols = [c for c in df.columns if c not in ["Date", "ticker", "target", "fwd_ret"]]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    data = df[feature_cols].values
    targets = df["target"].values
    fwd_ret_vals = df["fwd_ret"].values
    prices_vals = df["Close"].values

    for i in range(len(df) - window - 5):
        X.append(data[i:i+window])
        y.append(targets[i + window])
        fwd_rets.append(fwd_ret_vals[i + window])
        prices.append(prices_vals[i + window])

    return np.array(X), np.array(y), np.array(fwd_rets), np.array(prices)

def main(args):

	stocks = ["AAPL", "AMD", "GOOGL", "AMZN", "BA", "CAT", "NVDA", "ORCL", "RDDT", "^GSPC"]   # Technically stocks and ETFs

	dfs = []

	for s in stocks:
		df = y.download(s, period="10y", interval="1d", auto_adjust=True)
		df.reset_index(inplace=True)
		
		df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

		df["ticker"] = s
		if args.type == "time":
			df = df.dropna()
			X, y_, fwd_rets, price = build_sequ(df, 60)

			dfs.append((X, y_, fwd_rets, price))

		elif args.type == "tabular":

			df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        # Returns
			df["ret_1d"] = df["Close"].pct_change(1)   # 1 day return
			df["ret_5d"] = df["Close"].pct_change(5)   # 5 day return
			df["ret_10d"] = df["Close"].pct_change(10)   # 10 day return
			df["ret_20d"] = df["Close"].pct_change(20)

			df["fwd_ret"] = df["Close"].pct_change(5).shift(-5)

		# Moving Averages
			df["ma_10"] = df["Close"].rolling(10).mean()
			df["ma_50"] = df["Close"].rolling(50).mean()
			df["ma_200"] = df["Close"].rolling(200).mean()
			df["ma_ratio_10_50"] = df["ma_10"] / df["ma_50"]
			df["ma_ratio_50_200"] = df["ma_50"] / df["ma_200"]

		# Volatility
			df["volatility_5"] = df["ret_1d"].rolling(5).std()
			df["volatility_10"] = df["ret_1d"].rolling(10).std()
			df["volatility_20"] = df["ret_1d"].rolling(20).std()
			df["vol_ratio"] = df["volatility_5"] / df["volatility_20"]

		
			df["overnight_gap"] = df["Open"] / df["Close"].shift(1) - 1
			df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
			df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

			# 52-week high proximity (breakout momentum)
			df["high_52w"] = df["Close"].expanding().max()
			df["proximity_52w_high"] = df["Close"] / df["high_52w"]

		# Volume Features
			df["vol_change"] = df["Volume"].pct_change()
			df["vol_ma_10"] = df["Volume"].rolling(10).mean()
			df["vol_ma_20"] = df["Volume"].rolling(20).mean()
			df["vol_surprise"] = df["Volume"] / df["vol_ma_20"]

		# Relative Str Index
			df["RSI"] = RSI(df["Close"], 14)
			df["RSI_overb"] = (df["RSI"] > 70).astype(int)
			df["RSI_overs"] = (df["RSI"] < 30).astype(int)
			df["RSI_change"] = df["RSI"].diff()

		# Stochastic Oscillator
			#df["Sto_Osc"] = sto_osc(df["Close"], 14)
			#df["WillR"] = will_R(df["Close"], 14)

		# OBV
			df["OBV"] = OBV(df["Close"], df["Volume"])
			df["obv_change"] = df["OBV"].diff()
			df["obv_norm"] = (df["OBV"] - df["OBV"].rolling(50).mean()) / df["OBV"].rolling(50).std()
			df["obv_ma_10"] = df["OBV"].rolling(10).mean()
			df["obv_ratio"] = df["OBV"] / df["obv_ma_10"]

            # Relative performance vs market (alpha signal)
			#df["ret_rel_market"] = df["ret_5d"] - market_ret_5d

# Price momentum ranked cross-sectionally (very powerful)
			#df["mom_1m"] = df["Close"].pct_change(21)
			#df["mom_3m"] = df["Close"].pct_change(63)
			#df["mom_6m"] = df["Close"].pct_change(126)
			#df["mom_12m_skip1m"] = df["Close"].shift(21).pct_change(252)  # skip 1mo, classic factor

# Mean reversion
			#df["dist_from_ma50"] = (df["Close"] - df["ma_50"]) / df["ma_50"]
			#df["dist_from_ma200"] = (df["Close"] - df["ma_200"]) / df["ma_200"]

# Volatility-adjusted momentum
			#df["sharpe_5d"] = df["ret_1d"].rolling(5).mean() / (df["ret_1d"].rolling(5).std() + 1e-9)
			#df["sharpe_20d"] = df["ret_1d"].rolling(20).mean() / (df["ret_1d"].rolling(20).std() + 1e-9)

# Volume-price divergence
			#df["price_vol_diverge"] = df["ret_5d"] / (df["vol_surprise"] + 1e-9)

# Trend strength (ADX-like)
			#df["trend_consistency"] = df["ret_1d"].rolling(10).apply(lambda x: np.sum(x > 0) / len(x))

# Lagged returns (autocorrelation signal)
			for lag in [1, 2, 3, 5, 10]:
				df[f"ret_lag_{lag}"] = df["ret_1d"].shift(lag)
		
			k = 1.5

			df["target"] = 0
			df.loc[df["fwd_ret"] > 0.01, "target"] = 1
			df.loc[df["fwd_ret"] < -0.01, "target"] = -1

			#df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])

			df = df.dropna()
			dfs.append(df)
		
	if args.type == "tabular":
		full_df = pd.concat(dfs)
		full_df = pd.get_dummies(full_df, columns=["ticker"], prefix="stock")
		
		#print(full_df.corr()['target'].sort_values())
		
		full_df.to_csv("dataset.csv", index=False)

	elif args.type == "time":
		X_all = np.concatenate([x for x, _, __, ___ in dfs], axis=0)
		y_all = np.concatenate([y for _, y, __, ___ in dfs], axis=0); fwd_all = np.concatenate([f for _, _, f, ___ in dfs], axis=0); price = np.concatenate([p for _, _, ___, p in dfs], axis=0)

		np.save("X.npy", X_all)
		np.save("y.npy", y_all); np.save("fwd_ret.npy", fwd_all); np.save("prices.npy", price)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str)   # can be tabular or time
    args = parser.parse_args()
    
    main(args)
