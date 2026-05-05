import numpy as np
import argparse
import pickle
import json
from train import load_csv, train_test_split
import matplotlib.pyplot as plt
from strats import *

def pred_alpha(model, X_test, y_test, model_type, proba = None):
    alpha = 0
    if model_type == "Random Forest":
        if proba is None:
            proba = model.predict_probs(X_test)
            
        classes = np.array(model.classes_)

        idx_down = np.where(classes == -1)[0][0]
        idx_up   = np.where(classes == 1)[0][0]
        idx_side = np.where(classes == 0)[0][0]
        alpha = proba[:, idx_up] - proba[:, idx_down]

    elif model_type in ("SVM", "XGBoost"):
        preds = model.predict(X_test)
        alpha = preds.astype(np.float64)

    # z score normalization
    alpha = (alpha - np.mean(alpha)) / (np.std(alpha) + 1e-8)

    return alpha

class BackTest():
    def __init__(self, strat, alpha, y_true, vol, thresh = 0.5, k = 0.1, prices=None, transaction_cost=0.001, y_true_returns=None):
        self.alpha = alpha
        self.threshold = thresh
        self.k = k
        self.vol = vol
        self.strategy = self.build_strat(strat)
        self.y_true = np.array(y_true)
        self.prices = prices
        self.tc = transaction_cost
        self.fwd_rets = y_true_returns

    def build_strat(self, name):
        if name == "sign":
            return SignStrategy(self.alpha)
        elif name == "threshold":
            return ThresholdStrategy(self.alpha, threshold=self.threshold)
        elif name == "topk":
            return TopKStrategy(self.alpha, k=self.k)
        elif name == "volscaled":
            return VolScaledStrategy(self.alpha, self.vol)

    def find_rets(self):
        raw_returns = self.fwd_rets

        positions = self.strategy.get_positions()

        positions = np.roll(positions, 1)
        positions[0] = 0

        strategy_returns = positions * raw_returns

        turnover = np.abs(np.diff(positions, prepend=0))
        strategy_returns -= turnover * self.tc

        return np.maximum(strategy_returns, -0.9999)

    def run(self):
        returns = self.find_rets()
        cum_returns = np.cumprod(1 + returns)

        total_return = cum_returns[-1] - 1
        sharpe = self.sharpe(returns)
        max_dd = self.max_drawdown(cum_returns)
        hit_rate = np.mean(returns > 0)
        ir = self.information_ratio(returns)
        t_stat = self.t_stat(returns)
        sortino = self.sortino(returns)

        print(f"Total Return:    {total_return:.2%}")
        print(f"Sharpe Ratio:    {sharpe:.3f}")
        print(f"Max Drawdown:    {max_dd:.2%}")
        print(f"Hit Rate:        {hit_rate:.2%}")
        print(f"Information Raio {ir:.2f}")
        print(f"t stat           {t_stat:.2f}")
        print(f"Sortino          {sortino:.2f}")

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "ir": ir,
            "t_stat": t_stat,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "cum_returns": cum_returns,
        }

    def sharpe(self, returns, periods=252):
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(periods)

    def max_drawdown(self, cum_returns):
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

    def information_ratio(self, returns, benchmark=0):
        excess = returns - benchmark
        return np.mean(excess) / np.std(excess)

    def sortino(self, returns):
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0

        return np.mean(returns) / np.std(downside)

    def t_stat(self, returns):
        return np.mean(returns) / (np.std(returns) / np.sqrt(len(returns)))

    def cum_plot(self, cum_returns):
        plt.figure(figsize=(12, 5))
        plt.plot(cum_returns, label="Strategy")
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        plt.title("Cumulative Returns")
        plt.xlabel("Time Step")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.tight_layout()
        plt.show()
