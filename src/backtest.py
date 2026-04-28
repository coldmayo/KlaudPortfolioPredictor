import numpy as np
import argparse
import pickle
import json
from train import load_csv, train_test_split

def pred_alpha(model, X_test, y_test, model_type):
    if model_type == "Random Forest":
        proba = model.predict_probs(X_test)
        classes = np.array(model.classes_)

        # Rule out the 0 class
        idx_down = np.where(classes == -1)[0][0]
        idx_up   = np.where(classes ==  1)[0][0]
        alpha = proba[:, idx_up] - proba[:, idx_down]

    elif model_type in ("SVM", "XGBoost"):
        preds = model.predict(X_test)
        alpha = preds.astype(np.float64)

    return alpha

class BackTest():
    def __init__(self, alpha, y_true, prices=None, transaction_cost=0.001, y_true_returns=None):
        self.alpha = np.array(alpha)
        self.y_true = np.array(y_true)
        self.prices = prices
        self.tc = transaction_cost
        self.fwd_rets = y_true_returns

    def find_rets(self):
        raw_returns = self.fwd_rets
        strategy_returns = self.alpha * raw_returns
        turnover = np.abs(np.diff(self.alpha, prepend=0))
        strategy_returns = strategy_returns - (turnover * self.tc)
        strategy_returns = np.maximum(strategy_returns, -0.9999)
        return strategy_returns

    def run(self):
        returns = self.find_rets()
        cum_returns = np.cumprod(1 + returns)

        total_return = cum_returns[-1] - 1
        sharpe = self.sharpe(returns)
        max_dd = self.max_drawdown(cum_returns)
        hit_rate = np.mean(returns > 0)

        print(f"Total Return:    {total_return:.2%}")
        print(f"Sharpe Ratio:    {sharpe:.3f}")
        print(f"Max Drawdown:    {max_dd:.2%}")
        print(f"Hit Rate:        {hit_rate:.2%}")

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "cum_returns": cum_returns,
        }

    def sharpe(self, returns, periods=300):
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(periods)

    def max_drawdown(self, cum_returns):
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

def main(args):

    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {}

    data_pth = config.get("data_pth", "dataset.csv")

    X, y, dates = load_csv(data_pth)

    mask = y != 0
    X, y, dates = X[mask], y[mask], dates[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, dates, split_date="2022-01-01"
    )

    fwd_returns = X_test[:, 8]
    print(fwd_returns)

    # Load pkl file
    path = "../models/" + config.get("model_out", "SVM.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)

    alpha = pred_alpha(model, X_test, y_test, model_type=config.get("model_type", "Random Forest"))

    #print(X_test[:, 0])
    bt = BackTest(alpha=alpha, y_true=y_test, prices=X_test[:, 0], y_true_returns=fwd_returns)

    results = bt.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str)

    args = parser.parse_args()

    main(args)
