import numpy as np
import pandas as pd
import argparse
import json
from nn.RF import *
from nn.SVM import *

def train_test_split(x, y, dates, split_date):
    mask = dates < np.datetime64(split_date)

    X_train = x[mask]
    X_test = x[~mask]

    y_train = y[mask]
    y_test = y[~mask]

    return X_train, X_test, y_train, y_test

def load_csv(pth):
    df = pd.read_csv(pth)

    # Ensure Date exists and is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    else:
        raise ValueError("Dataset must contain a 'Date' column for time-based split.")

    # Split features / target
    X = df.drop(columns=["target", "Date"]).values
    y = df["target"].values
    dates = df["Date"].values

    X = X.astype(np.float64)
    y = y.astype(np.int64)

    return X, y, dates

def accuracy(y_test, preds):
    return np.mean(y_test == preds)

def signal_accuracy(y_true, y_pred):
    mask = y_pred != 0

    if np.sum(mask) == 0:
        return 0.0

    return np.mean(y_true[mask] == y_pred[mask])

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

    # Takes about 4 minuites to run on the chud machine (with defualt config)
    if config.get("model_type", "Random Forest") == "Random Forest":
        model = RForest(
            num_trees=config.get("num_trees", 10),
            max_depth=config.get("max_depth", 10),
            min_samples=config.get("min_samples", 2),
            max_features=config.get("max_features", None)
        )

        print("Training Random Forest...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy(y_test, preds)
        sig_acc = signal_accuracy(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Signal Accuracy: {sig_acc:.4f}")

    # Can train on my chud laptop for a neglagble amount of time
    elif config.get("model_type", "SVM") == "SVM":
        model = SVM(
            lr = config.get("lr", 0.001),
            lambda_p = config.get("lambda", 0.01),
            iters = config.get("iters", 100)
        )

        print("Training SVM...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy(y_test, preds)
        sig_acc = signal_accuracy(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Signal Accuracy: {sig_acc:.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str)
    #parser.add_argument("-e", "--epochs", type=int)

    args = parser.parse_args()

    main(args)
