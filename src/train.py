import numpy as np
import pandas as pd
import argparse
import json
from nn.RF import *

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

    return X, y, dates

def accuracy(y_test, preds):
    return np.mean(y_true == y_pred)

def main(args):

    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {}

    data_pth = config["data_pth"]
    X, y, dates = load_csv(data_pth)

    X_train, X_test, y_train, y_test = train_test_split(X, y, dates, split_date="2022-01-01")

    if config["model_type"] == "Random Forest":
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

        print(f"Accuracy: {acc:.4f}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str)
    #parser.add_argument("-e", "--epochs", type=int)

    args = parser.parse_args()

    main(args)
