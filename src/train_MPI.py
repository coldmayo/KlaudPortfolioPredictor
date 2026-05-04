import numpy as np
import pandas as pd
import argparse
import json
import pickle
import os
from mpi4py import MPI

from nn.RF_opt import RForest_MPI 
from nn.RF_skl import RForest_Sklearn
from backtest import pred_alpha, BackTest

def load_csv(pth, silent=False):
    df = pd.read_csv(pth)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    else:
        raise ValueError("Dataset must contain a 'Date' column.")

    if not silent:
        correlation_series = df.drop(columns=["Date", "fwd_ret"]).corr()["target"].sort_values(ascending=False)
        print("Feature Correlations with Target:")
        print(correlation_series.head(10))

    fwd_ret = df["fwd_ret"].values
    
    X = df.drop(columns=["target", "Date", "fwd_ret"]).values.astype(np.float64)
    y = df["target"].values.astype(np.int64)
    dates = df["Date"].values

    return X, y, dates, fwd_ret

def balance_classes(X, y, dates, fwd_ret, random_state=42):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    keep_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = np.sort(rng.choice(cls_indices, size=min_count, replace=False))
        keep_indices.append(chosen)

    keep_indices = np.sort(np.concatenate(keep_indices))
    return X[keep_indices], y[keep_indices], dates[keep_indices], fwd_ret[keep_indices]

def train_test_split(x, y, dates, fwd_ret, split_date, random_state=42):
    mask = dates < np.datetime64(split_date)
    X_train, X_test = x[mask], x[~mask]
    y_train, y_test = y[mask], y[~mask]
    fwd_ret_train, fwd_ret_test = fwd_ret[mask], fwd_ret[~mask]

    return X_train, X_test, y_train, y_test, fwd_ret_train, fwd_ret_test

def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    data_pth = config.get("data_pth", "dataset.csv")

    if rank == 0:
        X, y, dates, fwd_ret = load_csv(data_pth, silent=False)
        #mask = y != 0
        #X, y, dates = X[mask], y[mask], dates[mask]
    else:
        X = y = dates = fwd_ret = None

    X = comm.bcast(X, root=0)
    y = comm.bcast(y, root=0)
    dates  = comm.bcast(dates, root=0)
    fwd_ret = comm.bcast(fwd_ret, root=0)

    X, y, dates, fwd_ret = balance_classes(X, y, dates, fwd_ret)

    X_train, X_test, y_train, y_test, fwd_ret_train, fwd_ret_test = train_test_split(
        X, y, dates, fwd_ret, split_date=config.get("split_date", "2025-05-04")
    )

    model_type = config.get("model_type", "Random Forest MPI")
    
    if model_type == "Random Forest MPI":
        model = RForest_MPI(
            num_trees=config.get("num_trees", 45),
            max_depth=config.get("max_depth", 10),
            min_samples=config.get("min_samples", 2),
            max_features=config.get("max_features", None),
            comm=comm
        )

        if rank == 0:
            print(f"Starting Training: {model.num_trees} trees on {size} ranks...")

        model.fit(X_train, y_train)

        comm.Barrier()

        preds = model.predict(X_test)

        probas = model.predict_probs(X_test)

        if rank == 0:
            unique_preds, pred_counts = np.unique(preds, return_counts=True)
            unique_y, y_counts = np.unique(y_test, return_counts=True)
    
            print(f"True labels distribution: {dict(zip(unique_y, y_counts))}")
            print(f"Predictions distribution: {dict(zip(unique_preds, pred_counts))}")

            acc = np.mean(y_test == preds)
    
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, preds)
            print(f"\nConfusion Matrix:")
            print(f"            Predicted Down  Predicted Up")
            print(f"Actual Down     {cm[0,0]:6d}        {cm[0,1]:6d}")
            print(f"Actual Up       {cm[1,0]:6d}        {cm[1,1]:6d}")
    
            if cm[0,0] + cm[1,1] < cm[0,1] + cm[1,0]:
                print("\n WARNING: Predictions appear INVERTED!")
                print("   Model is predicting the opposite of the true labels.")
                print("   Flipping predictions...")
                preds = 1 - preds
                acc = np.mean(y_test == preds)
                print(f"Corrected Accuracy: {acc:.4f}")
        
                if probas is not None:
                    probas = probas[:, ::-1]
    
            from sklearn.metrics import balanced_accuracy_score, classification_report
            bal_acc = balanced_accuracy_score(y_test, preds)
            print(f"\nBalanced Accuracy: {bal_acc:.4f}")
            print(f"Standard Accuracy: {acc:.4f}")
    
            print("\nClassification Report:")
            print(classification_report(y_test, preds, target_names=['Down (-1)', 'Sideways (0)', 'Up (1)']))

        if rank == 0:
            print(f"\nFinal Validation Accuracy: {acc:.4f}")
            
            if "model_out" in config:
                path = "../models/"
                os.makedirs(path, exist_ok=True)
                full_path = os.path.join(path, config["model_out"])
                with open(full_path, "wb") as f:
                    pickle.dump(model, f)
                print(f"Model saved to {full_path}")

            print("Starting Backtesting...")
            alpha = pred_alpha(model, X_test, y_test, model_type="Random Forest", proba = probas)
            bt = BackTest(alpha=alpha, y_true=y_test, prices=X_test[:, 0], y_true_returns=fwd_ret_test)
            results = bt.run()

    elif model_type == "Random Forest Skl":
        model = RForest_Sklearn(
            num_trees=config.get("num_trees", 45),
            max_depth=config.get("max_depth", 10),
            min_samples=config.get("min_samples", 2),
            max_features=config.get("max_features", None),
            comm=comm
        )

        if rank == 0:
            print(f"Starting Training: {model.num_trees} trees on {size} ranks...")

        model.fit(X_train, y_train)

        comm.Barrier()

        preds = model.predict(X_test)

        probas = model.predict_probs(X_test)

        if rank == 0:
            unique_preds, pred_counts = np.unique(preds, return_counts=True)
            unique_y, y_counts = np.unique(y_test, return_counts=True)
    
            print(f"True labels distribution: {dict(zip(unique_y, y_counts))}")
            print(f"Predictions distribution: {dict(zip(unique_preds, pred_counts))}")

            acc = np.mean(y_test == preds)
    
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, preds)
            print(f"\nConfusion Matrix:")
            print(f"            Predicted Down  Predicted Up")
            print(f"Actual Down     {cm[0,0]:6d}        {cm[0,1]:6d}")
            print(f"Actual Up       {cm[1,0]:6d}        {cm[1,1]:6d}")
    
            if cm[0,0] + cm[1,1] < cm[0,1] + cm[1,0]:
                print("\n WARNING: Predictions appear INVERTED!")
                print("   Model is predicting the opposite of the true labels.")
                print("   Flipping predictions...")
                preds = 1 - preds
                acc = np.mean(y_test == preds)
                print(f"Corrected Accuracy: {acc:.4f}")
        
                if probas is not None:
                    probas = probas[:, ::-1]
    
            from sklearn.metrics import balanced_accuracy_score, classification_report
            bal_acc = balanced_accuracy_score(y_test, preds)
            print(f"\nBalanced Accuracy: {bal_acc:.4f}")
            print(f"Standard Accuracy: {acc:.4f}")
    
            print("\nClassification Report:")
            print(classification_report(y_test, preds, target_names=['Down (-1)', 'Sideways (0)', 'Up (1)']))

    comm.Barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()
    main(args)