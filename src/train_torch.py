import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import argparse
import json
import pickle
import os

from nn.LSTM_SKL import LSTMClassifier, StockDataset
from backtest import pred_alpha, BackTest

def main(args):
    
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
            
    X_pth = config.get("X_pth", "X.npy")
    y_pth = config.get("y_pth", "y.npy")
    hidden_size = config.get("hidden_size", 128)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.2)
    
    X = np.load(X_pth)
    y = np.load(y_pth)
    fwd_ret = np.load("fwd_ret.npy")
    prices = np.load("prices.npy")

    y = y+1   # map from -1, 0, 1 to 0, 1, 2

    # DO NOT SHUFFLE THIS IS TEMPORAL DATA
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    fwd_ret_train, fwd_ret_test = fwd_ret[:split], fwd_ret[split:]
    prices_train, prices_test = prices[:split], prices[split:]

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=False)
    val_loader   = DataLoader(StockDataset(X_val, y_val),   batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(inp_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers, out_size=3, dropout = dropout if num_layers > 1 else 0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Class weights
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(args.epochs):
        train_loss, train_acc = model.fit(train_loader, optimizer, criterion, device)
        val_loss, val_acc = model.evaluate(val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:02d}\n"
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} and "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    print("Starting Backtesting...")
    alpha = pred_alpha(model, X_val, y_val, model_type="LSTM")
    y_val_orig = y_val - 1
    for strat in ["sign", "threshold", "topk", "volscaled"]:
        print(f"Testing {strat} Strategy")
        bt = BackTest(strat = strat, alpha=alpha, y_true=y_val_orig, prices=prices_test, y_true_returns=fwd_ret_test)
        results = bt.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()

    main(args)