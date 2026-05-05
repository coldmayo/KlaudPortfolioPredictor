import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

from nn.LSTM import LSTMClassifier

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # long for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main(args):
    X = np.load("X.npy")
    y = np.load("y.npy")

    y = y+1   # map from -1, 0, 1 to 0, 1, 2

    # DO NOT SHUFFLE THIS IS TEMPORAL DATA
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=False)
    val_loader   = DataLoader(StockDataset(X_val, y_val),   batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(inp_size=X.shape[2], hidden_size=128, num_layers=2, out_size=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:02d}"
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-c", "--config", type=str)
    args = parser.parse_args()

    main(args)
