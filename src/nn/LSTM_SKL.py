import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # long for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, inp_size, hidden_size, num_layers, out_size, dropout):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.dropout = 0
        
        if dropout is not None:
            self.dropout = dropout

        self.model = nn.LSTM(inp_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, h0 = None, c0 = None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.model(x, (h0, c0))
        return self.fc(out[:, -1, :])

    def predict(self, X_or_loader, device, y_test=None):
        super().train(False)

        if isinstance(X_or_loader, np.ndarray):
            X_tensor = torch.tensor(X_or_loader, dtype=torch.float32)
            loader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=False)
        else:
            loader = X_or_loader
        
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(device)
                output = self(X_batch)
                preds = torch.argmax(output, dim=1).cpu().numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds)

    def fit(self, loader, opt, crit, device):
        super().train()
        total_loss, correct = 0, 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            opt.zero_grad()
            output = self(X_batch)          # forward pass
            loss = crit(output, y_batch)
            loss.backward()                  # autograd computes all gradients
        
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # prevents exploding gradients, important for LSTMs
        
            opt.step()                 # update weights

            total_loss += loss.item()
            correct += (output.argmax(dim=1) == y_batch).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    def evaluate(self, loader, crit, device):
        super().train(False)
        total_loss, correct = 0, 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = self(X_batch)
                loss = crit(output, y_batch)
                total_loss += loss.item()
                correct += (output.argmax(dim=1) == y_batch).sum().item()
                
        return total_loss / len(loader), correct / len(loader.dataset)
