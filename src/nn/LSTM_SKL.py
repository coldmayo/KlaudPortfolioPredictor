import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, inp_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.model = nn.LSTM(inp_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, h0 = None, c0 = None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.model(x, (h0, c0))
        return self.fc(out[:, -1, :])

    def train(self, loader, opt, crit, device):
        self.model.train()
        total_loss, correct = 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)          # forward pass
            loss = criterion(output, y_batch)
            loss.backward()                  # autograd computes all gradients
        
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevents exploding gradients, important for LSTMs
        
            optimizer.step()                 # update weights

            total_loss += loss.item()
            correct += (output.argmax(dim=1) == y_batch).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    def fit(self, loader, crit, device):
        model.eval()
        total_loss, correct = 0, 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                total_loss += loss.item()
                correct += (output.argmax(dim=1) == y_batch).sum().item()
                
        return total_loss / len(loader), correct / len(loader.dataset)
