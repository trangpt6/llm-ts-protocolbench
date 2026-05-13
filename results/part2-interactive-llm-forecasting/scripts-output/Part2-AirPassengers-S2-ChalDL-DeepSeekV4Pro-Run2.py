import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# read data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
target_col = "Passengers"
series = df[target_col].values.astype(np.float32)

# fixed split from Turn 0
train_size = 115
train_series = series[:train_size]
test_series = series[train_size:]

# hyperparameters (fixed)
seq_len = 12
input_size = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01
pred_len = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last time step output
        out = self.linear(out)
        return out

def create_sequences(data, seq_len):
    # data: 1D numpy array
    if len(data) <= seq_len:
        return torch.empty(0, seq_len), torch.empty(0, 1)  # not enough data
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

# expanding window forecast with retraining at every step
history = train_series.copy()  # 1D numpy array
forecasts = []

for step in range(len(test_series)):
    true_next = test_series[step]
    
    # prepare training data from all history available so far
    X_train, y_train = create_sequences(history, seq_len)
    
    if len(X_train) == 0:
        # fallback: not enough history to form a sequence, skip (should not happen with train_size>seq_len)
        forecasts.append(0.0)
        history = np.append(history, true_next)
        continue
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # re-initialize model and train from scratch
    model = GRUModel(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = batch_X.unsqueeze(-1)  # (batch, seq_len, input_size)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # predict next step
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(history[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)
        pred = model(last_seq).cpu().item()
    forecasts.append(pred)
    
    # update history with ground truth (rolling update with retraining)
    history = np.append(history, true_next)

print(forecasts)