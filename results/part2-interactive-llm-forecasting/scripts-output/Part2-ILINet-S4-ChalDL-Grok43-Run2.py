import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values.astype(np.float32)
train_size = 1044
train_data = target[:train_size]
test_data = target[train_size:]
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
def train_lstm(train_series):
    model = LSTMModel(1, hidden_size, num_layers, pred_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X, y = create_sequences(train_series, seq_len, pred_len)
    if len(X) == 0:
        return model
    X = torch.tensor(X).float().unsqueeze(-1)
    y = torch.tensor(y).float()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model
forecasts = []
current_train = list(train_data)
i = 0
while i < len(test_data):
    model = train_lstm(np.array(current_train))
    model.eval()
    last_seq = torch.tensor(current_train[-seq_len:]).float().unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(last_seq).numpy().flatten()
    steps = min(pred_len, len(test_data) - i)
    forecasts.extend(pred[:steps].tolist())
    current_train.extend(test_data[i:i+steps])
    i += steps
print(forecasts)