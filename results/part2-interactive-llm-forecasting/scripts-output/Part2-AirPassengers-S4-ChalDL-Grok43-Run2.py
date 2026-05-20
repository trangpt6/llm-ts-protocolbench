import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(float)
train_size = 115
known = data[:train_size].tolist()
test = data[train_size:].tolist()
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
def prepare_data(series, seq_len, pred_len):
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
def train_model(series, seq_len, pred_len, hidden_size, num_layers, epochs, batch_size, lr):
    if len(series) < seq_len + pred_len:
        return None
    X, y = prepare_data(series, seq_len, pred_len)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMModel(1, hidden_size, num_layers, pred_len)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model
def predict(model, last_seq, seq_len, pred_len):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(last_seq[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model(x)
        return pred.squeeze().tolist()
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001
forecasts = []
idx = 0
current_known = known.copy()
while idx < len(test):
    model = train_model(current_known, seq_len, pred_len, hidden_size, num_layers, epochs, batch_size, lr)
    if model is None:
        break
    pred = predict(model, current_known, seq_len, pred_len)
    remaining = len(test) - idx
    to_add = min(len(pred), remaining)
    forecasts.extend(pred[:to_add])
    true_block = test[idx:idx + to_add]
    current_known.extend(true_block)
    idx += to_add
print(forecasts)