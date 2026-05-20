import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
# set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values.astype(np.float32)
train_size = 1043
train = target[:train_size]
test = target[train_size:]
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001
device = torch.device('cpu')
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
forecasts = []
current_train = train.copy()
for start in range(0, len(test), pred_len):
    block = test[start:start+pred_len]
    if len(block) < pred_len:
        block = np.pad(block, (0, pred_len - len(block)), 'constant', constant_values=0)
    X_train, y_train = create_sequences(current_train, seq_len, pred_len)
    if len(X_train) == 0:
        X_train = np.zeros((1, seq_len, 1), dtype=np.float32)
        y_train = np.zeros((1, pred_len), dtype=np.float32)
    X_train = torch.tensor(X_train.reshape(-1, seq_len, 1), dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    last_seq = torch.tensor(current_train[-seq_len:].reshape(1, seq_len, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_block = model(last_seq).cpu().numpy().flatten()[:len(block)]
    forecasts.extend(pred_block.tolist())
    current_train = np.concatenate([current_train, block])
forecasts = forecasts[:261]
print(forecasts)