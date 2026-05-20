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
train = target[:train_size]
test = target[train_size:]
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01
device = torch.device('cpu')
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
forecasts = []
current_train = list(train)
for i in range(len(test)):
    if len(current_train) >= seq_len + pred_len:
        X_train, y_train = create_sequences(np.array(current_train), seq_len, pred_len)
        X_train = torch.tensor(X_train).unsqueeze(-1).to(device)
        y_train = torch.tensor(y_train).to(device)
        model = GRUModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
    else:
        model = GRUModel().to(device)
    model.eval()
    last_seq = torch.tensor(current_train[-seq_len:]).unsqueeze(0).unsqueeze(-1).float().to(device)
    with torch.no_grad():
        pred = model(last_seq).cpu().numpy().flatten()
    forecasts.append(float(pred[0]))
    current_train.append(test[i])
print(forecasts)