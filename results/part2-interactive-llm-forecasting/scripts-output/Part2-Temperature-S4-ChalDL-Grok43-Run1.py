import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values.astype(float)
non_zero = target[target != 0]
median_non_zero = np.median(non_zero)
target = np.where(target == 0, median_non_zero, target)
train_size = 2920
train = target[:train_size]
test = target[train_size:]
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
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001
X_train, y_train = create_sequences(train, seq_len, pred_len)
X_train = X_train.reshape(-1, seq_len, input_size)
y_train = y_train.reshape(-1, pred_len)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = LSTMModel(input_size, hidden_size, num_layers, pred_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
forecasts = []
current_data = list(train)
test_index = 0
while test_index < len(test):
    input_seq = np.array(current_data[-seq_len:]).reshape(1, seq_len, input_size)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32)
    with torch.no_grad():
        pred = model(input_tensor).numpy().flatten()
    remaining = len(test) - test_index
    steps = min(pred_len, remaining)
    block_forecast = pred[:steps]
    forecasts.extend(block_forecast.tolist())
    true_block = test[test_index:test_index + steps]
    current_data.extend(true_block)
    test_index += steps
    X_new, y_new = create_sequences(np.array(current_data), seq_len, pred_len)
    if len(X_new) > 0:
        X_new = X_new.reshape(-1, seq_len, input_size)
        y_new = y_new.reshape(-1, pred_len)
        X_new = torch.tensor(X_new, dtype=torch.float32)
        y_new = torch.tensor(y_new, dtype=torch.float32)
        dataset_new = TensorDataset(X_new, y_new)
        loader_new = DataLoader(dataset_new, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            for batch_x, batch_y in loader_new:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
print(forecasts)