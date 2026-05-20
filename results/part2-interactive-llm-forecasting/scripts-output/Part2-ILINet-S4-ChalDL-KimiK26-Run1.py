import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
target = df['% WEIGHTED ILI'].values.astype(np.float32)

train_size = 1040
train_data = target[:train_size]
test_data = target[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, seq_len)

train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(-1).to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

forecasts = []
current_window = train_scaled[-seq_len:].tolist()
remaining = len(test_scaled)
start_idx = 0

while start_idx < remaining:
    # Forecast next pred_len steps
    input_seq = torch.tensor(current_window[-seq_len:]).float().unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(input_seq).cpu().numpy().flatten()

    actual_block_len = min(pred_len, remaining - start_idx)
    pred_block = pred[:actual_block_len]

    # Inverse transform
    pred_block = scaler.inverse_transform(pred_block.reshape(-1, 1)).flatten().tolist()
    forecasts.extend(pred_block)

    # Use ground truth to update window
    true_block = test_scaled[start_idx:start_idx+actual_block_len]
    current_window.extend(true_block.tolist())
    if len(current_window) > seq_len:
        current_window = current_window[-seq_len:]

    start_idx += actual_block_len

    # Retrain model on the extended training set (including observed true values)
    # New training data = old train_scaled + observed test_scaled up to start_idx
    extended_train = np.concatenate([train_scaled, test_scaled[:start_idx]])
    X_new, y_new = create_sequences(extended_train, seq_len)
    new_dataset = TensorDataset(torch.tensor(X_new).float(), torch.tensor(y_new).float())
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in new_loader:
            X_batch = X_batch.unsqueeze(-1).to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

print(forecasts)