import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['OT'] = df['OT'].clip(lower=0)
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size].reset_index(drop=True)
test_df = df.iloc[train_size:].reset_index(drop=True)
for lag in range(1, 8):
    train_df[f'lag{lag}'] = train_df['OT'].shift(lag)
    test_df[f'lag{lag}'] = test_df['OT'].shift(lag)
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)
feature_cols = [f'lag{lag}' for lag in range(1, 8)]
seq_len = 12
pred_len = 24
def create_sequences(data, feature_cols, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[feature_cols].iloc[i:i+seq_len].values)
        y.append(data['OT'].iloc[i+seq_len:i+seq_len+pred_len].values)
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_df, feature_cols, seq_len, pred_len)
X_test, y_test = create_sequences(test_df, feature_cols, seq_len, pred_len)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
model = LSTMModel(7, 32, 1, 24)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
forecasts = []
history = train_df.copy()
for i in range(len(test_df) - seq_len - pred_len + 1):
    model.eval()
    with torch.no_grad():
        last_seq = history[feature_cols].iloc[-seq_len:].values
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)
        pred = model(last_seq).numpy().flatten()
        forecasts.extend(pred.tolist())
    true_val = test_df['OT'].iloc[i + seq_len]
    new_row = test_df.iloc[i + seq_len].copy()
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    for lag in range(1, 8):
        history[f'lag{lag}'] = history['OT'].shift(lag)
    history = history.dropna().reset_index(drop=True)
    model.train()
    X_upd, y_upd = create_sequences(history, feature_cols, seq_len, pred_len)
    if len(X_upd) > 0:
        X_upd = torch.tensor(X_upd, dtype=torch.float32)
        y_upd = torch.tensor(y_upd, dtype=torch.float32)
        upd_dataset = TensorDataset(X_upd, y_upd)
        upd_loader = DataLoader(upd_dataset, batch_size=32, shuffle=True)
        for epoch in range(3):
            for xb, yb in upd_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
print(forecasts)