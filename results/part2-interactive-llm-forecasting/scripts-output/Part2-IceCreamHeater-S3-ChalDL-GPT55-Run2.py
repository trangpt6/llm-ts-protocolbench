import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]

train_size = 158
test_size = 40
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:train_size + test_size].copy()

params = {'input_size': 2, 'seq_len': 6, 'pred_len': 12, 'hidden_size': 16, 'num_layers': 1, 'epochs': 5, 'batch_size': 8, 'lr': 0.01}

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(data_values, target_values, seq_len, pred_len):
    X, y = [], []
    n = len(data_values)
    for i in range(n - seq_len - pred_len + 1):
        X.append(data_values[i:i + seq_len])
        y.append(target_values[i + seq_len:i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_and_predict(history_df):
    feature_values = history_df[feature_cols].values.astype(np.float32)
    target_values = history_df[target_col].values.astype(np.float32)

    feature_mean = feature_values.mean(axis=0)
    feature_std = feature_values.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    target_mean = target_values.mean()
    target_std = target_values.std()
    if target_std == 0:
        target_std = 1.0

    scaled_features = (feature_values - feature_mean) / feature_std
    scaled_target = (target_values - target_mean) / target_std

    X, y = make_sequences(scaled_features, scaled_target, params["seq_len"], params["pred_len"])

    model = LSTMForecaster(params["input_size"], params["hidden_size"], params["num_layers"], params["pred_len"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)

    model.train()
    for _ in range(params["epochs"]):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    last_seq = scaled_features[-params["seq_len"]:]
    x_input = torch.tensor(last_seq.reshape(1, params["seq_len"], params["input_size"]), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_input).cpu().numpy().reshape(-1)

    pred = pred_scaled * target_std + target_mean
    return float(pred[0])

history_df = train_df.copy()
forecasts = []

for i in range(test_size):
    yhat = train_and_predict(history_df)
    forecasts.append(yhat)
    history_df = pd.concat([history_df, test_df.iloc[[i]]], ignore_index=True)

print(forecasts)