import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use CPU fallback safely.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

feature_cols = ["Heater", "Ice cream"]
target_col = "Ice cream"

train_size = 158
test_size = 40

all_features = df[feature_cols].values.astype(np.float32)
target_index = feature_cols.index(target_col)

seq_len = 6
pred_len = 12
input_size = 2
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(features_scaled, target_scaled):
    X, y = [], []
    n = len(features_scaled)
    for i in range(n - seq_len - pred_len + 1):
        X.append(features_scaled[i:i + seq_len])
        y.append(target_scaled[i + seq_len:i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

forecasts = []

for step in range(test_size):
    current_end = train_size + step
    current_data = all_features[:current_end]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    features_scaled = scaler_x.fit_transform(current_data).astype(np.float32)
    target_scaled = scaler_y.fit_transform(current_data[:, target_index].reshape(-1, 1)).reshape(-1).astype(np.float32)

    X_train, y_train = make_sequences(features_scaled, target_scaled)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(42)
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, pred_len=pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    latest_window = features_scaled[-seq_len:].reshape(1, seq_len, input_size)
    latest_tensor = torch.tensor(latest_window, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(latest_tensor).cpu().numpy().reshape(-1)

    pred_unscaled = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    forecasts.append(float(pred_unscaled[0]))

print(forecasts)