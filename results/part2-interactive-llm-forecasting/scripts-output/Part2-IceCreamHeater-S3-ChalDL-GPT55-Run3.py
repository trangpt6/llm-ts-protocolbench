import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

train_size = 158
test_size = 40
feature_cols = ["Heater", "Ice cream"]
target_col = "Ice cream"

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

def make_sequences(data):
    values_x = data[feature_cols].to_numpy(dtype=np.float32)
    values_y = data[target_col].to_numpy(dtype=np.float32)
    xs = []
    ys = []
    max_start = len(data) - seq_len - pred_len + 1
    for start in range(max_start):
        xs.append(values_x[start:start + seq_len])
        ys.append(values_y[start + seq_len:start + seq_len + pred_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

forecasts = []

 for_step_range = range(test_size)
for step in for_step_range:
    available = df.iloc[:train_size + step].copy()
    x_train, y_train = make_sequences(available)

    model = LSTMForecaster(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, pred_len=pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

    recent_x = available[feature_cols].to_numpy(dtype=np.float32)[-seq_len:]
    recent_x = torch.tensor(recent_x, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_12 = model(recent_x).cpu().numpy().reshape(-1)
    forecasts.append(float(pred_12[0]))

print(forecasts)