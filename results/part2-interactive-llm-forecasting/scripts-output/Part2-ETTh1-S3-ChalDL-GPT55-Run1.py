import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

Set random seeds for reproducibility

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

Use CPU if CUDA is unavailable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/ETTh1.csv')
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in df.columns if c != "date"]
target_col = "OT"
target_idx = feature_cols.index(target_col)

train_size = 13936
test_size = 3484

values = df[feature_cols].astype(float).values
train_values = values[:train_size]
test_values = values[train_size:train_size + test_size]

input_size = 7
seq_len = 12
pred_len = 24
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

class LSTMForecaster(nn.Module):
def init(self, input_size, hidden_size, num_layers, pred_len):
super().init()
self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
self.fc = nn.Linear(hidden_size, pred_len)

def forward(self, x):
    out, _ = self.lstm(x)
    out = out[:, -1, :]
    return self.fc(out)

def make_windows(arr, seq_len, pred_len, target_idx):
x_list = []
y_list = []
max_start = len(arr) - seq_len - pred_len + 1
for i in range(max_start):
x_list.append(arr[i:i + seq_len])
y_list.append(arr[i + seq_len:i + seq_len + pred_len, target_idx])
return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)

def train_and_predict(history_values):
scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaled_history = scaler_x.fit_transform(history_values)
y_history = history_values[:, target_idx:target_idx + 1]
scaler_y.fit(y_history)

x_train, y_train_raw = make_windows(history_values, seq_len, pred_len, target_idx)
x_train_scaled, _ = make_windows(scaled_history, seq_len, pred_len, target_idx)
y_train_scaled = scaler_y.transform(y_train_raw.reshape(-1, 1)).reshape(y_train_raw.shape)

x_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = LSTMForecaster(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, pred_len=pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

model.train()
for _ in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

last_window = history_values[-seq_len:]
last_window_scaled = scaler_x.transform(last_window)
x_input = torch.tensor(last_window_scaled.reshape(1, seq_len, input_size), dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    pred_scaled = model(x_input).cpu().numpy().reshape(-1, 1)

pred = scaler_y.inverse_transform(pred_scaled).reshape(-1)
return float(pred[0])

forecasts = []
history = train_values.copy()

for i in range(test_size):
y_pred = train_and_predict(history)
forecasts.append(y_pred)
history = np.vstack([history, test_values[i:i + 1]])

print(forecasts)