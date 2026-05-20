import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

Set random seeds for reproducibility.

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

Use CUDA if available, otherwise safely fall back to CPU.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/ETTh1.csv')
feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
target_col = "OT"
data = df[feature_cols].to_numpy(dtype=np.float32)

train_size = 13936
test_size = 3484
seq_len = 168
pred_len = 168
input_size = 7
hidden_size = 64
num_layers = 4
kernel_size = 4
dilations = [1, 2, 4, 8, 16, 32]
epochs = 20
batch_size = 32
lr = 0.001

class Chomp1d(nn.Module):
def init(self, chomp_size):
super().init()
self.chomp_size = chomp_size

def forward(self, x):
    if self.chomp_size == 0:
        return x
    return x[:, :, :-self.chomp_size]

class TCN(nn.Module):
def init(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
super().init()
layers = []
in_channels = input_size
for i in range(num_layers):
dilation = dilations[i]
padding = (kernel_size - 1) * dilation
layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size, padding=padding, dilation=dilation))
layers.append(Chomp1d(padding))
layers.append(nn.ReLU())
in_channels = hidden_size
self.network = nn.Sequential(*layers)
self.head = nn.Linear(hidden_size, pred_len)

def forward(self, x):
    x = x.transpose(1, 2)
    h = self.network(x)
    h = h[:, :, -1]
    return self.head(h)

def make_windows(scaled_data):
n = len(scaled_data) - seq_len - pred_len + 1
x = np.empty((n, seq_len, input_size), dtype=np.float32)
y = np.empty((n, pred_len), dtype=np.float32)
target_idx = feature_cols.index(target_col)
for i in range(n):
x[i] = scaled_data[i:i + seq_len]
y[i] = scaled_data[i + seq_len:i + seq_len + pred_len, target_idx]
return x, y

def train_model(available_data):
mean = available_data.mean(axis=0)
std = available_data.std(axis=0)
std[std == 0] = 1.0
scaled = (available_data - mean) / std
x_np, y_np = make_windows(scaled)
x_tensor = torch.from_numpy(x_np).to(device)
y_tensor = torch.from_numpy(y_np).to(device)
model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
model.train()
for _ in range(epochs):
indices = np.arange(len(x_np))
np.random.shuffle(indices)
for start in range(0, len(indices), batch_size):
batch_idx = indices[start:start + batch_size]
xb = x_tensor[batch_idx]
yb = y_tensor[batch_idx]
optimizer.zero_grad()
loss = loss_fn(model(xb), yb)
loss.backward()
optimizer.step()
return model, mean, std

forecasts = []
current_end = train_size

while len(forecasts) < test_size:
available_data = data[:current_end]
model, mean, std = train_model(available_data)
context = data[current_end - seq_len:current_end]
context_scaled = (context - mean) / std
x_input = torch.from_numpy(context_scaled.astype(np.float32)).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
pred_scaled = model(x_input).cpu().numpy().reshape(-1)
target_idx = feature_cols.index(target_col)
pred = pred_scaled * std[target_idx] + mean[target_idx]
remaining = test_size - len(forecasts)
take = min(pred_len, remaining)
forecasts.extend(pred[:take].astype(float).tolist())
current_end += take

print(forecasts)