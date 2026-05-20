import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

Set random seeds for reproducibility.

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

Use CPU fallback if CUDA is unavailable.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/ETTh1.csv')
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
target_col = "OT"
target_idx = feature_cols.index(target_col)

train_size = 13936
test_size = 3484
seq_len = 12
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

values = df[feature_cols].astype(np.float32).values
test_values = values[train_size:train_size + test_size]

class GRUModel(nn.Module):
def init(self):
super().init()
self.gru = nn.GRU(input_size=7, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
self.fc = nn.Linear(hidden_size, 1)

def forward(self, x):
    out, _ = self.gru(x)
    out = out[:, -1, :]
    return self.fc(out)

def make_windows(arr):
x_list = []
y_list = []
for i in range(len(arr) - seq_len):
x_list.append(arr[i:i + seq_len])
y_list.append(arr[i + seq_len, target_idx])
x = torch.tensor(np.array(x_list), dtype=torch.float32)
y = torch.tensor(np.array(y_list), dtype=torch.float32).view(-1, 1)
return x, y

def train_model(history):
model = GRUModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
x_train, y_train = make_windows(history)
dataset = TensorDataset(x_train, y_train)
generator = torch.Generator()
generator.manual_seed(42)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
model.train()
for _ in range(epochs):
for xb, yb in loader:
xb = xb.to(device)
yb = yb.to(device)
optimizer.zero_grad()
loss = criterion(model(xb), yb)
loss.backward()
optimizer.step()
return model

history = values[:train_size].copy()
forecasts = []

for step in range(test_size):
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
model = train_model(history)
x_pred = torch.tensor(history[-seq_len:].reshape(1, seq_len, 7), dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
pred = model(x_pred).cpu().numpy().reshape(-1)[0].item()
forecasts.append(float(pred))
history = np.vstack([history, test_values[step]])

print(forecasts)