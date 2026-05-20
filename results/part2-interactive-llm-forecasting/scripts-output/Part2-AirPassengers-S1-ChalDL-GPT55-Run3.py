import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Use CPU fallback if CUDA is unavailable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
train_values = values[:train_size]

input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 64
num_layers = 2
epochs = 50
batch_size = 16
lr = 0.001

X = []
y = []
for i in range(len(train_values) - seq_len - pred_len + 1):
    X.append(train_values[i:i + seq_len])
    y.append(train_values[i + seq_len:i + seq_len + pred_len])

X = np.array(X, dtype=np.float32).reshape(-1, seq_len, input_size)
y = np.array(y, dtype=np.float32).reshape(-1, pred_len)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
generator = torch.Generator()
generator.manual_seed(42)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

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

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, pred_len=pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for _ in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

model.eval()
history = list(train_values.astype(float))
forecasts = []
with torch.no_grad():
    for _ in range(test_size):
        x_input = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, input_size)
        x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
        pred = model(x_tensor).cpu().numpy().reshape(-1)[0].item()
        forecasts.append(pred)
        history.append(pred)

print(forecasts)