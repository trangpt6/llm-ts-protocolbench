import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
values = df[["Heater", "Ice cream"]].astype(np.float32).values

train_size = 158
test_size = 40

input_size = 2
seq_len = 6
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(data):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len, :])
        y.append(data[i + seq_len, 1])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y

forecasts = []

for step in range(test_size):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    end_idx = train_size + step
    history = values[:end_idx, :]

    X_train, y_train = make_sequences(history)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = GRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    x_next = history[-seq_len:, :].reshape(1, seq_len, input_size)
    x_next_tensor = torch.tensor(x_next, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        forecast = model(x_next_tensor).cpu().numpy().reshape(-1)[0]

    forecasts.append(float(forecast))

print(forecasts)