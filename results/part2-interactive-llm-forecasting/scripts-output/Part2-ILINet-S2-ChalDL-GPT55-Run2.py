import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
seq_len = 13
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

train_size = 1040
test_size = 261

df = pd.read_csv(r'../../../data/ILINet.csv')
df = df[["DATE", "% WEIGHTED ILI"]].copy()
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

values = df["% WEIGHTED ILI"].astype(float).to_numpy()
train_values = values[:train_size]
test_values = values[train_size:train_size + test_size]

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(series, seq_len):
    x_list = []
    y_list = []
    for i in range(len(series) - seq_len):
        x_list.append(series[i:i + seq_len])
        y_list.append(series[i + seq_len])
    x = np.array(x_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return x, y

def train_model(history):
    torch.manual_seed(42)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    x_train, y_train = make_sequences(np.array(history, dtype=np.float32), seq_len)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
    return model

history = list(train_values.astype(np.float32))
forecasts = []

 for i in range(test_size):
    model = train_model(history)
    x_input = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy().reshape(-1)[0]
    forecasts.append(float(pred))
    history.append(float(test_values[i]))

print(forecasts)