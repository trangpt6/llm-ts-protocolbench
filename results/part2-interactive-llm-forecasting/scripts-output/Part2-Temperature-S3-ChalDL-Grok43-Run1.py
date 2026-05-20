import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
def train_model(model, history, seq_len, pred_len, epochs, batch_size, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_list = []
    y_list = []
    for i in range(len(history) - seq_len - pred_len + 1):
        X_list.append(history[i:i+seq_len])
        y_list.append(history[i+seq_len:i+seq_len+pred_len])
    if len(X_list) == 0:
        return
    X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y_list, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
data = pd.read_csv(r'../../../data/Temperature.csv')
temps = data['Daily minimum temperatures'].values.astype(float)
train_temps = temps[:2920]
non_zero = train_temps[train_temps != 0]
median_val = np.median(non_zero) if len(non_zero) > 0 else 0.0
temps = np.where(temps == 0, median_val, temps)
train_size = 2920
history = temps[:train_size].tolist()
test = temps[train_size:].tolist()
forecasts = []
model = GRUForecaster(1, 16, 1, 7)
for step in range(len(test)):
    train_model(model, history, 7, 7, 5, 16, 0.01)
    if len(history) >= 7:
        input_seq = torch.tensor(history[-7:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(input_seq).squeeze().tolist()
        next_forecast = pred[0]
    else:
        next_forecast = history[-1]
    forecasts.append(next_forecast)
    history.append(test[step])
print(forecasts)