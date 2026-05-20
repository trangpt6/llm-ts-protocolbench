import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['OT'] = df['OT'].clip(lower=0)
train_size = 14016
train = df['OT'].iloc[:train_size].values.tolist()
test = df['OT'].iloc[train_size:].values.tolist()
input_size = 7
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
def create_sequences(data, seq_len, input_size):
    X, y = [], []
    for i in range(len(data) - seq_len - input_size + 1):
        seq = [data[i+j:i+j+input_size] for j in range(seq_len)]
        X.append(seq)
        y.append(data[i+seq_len+input_size-1])
    return np.array(X), np.array(y)
current_data = train[:]
forecasts = []
for i in range(len(test)):
    if len(current_data) < seq_len + input_size:
        forecasts.append(current_data[-1])
        current_data.append(test[i])
        continue
    X_train, y_train = create_sequences(current_data, seq_len, input_size)
    model = GRUModel(input_size, hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = current_data[-(seq_len + input_size - 1):]
    pred_input = [last_seq[j:j+input_size] for j in range(seq_len)]
    pred_input = torch.tensor([pred_input], dtype=torch.float32)
    with torch.no_grad():
        pred = model(pred_input).item()
    forecasts.append(pred)
    current_data.append(test[i])
print(forecasts)