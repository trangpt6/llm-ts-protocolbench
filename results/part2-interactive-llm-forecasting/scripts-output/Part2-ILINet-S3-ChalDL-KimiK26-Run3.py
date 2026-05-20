import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cpu')

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)
target = df['% WEIGHTED ILI'].values.astype(np.float32)

total = len(target)
train_size = int(0.8 * total)
test_size = total - train_size
train_data = target[:train_size]
test_data = target[train_size:]

seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
lr = 0.01
epochs = 5
batch_size = 16
input_size = 1

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

def train_model(X_train, y_train):
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_train, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

# initial training on entire train set
X_train_init, y_train_init = create_sequences(train_data, seq_len)
model = train_model(X_train_init, y_train_init)

# rolling forecast over test set
current_data = list(train_data)
forecasts = []
for t in range(test_size):
    # use last seq_len points to predict pred_len steps
    input_seq = np.array(current_data[-seq_len:]).reshape(1, seq_len, 1)
    with torch.no_grad():
        pred = model(torch.tensor(input_seq, dtype=torch.float32).to(device))
    pred = pred.cpu().numpy().flatten()
    # store first step forecast (one-step ahead)
    forecasts.append(pred[0])
    # update current_data with true test value
    true_val = test_data[t]
    current_data.append(true_val)
    # retrain model with all available data (include the new true value)
    # we need to rebuild the sequences from the entire current_data up to now
    # careful: we must have enough for seq_len+pred_len
    if len(current_data) >= seq_len + pred_len:
        X_curr, y_curr = create_sequences(np.array(current_data), seq_len)
        model = train_model(X_curr, y_curr)

print(forecasts)