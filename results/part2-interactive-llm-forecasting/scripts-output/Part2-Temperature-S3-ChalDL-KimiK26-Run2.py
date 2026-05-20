import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# load and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# extract target series
series = df['Daily minimum temperatures'].values.astype(np.float32)

# chronological split: first 80% train
train_size = 2920
train_series = series[:train_size]
test_series = series[train_size:]

# fixed hyperparameters
input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01
device = torch.device('cpu')

# define GRU model
class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# helper to build sliding windows for multi-step supervised learning
def make_sequences(data, seq_len, pred_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:(i + seq_len)]
        y = data[(i + seq_len):(i + seq_len + pred_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# rolling forecast: retrain at every step, use ground truth after each prediction
forecasts = []
history = train_series.tolist()

for step in range(len(test_series)):
    data = np.array(history, dtype=np.float32)
    X, y = make_sequences(data, seq_len, pred_len)

    X_tensor = torch.from_numpy(X).unsqueeze(-1).to(device)
    y_tensor = torch.from_numpy(y).to(device)

    model = GRUModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_tensor[idx]
            batch_y = y_tensor[idx]
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = torch.from_numpy(data[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(last_seq).cpu().numpy().flatten()

    # append the one-step-ahead component to cover the test set point-by-point
    forecasts.append(float(pred[0]))

    # update history with observed true value (ground truth enabled)
    history.append(float(test_series[step]))

print(forecasts)