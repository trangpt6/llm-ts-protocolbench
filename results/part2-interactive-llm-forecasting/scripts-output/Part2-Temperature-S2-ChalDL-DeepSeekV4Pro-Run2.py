import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
df = df.set_index('Date').sort_index()
df.columns = ['temp']

# Preprocess: insert missing 1988-12-31 if not present
if pd.Timestamp('1988-12-31') not in df.index:
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D'))
    df['temp'] = df['temp'].interpolate(method='linear')

# Chronological split based on fixed dates
train_end = pd.Timestamp('1988-12-29')
test_start = pd.Timestamp('1988-12-30')
train = df[:train_end].copy()
test = df[test_start:].copy()
assert len(train) == 2920
assert len(test) == 732  # After imputation

# Normalization: global min-max to avoid leakage, using only train data
tmin = train['temp'].min()
tmax = train['temp'].max()

def scale(s):
    return (s - tmin) / (tmax - tmin) if tmax != tmin else s

def inverse_scale(s_scaled):
    return s_scaled * (tmax - tmin) + tmin

train_scaled = scale(train['temp']).values
test_true = test['temp'].values

# Sliding window creation
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

# GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # take last output
        last = out[:, -1, :]
        return self.fc(last)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Rolling forecast with retraining at each step
forecasts = []
history_scaled = list(train_scaled)  # will be updated with true test values

for t in range(len(test_true)):
    # Prepare current history array
    hist_arr = np.array(history_scaled + list(test_true[:t]))  # true values up to t-1
    # Need at least seq_len+1 points
    seq_len = 7
    if len(hist_arr) >= seq_len+1:
        X, y = create_sequences(hist_arr, seq_len)
    else:
        # Not enough data, fallback to last value (should not happen because train has 2920>7)
        forecasts.append(history_scaled[-1])
        continue

    # Train model
    model = GRUModel(input_size=1, hidden_size=16, num_layers=1, output_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    batch_size = 16

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)  # (N, seq_len, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)

    model.train()
    for epoch in range(5):
        permutation = torch.randperm(len(X_tensor))
        for i in range(0, len(X_tensor), batch_size):
            indices = permutation[i:i+batch_size]
            xb = X_tensor[indices]
            yb = y_tensor[indices]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # Predict next
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(hist_arr[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)
        pred_scaled = model(last_seq).item()
    forecasts.append(pred_scaled)

# Inverse transform forecasts
forecasts = inverse_scale(np.array(forecasts)).tolist()
print(forecasts)