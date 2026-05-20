import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Reproducibility seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load and preprocess
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.set_index('DATE', inplace=True)
series = df['% WEIGHTED ILI'].astype(float)

# Resample to weekly Sunday frequency, keep original values, introduce NaN for missing Sundays
series = series.resample('W-SUN').asfreq()

# Linear interpolation to fill missing single weeks (gaps) but not off-season zero blocks
series = series.interpolate(method='linear', limit_direction='both')

# Split exactly at given dates (chronological split defined in Turn 0)
train_series = series[series.index <= '2017-11-19']
test_series = series[series.index >= '2017-11-26']

train_data = train_series.values.astype(np.float32)
test_data = test_series.values.astype(np.float32)

# Normalize using train range (MinMax scaling to [0,1])
min_val = train_data.min()
max_val = train_data.max()
scale_range = max_val - min_val
if scale_range == 0:
    scale_range = 1
train_data_norm = (train_data - min_val) / scale_range
test_data_norm = (test_data - min_val) / scale_range

# Fixed hyperparameters
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, pred_len=52):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last_hidden)  # (batch, pred_len)
        return out

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

def train_model(train_data_np):
    X, y = create_sequences(train_data_np, seq_len, pred_len)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)  # (samples, seq_len, 1)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    model = LSTMForecaster(input_size=1, hidden_size=hidden_size, num_layers=num_layers, pred_len=pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Initial training on train set
model = train_model(train_data_norm)

all_forecasts_norm = []
current_train = train_data_norm.copy()  # array that will be expanded with ground truth

remaining_test_len = len(test_data_norm)
test_idx = 0

while test_idx < remaining_test_len:
    # Forecast next block: use last seq_len values from current training data
    input_seq = current_train[-seq_len:]  # shape (seq_len,)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    model.eval()
    with torch.no_grad():
        pred_norm = model(input_tensor).cpu().numpy().flatten()  # (pred_len,)

    # Determine how many forecasts we actually need for this block
    need = min(pred_len, remaining_test_len - test_idx)
    all_forecasts_norm.append(pred_norm[:need])

    # Get ground truth for this block
    true_block = test_data_norm[test_idx:test_idx+need]
    # Append true block to current training data for future retraining
    current_train = np.concatenate([current_train, true_block])

    # Retrain model on expanded data if there are more blocks to forecast
    if test_idx + need < remaining_test_len:
        model = train_model(current_train)

    test_idx += need

# Concatenate all normalized forecasts and inverse transform
forecasts_norm = np.concatenate(all_forecasts_norm)
forecasts = forecasts_norm * scale_range + min_val

# Ensure exact length of test set
forecasts = forecasts[:len(test_series)].tolist()
print(forecasts)