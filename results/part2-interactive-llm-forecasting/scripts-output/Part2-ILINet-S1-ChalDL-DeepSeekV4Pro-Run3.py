import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df = df.set_index('DATE')

# Preprocessing: reindex to complete weekly frequency (Sunday) and forward fill
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_index, method='ffill')

# Target column
target_col = '% WEIGHTED ILI'
series = df[target_col].values.astype(np.float32)

# Chronological split: train up to 2017-11-26, test from 2017-12-03
train_mask = df.index <= '2017-11-26'
test_mask = df.index >= '2017-12-03'
train_series = series[train_mask]
test_series = series[test_mask]

# Hyperparameters (fixed)
SEQ_LEN = 52
PRED_LEN = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 3
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4]
INPUT_SIZE = 1
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001

# Prepare training sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y).reshape(-1, 1)

X_train, y_train = create_sequences(train_series, SEQ_LEN)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# TCN model with causal convolutions
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # remove trailing padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = input_size
        for d in dilations[:num_layers]:
            self.layers.append(CausalConv1d(in_channels, hidden_size, kernel_size, d))
            in_channels = hidden_size
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> transpose to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = torch.relu(layer(x))
        # global pooling: take last timestep
        x = x[:, :, -1]
        return self.linear(x)

model = TCN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, KERNEL_SIZE, DILATIONS).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # optional: no print inside final script

# Recursive forecasting over test set
model.eval()
forecasts = []
# Initial input window: last SEQ_LEN points from training set
input_window = train_series[-SEQ_LEN:].copy()
with torch.no_grad():
    for i in range(len(test_series)):
        x = torch.tensor(input_window.reshape(1, SEQ_LEN, 1), dtype=torch.float32).to(device)
        pred = model(x).cpu().item()
        forecasts.append(pred)
        # update window: drop first, append prediction
        input_window = np.append(input_window[1:], pred)

forecasts = [float(f) for f in forecasts]
print(forecasts)