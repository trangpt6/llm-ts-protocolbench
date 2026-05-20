import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Read CSV
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df.set_index('date', inplace=True)
df = df.asfreq('H')

# Preprocessing: Replace constant-day artifacts with NaN and interpolate
artifact_dates = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31',
    '2018-01-31', '2018-03-31', '2018-05-31'
]
for d in artifact_dates:
    if d in df.index.floor('D'):
        df.loc[d, 'OT'] = np.nan
df['OT'] = df['OT'].interpolate(method='linear')

# Split: first 14016 train, rest test
train_len = 14016
train = df.iloc[:train_len].copy()
test = df.iloc[train_len:].copy()

# Create lag features (7 lags: t-1 to t-7)
def create_lag_features(series, lags=7):
    df_feat = pd.DataFrame(index=series.index)
    for i in range(1, lags+1):
        df_feat[f'lag_{i}'] = series.shift(i)
    return df_feat

full_series = df['OT']
full_features = create_lag_features(full_series, lags=7)
full_features = full_features.dropna()

# Align target (OT) with features (no NaN)
full_target = full_series.loc[full_features.index]

# Train/test split for features/target based on date
train_features = full_features.loc[:train.index[-1]]
train_target = full_target.loc[:train.index[-1]]
test_features = full_features.loc[test.index[0]:]
test_target = full_target.loc[test.index[0]:]

# Model definition
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dilations, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = dilations[i] if i < len(dilations) else 1
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation, padding=(kernel_size-1)*dilation, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> need (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        # take last time step
        y = y[:, :, -1]
        return self.linear(y)

# Fixed hyperparameters
input_size = 7
seq_len = 168
pred_len = 168
hidden_size = 64
num_layers = 4
kernel_size = 4
dilations = [1, 2, 4, 8, 16, 32]
# limit dilations to num_layers
dilations = dilations[:num_layers]
output_size = pred_len

# Create sliding windows from feature/target arrays
def create_windows(features, target, seq_len, pred_len):
    X_list, y_list = [], []
    n = len(features)
    for i in range(n - seq_len - pred_len + 1):
        X_list.append(features.iloc[i:i+seq_len].values)
        y_list.append(target.iloc[i+seq_len:i+seq_len+pred_len].values)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# Training function
def train_model(model, X_train, y_train, epochs, batch_size, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

# Forecasting process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initial training data: train set only
current_features = train_features.copy()
current_target = train_target.copy()
test_blocks = []
remaining_test = test_target.copy()
test_len = len(test_target)
for block_start in range(0, test_len, pred_len):
    block_end = min(block_start + pred_len, test_len)
    block_len = block_end - block_start
    # Create windows from current features/target
    X, y = create_windows(current_features, current_target, seq_len, pred_len)
    # Initialize model
    model = TCN(input_size=input_size, output_size=output_size,
                num_channels=[hidden_size]*num_layers, kernel_size=kernel_size, dilations=dilations)
    # Train
    train_model(model, X, y, epochs=20, batch_size=32, lr=0.001, device=device)
    # Prepare input for prediction: use the last seq_len steps of current_features
    # Actually we need the features for the last seq_len time steps to predict the next block.
    # We can take the features aligned with the target's last seq_len indices.
    # The current_target index tells us the last available target value.
    # The features for the last seq_len time steps are those corresponding to the last seq_len indices in current_target.
    last_seq = current_features.iloc[-seq_len:].values
    X_pred = np.expand_dims(last_seq, axis=0).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_pred).to(device)).cpu().numpy()[0]
    # Keep only needed length
    pred = pred[:block_len]
    test_blocks.append(pred)
    # Append true block to current training data
    true_block_values = remaining_target.iloc[block_start:block_end].values
    # Add new features (need to compute features for the new points)
    # We'll extend the full_features/target arrays, but simpler: we have current_features/current_target as a continuous aligned dataframe.
    # We can create new indices for the added true block and compute their lag features using full series.
    # Easiest: Reconstruct the aligned features/target data by expanding from the full series up to the new data.
    # For efficiency, we can just append new target values and then recalculate features for those new points using previous values.
    # Since we only need to extend current_features by adding rows for the new block, we can compute the lag features for those time points using the original full series.
    # The full series is available up to the end of test. We can compute the features for the entire test set initially, but for ground-truth update, we only need to extend the training set with the observed block.
    # We'll just rebuild the current_features and current_target by taking from the initially computed full_features/target up to the new end index.
    new_end_idx = train_len + block_end
    current_features = full_features.iloc[:new_end_idx]
    current_target = full_target.iloc[:new_end_idx]

# Combine all block forecasts into flat list
forecasts = np.concatenate(test_blocks).tolist()
print(forecasts)