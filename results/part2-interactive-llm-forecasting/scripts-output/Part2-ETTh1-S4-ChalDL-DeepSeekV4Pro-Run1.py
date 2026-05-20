import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read CSV
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date')

# Preprocessing: detect constant blocks at end-of-month days
# days with all same value (sensor freeze)
df_prep = df.copy()
for date in df.index.to_period('D').unique():
    day_data = df_prep[df_prep.index.to_period('D') == date]
    if len(day_data) == 24 and day_data['OT'].nunique() == 1:
        df_prep.loc[day_data.index, 'OT'] = np.nan

# Convert to numeric, drop NaNs for sequence building later

target = df_prep['OT'].values.astype(np.float32)
timestamps = df_prep.index

# Train/test split
N = len(target)
train_size = 14016  # fixed from Turn 0
train_target = target[:train_size]
test_target = target[train_size:]

# Build lag feature matrix: at each time step, features = [OT[t], OT[t-1], ..., OT[t-6]]
# We need at least lag 6, so valid starting index = 6
def create_lag_matrix(data):
    n = len(data)
    mat = np.zeros((n, 7), dtype=np.float32) * np.nan
    for i in range(6, n):
        mat[i, :] = [data[i], data[i-1], data[i-2], data[i-3], data[i-4], data[i-5], data[i-6]]
    return mat

full_features = create_lag_matrix(target)
# Only rows without NaN are usable
valid_mask = np.all(~np.isnan(full_features), axis=1)
valid_indices = np.where(valid_mask)[0]

# Create sequences: X: (seq_len=168, 7), Y: (pred_len=168,)
def create_sequences(features, target_vals, indices, seq_len, pred_len):
    X_list, Y_list = [], []
    # Indices are valid positions in the full time series
    for idx in indices:
        if idx + pred_len < len(target_vals) and idx >= seq_len - 1:
            # Input window from idx - seq_len + 1 to idx inclusive
            X_seq = features[idx - seq_len + 1 : idx + 1]  # shape (seq_len, 7)
            Y_seq = target_vals[idx + 1 : idx + 1 + pred_len]  # shape (pred_len,)
            if not np.isnan(X_seq).any() and not np.isnan(Y_seq).any():
                X_list.append(X_seq)
                Y_list.append(Y_seq)
    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y

seq_len = 168
pred_len = 168
horizon = 168

# TCN block definition
class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        out = self.relu(out)
        # Causal cropping: remove extra padding at the end to keep sequence length
        out = out[:, :, :-self.padding]
        return out

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, output_len):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = [input_size] + [hidden_size] * num_layers
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 2**i
            self.layers.append(TCNResidualBlock(channels[i], channels[i+1], kernel_size, dilation))
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        # x: (B, seq_len, input_size) -> transpose to (B, input_size, seq_len)
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        # After TCN layers, x shape: (B, hidden_size, seq_len)
        # Pool or take last element: use last time step
        x_last = x[:, :, -1]  # (B, hidden_size)
        out = self.fc(x_last)  # (B, output_len)
        return out

# Initialize TCN model
input_size = 7
hidden_size = 64
num_layers = 4
kernel_size = 4
dilations = [1, 2, 4, 8]  # using first 4 dilations
model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)

# Training function
def train_model(trainX, trainY, epochs=20, batch_size=32, lr=0.001):
    model.train()
    dataset = TensorDataset(torch.tensor(trainX).float(), torch.tensor(trainY).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    # return model

# Main block-wise forecasting loop
forecasts = []
current_train_end = train_size  # index in full target array (end exclusive)
# For retraining, we will use data up to current_train_end

# Create initial training sequences using all valid indices up to current_train_end
full_features_mat = full_features  # np array (N, 7)
target_vals = target

# Helper to build X,Y from data up to an end index (exclusive)
def build_dataset(end_idx):
    mask = valid_indices < end_idx
    indices = valid_indices[mask]
    X, Y = create_sequences(full_features_mat, target_vals, indices, seq_len, pred_len)
    return X, Y

# Start block-wise forecasting
num_full_blocks = 3504 // horizon  # horizon=168, 20 blocks
remaining = 3504 % horizon

for block in range(num_full_blocks):
    # Train on data up to current_train_end
    X_train, Y_train = build_dataset(current_train_end)
    train_model(X_train, Y_train, epochs=20, batch_size=32, lr=0.001)
    # Forecast next block: we need the last seq_len valid feature vectors up to current_train_end
    # The last valid input window ends at current_train_end-1
    # Ensure we have enough valid indices to construct the input sequence
    # Build the single input sequence: from current_train_end - seq_len to current_train_end-1
    # Using full_features_mat
    input_seq = full_features_mat[current_train_end - seq_len : current_train_end]  # (seq_len, 7)
    if np.isnan(input_seq).any():
        # fallback: find latest valid sequence of length seq_len before current_train_end
        # But due to NaNs, we may need to skip, but we assume valid
        # If NaN, we could replace with last known good, but skipping for simplicity
        raise ValueError("NaN in input sequence")
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(input_seq).unsqueeze(0).float().to(device)  # (1, seq_len, 7)
        pred = model(x_tensor).cpu().numpy().flatten()  # (pred_len,)
    forecasts.extend(pred.tolist())
    # Update ground truth: the true values for this block are test_target[block*horizon: (block+1)*horizon]
    # However, the test_target is a separate array starting from index train_size.
    # We need to incorporate them into the full target array? We'll extend our full target array with true values.
    # We have the original target array of length N. The test part starts at train_size. So we can just use the original target array, which already contains the true future values (it's the full dataset). But we must not use future true values beyond the block we have already "seen". So we can pretend that at each block, after forecasting, we reveal the true values for that block from the original full dataset (which is known). So we just need to move the train end.
    current_train_end += horizon  # we have now "revealed" the next 168 values
    # Note: the model is retrained on the expanded dataset in the next loop iteration.

# Last partial block
if remaining > 0:
    X_train, Y_train = build_dataset(current_train_end)
    # adjust pred_len for last block
    old_pred_len = model.fc.out_features
    if old_pred_len != remaining:
        # Since we cannot easily change output length, we forecast horizon steps and truncate
        # So we just train normally and then take first remaining predictions
        pass
    train_model(X_train, Y_train, epochs=20, batch_size=32, lr=0.001)
    input_seq = full_features_mat[current_train_end - seq_len : current_train_end]
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(input_seq).unsqueeze(0).float().to(device)
        pred = model(x_tensor).cpu().numpy().flatten()
    forecasts.extend(pred[:remaining].tolist())

# Output final forecasts
forecasts = [float(v) for v in forecasts]
print(forecasts)