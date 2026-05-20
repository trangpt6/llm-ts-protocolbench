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

# Read the CSV file
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])

# Set date as index
df = df.set_index('date')

# Mark constant-value days as NaN (list of date strings)
constant_dates = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31',
    '2018-03-31', '2018-05-31'
]
for d in constant_dates:
    df.loc[d, 'OT'] = np.nan

# Split into train and test (80/20 chronological)
total_len = len(df)
train_len = int(0.8 * total_len)  # 14016
train_df = df.iloc[:train_len]
test_df = df.iloc[train_len:]

# Feature engineering: create 7 lagged versions of OT
input_size = 7
for lag in range(1, input_size):
    df[f'OT_lag{lag}'] = df['OT'].shift(lag)
# The original OT is lag0
features_cols = ['OT'] + [f'OT_lag{lag}' for lag in range(1, input_size)]
df_features = df[features_cols]

# Extract training target and features
train_ot = df_features.loc[train_df.index, 'OT'].values
train_feat = df_features.loc[train_df.index, features_cols].values

seq_len = 48
pred_len = 1  # one-step ahead

# Create sequences for training (skip windows containing NaN)
X_train = []
y_train = []
for i in range(len(train_feat) - seq_len - pred_len + 1):
    window = train_feat[i:i+seq_len]
    target = train_ot[i+seq_len:i+seq_len+pred_len]
    if np.isnan(window).any() or np.isnan(target).any():
        continue
    X_train.append(window)
    y_train.append(target)
X_train = np.array(X_train, dtype=np.float32)  # shape (num_samples, seq_len, 7)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)

# Define TCN residual block
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                    dilation=dilation_size, padding=(kernel_size-1)*dilation_size,
                                    dropout=0.0)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # take last timestep
        out = out[:, :, -1]
        return self.linear(out)

# Hyperparameters
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4]  # using first 3 from the list to match num_layers
num_channels = [hidden_size] * num_layers
output_size = pred_len

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(input_size=input_size, output_size=output_size, num_channels=num_channels,
            kernel_size=kernel_size, dilations=dilations).to(device)

# Training parameters
epochs = 30
batch_size = 64
lr = 0.001

# Create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    # optional: print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_dataset):.6f}')

# Prepare for recursive forecasting
model.eval()

# Get the last seq_len rows from the training set (raw OT, before feature engineering)
train_ot_series = train_df['OT'].values  # may contain NaN (constant days)
# Find the last seq_len non-NaN values before the test start.
# We'll take the tail of the training set (of length train_len) but we need to handle NaN.
# Starting from the end, collect values until we have seq_len non-NaN.
last_ot_seq = []
idx = train_len - 1
while len(last_ot_seq) < seq_len:
    val = train_ot_series[idx]
    if not np.isnan(val):
        last_ot_seq.insert(0, val)
    idx -= 1
last_ot_seq = np.array(last_ot_seq, dtype=np.float32)

# Also need the corresponding lagged features for the initial window.
# We can compute the last seq_len OT values and then construct the feature matrix.
# Given the OT series from the training set (with NaN), we need the date-aligned features.
# Better approach: use the original df_features for the training portion and take the last seq_len non-NaN rows.
train_features_df = df_features.iloc[:train_len]
# We need to find the last seq_len rows where all feature columns are not NaN.
valid_indices = train_features_df.dropna().index
# Get the last seq_len valid timestamps before the test start: these must be the most recent timestamps.
# Since constant days are end-of-month, the valid tail will be after last constant day in training.
# The last constant day in training is 2018-01-31. So we can safely take the last seq_len rows of the training set after dropping NaN.
train_features_valid = train_features_df.dropna()
last_window_features = train_features_valid.iloc[-seq_len:].values.astype(np.float32)

# Forecast recursively
test_len = len(test_df)
forecasts = []
current_window_features = last_window_features  # shape (seq_len, 7)
current_ot_series = list(last_ot_seq)  # for feature updates

with torch.no_grad():
    for step in range(test_len):
        input_tensor = torch.tensor(current_window_features).unsqueeze(0).to(device)  # (1, seq_len, 7)
        pred = model(input_tensor).item()
        forecasts.append(pred)
        # Update OT series for feature computation (shift lags)
        current_ot_series.append(pred)
        # Compute new features from the updated OT list
        # We need the last seq_len OT values to compute features for next step.
        # The feature columns: [OT(t), OT(t-1), ..., OT(t-6)] for the most recent seq_len timesteps.
        # For the next window, the new OT vector of length seq_len is the previous window shifted by 1, appending the predicted value.
        # So we can keep only the last seq_len OT values and compute features accordingly.
        # To simplify, we can construct the feature matrix for the new window:
        # Use the current OT series of length >= seq_len, take last seq_len values.
        recent_ot = np.array(current_ot_series[-seq_len:], dtype=np.float32)
        new_window = np.zeros((seq_len, input_size), dtype=np.float32)
        new_window[:, 0] = recent_ot
        for lag in range(1, input_size):
            new_window[lag:, lag] = recent_ot[:-lag]  # shifted
            new_window[:lag, lag] = np.nan  # lead with NaN, but these will be at the beginning, not used because window is the last seq_len; actually the initial few rows of the window will have NaN in lags, but we'll fill with 0? However, we need a full non-NaN window. Since we have the complete recent_ot, we can compute lagged values for all positions if we have enough history. The OT series from training provided seq_len values, plus the predictions so far, so we have enough history. We'll compute properly by using the full OT series of length >= seq_len + max_lag, and construct the window for the last seq_len steps.
        # Better: maintain the full OT series including past, then at each step generate the window using the most recent seq_len steps with their lags computed from the full series.
        # We'll implement a rolling approach using a pandas Series of OT (including both training and predicted). We'll initialize a series with the training data up to the window's end, then append predictions.
        # However, in this loop we can avoid heavy pandas; we already have current_ot_