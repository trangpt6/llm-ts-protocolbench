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

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').asfreq('H')

# Detect constant-value days and replace with NaN
daily_groups = df.groupby(df.index.date)
constant_days_mask = np.zeros(len(df), dtype=bool)
for day, group in daily_groups:
    if group['OT'].nunique() == 1:
        constant_days_mask[df.index.date == day] = True
df.loc[constant_days_mask, 'OT'] = np.nan

# Create time features for all timestamps
def add_time_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

df = add_time_features(df)

# Define feature columns (target OT + time features)
feature_cols = ['OT', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']

# -------------------------------
# 2. Train/Test split
# -------------------------------
train_size = 14016
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Scale target OT using training set statistics
target_mean = train_df['OT'].mean()
target_std = train_df['OT'].std()
train_df['OT_scaled'] = (train_df['OT'] - target_mean) / target_std
test_df['OT_scaled'] = (test_df['OT'] - target_mean) / target_std

# Replace scaled OT for feature column
train_df_clean = train_df.copy()
train_df_clean['OT'] = train_df_clean['OT_scaled']
test_df_clean = test_df.copy()
test_df_clean['OT'] = test_df_clean['OT_scaled']

# -------------------------------
# 3. Build sequences for TCN
# -------------------------------
seq_len = 48
pred_len = 1

def create_sequences(data, seq_len):
    X, y = [], []
    values = data[feature_cols].values
    targets = data['OT'].values  # scaled
    for i in range(len(values) - seq_len):
        # Check if window contains any NaN
        window = values[i:i+seq_len]
        if np.isnan(window).any() or np.isnan(targets[i+seq_len]):
            continue
        X.append(window)
        y.append(targets[i+seq_len])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_df_clean, seq_len)
# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# -------------------------------
# 4. TCN Model Definition
# -------------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
    def forward(self, x):
        out = self.conv(x)
        # causal: remove future padding from the end
        return out[:, :, :-self.padding]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super().__init__()
        layers = []
        for i, d in enumerate(dilations[:num_layers]):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, d))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # out shape: (batch, hidden_size, seq_len); take last time step
        out = out[:, :, -1]
        return self.fc(out).squeeze(-1)

model = TCN(input_size=7, hidden_size=64, num_layers=3, kernel_size=3,
            dilations=[1, 2, 4, 8, 16])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 5. Train model
# -------------------------------
epochs = 30
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    # No validation, just train

# -------------------------------
# 6. Recursive forecasting over test set
# -------------------------------
model.eval()
# Prepare initial window: last seq_len steps of training data (scaled, no NaN)
train_values_scaled = train_df_clean[feature_cols].values
# Find the last seq_len valid steps (no NaN in OT)
valid_indices = ~np.isnan(train_df_clean['OT'].values)
last_valid_seq = train_values_scaled[valid_indices][-seq_len:]
current_seq = last_valid_seq.copy()

test_dates = test_df.index
forecasts_scaled = []
# Precompute future time features for all test steps (they are known)
test_features = test_df_clean[feature_cols].values  # but OT column is scaled (we don't use actual test OT)
# We'll only use time features and fill in predicted OT for the OT column.
# So we need to start with last OT from training, and then for each step, predict next OT, then update current_seq.

# Extract the OT column index (0) and time feature columns (1..6)
past_seq = current_seq
for i in range(len(test_df)):
    # The current_seq has shape (seq_len, 7). The last time step corresponds to the actual time just before the prediction.
    # To predict step i, we use current_seq as input.
    input_tensor = torch.tensor(past_seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(input_tensor).item()
    forecasts_scaled.append(pred_scaled)
    # Prepare the feature vector for the newly predicted time step
    # We need the actual time features for the predicted timestamp, which are known from test_df.
    # The OT value we set as the prediction (scaled).
    new_row_features = test_features[i].copy()
    new_row_features[0] = pred_scaled  # replace OT with predicted (scaled)
    # Remove the oldest step and append the new step
    past_seq = np.concatenate([past_seq[1:], new_row_features[np.newaxis, :]], axis=0)

# Inverse transform forecasts to original scale
forecasts = np.array(forecasts_scaled) * target_std + target_mean
forecasts = forecasts.tolist()

print(forecasts)