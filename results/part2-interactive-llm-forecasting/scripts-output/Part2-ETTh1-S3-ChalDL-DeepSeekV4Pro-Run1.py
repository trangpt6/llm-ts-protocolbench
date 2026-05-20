import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Preprocessing: identify constant-value days (all 24 hourly values identical)
df['date_only'] = df['date'].dt.date
daily_count = df.groupby('date_only')['OT'].nunique()
constant_days = daily_count[daily_count == 1].index
df.loc[df['date_only'].isin(constant_days), 'OT'] = np.nan
df.drop('date_only', axis=1, inplace=True)
df['OT'] = df['OT'].interpolate(method='linear')

# Split
n = len(df)
train_size = int(0.8 * n)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Feature engineering
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['lag1'] = df['OT'].shift(1)
df['lag2'] = df['OT'].shift(2)
df['lag3'] = df['OT'].shift(3)

feature_cols = ['OT', 'hour', 'dayofweek', 'month', 'lag1', 'lag2', 'lag3']
# Drop rows with NaN from lags (first 3)
df_feat = df.dropna(subset=feature_cols).reset_index(drop=True)
# Store original index mapping: feat_index = orig_index - 3 (because we dropped first 3)
# Actually after dropping first 3 rows, the index correspondence: feat df index i corresponds to original index i+3.

ot_series = df['OT'].values  # full preprocessed OT, unchanged
feat_values = df_feat[feature_cols].values.astype(np.float32)
feat_dates = df_feat['date'].values

# Build LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out: (batch, seq_len, hidden)
        # Use last time step's output
        out = out[:, -1, :]
        return self.fc(out)

input_size = 7
seq_len = 12
pred_len = 24
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

# Prepare arrays for rolling forecast
forecasts = []

# We'll maintain a list of all observed feature vectors up to current time,
# using the global feat_values array but we need to add test points as they become observed.
# For convenience, we can precompute the feature vectors for all test points as well,
# because they only depend on past OT values. That's safe.
# So we have feats for all indices from 0 to len(df_feat)-1, corresponding to original indices 3..n-1.

# The original index of the first test point is train_size.
# The corresponding feat index is train_size - 3 (if train_size >= 3, which it is).

start_test_feat_idx = train_size - 3

# In the loop, current_time original index t starts from train_size.
for t in range(train_size, n):
    # Determine last available feat index (up to t-1)
    last_feat_idx = t - 3 - 1  # because feat index = orig_idx - 3; we can use up to orig_idx-1 which corresponds to feat idx (t-1)-3 = t-4.
    # but careful: the feature at original index t-1 uses OT up to t-1, so it is available. Its feat index is (t-1)-3 = t-4.
    # So feature indices from 0 to last_feat_idx are available.
    if last_feat_idx < seq_len + pred_len - 1:
        # Not enough data to train, skip (shouldn't happen)
        continue

    # Build training data: we can train using all windows where the target ends by last_feat_idx.
    # We'll create sequences from available feats: for start s from 0 to last_feat_idx - seq_len - pred_len + 1
    # input: feats[s : s+seq_len], target: original ot from (orig index = (s+3)+seq_len to (s+3)+seq_len+pred_len-1)
    X_list = []
    y_list = []
    max_s = last_feat_idx - seq_len - pred_len + 1
    for s in range(max_s + 1):
        inp = feat_values[s : s+seq_len]
        # target indices in original: orig_start = (s + 3) + seq_len
        tgt_start = s + 3 + seq_len
        tgt = ot_series[tgt_start : tgt_start + pred_len]
        if len(tgt) == pred_len:
            X_list.append(inp)
            y_list.append(tgt)
    if len(X_list) == 0:
        continue

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)

    # Train model
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            indices = perm[i:i+batch_size]
            xb = torch.tensor(X[indices], dtype=torch.float32).to(device)
            yb = torch.tensor(y[indices], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # Prepare input for prediction: use last seq_len feature vectors (up to t-1)
    # orig index of last available is t-1, feat idx = t-1-3 = t-4.
    pred_input_feat_idx = last_feat_idx  # which is (t-1)-3 = t-4
    input_window = feat_values[pred_input_feat_idx - seq_len + 1 : pred_input_feat_idx + 1]
    input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        future_pred = model(input_tensor).cpu().numpy().flatten()
    forecast_val = future_pred[0]
    forecasts.append(forecast_val)

    # Now add the true feature vector for time t to the growing dataset.
    # We need to compute feature vector for index t using true OT[t] and existing historical data.
    # feat index for t is t - 3. We can create a new row and append to feats array.
    new_row = np.array([
        ot_series[t],
        df['hour'].iloc[t],
        df['dayofweek'].iloc[t],
        df['month'].iloc[t],
        ot_series[t-1] if t-1 >= 0 else np.nan,
        ot_series[t-2] if t-2 >= 0 else np.nan,
        ot_series[t-3] if t-3 >= 0 else np.nan
    ], dtype=np.float32)
    # Append to feat_values array (making it bigger)
    feat_values = np.vstack([feat_values, new_row])

print(forecasts)