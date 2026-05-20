import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read dataset
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
target_col = 'OT'

# Preprocessing: replace constant-value blocks on month-end days with NaN
# (where all 24 hours have identical value) and replace negative values with NaN
df['date_only'] = df['date'].dt.date
daily_std = df.groupby('date_only')[target_col].transform('std')
df.loc[daily_std == 0, target_col] = np.nan
df.loc[df[target_col] < 0, target_col] = np.nan
df = df.drop(columns=['date_only'])
# Forward-fill to remove NaN for continuous sequence
df[target_col] = df[target_col].ffill()

# Create time-based features
def create_features(dates):
    df_feat = pd.DataFrame(index=dates)
    hour = dates.hour
    dayofweek = dates.dayofweek
    month = dates.month
    dayofyear = dates.dayofyear
    # Cyclical encoding for 6 dims plus target will be added later
    df_feat['sin_hour'] = np.sin(2 * np.pi * hour / 23.0)  # use 23 to avoid exact wrap-around issues
    df_feat['cos_hour'] = np.cos(2 * np.pi * hour / 23.0)
    df_feat['sin_dow'] = np.sin(2 * np.pi * dayofweek / 6.0)
    df_feat['cos_dow'] = np.cos(2 * np.pi * dayofweek / 6.0)
    df_feat['sin_doy'] = np.sin(2 * np.pi * dayofyear / 365.0)
    df_feat['cos_doy'] = np.cos(2 * np.pi * dayofyear / 365.0)
    return df_feat

features = create_features(df['date'])
features[target_col] = df[target_col].values

# Split train/test
train_size = 14016
train = features.iloc[:train_size]
test = features.iloc[train_size:]

# Standardize target
scaler = StandardScaler()
train_target = scaler.fit_transform(train[[target_col]])
test_target = scaler.transform(test[[target_col]])

# Replace target in features with scaled version
train = train.copy()
train[target_col] = train_target.flatten()
test = test.copy()
test[target_col] = test_target.flatten()

# Create sequences
def create_sequences(data, seq_len=48):
    X, y = [], []
    cols = data.columns.tolist()
    target_idx = cols.index(target_col)
    vals = data.values
    for i in range(len(vals) - seq_len):
        x_seq = vals[i:i+seq_len]  # shape: (seq_len, num_features)
        y_val = vals[i+seq_len, target_idx]
        X.append(x_seq)
        y.append(y_val)
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train, seq_len=48)
X_test, y_test = create_sequences(test, seq_len=48)

input_size = X_train.shape[2]  # should be 7

import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=(kernel_size-1)*dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=(kernel_size-1)*dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + out)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3, kernel_size=3, dilations=[1,2,4], dropout=0.1):
        super().__init__()
        layers = []
        self.hidden_size = hidden_size
        in_channels = input_size
        for i in range(num_layers):
            dil = dilations[i % len(dilations)]
            layers.append(TCNBlock(in_channels, hidden_size, kernel_size, dil, dropout))
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # take last time step
        out = out[:, :, -1]
        out = self.linear(out)
        return out.squeeze(-1)

# Use provided dilations list truncated to num_layers
dil_list = [1, 2, 4, 8, 16][:3]  # use first 3 for num_layers=3
model = TCN(input_size=input_size, hidden_size=64, num_layers=3, kernel_size=3, dilations=dil_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
BATCH_SIZE = 64
EPOCHS = 30
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                               torch.tensor(y_train, dtype=torch.float32))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_dataset)
    # optional: print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}')  # can comment out if needed

# Recursive forecasting over test set
model.eval()
forecasts_scaled = []
current_seq = train.values[-48:].copy()  # last 48 steps of training set (scaled)
current_dates = df['date'].iloc[train_size:train_size+3504]  # test dates

with torch.no_grad():
    for i in range(3504):
        # create input features for current sequence: (1, seq_len, input_size)
        inp = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred_scaled = model(inp).cpu().item()
        forecasts_scaled.append(pred_scaled)
        # Prepare next step's features
        # We need to create the feature vector for the next time step (which is test row i)
        # The next row's features are from test DataFrame row i
        next_features = test.iloc[i].values.copy()  # (input_size,)
        # Overwrite the target column with the predicted (scaled) value
        next_features[test.columns.get_loc(target_col)] = pred_scaled
        # Update the sequence: drop first row, append new row
        current_seq = np.vstack([current_seq[1:], next_features])

# Inverse scale the forecasts
forecasts_scaled_arr = np.array(forecasts_scaled).reshape(-1, 1)
forecasts = scaler.inverse_transform(forecasts_scaled_arr).flatten().tolist()

print(forecasts)