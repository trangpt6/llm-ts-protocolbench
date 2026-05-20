import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[['OT']]

# Preprocessing: replace artificial filler days with NaN
filler_days = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31',
    '2018-03-31', '2018-05-31'
]
for day in filler_days:
    df.loc[day, 'OT'] = np.nan

# Split into train and test (chronological)
train_size = 14016
test_size = 3504
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:train_size+test_size].copy()
full_series = df.copy()

# Create 7 lag features: OT shifted by 0 to 6 lags
for lag in range(1, 7):
    full_series[f'lag_{lag}'] = full_series['OT'].shift(lag)
full_series = full_series.rename(columns={'OT': 'lag_0'})
feature_cols = [f'lag_{i}' for i in range(7)]
# Drop rows with any NaN in features (these will be the first 6 rows due to lag and the filler days NaNs)
full_series_valid = full_series.dropna()

# Extract valid OT series and feature matrix
ot_valid = full_series_valid['lag_0'].values
features_valid = full_series_valid[feature_cols].values

# Prepare sequences for training
seq_len = 12
pred_len = 24
input_size = 7

def create_sequences(features, targets, seq_len, pred_len):
    X, y = [], []
    # i is the last index of the input window, target starts from i+1 to i+pred_len
    for i in range(seq_len - 1, len(features) - pred_len):
        X.append(features[i - seq_len + 1 : i + 1])
        y.append(targets[i + 1 : i + 1 + pred_len])
    return np.array(X), np.array(y)

X_full, y_full = create_sequences(features_valid, ot_valid, seq_len, pred_len)

# Determine how many sequences belong to training set (based on original index alignment)
# We'll map indices: the valid series indices correspond to original df indices.
# Simpler: we can just take the first part of the valid series that corresponds to training period.
# The training set covers indices 0 to 14015. After dropna, valid indices are a subset.
# We'll split based on the original index values of full_series_valid.
train_valid_mask = full_series_valid.index < df.index[train_size]
X_train = X_full[:train_valid_mask.sum() - pred_len]
y_train = y_full[:train_valid_mask.sum() - pred_len]

# Build LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size)
        out = self.fc(h_n[-1])
        return out

hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Function to train model
def train_model(model, X, y, epochs, batch_size):
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return model

# Initial training
model = train_model(model, X_train, y_train, epochs, batch_size)

# Forecasting loop
forecasts = []
# We'll maintain a growing series of valid features for retraining.
# Starting from the end of training set, we have all features up to train_size-1.
# To generate the first forecast, we need the last seq_len features from training set.
# We'll use the full_series_valid features up to the last train index.

# Get the last valid index before test set
train_end_idx = full_series_valid.index.get_loc(df.index[train_size - 1], method='ffill')
# The feature matrix up to train_end_idx is like features_valid[:train_end_idx+1]
# The corresponding target is ot_valid[:train_end_idx+1]
# We'll keep these arrays and extend them during forecasting.

current_features = features_valid[:train_end_idx + 1].copy()
current_targets = ot_valid[:train_end_idx + 1].copy()

# Iterate over test set
test_dates = df.index[train_size:train_size + test_size]
for i, test_date in enumerate(test_dates):
    # Prepare input window: last seq_len features
    input_seq = current_features[-seq_len:]
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(input_tensor).cpu().numpy().flatten()
    # We take the first forecast (1-step ahead) as the prediction for this test point
    forecast = preds[0]
    forecasts.append(forecast)
    # Observe true value (if not NaN)
    true_val = test['OT'].iloc[i]
    if not np.isnan(true_val):
        # Add this new observation to the series
        # Construct new feature row: [true_val, lag1, lag2, ..., lag6]
        # lag1 = previous OT, which is current_targets[-1]
        # lag2 = current_targets[-2], etc.
        new_features = [true_val]
        for lag in range(1, 7):
            if len(current_targets) >= lag + 1:
                new_features.append(current_targets[-lag])
            else:
                new_features.append(np.nan)
        new_features = np.array(new_features)
        # Append to current arrays
        current_features = np.vstack([current_features, new_features.reshape(1, -1)])
        current_targets = np.append(current_targets, true_val)
        # Retrain model with the expanded training data
        # Rebuild sequences from all valid data
        X_all, y_all = create_sequences(current_features, current_targets, seq_len, pred_len)
        if len(X_all) > 0:
            # Reinitialize optimizer for retraining
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model = train_model(model, X_all, y_all, epochs, batch_size)
    else:
        # If NaN, we cannot update with true value, just skip retraining
        # and use the current model. The input sequence will not be extended with this NaN point,
        # so we rely on the last valid features. This may cause the input window to be older than desired.
        pass

print(forecasts)