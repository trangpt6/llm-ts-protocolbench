import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
total_rows = len(df)

# Anomalous days with constant values from Turn 1
anomalous_dates = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31',
    '2018-03-31', '2018-05-31'
]

# Chronological split
train_size = 14016
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Replace constant values with NaN only in training set
for d in anomalous_dates:
    mask = train['date'].dt.strftime('%Y-%m-%d') == d
    train.loc[mask, 'OT'] = np.nan

# Create feature columns for the entire dataset (with OT from train having NaN)
df_full = df.copy()
df_full['hour'] = df_full['date'].dt.hour
df_full['dayofweek'] = df_full['date'].dt.dayofweek
df_full['day'] = df_full['date'].dt.day
df_full['month'] = df_full['date'].dt.month
df_full['is_weekend'] = (df_full['dayofweek'] >= 5).astype(float)
df_full['sin_hour'] = np.sin(2 * np.pi * df_full['hour'] / 24)

# We need to replace the NaN OT in train rows with NaN for scaling, but we will forward fill later
# First, prepare features for scaling using only clean training data
train_clean = train.dropna(subset=['OT'])
scaler = StandardScaler()
feature_cols = ['OT', 'hour', 'dayofweek', 'day', 'month', 'is_weekend', 'sin_hour']
scaler.fit(train_clean[feature_cols])

# Scale the full dataset
scaled = scaler.transform(df_full[feature_cols])
scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df_full.index)

# Forward fill any NaN in the scaled data to eliminate NaN for sequence input (only OT had NaN)
scaled_df = scaled_df.fillna(method='ffill').fillna(method='bfill')

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Fixed hyperparameters
input_size = 7
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_sequences(data, end_idx):
    # data: scaled DataFrame with features
    # end_idx: last index to include as target (i.e., we need sequences to predict target at indices from seq_len to end_idx)
    # Returns X (N, seq_len, input_size), y (N,), indices of targets
    X_list = []
    y_list = []
    for i in range(seq_len, end_idx + 1):
        seq = data.iloc[i - seq_len:i]
        target = data.iloc[i]['OT']
        # Check if any NaN in sequence or target (target NaN if from original NaN OT)
        if seq.isnull().any().any() or np.isnan(target):
            continue
        X_list.append(seq.values)
        y_list.append(target)
    if len(X_list) == 0:
        return None, None
    X = np.array(X_list)
    y = np.array(y_list)
    return torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)

def train_model(model, X_train, y_train, epochs, batch_size, lr):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

# Initial training set: up to train_end index (14015)
train_end_idx = train_size - 1  # 14015
X_train, y_train = build_sequences(scaled_df, train_end_idx)

model = GRUModel(input_size, hidden_size, num_layers, 1).to(device)
if X_train is not None:
    train_model(model, X_train, y_train, epochs, batch_size, lr)
else:
    raise ValueError("No clean training sequences available")

forecasts = []
test_start_idx = train_size  # 14016
for step in range(len(test)):
    test_idx = test_start_idx + step  # index of the target we need to predict
    # Build input sequence using last seq_len points up to test_idx-1
    if test_idx < seq_len:
        raise ValueError("Not enough history to predict")
    seq = scaled_df.iloc[test_idx - seq_len:test_idx].values
    if np.isnan(seq).any():
        raise ValueError("NaN in input sequence at step {}".format(step))
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x).item()
    # Inverse transform only the OT dimension (the first column) using scaler mean and scale
    # The scaler was fit on all features, but we only need to inverse transform OT.
    # We'll construct a zeros array with same dimension, place pred_scaled, inverse transform, take first column.
    temp = np.zeros((1, len(feature_cols)))
    temp[0, 0] = pred_scaled
    pred_original = scaler.inverse_transform(temp)[0, 0]
    forecasts.append(pred_original)

    # Obtain true value for this test point
    true_val = test.iloc[step]['OT']
    # Add this true point to the training data for future steps by expanding scaled_df
    # We need to add a new row to scaled_df with the scaled true OT and time features.
    # Get the date from the test row
    date = test.iloc[step]['date']
    hour = date.hour
    dayofweek = date.dayofweek
    day = date.day
    month = date.month
    is_weekend = 1.0 if dayofweek >= 5 else 0.0
    sin_hour = np.sin(2 * np.pi * hour / 24)
    # Scale the true OT using the same scaler
    true_scaled = scaler.transform([[true_val, hour, dayofweek, day, month, is_weekend, sin_hour]])[0]
    # Append to scaled_df at the correct index (which is test_idx, same as original index)
    new_row = pd.Series(true_scaled, index=feature_cols, name=test_idx)
    scaled_df = scaled_df._append(new_row)
    scaled_df.sort_index(inplace=True)  # ensure order

    # Retrain the model after adding the new true observation
    # Expand training data end_idx to include this new point (test_idx)
    X_train, y_train = build_sequences(scaled_df, test_idx)
    if X_train is not None:
        model = GRUModel(input_size, hidden_size, num_layers, 1).to(device)
        train_model(model, X_train, y_train, epochs, batch_size, lr)

print(forecasts)