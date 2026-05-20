import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')
target_col = 'OT'

# Identify month-end days with constant values (std == 0)
daily_std = df[target_col].resample('D').std()
constant_dates = daily_std[daily_std == 0.0].index

# Mark those rows in the training period for removal; keep test rows as is
train_end = pd.Timestamp('2017-08-05 23:00:00')
for d in constant_dates:
    day_mask = (df.index >= d) & (df.index < d + pd.Timedelta(days=1))
    if day_mask.any() and df[day_mask].index[0] <= train_end:
        df.loc[day_mask, target_col] = np.nan

# Create lag features (t-1 to t-7) as new OT does not include NaNs after upcoming drop
# To compute lags properly, we must first shift the target and then drop NaNs
df['lag1'] = df[target_col].shift(1)
df['lag2'] = df[target_col].shift(2)
df['lag3'] = df[target_col].shift(3)
df['lag4'] = df[target_col].shift(4)
df['lag5'] = df[target_col].shift(5)
df['lag6'] = df[target_col].shift(6)
df['lag7'] = df[target_col].shift(7)

# Split into train and test based on original timestamps
train_mask = df.index <= train_end
test_mask = df.index > train_end

# Before dropping, separate targets and features
y_all = df[target_col].copy()
X_all = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7']].copy()

# Apply the same NaN rows to X_all (they already are NaN due to lag shift)
# For training, we will drop rows where y_train is NaN or any lag is NaN
X_train = X_all.loc[train_mask].dropna()
y_train = y_all.loc[train_mask].dropna()
# Ensure index alignment
common_idx = X_train.index.intersection(y_train.index)
X_train = X_train.loc[common_idx]
y_train = y_train.loc[common_idx]

# Test data: keep all rows, even constant days (they are not NaN)
X_test = X_all.loc[test_mask]
y_test = y_all.loc[test_mask]

# Convert to numpy arrays
X_train_np = X_train.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

# Hyperparameters
input_size = 7
seq_len = 12
pred_len = 1  # one-step ahead
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])  # last time step
        return out

# Function to create sequences from feature array X and target y
def create_sequences(X, y, seq_len):
    X_seq = []
    y_seq = []
    for i in range(seq_len - 1, len(X) - 1):
        X_seq.append(X[i - seq_len + 1 : i + 1])
        y_seq.append(y[i + 1])
    return np.stack(X_seq), np.array(y_seq)

# Initial training data sequences
X_train_seq, y_train_seq = create_sequences(X_train_np, y_train_np, seq_len)

# Convert to tensor datasets
train_dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
device = torch.device('cpu')
model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train initial model for 3 epochs
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Forecasting loop: one-step ahead prediction with retraining after each true observation
forecasts = []
combined_X = list(X_train_np)
combined_y = list(y_train_np)

for step in range(len(y_test_np)):
    # Prepare input sequence from combined history (last seq_len rows)
    if len(combined_X) < seq_len:
        # Not enough history; skip? But we have enough from training
        pass

    # Use the most recent seq_len rows of combined_X to predict next
    input_seq = np.array(combined_X[-seq_len:]).reshape(1, seq_len, input_size)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(input_seq).to(device)).cpu().numpy()[0, 0]
    forecasts.append(float(pred))

    # Append the true test observation to combined data
    new_x = X_test_np[step]  # lag features for this test point, computed from true previous values
    new_y = y_test_np[step]
    combined_X.append(new_x)
    combined_y.append(new_y)

    # Retrain model from scratch on all combined data
    combined_X_arr = np.array(combined_X)
    combined_y_arr = np.array(combined_y)
    if len(combined_X_arr) > seq_len + 1:
        X_full_seq, y_full_seq = create_sequences(combined_X_arr, combined_y_arr, seq_len)
    else:
        # Not enough data to retrain yet; just move on
        continue

    dataset_full = TensorDataset(torch.tensor(X_full_seq), torch.tensor(y_full_seq).unsqueeze(1))
    loader_full = DataLoader(dataset_full, batch_size=batch_size, shuffle=True)

    # Reinitialize model and optimizer for retraining
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for inputs, targets in loader_full:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# Output the forecast list exactly as required
print(forecasts)