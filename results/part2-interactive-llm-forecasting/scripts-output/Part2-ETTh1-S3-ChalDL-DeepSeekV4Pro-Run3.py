import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# LSTM model definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters from Turn 2
INPUT_SIZE = 7
SEQ_LEN = 12
PRED_LEN = 24
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 3
BATCH_SIZE = 32
LR = 0.005

# Read data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').asfreq('H')

# Preprocessing: replace artificial constant end-of-month days with NaN
constant_days = [
    '2016-07-31','2016-08-31','2016-10-31','2016-12-31',
    '2017-01-31','2017-03-31','2017-05-31','2017-07-31',
    '2017-08-31','2017-10-31','2017-12-31',
    '2018-01-31','2018-03-31','2018-05-31'
]
for d in constant_days:
    if d in df.index.strftime('%Y-%m-%d'):
        df.loc[d, 'OT'] = np.nan

# Forward fill NaN introduced by replacement
df['OT'] = df['OT'].ffill()

# Create lag features to achieve input_size=7
for lag in range(7):
    df[f'OT_lag{lag}'] = df['OT'].shift(lag)
df = df.dropna()  # remove rows with NaN from lagging

# Chronological split point: 14016 total timesteps training (index 0..14015), test from 14016
train_end_idx = 14015
test_start_idx = 14016
total_rows = len(df)

# Prepare data
data_values = df[[f'OT_lag{i}' for i in range(7)]].values.astype(np.float32)
target_values = df['OT'].values.astype(np.float32)

# Function to create sequences for training
def create_sequences(X, y, seq_len, pred_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(X_seq), np.array(y_seq)

# Forecast list
forecasts = []

# Iterate through test set, retraining at each step
for t in range(test_start_idx, total_rows):
    # Use all data up to current index t (inclusive) for training
    X_train_full = data_values[:t+1]
    y_train_full = target_values[:t+1]

    # Create training sequences from available data
    X_train_seq, y_train_seq = create_sequences(X_train_full, y_train_full, SEQ_LEN, PRED_LEN)

    # If not enough data to form any sequence, skip (should not happen after initial few steps)
    if len(X_train_seq) == 0:
        # Use last available lagged value repeated as fallback forecast
        last_val = target_values[t]
        pred = np.full(PRED_LEN, last_val, dtype=np.float32)
        forecasts.append(pred[0])
        continue

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train model
    model.train()
    for epoch in range(EPOCHS):
        permutation = torch.randperm(len(X_train_tensor))
        for i in range(0, len(X_train_tensor), BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_X = X_train_tensor[indices].to(device)
            batch_y = y_train_tensor[indices].to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Prepare input for prediction: the last SEQ_LEN feature vectors
    X_pred = data_values[t-SEQ_LEN+1:t+1]  # shape (SEQ_LEN, 7)
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).unsqueeze(0).to(device)  # (1, SEQ_LEN, 7)

    model.eval()
    with torch.no_grad():
        pred = model(X_pred_tensor).cpu().numpy().flatten()

    # Store the first prediction (horizon=1) as the forecast for this test step
    forecasts.append(float(pred[0]))

print(forecasts)