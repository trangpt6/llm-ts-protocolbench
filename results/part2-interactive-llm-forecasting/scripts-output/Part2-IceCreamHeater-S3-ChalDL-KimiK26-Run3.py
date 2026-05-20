import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Preprocessing: minimal, keep all values as-is
target_col = 'Ice cream'
heater_col = 'Heater'

# Split: first 157 for train, rest for test
train_size = 157
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# Fixed hyperparameters
input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

device = torch.device('cpu')

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Prepare sequences
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 0])
    return np.array(X), np.array(y)

# Normalize data
def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

def denormalize(data, mean, std):
    return data * (std + 1e-8) + mean

# Rolling forecast with retraining
all_forecasts = []
history = train_df.copy()

# We need to cover test set of 40 points with rolling 12-step forecasts
# At each step t in test, we predict t to t+11, but only store what we need

test_start_idx = train_size
test_end_idx = len(df)

for t in range(test_start_idx, test_end_idx):
    # Build current dataset from history
    current_data = history[[target_col, heater_col]].values
    
    # Compute normalization on current history
    mean = current_data.mean(axis=0)
    std = current_data.std(axis=0)
    
    norm_data = normalize(current_data, mean, std)
    
    # Need at least seq_len + pred_len points to create one training sample
    if len(norm_data) >= seq_len + pred_len:
        X, y = create_sequences(norm_data, seq_len, pred_len)
    else:
        # Pad if needed (should not happen given our data size)
        pad_len = seq_len + pred_len - len(norm_data)
        padded = np.vstack([norm_data, np.tile(norm_data[-1], (pad_len, 1))])
        X, y = create_sequences(padded, seq_len, pred_len)
    
    if len(X) == 0:
        # Fallback: use last known values
        last_val = current_data[-1, 0]
        all_forecasts.extend([last_val] * pred_len)
        # Update history with true value if available
        if t < test_end_idx:
            history = pd.concat([history, df.iloc[[t]]], ignore_index=True)
        continue
    
    # Train model
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Predict: use last seq_len points from history
    model.eval()
    last_seq = norm_data[-seq_len:]
    last_seq_tensor = torch.FloatTensor(last_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_norm = model(last_seq_tensor).cpu().numpy()[0]
    
    pred = denormalize(pred_norm, mean[0], std[0])
    
    # Store forecast for current time step (the first step of the 12-step prediction)
    # We need forecasts for each test point, so we take the first prediction
    # But we need to ensure we cover all 40 test points
    step_in_test = t - test_start_idx
    # For the forecast at test position step_in_test, we need the prediction for that step
    # From predictions made at various origins
    
    # We collect all 12-step forecasts and later extract the appropriate ones
    all_forecasts.append({
        'origin': t,
        'pred': pred.tolist()
    })
    
    # Update history with true value (ground truth enabled)
    if t < test_end_idx:
        history = pd.concat([history, df.iloc[[t]]], ignore_index=True)

# Extract final forecasts using rolling strategy
# For each test point i (0 to 39), find the most recent prediction that covers it
final_forecasts = []
for i in range(len(test_df)):
    target_time = test_start_idx + i
    # Find predictions from origins that cover this time
    # A prediction from origin o covers times o to o+pred_len-1
    # We want the prediction for time target_time from the latest origin <= target_time
    best_pred = None
    best_origin = -1
    for fc in all_forecasts:
        origin = fc['origin']
        if origin <= target_time and origin + pred_len > target_time:
            if origin > best_origin:
                best_origin = origin
                pred_idx = target_time - origin
                best_pred = fc['pred'][pred_idx]
    if best_pred is not None:
        final_forecasts.append(best_pred)
    else:
        # Fallback
        final_forecasts.append(float(train_df[target_col].iloc[-1]))

print(final_forecasts)