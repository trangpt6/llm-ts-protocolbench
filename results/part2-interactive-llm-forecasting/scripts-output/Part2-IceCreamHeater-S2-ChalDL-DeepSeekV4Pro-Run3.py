import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Device fallback
device = torch.device('cpu')

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']  # input_size=2, both series used as features

# No preprocessing needed per Turn 1

# Split definition from Turn 0
total = len(df)
train_size = int(0.8 * total)  # 158
# train indices: 0 to 157 inclusive, test indices: 158 to 197

# Hyperparameters
input_size = 2
seq_len = 6
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

# GRU model definition
class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, pred_len)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # take the last time step output
        out = out[:, -1, :]  # (batch, hidden_size)
        out = self.linear(out)  # (batch, pred_len)
        return out

# Helper: create sliding window dataset from a DataFrame containing both features
def create_sequences(data, seq_len):
    X, y = [], []
    values = data[feature_cols].values
    target_vals = data[target_col].values
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(target_vals[i+seq_len])
    return np.array(X), np.array(y).reshape(-1, 1)

# Forecast list
forecasts = []

# Initial training data: rows 0 to train_size-1 (158 points)
train_data = df.iloc[:train_size].copy()

# We will iteratively predict the test set (indices 158 to 197)
for step in range(len(df) - train_size):  # 40 iterations
    # Prepare training sequences from current train_data
    X_train, y_train = create_sequences(train_data, seq_len)
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    # Initialize model and optimizer
    model = GRUForecaster(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train for defined epochs
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_t[indices]
            batch_y = y_train_t[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Extract the last seq_len window from the current train_data to predict the next step
    # The prediction point is just after the last index in train_data.
    last_seq = train_data[feature_cols].values[-seq_len:]  # shape (seq_len, input_size)
    X_pred = torch.FloatTensor(last_seq.reshape(1, seq_len, input_size)).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X_pred).cpu().numpy().flatten()[0]
    forecasts.append(pred)
    
    # Append the true test observation to train_data for the next step
    # The true row is at index train_size + step
    true_row = df.iloc[train_size + step:train_size + step + 1]
    train_data = pd.concat([train_data, true_row], ignore_index=True)

# Output final forecast list
print(forecasts)