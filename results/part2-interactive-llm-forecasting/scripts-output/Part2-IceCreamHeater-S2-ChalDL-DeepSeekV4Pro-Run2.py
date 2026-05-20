import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
# Parse Month column as datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()

# Use both Heater and Ice cream as features, target is Ice cream
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

# Split
train_size = 158
train_data = data[:train_size]
test_data = data[train_size:]

# Define GRU model
class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(GRUForecaster, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Use last output
        return self.linear(out[:, -1, :])

# Hyperparameters
seq_len = 6
input_size = 2
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

# Prepare training data for initial train
def create_sequences(data_array, seq_len):
    X, y = [], []
    for i in range(len(data_array) - seq_len):
        X.append(data_array[i:i+seq_len, :])      # all 2 features
        y.append(data_array[i+seq_len, 1])        # Ice cream target
    return np.array(X), np.array(y)

X_train_full, y_train_full = create_sequences(train_data, seq_len)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initial model training
model = GRUForecaster(input_size, hidden_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train initial model
def train_model(model, X_tensor, y_tensor, epochs, batch_size, device):
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()

# Convert to tensors
X_tensor = torch.from_numpy(X_train_full).float()
y_tensor = torch.from_numpy(y_train_full).float()
train_model(model, X_tensor, y_tensor, epochs, batch_size, device)

# Rolling forecast
forecasts = []
current_buffer = train_data.copy()  # shape (N,2); will grow

for step in range(len(test_data)):
    # Prepare last seq_len rows of current_buffer to predict next Ice cream
    if len(current_buffer) < seq_len:
        raise ValueError("Not enough data to form sequence")
    input_seq = current_buffer[-seq_len:]  # (seq_len, 2)
    input_seq_tensor = torch.from_numpy(input_seq).float().unsqueeze(0).to(device)  # (1, seq_len, 2)
    with torch.no_grad():
        pred = model(input_seq_tensor).item()
    forecasts.append(pred)

    # Now incorporate true observation for this test timestep (ground truth)
    new_row = test_data[step].copy()
    current_buffer = np.vstack([current_buffer, new_row])

    # Retrain model with expanded data
    X_update, y_update = create_sequences(current_buffer, seq_len)
    X_tensor = torch.from_numpy(X_update).float()
    y_tensor = torch.from_numpy(y_update).float()
    # Re-initialize optimizer and model? Per setup, retrain means update model.
    # We reinitialize model to avoid bias from previous lr? We'll just re-train from scratch on expanded data.
    # Better: reinitialize model weights and optimizer each retrain to avoid overfitting to older runs.
    model = GRUForecaster(input_size, hidden_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model, X_tensor, y_tensor, epochs, batch_size, device)

print(forecasts)