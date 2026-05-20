import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read CSV
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False)
df = df.set_index('Date').sort_index()

# Reindex to full daily range
full_range = pd.date_range(start='1981-01-01', end='1990-12-31', freq='D')
df = df.reindex(full_range)

# Linear interpolation for missing values
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='linear')

# Extract values
vals = df['Daily minimum temperatures'].values.astype(np.float32)

# Chronological split
train_size = 2921
test_size = 731

train_vals = vals[:train_size]
test_vals = vals[train_size:]

# Model definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Fixed hyperparameters
seq_len = 7
pred_len = 1
input_size = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare initial history as list (will grow)
history = train_vals.tolist()

forecasts = []

# Training function
def train_model(data, epochs, batch_size, lr):
    # Create sequences
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = GRUModel(input_size, hidden_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Forecast loop
for i in range(test_size):
    # Form input from last seq_len history values
    input_seq = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
    input_tensor = torch.tensor(input_seq).to(device)
    
    # Retrain model on current history
    model = train_model(history, epochs, batch_size, lr)
    
    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).cpu().item()
    forecasts.append(pred)
    
    # Append true test value to history
    history.append(test_vals[i])

print(forecasts)