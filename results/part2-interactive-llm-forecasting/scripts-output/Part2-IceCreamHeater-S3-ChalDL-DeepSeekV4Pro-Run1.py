import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)
n_total = len(data)

# Split strictly chronologically
train_size = 158
test_size = n_total - train_size

# Hyperparameters
input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (seq_len, batch, input_size)
        out, _ = self.lstm(x)
        # Use last time step hidden state
        last_hidden = out[-1]  # (batch, hidden_size)
        return self.fc(last_hidden)  # (batch, output_size)

# Function to create training sequences
def create_sequences(arr, seq_len, pred_len):
    X, y = [], []
    for i in range(seq_len, len(arr) - pred_len + 1):
        X.append(arr[i-seq_len:i])
        y.append(arr[i:i+pred_len, 1])  # target is Ice cream
    return np.array(X), np.array(y)

forecasts = []

# Initial training data up to index train_size-1 (inclusive)
current_end = train_size - 1

for step in range(test_size):
    test_idx = train_size + step  # current test index (0-based)
    
    # Data available up to (but not including) test_idx
    available = data[:test_idx].copy()  # shape (current_length, 2)
    
    # Create training sequences
    X_train, y_train = create_sequences(available, seq_len, pred_len)
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(1, 0, 2)  # (seq_len, num_samples, input_size)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    # Initialize model
    model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train for specified epochs
    n_samples = X_train_t.shape[1]
    for epoch in range(epochs):
        # Shuffle indices
        permutation = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_t[:, indices, :].to(device)
            batch_y = y_train_t[indices].to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    # Prepare input for prediction: last seq_len observations
    input_seq = available[-seq_len:]  # (seq_len, 2)
    input_seq_t = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(1).to(device)  # (seq_len, 1, input_size)
    
    model.eval()
    with torch.no_grad():
        pred = model(input_seq_t).cpu().numpy().flatten()  # (pred_len,)
    # Store only the first predicted value (next time step)
    forecast_value = float(pred[0])
    forecasts.append(forecast_value)
    
    # Observe ground truth: the actual row at test_idx
    # This updates the available data for the next iteration
    # The true value is already in the full data array; we just allow it for next retraining.
    # No action needed because we slice data[:test_idx+1] in next iteration.
    # But careful: available already excludes test_idx, so next iteration will include it.

print(forecasts)