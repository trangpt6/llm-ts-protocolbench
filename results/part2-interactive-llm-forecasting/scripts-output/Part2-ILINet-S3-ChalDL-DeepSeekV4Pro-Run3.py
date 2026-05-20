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
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing: drop empty columns (from Turn1)
df = df.drop(columns=['AGE 25-49', 'AGE 25-64'], errors='ignore')

# Ensure chronological order
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

# Extract target series
target_col = '% WEIGHTED ILI'
y = df[target_col].values.astype(np.float32)

# Fixed split
train_size = 1050
test_size = 263
y_train = y[:train_size]
y_test = y[train_size:train_size+test_size]

# Fixed hyperparameters
input_size = 1
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)          # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]           # take last timestep hidden state
        out = self.fc(out)            # (batch, pred_len)
        return out

# Create training sequences from a given series
def create_sequences(series, seq_len, pred_len):
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cumulative data starts with training set
cumulative = y_train.tolist()

# Forecast collection
forecasts = []

for i in range(test_size):
    # Current available history
    data = np.array(cumulative, dtype=np.float32)
    
    # Prepare training sequences
    X, Y = create_sequences(data, seq_len, pred_len)
    if len(X) == 0:
        # Fallback: cannot create sequences, use last value as naive forecast
        forecasts.append(float(data[-1]))
        cumulative.append(float(y_test[i]))
        continue
    
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)                 # (N, pred_len)
    
    # Initialize fresh model for each step
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train for 5 epochs
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(X_tensor))
        for idx in range(0, len(X_tensor), batch_size):
            batch_idx = permutation[idx:idx+batch_size]
            batch_x = X_tensor[batch_idx].to(device)
            batch_y = Y_tensor[batch_idx].to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Predict next horizon using the last seq_len values
    last_input = torch.tensor(data[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(last_input).cpu().numpy().flatten()  # shape (pred_len,)
    # Take first step forecast (one-step ahead for the current test index)
    forecast = float(preds[0])
    forecasts.append(forecast)
    
    # Append true observation to cumulative data for next step
    cumulative.append(float(y_test[i]))

print(forecasts)