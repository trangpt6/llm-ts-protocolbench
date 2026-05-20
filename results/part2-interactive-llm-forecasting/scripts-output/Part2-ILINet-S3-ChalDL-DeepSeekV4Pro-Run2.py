import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'], dayfirst=False)
df = df.sort_values('DATE').reset_index(drop=True)

# Preprocessing: drop empty columns
if 'AGE 25-49' in df.columns:
    df.drop(columns=['AGE 25-49'], inplace=True)
if 'AGE 25-64' in df.columns:
    df.drop(columns=['AGE 25-64'], inplace=True)

# Extract target
target_col = '% WEIGHTED ILI'
series = df[target_col].astype(float).values

# Chronological split (fixed: first 1050 train, next 263 test)
train_size = 1050
test_size = 263
train_series = series[:train_size].copy()
test_series = series[train_size:train_size+test_size].copy()

# Hyperparameters
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# GRU model definition
class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUForecaster, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # take last hidden state output
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Function to create sliding window datasets
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# Rolling forecast with retraining at each step
forecasts = []
current_train = train_series.copy()

for step in range(test_size):
    # Prepare training data from current_train
    x_train, y_train = create_sequences(current_train, seq_len, pred_len)
    if len(x_train) == 0:
        raise ValueError("Not enough training data to create sequences")
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)  # (samples, seq_len, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Build and train model
    model = GRUForecaster(input_size=1, hidden_size=hidden_size,
                          num_layers=num_layers, output_size=pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Simple batching
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Predict next 4 steps using last seq_len points of current_train
    model.eval()
    with torch.no_grad():
        last_input = torch.tensor(current_train[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)
        pred = model(last_input).cpu().numpy().flatten()
    # Take first predicted value as forecast for the next time step
    forecasts.append(pred[0])

    # Update training set with true value (ground truth)
    true_val = test_series[step]
    current_train = np.append(current_train, true_val)

# Output final forecast list
print(forecasts)