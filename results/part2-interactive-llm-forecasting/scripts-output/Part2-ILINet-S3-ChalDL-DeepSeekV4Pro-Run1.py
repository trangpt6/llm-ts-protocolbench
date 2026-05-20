import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read and preprocess data exactly as per Turn 1
df = pd.read_csv(r'../../../data/ILINet.csv')
# Drop entirely empty columns AGE 25-49 and AGE 25-64
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True, errors='ignore')
target_col = '% WEIGHTED ILI'
series = df[target_col].values.astype(np.float32)

# Chronological split
train_size = 1050
test_size = 263

# Fixed hyperparameters from Turn 2
input_size = 1
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Use last hidden state for prediction
        last_out = out[:, -1, :]
        return self.fc(last_out)

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(train_series):
    # Create sequences from current training series
    X, y = create_sequences(train_series, seq_len, pred_len)
    X = torch.tensor(X.reshape(-1, seq_len, input_size)).to(device)
    y = torch.tensor(y).to(device)

    model = GRUModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X[indices]
            batch_y = y[indices]
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

# Start rolling forecast
train_series = series[:train_size].copy()
all_forecasts = []

for step in range(test_size):
    # Train model on current training series
    model = train_model(train_series)
    # Prepare last seq_len values for prediction
    last_seq = torch.tensor(train_series[-seq_len:].reshape(1, seq_len, input_size)).to(device)
    with torch.no_grad():
        pred = model(last_seq).cpu().numpy()[0]  # shape (pred_len,)
    # Store the first predicted value (1-step ahead)
    all_forecasts.append(float(pred[0]))
    # Append ground truth of this test point (enabled use of ground truth during inference)
    true_val = series[train_size + step]
    train_series = np.append(train_series, true_val)

# The forecasts list covers the entire test set
print(all_forecasts)