import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read CSV and preprocess
df = pd.read_csv(r'../../../data/ILINet.csv')
df = df.drop(columns=['AGE 25-49', 'AGE 25-64'])
y = df['% WEIGHTED ILI'].values.astype(np.float32)

# Split into train and test
train_size = int(0.8 * len(df))
y_train = y[:train_size]
y_test = y[train_size:]

# Scale using training data
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

# Hyperparameters
seq_len = 52
pred_len = 52
input_size = 1
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

# LSTM model definition
class BlockLSTM(nn.Module):
    def __init__(self):
        super(BlockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

# Create sequences from a series
def create_sequences(data):
    X, y_seq = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y_seq.append(data[i+seq_len:i+seq_len+pred_len])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y_seq, dtype=torch.float32)

# Train model
def train_model(train_data):
    model = BlockLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_tensor, y_tensor = create_sequences(train_data)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
    return model

# Forecast a block
def forecast_next(model, last_sequence):
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model(inp).squeeze(0).numpy()
    return pred

# Initial training on full training set
train_data = y_train_scaled.copy()
model = train_model(train_data)

# Forecasting block-wise with retraining
forecasts_scaled = []
remaining = len(y_test_scaled)
current_idx = 0

while remaining > 0:
    # Input sequence is the last seq_len values of current training data (scaled)
    input_seq = train_data[-seq_len:]
    pred_block = forecast_next(model, input_seq)
    
    # How many steps to take in this block
    take = min(pred_len, remaining)
    forecasts_scaled.extend(pred_block[:take])
    remaining -= take
    current_idx += take
    
    if remaining > 0:
        # Append true test values (scaled) that we just predicted to training data
        true_block = y_test_scaled[current_idx - take:current_idx]
        train_data = np.concatenate([train_data, true_block])
        # Retrain model from scratch on extended data
        model = train_model(train_data)

# Inverse transform to original scale
forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten().tolist()
print(forecasts)