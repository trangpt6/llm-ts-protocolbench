import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read the CSV
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False)

# Preprocessing: reindex to complete daily range and forward fill missing dates
df.set_index('Date', inplace=True)
full_range = pd.date_range(start='1981-01-01', end='1990-12-31', freq='D')
df = df.reindex(full_range, method='ffill')
df.index.name = 'Date'

# Train/test split exactly as in Turn 0
train_series = df.loc[:'1988-12-30', 'Daily minimum temperatures'].values.astype(np.float32)
test_series = df.loc['1989-01-01':, 'Daily minimum temperatures'].values.astype(np.float32)

# Hyperparameters
input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use last hidden state to predict future sequence
        return out

# Custom Dataset for sliding windows
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len, pred_len):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len].reshape(-1, 1)
        y = self.series[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Training function
def train_model(train_data):
    dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
    return model

# Forecasting loop with retraining at each step
available_series = train_series.copy().tolist()
forecasts = []

for i in range(len(test_series)):
    # Train model on all available data
    current_train = np.array(available_series, dtype=np.float32)
    model = train_model(current_train)
    
    # Prepare input for prediction: last seq_len values
    last_seq = torch.tensor(current_train[-seq_len:].reshape(1, seq_len, 1), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).cpu().numpy().flatten()
    forecast_val = pred[0]
    forecasts.append(forecast_val)
    
    # Incorporate true value of this test day for next step
    available_series.append(test_series[i])

print(forecasts)