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

# Read dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Preprocessing: winsorize anomalous spikes
df.loc[df['Month'] == '2011-04-01', 'Ice cream'] = 25
df.loc[df['Month'] == '2016-12-01', 'Ice cream'] = 25

# Split: first 157 for train, remaining 40 for test
train_size = 157
test_size = 40
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

# Prepare data: use both Heater and Ice cream as features
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 1])  # predict Ice cream
    return np.array(X), np.array(y)

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

# Normalize data using training statistics
all_data = df[['Heater', 'Ice cream']].values.astype(np.float32)
train_data = all_data[:train_size]
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
std[std == 0] = 1
normalized_data = (all_data - mean) / std

# Rolling forecast with retraining
forecasts = []
current_train_end = train_size

for step in range(test_size):
    # Build training sequences from all data up to current point
    train_slice = normalized_data[:current_train_end]
    X_train, y_train = create_sequences(train_slice, seq_len, pred_len)
    
    if len(X_train) == 0:
        break
    
    # Train model
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Predict: use last seq_len observations
    model.eval()
    with torch.no_grad():
        x_input = normalized_data[current_train_end - seq_len:current_train_end]
        x_input = torch.FloatTensor(x_input).unsqueeze(0).to(device)
        pred = model(x_input).cpu().numpy()[0]
    
    # Denormalize prediction
    pred_denorm = pred * std[1] + mean[1]
    
    # For this step, we need forecast for position current_train_end (0-indexed in all_data)
    # pred[0] corresponds to forecast for next step
    forecasts.append(float(pred_denorm[0]))
    
    # Update: add true value to training data for next iteration
    current_train_end += 1

print(forecasts)