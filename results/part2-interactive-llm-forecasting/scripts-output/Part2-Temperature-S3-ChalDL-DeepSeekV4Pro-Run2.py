import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Read data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False)
df = df.sort_values('Date').reset_index(drop=True)
series = df['Daily minimum temperatures'].values.astype(np.float32)

# Split
train_size = 2921
train = series[:train_size]
test = series[train_size:]

# Hyperparameters
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01
input_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data):
    X, y = [], []
    for i in range(len(data)-seq_len-pred_len+1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32)
    return X, y

forecasts = []
current_train = train.copy()

for i in range(len(test)):
    # Prepare data
    X, y = create_sequences(current_train)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    # Predict next 7 days
    model.eval()
    with torch.no_grad():
        last_seq = current_train[-seq_len:].reshape(1, seq_len, 1)
        last_seq = torch.from_numpy(last_seq).float().to(device)
        pred = model(last_seq).cpu().numpy().flatten()
    forecast_today = pred[0]
    forecasts.append(float(forecast_today))

    # Add true value to training for next step
    current_train = np.concatenate([current_train, [test[i]]])

print(forecasts)