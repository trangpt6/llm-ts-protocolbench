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
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
target_col = 'Daily minimum temperatures'
data = df[target_col].values.astype(np.float32)

# Fixed split (Turn 0)
train_size = int(0.8 * len(data))  # 2921
train_data = data[:train_size]
test_data = data[train_size:]

# Model definition
class GRUForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, pred_len=7):
        super(GRUForecaster, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)
        out = self.fc(h_n[-1])  # take last layer's final hidden state
        return out  # (batch, pred_len)

# Training function
def train_model(series, seq_len=7, pred_len=7, hidden_size=16, num_layers=1,
                epochs=5, batch_size=16, lr=0.01):
    # Build input-output pairs
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
    y = torch.tensor(y, dtype=torch.float32)               # (N, pred_len)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRUForecaster(input_size=1, hidden_size=hidden_size,
                          num_layers=num_layers, pred_len=pred_len)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    return model

# Fixed hyperparameters
params = {
    'input_size': 1,
    'seq_len': 7,
    'pred_len': 7,
    'hidden_size': 16,
    'num_layers': 1,
    'epochs': 5,
    'batch_size': 16,
    'lr': 0.01
}

forecasts = []
current_series = train_data.copy()

for i in range(len(test_data)):
    # Retrain model using all available data up to the day before the forecast origin
    model = train_model(current_series,
                        seq_len=params['seq_len'],
                        pred_len=params['pred_len'],
                        hidden_size=params['hidden_size'],
                        num_layers=params['num_layers'],
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        lr=params['lr'])

    # Prepare last sequence
    last_input = current_series[-params['seq_len']:].reshape(1, params['seq_len'], 1)
    with torch.no_grad():
        preds = model(torch.tensor(last_input, dtype=torch.float32)).numpy().flatten()
    # Record 1-step-ahead forecast
    forecasts.append(float(preds[0]))

    # Append newly observed true value (after prediction)
    current_series = np.append(current_series, test_data[i])

print(forecasts)