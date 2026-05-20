import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 1
            in_channels = input_size if i == 0 else hidden_size
            self.layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation))
            self.layers.append(nn.ReLU())
        self.fc = nn.Linear(hidden_size, 12)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = x[:, :, -1]
        x = self.fc(x)
        return x
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(float)
train_size = 115
train_data = data[:train_size].tolist()
test_data = data[train_size:].tolist()
def create_sequences(data, seq_len, pred_len):
    X = []
    y = []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len : i+seq_len+pred_len])
    return np.array(X), np.array(y)
hyperparams = {'input_size': 1, 'seq_len': 12, 'pred_len': 12, 'hidden_size': 32, 'num_layers': 2, 'epochs': 10, 'batch_size': 8, 'lr': 0.01, 'kernel_size': 3, 'dilations': [1, 2, 4, 8]}
forecasts = []
current_data = train_data.copy()
for step in range(len(test_data)):
    X_train, y_train = create_sequences(current_data, hyperparams['seq_len'], hyperparams['pred_len'])
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    model = TCN(hyperparams['input_size'], hyperparams['hidden_size'], hyperparams['num_layers'], hyperparams['kernel_size'], hyperparams['dilations'])
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(hyperparams['epochs']):
        for i in range(0, len(X_train), hyperparams['batch_size']):
            batch_X = X_train[i:i+hyperparams['batch_size']]
            batch_y = y_train[i:i+hyperparams['batch_size']]
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        last_window = torch.tensor(current_data[-12:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model(last_window)
        next_forecast = pred[0, 0].item()
    forecasts.append(next_forecast)
    current_data.append(test_data[step])
print(forecasts)