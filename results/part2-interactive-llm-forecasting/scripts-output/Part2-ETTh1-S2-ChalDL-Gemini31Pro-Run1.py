import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
df = pd.read_csv(r'../../../data/ETTh1.csv')

# Preprocessing
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

data = df['OT'].values

# Hyperparameters
input_size = 7
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

# Feature engineering to match input_size
features = np.zeros((len(data), input_size))
features[:, 0] = data

# Train/test split
train_size = int(0.8 * len(data))
test_size = len(data) - train_size

# Model definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])
        return out

# Dataset creation function
def create_dataset(ts_features, ts_target, seq_len):
    X, y = [], []
    for i in range(len(ts_features) - seq_len):
        X.append(ts_features[i:i+seq_len])
        y.append(ts_target[i+seq_len])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

forecasts = []

# Rolling update and retraining
for i in range(test_size):
    current_features = features[:train_size + i]
    current_target = data[:train_size + i]
    
    X_train, y_train = create_dataset(current_features, current_target, seq_len)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        x_pred = torch.tensor(current_features[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x_pred)
        forecasts.append(pred.item())

print(forecasts)