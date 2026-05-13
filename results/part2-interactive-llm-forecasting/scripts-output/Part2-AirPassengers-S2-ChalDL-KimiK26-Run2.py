import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Read the dataset from the local CSV file
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# Extract the primary target series; no preprocessing is needed per Turn 1
series = df['Passengers'].values.astype(np.float32)
series = series.reshape(-1, 1)

# Define the chronological train/test split as decided in Turn 0
train_size = 115
test_size = 29

# Fixed hyperparameters from Turn 2
input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01

# Use CPU if CUDA is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(data):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return torch.from_numpy(np.array(xs)), torch.from_numpy(np.array(ys))

forecasts = []
current_train = series[:train_size].copy()

for step in range(test_size):
    X_train, y_train = create_sequences(current_train)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = GRUModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        x_input = current_train[-seq_len:]
        x_input = torch.from_numpy(x_input).unsqueeze(0).to(device)
        pred = model(x_input)
        forecasts.append(pred.cpu().item())
    
    true_value = series[train_size + step]
    current_train = np.concatenate([current_train, true_value.reshape(1, -1)], axis=0)

print(forecasts)