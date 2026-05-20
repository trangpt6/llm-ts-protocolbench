import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)

# Hyperparameters
seq_len = 12
pred_len = 1
epochs = 10
batch_size = 8
lr = 0.01
train_size = 115
test_size = 29

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

forecasts = []

# Rolling update and retraining
for step in range(test_size):
    # Current historical data up to the current step
    current_train = data[:train_size + step]
    
    # Prepare sequences
    X_train, y_train = [], []
    for i in range(len(current_train) - seq_len):
        X_train.append(current_train[i:i+seq_len])
        y_train.append(current_train[i+seq_len])
        
    X_train_tensor = torch.tensor(np.array(X_train)).unsqueeze(-1).to(device)
    y_train_tensor = torch.tensor(np.array(y_train)).unsqueeze(-1).to(device)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Re-initialize model for retraining at every step
    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train model
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Forecast next step
    model.eval()
    with torch.no_grad():
        last_seq = current_train[-seq_len:]
        last_seq_tensor = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(last_seq_tensor)
        forecasts.append(float(pred.item()))

# Print final forecasts
print(forecasts)