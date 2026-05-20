import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset

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
input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01

# Split sizes
train_size = 115
test_size = len(data) - train_size

# Model definition
class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Function to create sequences
def create_sequences(ts, seq_length):
    X, y = [], []
    for i in range(len(ts) - seq_length):
        X.append(ts[i:i+seq_length])
        y.append(ts[i+seq_length])
    return np.array(X), np.array(y)

forecasts = []
history = list(data[:train_size])

# Rolling forecast with retraining
for t in range(test_size):
    # Prepare training data from current history
    X_train, y_train = create_sequences(history, seq_len)
    
    # Reshape for GRU: (batch, seq_len, input_size)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and optimizer for retraining
    model = GRUModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
        x_input = np.array(history[-seq_len:], dtype=np.float32)
        x_input = torch.tensor(x_input).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(x_input).item()
        
    forecasts.append(pred)
    
    # Update history with ground truth for the next step
    history.append(data[train_size + t])

print(forecasts)