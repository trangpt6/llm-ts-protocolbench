import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
# Features: Heater and Ice cream (input_size=2)
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

# Split data
train_size = 158
test_size = 40
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
input_size = 2
seq_len = 6
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len, 1] # Target is Ice cream
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Initial training data
current_train_data = list(train_data)
forecasts = []

# Rolling update forecasting
for i in range(test_size):
    # Prepare training sequences
    train_arr = np.array(current_train_data)
    X_train, y_train = create_sequences(train_arr, seq_len)
    
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).view(-1, 1)
    
    # Initialize model and optimizer
    model = GRUModel(input_size, hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
    # Forecast one step ahead
    model.eval()
    # Input for prediction is the last seq_len observations from current_train_data
    last_seq = np.array(current_train_data[-seq_len:]).reshape(1, seq_len, input_size)
    last_seq_t = torch.tensor(last_seq).float()
    
    with torch.no_grad():
        pred = model(last_seq_t).item()
        forecasts.append(pred)
        
    # Update training data with ground truth for next step
    current_train_data.append(test_data[i])

print(forecasts)