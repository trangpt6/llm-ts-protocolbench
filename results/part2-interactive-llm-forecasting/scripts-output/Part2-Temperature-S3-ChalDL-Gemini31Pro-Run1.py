import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index.freq = 'D'
data = df['Daily minimum temperatures'].values.reshape(-1, 1)

# Train/test split
train_size = 2921
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# Model definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Rolling forecast setup
scaler = StandardScaler()
history = list(train_data.flatten())
test_values = list(test_data.flatten())
forecasts = []

for t in range(len(test_values)):
    # Prepare current history
    current_history = np.array(history).reshape(-1, 1)
    scaled_history = scaler.fit_transform(current_history)
    
    X_train, y_train = [], []
    for i in range(len(scaled_history) - seq_len - pred_len + 1):
        X_train.append(scaled_history[i:i+seq_len])
        y_train.append(scaled_history[i+seq_len:i+seq_len+pred_len])
        
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).squeeze(-1).to(device)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize and train model
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Forecast
    model.eval()
    last_seq = scaled_history[-seq_len:]
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_scaled = model(last_seq_tensor).cpu().numpy()
        
    pred_scaled = pred_scaled.reshape(-1, 1)
    pred_unscaled = scaler.inverse_transform(pred_scaled).flatten()
    
    # Store the 1-step ahead prediction to match test set length
    forecasts.append(float(pred_unscaled[0]))
    
    # Update history with the true value for the next iteration
    history.append(test_values[t])

print(forecasts)