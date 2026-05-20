import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device configuration fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Apply preprocessing steps from Turn 1
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index.freq = 'D'

# Extract target column
target_col = 'Daily minimum temperatures'
data = df[target_col].values

# Define train/test split strictly chronologically
train_size = 2921
train_data = data[:train_size]
test_data = data[train_size:]

# Fixed hyperparameters from Turn 2
seq_len = 7
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01
input_size = 1

# Define PyTorch Dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, ts_data, seq_length):
        self.ts_data = ts_data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.ts_data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.ts_data[idx : idx + self.seq_length]
        y = self.ts_data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize history with training data
history = list(train_data)
predictions = []

# Rolling update forecasting
for i in range(len(test_data)):
    # Prepare current historical data
    current_train = np.array(history)
    dataset = TimeSeriesDataset(current_train, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Re-initialize model for retraining at every time step
    model = GRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, pred_len=pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Forecast one-step ahead
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(current_train[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(last_seq)
        pred_value = pred.item()
        predictions.append(pred_value)
        
    # Update history with the true value for the next step
    history.append(test_data[i])

# Print final forecast list
print(predictions)