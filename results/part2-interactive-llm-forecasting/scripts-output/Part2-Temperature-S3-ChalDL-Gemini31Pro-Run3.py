import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df['Daily minimum temperatures'].values.astype(np.float32)

# Train/test split
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

# Minimal preprocessing: scaling based on train data
train_raw = data[:train_size]
data_mean = np.mean(train_raw)
data_std = np.std(train_raw)
data_scaled = (data - data_mean) / data_std

# Hyperparameters
input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# Sequence generation function
def create_sequences(ts, seq_len, pred_len):
    X, y = [], []
    for i in range(len(ts) - seq_len - pred_len + 1):
        X.append(ts[i:i+seq_len])
        y.append(ts[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

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

# Initialize model, loss, optimizer
model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Initial training
current_train_data = list(data_scaled[:train_size])
X_train, y_train = create_sequences(current_train_data, seq_len, pred_len)
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Rolling forecast and retraining
forecasts = []

for i in range(test_size):
    model.eval()
    last_seq = current_train_data[-seq_len:]
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        pred = model(last_seq_t).cpu().numpy()[0]
    
    # Extract the 1-step ahead prediction for the continuous forecast list
    pred_unscaled = pred[0] * data_std + data_mean
    forecasts.append(float(pred_unscaled))
    
    # Update training data with the true value
    true_val = data_scaled[train_size + i]
    current_train_data.append(true_val)
    
    # Retrain model
    X_retrain, y_retrain = create_sequences(current_train_data, seq_len, pred_len)
    X_retrain_t = torch.tensor(X_retrain, dtype=torch.float32).unsqueeze(-1).to(device)
    y_retrain_t = torch.tensor(y_retrain, dtype=torch.float32).to(device)
    
    retrain_dataset = TensorDataset(X_retrain_t, y_retrain_t)
    retrain_loader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in retrain_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

print(forecasts)