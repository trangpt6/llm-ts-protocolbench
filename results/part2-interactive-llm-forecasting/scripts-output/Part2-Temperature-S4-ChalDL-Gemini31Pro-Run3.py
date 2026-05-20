import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing from Turn 1
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Target column
target_col = 'Daily minimum temperatures'
data = df[target_col].values

# Split
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

# Model definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data_array, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(data_array) - seq_len - pred_len + 1):
        xs.append(data_array[i:(i + seq_len)])
        ys.append(data_array[(i + seq_len):(i + seq_len + pred_len)])
    return np.array(xs), np.array(ys)

def train_model(model, train_array):
    xs, ys = create_sequences(train_array, seq_len, pred_len)
    if len(xs) == 0:
        return
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(xs, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(ys, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

# Initial training
model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len).to(device)
current_train = list(train_data)
train_model(model, current_train)

# Forecasting
forecasts = []
block_size = 30

for i in range(0, len(test_data), block_size):
    # Predict
    x_input = current_train[-seq_len:]
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor)
    pred_list = pred[0].cpu().tolist()
    
    # Determine how many steps we actually need for this block
    steps_needed = min(block_size, len(test_data) - i)
    forecasts.extend(pred_list[:steps_needed])
    
    # Update training data with ground truth
    block_true = test_data[i : i + steps_needed]
    current_train.extend(block_true)
    
    # Retrain if there are more blocks to predict
    if i + block_size < len(test_data):
        model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len).to(device)
        train_model(model, current_train)

print(forecasts)