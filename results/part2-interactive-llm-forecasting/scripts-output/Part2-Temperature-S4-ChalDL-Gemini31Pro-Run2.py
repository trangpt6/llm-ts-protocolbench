import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.resample('D').interpolate(method='linear')
data = df['Daily minimum temperatures'].values

# Train/test split
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
train_data = data[:train_size]
test_data = data[train_size:]

# Define sequence creation function
def create_sequences(seq_data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(seq_data) - seq_len - pred_len + 1):
        X.append(seq_data[i:i+seq_len])
        y.append(seq_data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Define training function
def train_model(model, X_train, y_train, epochs, batch_size, lr, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X.unsqueeze(-1))
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Hyperparameters
seq_len = 14
pred_len = 30
input_size = 1
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

# Initial training
X_train, y_train = create_sequences(train_data, seq_len, pred_len)
model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
model = train_model(model, X_train, y_train, epochs, batch_size, lr, device)

# Forecasting with block-wise rolling update
forecasts = []
current_history = list(train_data)
test_idx = 0

while test_idx < len(test_data):
    model.eval()
    last_seq = np.array(current_history[-seq_len:], dtype=np.float32)
    last_seq_tensor = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        pred = model(last_seq_tensor).squeeze().cpu().numpy()
    
    steps_to_take = min(pred_len, len(test_data) - test_idx)
    
    if pred.ndim == 0:
        pred_list = [float(pred)]
    else:
        pred_list = pred.tolist()
        
    forecasts.extend(pred_list[:steps_to_take])
    
    true_block = test_data[test_idx : test_idx + steps_to_take]
    current_history.extend(true_block)
    
    test_idx += steps_to_take
    
    if test_idx < len(test_data):
        X_train_new, y_train_new = create_sequences(current_history, seq_len, pred_len)
        model = train_model(model, X_train_new, y_train_new, epochs, batch_size, lr, device)

print(forecasts)