import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Fallback to CPU
device = torch.device('cpu')

# Read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# Preprocessing: Parse Month as datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Extract values
data_values = df[['Heater', 'Ice cream']].values

# Hyperparameters
input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

train_size = 158
test_size = len(df) - train_size

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

forecasts = []

for t in range(test_size):
    current_data = data_values[:train_size + t]
    
    X_train = []
    y_train = []
    for i in range(len(current_data) - seq_len - pred_len + 1):
        X_train.append(current_data[i : i + seq_len, :])
        y_train.append(current_data[i + seq_len : i + seq_len + pred_len, 1])
        
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    X_test = torch.tensor(current_data[-seq_len:, :], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(X_test).squeeze(0).cpu().numpy()
        
    forecasts.append(float(pred[0]))

print(forecasts)