import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)

# Split data
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01

# Define model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Create sequences
def create_sequences(seq_data, seq_len):
    X = []
    y = []
    for i in range(len(seq_data) - seq_len):
        X.append(seq_data[i:i+seq_len])
        y.append(seq_data[i+seq_len])
    return np.array(X), np.array(y)

forecasts = []
history = list(train_data)

# Rolling forecast loop
for t in range(len(test_data)):
    X_train, y_train = create_sequences(history, seq_len)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
    
    torch.manual_seed(42)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    last_seq = history[-seq_len:]
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        pred = model(last_seq_tensor)
        
    forecasts.append(float(pred.item()))
    history.append(test_data[t])

print(forecasts)