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
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

train_size = 158
test_size = 40
seq_len = 6
pred_len = 1
input_size = 2
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

forecasts = []
current_train_data = data[:train_size]
test_data = data[train_size:]

for t in range(test_size):
    # Create sequences for current training data
    X_train, y_train = create_sequences(current_train_data, seq_len)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Initialize model
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Predict next step
    model.eval()
    last_seq = current_train_data[-seq_len:]
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(last_seq_tensor)
        forecasts.append(pred.item())
        
    # Update training data with the true value from test data
    current_train_data = np.vstack((current_train_data, test_data[t]))

print(forecasts)