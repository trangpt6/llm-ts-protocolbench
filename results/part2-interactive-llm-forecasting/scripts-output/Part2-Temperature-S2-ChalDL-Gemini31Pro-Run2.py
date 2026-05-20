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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing: Parse Date, set as index, assign daily frequency
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index.freq = 'D'

# Extract target values
data = df['Daily minimum temperatures'].values

# Split parameters
train_size = 2920
test_size = len(data) - train_size

# Hyperparameters
seq_len = 7
pred_len = 1
input_size = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# Define GRU Model
class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

model = GRUNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Function to create sequences
def create_sequences(ts, seq_length):
    X, y = [], []
    for i in range(len(ts) - seq_length):
        X.append(ts[i:i+seq_length])
        y.append(ts[i+seq_length])
    return np.array(X), np.array(y)

# Initial history is the training set
history = data[:train_size].tolist()
test_data = data[train_size:]

forecasts = []

# Rolling forecast with retraining
for t in range(len(test_data)):
    # Create dataset from current history
    X_train, y_train = create_sequences(history, seq_len)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Retrain model
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
        last_seq = history[-seq_len:]
        last_seq_t = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(last_seq_t)
        forecasts.append(pred.item())
        
    # Update history with ground truth for the next step
    history.append(test_data[t])

# Print only the final forecast list
print(forecasts)