import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)

# Hyperparameters
input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001

train_size = 115
test_size = len(data) - train_size

# Model definition
class LSTMForecaster(nn.Module):
    def __init__(self):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(ts_data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(ts_data) - seq_length - pred_length + 1):
        X.append(ts_data[i:(i + seq_length)])
        y.append(ts_data[(i + seq_length):(i + seq_length + pred_length)])
    return np.array(X), np.array(y)

def train_and_predict(train_series):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
    
    X_train, y_train = create_sequences(train_scaled, seq_len, pred_len)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = LSTMForecaster().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    last_seq = train_scaled[-seq_len:]
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_scaled = model(last_seq_tensor)
        pred_scaled = pred_scaled.cpu().numpy().reshape(-1, 1)
        
    pred = scaler.inverse_transform(pred_scaled).flatten()
    return pred

# Rolling forecast logic
predictions = []
current_train_end = train_size

while len(predictions) < test_size:
    current_train = data[:current_train_end]
    
    # Train and predict the next block
    block_preds = train_and_predict(current_train)
    
    # Determine how many predictions we actually need from this block
    needed = min(pred_len, test_size - len(predictions))
    predictions.extend(block_preds[:needed].tolist())
    
    # Move the training window forward by the block size
    current_train_end += pred_len

print(predictions)