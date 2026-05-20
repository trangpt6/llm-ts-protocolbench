import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Apply preprocessing steps
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df['Daily minimum temperatures'].values

# Split data chronologically 80/20
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Fixed hyperparameters
input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Helper function to create sequences
def create_sequences(data_array, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data_array) - seq_length - pred_length + 1):
        X.append(data_array[i : i + seq_length])
        y.append(data_array[i + seq_length : i + seq_length + pred_length])
    return np.array(X), np.array(y)

# Initialize variables for rolling block-wise forecast
current_train = list(train_data)
forecasts = []
test_idx = 0

# Rolling forecast loop
while test_idx < len(test_data):
    # Scale data
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(np.array(current_train).reshape(-1, 1)).flatten()
    
    # Create sequences
    X_train, y_train = create_sequences(scaled_train, seq_len, pred_len)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    # DataLoader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model for retraining
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Inference
    model.eval()
    last_seq = scaled_train[-seq_len:]
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        pred_scaled = model(last_seq_t).cpu().numpy().flatten()
        
    # Inverse transform
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    # Determine steps to take for the current block
    steps_to_take = min(pred_len, len(test_data) - test_idx)
    
    # Store forecasts
    forecasts.extend(pred[:steps_to_take].tolist())
    
    # Update training data with ground truth for the next block
    current_train.extend(test_data[test_idx : test_idx + steps_to_take])
    
    # Advance index
    test_idx += steps_to_take

# Print final forecasts
print(forecasts)