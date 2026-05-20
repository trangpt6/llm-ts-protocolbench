import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import StandardScaler

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

# Preprocessing steps from Turn 1
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Extract target variable
data = df['Daily minimum temperatures'].values

# Train/test split
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
train_data = data[:train_size]
test_data = data[train_size:]

# Fixed hyperparameters
input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# Model definition
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

# Function to create sequences
def create_sequences(data_array, seq_length, pred_length):
    xs = []
    ys = []
    for i in range(len(data_array) - seq_length - pred_length + 1):
        xs.append(data_array[i:(i + seq_length)])
        ys.append(data_array[(i + seq_length):(i + seq_length + pred_length), 0])
    return np.array(xs), np.array(ys)

# Initialize history and forecasts
history = list(train_data)
forecasts = []

# Rolling update and forecasting
for i in range(len(test_data)):
    # Scale data
    scaler = StandardScaler()
    scaled_history = scaler.fit_transform(np.array(history).reshape(-1, 1))
    
    # Create training sequences
    X_train, y_train = create_sequences(scaled_history, seq_len, pred_len)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    # Initialize model for retraining at every time step
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size(0))
        for b in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[b:b+batch_size]
            batch_X = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Forecast
    model.eval()
    with torch.no_grad():
        last_seq = scaled_history[-seq_len:]
        last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred_scaled = model(last_seq_tensor).cpu().numpy()[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
    # Store the 1-step ahead forecast to cover the test set exactly
    forecasts.append(float(pred[0]))
    
    # Update history with the true value
    history.append(float(test_data[i]))

# Print final forecasts
print(forecasts)