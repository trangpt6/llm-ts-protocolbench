import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
target_col = '% WEIGHTED ILI'

# Preprocessing: Handle missing timestamp 2002-01-06
df = df.set_index('DATE').resample('W-SUN').asfreq()
df[target_col] = df[target_col].interpolate(method='linear')

# Preprocessing: Fill exogenous columns with 0 as per Turn 1
df = df.fillna(0)

# Define split based on Turn 0
train_size = 1044
test_size = 261
full_data = df[target_col].values.reshape(-1, 1)

# Scaling
scaler = StandardScaler()
scaler.fit(full_data[:train_size])

# GRU Model Definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters from Turn 2
seq_len = 13
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

forecasts = []

# Rolling forecast with retraining at each step
for i in range(test_size):
    # Current training data includes all observations up to the current test point
    current_train_end = train_size + i
    train_data = full_data[:current_train_end]
    scaled_train = scaler.transform(train_data)
    
    # Prepare sequences
    X_train, y_train = [], []
    for j in range(len(scaled_train) - seq_len):
        X_train.append(scaled_train[j:j+seq_len])
        y_train.append(scaled_train[j+seq_len])
    
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    
    # Initialize model, loss, and optimizer
    model = GRUModel(1, hidden_size, num_layers, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for b in range(0, len(X_train), batch_size):
            X_batch = X_train[b:b+batch_size]
            y_batch = y_train[b:b+batch_size]
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Inference for the next step
    model.eval()
    # Use the last seq_len observations to predict the next value
    input_seq = scaled_train[-seq_len:].reshape(1, seq_len, 1)
    input_seq = torch.tensor(input_seq, dtype=torch.float32)
    
    with torch.no_grad():
        pred = model(input_seq)
        # Inverse transform to original scale
        pred_val = scaler.inverse_transform(pred.numpy())[0][0]
        forecasts.append(float(pred_val))

# Output the final forecast list
print(forecasts)