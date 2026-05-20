import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read CSV
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing: drop empty columns
if 'AGE 25-49' in df.columns:
    df.drop(columns=['AGE 25-49'], inplace=True)
if 'AGE 25-64' in df.columns:
    df.drop(columns=['AGE 25-64'], inplace=True)

# Extract target
target_col = '% WEIGHTED ILI'
series = df[target_col].values.astype(np.float32)

# Chronological split
train_size = 1050
test_size = len(series) - train_size

# Fixed hyperparameters
SEQ_LEN = 13
PRED_LEN = 1
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 5
BATCH_SIZE = 16
LR = 0.01

# GRU Model
class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUForecaster, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # Take last hidden state output
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

# Function to create sequences
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(data_series, seq_len, epochs, batch_size, lr):
    model = GRUForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X, y = create_sequences(data_series, seq_len, PRED_LEN)
    if len(X) == 0:
        return model  # no training possible, return random init
    
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Forecasting with rolling update and retraining
history = series[:train_size].copy()
forecasts = []

# Current time steps for test (positions in original series)
for t in range(train_size, len(series)):
    # Input: last SEQ_LEN values from history (up to t-1)
    input_seq = history[-SEQ_LEN:]
    # Convert to tensor
    model_input = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    # Train model on all data available so far (history)
    model = train_model(history, SEQ_LEN, EPOCHS, BATCH_SIZE, LR)
    model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(model_input).item()
    forecasts.append(pred)
    
    # Append true value (since ground truth enabled after prediction)
    true_val = series[t]
    history = np.append(history, true_val)

# Print only the forecast list
print(forecasts)