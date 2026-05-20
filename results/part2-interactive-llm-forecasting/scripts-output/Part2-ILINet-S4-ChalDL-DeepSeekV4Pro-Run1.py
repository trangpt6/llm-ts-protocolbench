import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- Load and preprocess data -----
df = pd.read_csv(r'../../../data/ILINet.csv')
# Drop entirely empty column if present
if 'AGE 25-49' in df.columns:
    df.drop(columns=['AGE 25-49'], inplace=True)
# Extract target
target = df['% WEIGHTED ILI'].values.astype(np.float32)

# Train/test split (Turn 0)
train_size = 1050
test_size = len(target) - train_size
train_data = target[:train_size]
test_data = target[train_size:]

# Parameters from Turn 2
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

# ----- LSTM model definition -----
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(seq_len * hidden_size, pred_len)
    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # out shape: (batch, seq_len, hidden_size)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)

# ----- Scaling and sequence creation -----
def create_sequences(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(Y)

# ----- Training function -----
def train_model(model, X_train, Y_train, epochs, batch_size, lr):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    # No return needed, model updated in-place

# ----- Iterative block-wise forecasting with retraining -----
# We will maintain all known true values (initial train + previously observed test blocks)
known = train_data.copy()
forecasts = []
block_start = 0  # index within test set
num_blocks = int(np.ceil(test_size / pred_len))  # total blocks needed

for blk in range(num_blocks):
    # Determine how many steps to forecast in this block
    steps_to_forecast = min(pred_len, test_size - block_start)
    
    # Retrain model from scratch on all currently known data
    # Scale using current known data
    min_val = known.min()
    max_val = known.max()
    if max_val - min_val < 1e-8:
        scaled_known = known - min_val
    else:
        scaled_known = (known - min_val) / (max_val - min_val)
    X_train_scaled, Y_train_scaled = create_sequences(scaled_known, seq_len, pred_len)
    model = LSTMForecaster().to(device)
    train_model(model, X_train_scaled, Y_train_scaled, epochs, batch_size, lr)
    
    # Prepare input sequence: last seq_len of scaled known values
    last_seq = scaled_known[-seq_len:]
    input_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    # Forecast future scaled values
    model.eval()
    with torch.no_grad():
        pred_scaled = model(input_seq).cpu().numpy().flatten()
    # Inverse scaling
    pred = pred_scaled
    if max_val - min_val >= 1e-8:
        pred = pred * (max_val - min_val) + min_val
    pred = pred.astype(np.float32)
    
    # Keep only needed forecast steps for this block
    block_pred = pred[:steps_to_forecast]
    forecasts.extend(block_pred.tolist())
    
    # Update known data with true values from this block (use ground truth)
    block_true = test_data[block_start:block_start + steps_to_forecast]
    known = np.concatenate([known, block_true])
    
    block_start += steps_to_forecast

# Output final forecast list
print(forecasts)