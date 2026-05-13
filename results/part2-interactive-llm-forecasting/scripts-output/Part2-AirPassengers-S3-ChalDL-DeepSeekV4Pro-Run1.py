import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'])
series = df['Passengers'].values.astype(np.float32)

# Split
train_size = 115
train_series = series[:train_size]
test_series = series[train_size:]

# Hyperparameters
input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 2
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 10
batch_size = 8
lr = 0.01

# TCN model definition
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = input_size
        for i in range(num_layers):
            dilation = dilations[i]
            padding = (kernel_size - 1) * dilation
            self.layers.append(
                nn.Conv1d(in_channels, hidden_size, kernel_size,
                          padding=padding, dilation=dilation)
            )
            in_channels = hidden_size
        # Final 1x1 convolution to reduce hidden_size to input_size
        self.output_conv = nn.Conv1d(hidden_size, input_size, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, input_size, seq_len)
        for layer in self.layers:
            x = self.relu(layer(x))
        # After last layer, apply output conv
        x = self.output_conv(x)   # shape: (batch, input_size, seq_len)
        x = x[:, 0, :]            # shape: (batch, seq_len)
        return x

def create_sequences(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    if len(X) == 0:
        return None, None
    return np.array(X), np.array(Y)

def train_model(history):
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X, Y = create_sequences(history, seq_len, pred_len)
    if X is None:
        return model
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2).to(device)  # (samples, seq_len, 1) -> permute to (samples, 1, seq_len)
    X_tensor = X_tensor.permute(0, 2, 1)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Rolling forecasting with retraining
current_history = train_series.copy()
forecasts = []

for i in range(len(test_series)):
    # Retrain model on current history
    model = train_model(current_history)
    # Prepare input: last seq_len points
    X_input = current_history[-seq_len:].reshape(1, 1, seq_len)  # (1, input_size, seq_len)
    X_input = torch.tensor(X_input, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X_input).squeeze().cpu().numpy()  # shape (pred_len,)
    # Take first predicted value as 1-step-ahead forecast
    forecast_val = pred[0]
    forecasts.append(forecast_val)
    # Append true test value to history for next step
    current_history = np.append(current_history, test_series[i])

print(forecasts)