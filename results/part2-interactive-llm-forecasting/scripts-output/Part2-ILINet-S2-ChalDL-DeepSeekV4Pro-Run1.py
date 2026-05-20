import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
df = pd.read_csv(r'../../../data/ILINet.csv')
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True)
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)
df.reset_index(drop=True, inplace=True)

target = '% WEIGHTED ILI'
data = df[target].values.astype(np.float32)

# Chronological split
total = len(data)
train_size = int(0.8 * total)  # 1050
train_data = data[:train_size]
test_data = data[train_size:]

# Fixed hyperparameters
seq_len = 13
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

def create_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

def train_model(train_series, model, optimizer, criterion):
    model.train()
    X_train, y_train = create_sequences(train_series, seq_len)
    if len(X_train) == 0:
        return
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device).unsqueeze(-1)  # add input_size dim
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

# Initial training on train_data only
model = GRUModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

current_series = train_data.copy()
train_model(current_series, model, optimizer, criterion)

# Forecasting with rolling retraining
forecasts = []
for i in range(len(test_data)):
    # Predict one step ahead
    model.eval()
    with torch.no_grad():
        # Use last seq_len points from current_series
        seq = torch.tensor(current_series[-seq_len:], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(-1)
        pred = model(seq).item()
    forecasts.append(pred)
    # Update current_series with true test value
    current_series = np.append(current_series, test_data[i])
    # Retrain model on expanded series
    model = GRUModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(current_series, model, optimizer, criterion)

print(forecasts)