import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Read data
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values.astype(np.float32)

# Parameters
seq_len = 13
pred_len = 4
train_size = 1044
test_size = 261
hidden_size = 32
num_layers = 1
batch_size = 16
epochs = 5
lr = 0.01

# Device
device = torch.device('cpu')

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

def train_model(train_data):
    X, y = create_sequences(train_data, seq_len, pred_len)
    X = torch.tensor(X).float().unsqueeze(-1).to(device)  # (samples, seq_len, 1)
    y = torch.tensor(y).float().to(device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRU(1, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Rolling forecast
forecasts = []
for i in range(test_size):
    # Training data includes all available up to current test step
    train_data = target[:train_size + i]
    model = train_model(train_data)
    # Prepare input: last seq_len values from training data
    input_seq = train_data[-seq_len:].reshape(1, seq_len, 1)  # (1, seq_len, 1)
    input_tensor = torch.tensor(input_seq).float().to(device)
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)  # (1, pred_len)
        forecast = pred[0, 0].item()  # first step of 4-step prediction
    forecasts.append(forecast)
    # Ground truth is used implicitly because training data includes the true value at this step

print(forecasts)