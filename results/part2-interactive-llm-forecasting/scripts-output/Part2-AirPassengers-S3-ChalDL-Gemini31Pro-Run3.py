import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                               padding=(kernel_size-1) * dilation_size)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        y1 = self.network(x.transpose(1, 2))
        return self.linear(y1[:, :, -1])

def create_sequences(data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        xs.append(data[i:(i + seq_len)])
        ys.append(data[(i + seq_len):(i + seq_len + pred_len)])
    return np.array(xs), np.array(ys)

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(float).reshape(-1, 1)

# Split
train_size = 115
test_size = 29
train_data = data[:train_size]
test_data = data[train_size:]

# Setup parameters
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 2
lr = 0.01
epochs = 10
batch_size = 8
kernel_size = 3
num_channels = [hidden_size] * num_layers

device = torch.device('cpu')
all_forecasts = []

# Rolling update with retraining
# We need to predict test_size points.
# At each step t in test set, we use data up to t to predict t+1...t+12.
# We only record the first prediction of each horizon to cover the test set.
current_full_data = list(train_data.flatten())

for i in range(test_size):
    # Prepare training data from current_full_data
    train_arr = np.array(current_full_data).reshape(-1, 1)
    X_train, y_train = create_sequences(train_arr, seq_len, pred_len)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).squeeze(-1).to(device)
    
    # Initialize and train model
    model = TCNModel(1, pred_len, num_channels, kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for j in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[j:j+batch_size]
            batch_y = y_train_t[j:j+batch_size]
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
    # Forecast
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(train_arr[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(last_seq).cpu().numpy().flatten()
        # We take the first step of the multi-step forecast as the prediction for the current test point
        all_forecasts.append(float(pred[0]))
        
    # Update with ground truth
    current_full_data.append(test_data[i][0])

print(all_forecasts)