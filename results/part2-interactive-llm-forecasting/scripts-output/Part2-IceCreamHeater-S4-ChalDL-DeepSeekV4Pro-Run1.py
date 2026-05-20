import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()

# Define target and features
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']

# Split parameters
total = len(df)
train_size = int(0.8 * total)
test_size = total - train_size
train_end_idx = train_size  # 158

# Fixed hyperparameters
seq_len = 12
pred_len = 12
input_size = 2
hidden_size = 32
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8][:num_layers]  # use first 3
epochs = 30
batch_size = 16
lr = 0.001

# TCN model
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        return out + residual

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        in_channels = input_size
        for i in range(num_layers):
            layers.append(TCNBlock(in_channels, hidden_size, kernel_size, dilations[i]))
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        out = self.network(x)   # (batch, hidden_size, seq_len)
        out = out[:, :, -1]     # last time step
        out = self.fc(out)      # (batch, pred_len)
        return out

def train_tcn(model, X, y, epochs, batch_size, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    model.cpu()

# Forecasting loop
forecasts = []
offset = 0
while offset < test_size:
    current_train_end = train_size + offset
    current_train_df = df.iloc[:current_train_end]
    train_values = current_train_df[feature_cols].values.astype(np.float32)

    scaler = MinMaxScaler()
    scaler.fit(train_values)
    scaled = scaler.transform(train_values)

    X_list, y_list = [], []
    for i in range(len(scaled) - seq_len - pred_len + 1):
        X_list.append(scaled[i:i+seq_len])
        y_list.append(scaled[i+seq_len:i+seq_len+pred_len, 1])
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)

    model = TCN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                kernel_size=kernel_size, dilations=dilations, pred_len=pred_len)
    train_tcn(model, X_arr, y_arr, epochs=epochs, batch_size=batch_size, lr=lr)

    last_seq_scaled = scaled[-seq_len:]
    X_input = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_input).numpy().flatten()

    dummy = np.zeros((pred_len, 2))
    dummy[:, 1] = pred_scaled
    pred_inv = scaler.inverse_transform(dummy)[:, 1]

    remaining = test_size - offset
    take = min(remaining, pred_len)
    forecasts.extend(pred_inv[:take].tolist())
    offset += take

print(forecasts)