import pandas as pd
import numpy as np
import torch
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
non_target_cols = [col for col in df.columns if col != '% WEIGHTED ILI']
df[non_target_cols] = df[non_target_cols].fillna(method='ffill')
target = df['% WEIGHTED ILI'].values.astype(np.float32)
train_size = 1018
train = target[:train_size]
test = target[train_size:]
class TemporalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = torch.nn.ReLU()
        self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)
class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 1
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size, dilation))
        self.network = torch.nn.Sequential(*layers)
        self.linear = torch.nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out[:, :, -1]
        return self.linear(out)
input_size = 1
seq_len = 52
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32]
epochs = 50
batch_size = 32
lr = 0.001
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train, seq_len)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations)
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()
model.train()
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size].to(device)
        batch_y = y_train[i:i+batch_size].to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
model.eval()
forecasts = []
current_window = train[-seq_len:].tolist()
for _ in range(len(test)):
    input_seq = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(input_seq).item()
    forecasts.append(pred)
    current_window = current_window[1:] + [pred]
print(forecasts)