import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
features = ['Heater', 'Ice cream']
seq_len = 12
pred_len = 1
train_size = 158
train_data = df.iloc[:train_size][features].values
test_data = df.iloc[train_size:][features].values
def create_sequences(data, seq_len, pred_len):
    X = []
    y = []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len].reshape(-1))
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_data, seq_len, pred_len)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilations[i]
            in_ch = input_size if i == 0 else hidden_size
            self.layers.append(TCNBlock(in_ch, hidden_size, kernel_size, dilation))
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = x[:, :, -1]
        out = self.fc(x)
        return out
model = TCN(2, 32, 2, 3, [1, 2, 4, 8], 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
for epoch in range(50):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
current_seq = train_data[-seq_len:].copy()
forecasts = []
model.eval()
with torch.no_grad():
    for _ in range(len(test_data)):
        input_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
        pred = model(input_tensor)
        pred_np = pred.squeeze(0).numpy()
        forecasts.append(float(pred_np[1]))
        current_seq = np.vstack([current_seq[1:], pred_np])
print(forecasts)