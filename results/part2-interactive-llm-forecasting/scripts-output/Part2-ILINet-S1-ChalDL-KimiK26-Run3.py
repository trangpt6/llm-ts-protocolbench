import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# read dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

# preprocessing: remove rows where target is exactly 0.0 and all other columns are exactly 0.0
target_zero = df['% WEIGHTED ILI'] == 0.0
other_cols = df.drop(columns=['DATE', '% WEIGHTED ILI'])
for col in other_cols.columns:
    other_cols[col] = pd.to_numeric(other_cols[col], errors='coerce')
all_others_zero = (other_cols == 0.0).all(axis=1)
df = df[~(target_zero & all_others_zero)].copy()
df = df.sort_values('DATE').reset_index(drop=True)

# strict chronological split per Turn 0
train_df = df[df['DATE'] <= '2017-09-24'].copy()
test_df = df[df['DATE'] >= '2017-10-01'].copy()

train_y = train_df['% WEIGHTED ILI'].values.astype(np.float32)
test_y = test_df['% WEIGHTED ILI'].values.astype(np.float32)

seq_len = 52
pred_len = 1

# sliding windows for training
def create_sequences(data, seq_len, pred_len):
    X = []
    y = []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

train_X, train_Y = create_sequences(train_y, seq_len, pred_len)
train_X = torch.from_numpy(train_X).unsqueeze(-1)
train_Y = torch.from_numpy(train_Y).unsqueeze(-1)

dataset = torch.utils.data.TensorDataset(train_X, train_Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# TCN definition
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        out = self.conv(x)
        out = self.relu(out)
        return out

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        self.pred_len = pred_len
        layers = []
        in_ch = input_size
        for i in range(num_layers):
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, dilations[i]))
            in_ch = hidden_size
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len * input_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.net(x)
        out = out[:, :, -1]
        out = self.fc(out)
        out = out.view(out.size(0), self.pred_len, -1)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dilations = [1, 2, 4, 8, 16, 32][:3]
model = TCN(input_size=1, hidden_size=64, num_layers=3, kernel_size=3, dilations=dilations, pred_len=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training
model.train()
for epoch in range(50):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

# recursive one-step ahead forecasting without ground truth
model.eval()
history = list(train_y[-seq_len:])
forecasts = []
with torch.no_grad():
    for _ in range(len(test_y)):
        x = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(x)
        val = pred.item()
        forecasts.append(val)
        history.append(val)

print(forecasts)