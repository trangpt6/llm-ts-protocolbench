import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# read dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# no preprocessing needed per Turn 1

# fixed parameters
target_col = 'Ice cream'
train_size = 157
test_size = 40
seq_len = 12
pred_len = 12
input_size = 2
hidden_size = 32
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 30
batch_size = 16
lr = 0.001

# split
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# normalize using training data stats
train_target = train_df[target_col].values
train_mean = train_target.mean()
train_std = train_target.std() + 1e-8

def normalize(x):
    return (x - train_mean) / train_std

def denormalize(x):
    return x * train_std + train_mean

# build sequences with both features (Heater and Ice cream)
def build_sequences(data_df, seq_len, pred_len):
    X, y = [], []
    vals = data_df[['Heater', target_col]].values
    for i in range(len(vals) - seq_len - pred_len + 1):
        X.append(vals[i:i+seq_len])
        y.append(vals[i+seq_len:i+seq_len+pred_len, 1])  # predict Ice cream
    return np.array(X), np.array(y)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        out = self.conv(x)
        # causal: remove extra padding from right
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.norm(out)
        return out + x if x.size(1) == out.size(1) else out

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, seq_len, pred_len):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            d = dilations[i % len(dilations)]
            self.blocks.append(TCNBlock(hidden_size, hidden_size, kernel_size, d))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)  # (batch, hidden_size)
        out = self.fc(x)  # (batch, pred_len)
        return out

device = torch.device('cpu')

def train_model(train_X, train_y):
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, seq_len, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X),
        torch.FloatTensor(train_y)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

def predict_block(model, hist_X):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(hist_X).unsqueeze(0).to(device)
        pred = model(x)
        return pred.cpu().numpy().flatten()

# normalize training data
train_df_norm = train_df.copy()
train_df_norm['Heater'] = normalize(train_df['Heater'].values)
train_df_norm[target_col] = normalize(train_df[target_col].values)

# initial training
train_X, train_y = build_sequences(train_df_norm, seq_len, pred_len)
train_y = normalize(train_y)  # train_y already normalized since built from normalized data
# actually train_y is already normalized because target col is normalized
# rebuild to be safe
train_X, train_y = build_sequences(train_df_norm, seq_len, pred_len)

model = train_model(train_X, train_y)

# block-wise rolling forecast
all_forecasts = []
current_data = train_df_norm.copy()

test_months = test_df['Month'].values
test_heater = test_df['Heater'].values
test_target = test_df[target_col].values

block_start = 0
while block_start < test_size:
    block_end = min(block_start + pred_len, test_size)
    block_len = block_end - block_start
    
    # prepare input sequence from current_data (last seq_len observations)
    hist = current_data[['Heater', target_col]].values[-seq_len:]
    pred_norm = predict_block(model, hist)
    
    # denormalize predictions
    pred = denormalize(pred_norm[:block_len])
    all_forecasts.extend(pred.tolist())
    
    # update with ground truth for next block
    for i in range(block_len):
        idx = block_start + i
        new_row = pd.DataFrame({
            'Month': [test_months[idx]],
            'Heater': [normalize(test_heater[idx])],
            target_col: [normalize(test_target[idx])]
        })
        current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    block_start = block_end
    
    # retrain if more blocks remain
    if block_start < test_size:
        train_X, train_y = build_sequences(current_data, seq_len, pred_len)
        model = train_model(train_X, train_y)

print(all_forecasts)