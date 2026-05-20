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

# preprocessing: log transform target
df['Ice_cream_log'] = np.log(df['Ice cream'])

# split
train_size = 158
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# fixed hyperparameters
input_size = 2
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 30
batch_size = 16
lr = 0.001

# build sequences with both features
def build_sequences(data_df, seq_len, pred_len):
    X, y = [], []
    vals = data_df[['Heater', 'Ice_cream_log']].values
    target = data_df['Ice_cream_log'].values
    n = len(data_df)
    for i in range(n - seq_len - pred_len + 1):
        X.append(vals[i:i+seq_len])
        y.append(target[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# TCN model
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, seq_len, pred_len, hidden_size, num_layers, kernel_size, dilations):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            out_ch = hidden_size
            dilation = dilations[i % len(dilations)]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size * seq_len, pred_len)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        out = self.network(x)   # (batch, hidden_size, seq_len)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)

device = torch.device('cpu')

def train_model(model, X, y, epochs, batch_size, lr):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        pred = model(X_t)
    return pred.cpu().numpy()

# initial training on train set
X_train, y_train = build_sequences(train_df, seq_len, pred_len)
model = TCN(input_size, seq_len, pred_len, hidden_size, num_layers, kernel_size, dilations)
model = train_model(model, X_train, y_train, epochs, batch_size, lr)

# block-wise rolling forecast on test set
all_forecasts = []
current_data = pd.concat([train_df, test_df], ignore_index=True)
test_start_idx = train_size

# number of blocks: cover 40 test points with blocks of 12
num_blocks = int(np.ceil(len(test_df) / pred_len))

for block in range(num_blocks):
    block_start = test_start_idx + block * pred_len
    block_end = min(block_start + pred_len, len(current_data))
    actual_pred_len = block_end - block_start
    
    # create input sequence ending right before block_start
    seq_start = block_start - seq_len
    seq = current_data[['Heater', 'Ice_cream_log']].iloc[seq_start:block_start].values
    X_input = seq.reshape(1, seq_len, input_size)
    
    # predict
    pred = predict(model, X_input)[0][:actual_pred_len]
    
    # store forecasts (inverse log transform)
    pred_orig = np.exp(pred)
    all_forecasts.extend(pred_orig.tolist())
    
    # update with ground truth for next block (if not last block)
    if block_end < len(current_data):
        # retrain model with updated data including true values up to block_end
        update_df = current_data.iloc[:block_end].copy()
        X_update, y_update = build_sequences(update_df, seq_len, pred_len)
        if len(X_update) > 0:
            model = TCN(input_size, seq_len, pred_len, hidden_size, num_layers, kernel_size, dilations)
            model = train_model(model, X_update, y_update, epochs, batch_size, lr)

# ensure exactly 40 forecasts
forecasts = all_forecasts[:40]
print(forecasts)