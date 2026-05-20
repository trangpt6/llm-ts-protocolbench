import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# fallback to cpu if cuda is unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
values = df['Daily minimum temperatures'].values.astype(np.float32)

# chronological split: first 80% train, remaining 20% test
train_size = 2920
train_data = values[:train_size]
test_data = values[train_size:]

# create sliding windows for training
seq_len = 14
pred_len = 1
X_train = []
y_train = []
for i in range(seq_len, len(train_data)):
    X_train.append(train_data[i - seq_len:i])
    y_train.append(train_data[i])
X_train = np.array(X_train).reshape(-1, seq_len, 1)
y_train = np.array(y_train).reshape(-1, pred_len)

# convert to tensors and create dataloader
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# define TCN with dilated causal convolutions
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super(TCN, self).__init__()
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        in_ch = input_size
        for i in range(num_layers):
            d = dilations[i]
            self.convs.append(nn.Conv1d(in_ch, hidden_size, kernel_size, dilation=d))
            self.acts.append(nn.ReLU())
            in_ch = hidden_size
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)
        for conv, act in zip(self.convs, self.acts):
            p = (conv.kernel_size[0] - 1) * conv.dilation[0]
            x = nn.functional.pad(x, (p, 0))
            x = conv(x)
            x = act(x)
        x = x[:, :, -1]
        return self.fc(x)

# initialize model, loss, and optimizer
model = TCN(input_size=1, hidden_size=64, num_layers=3, kernel_size=3, dilations=[1, 2, 4, 8], pred_len=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train once on training set
model.train()
for epoch in range(50):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# recursive one-step ahead forecasting over full test set without ground truth updates
model.eval()
history = train_data[-seq_len:].tolist()
forecasts = []
with torch.no_grad():
    for _ in range(len(test_data)):
        x = torch.tensor(history[-seq_len:]).float().view(1, seq_len, 1).to(device)
        pred = model(x).item()
        forecasts.append(pred)
        history.append(pred)

print(forecasts)