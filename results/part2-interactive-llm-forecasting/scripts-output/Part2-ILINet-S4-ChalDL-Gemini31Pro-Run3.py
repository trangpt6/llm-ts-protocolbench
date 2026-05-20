import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import random

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
target_col = '% WEIGHTED ILI'

# Preprocessing: Handle missing timestamp 2002-01-06
df = df.set_index('DATE').sort_index()
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_range)
df[target_col] = df[target_col].interpolate(method='linear')
# Fill exogenous missing values with 0 as per Turn 1
df = df.fillna(0)

# Split data
train_size = 1044
test_size = 261
train_data = df.iloc[:train_size][target_col].values.reshape(-1, 1)
test_data = df.iloc[train_size:][target_col].values.reshape(-1, 1)

# Scaling
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 1
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

def create_sequences(data, seq_l, pred_l):
    xs, ys = [], []
    for i in range(len(data) - seq_l - pred_l + 1):
        x = data[i:(i + seq_l)]
        y = data[(i + seq_l):(i + seq_l + pred_l)]
        xs.append(x)
        ys.append(y.flatten())
    return np.array(xs), np.array(ys)

def train_model(model, train_x, train_y):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

# Forecasting loop
all_forecasts = []
current_train_scaled = list(train_scaled.flatten())
test_scaled = scaler.transform(test_data).flatten()

# Block-wise rolling update
num_blocks = int(np.ceil(test_size / pred_len))

for b in range(num_blocks):
    # Prepare training data for current block
    train_arr = np.array(current_train_scaled).reshape(-1, 1)
    tx, ty = create_sequences(train_arr, seq_len, pred_len)
    
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len)
    train_model(model, tx, ty)
    
    # Inference
    model.eval()
    with torch.no_grad():
        input_seq = torch.FloatTensor(np.array(current_train_scaled[-seq_len:]).reshape(1, seq_len, 1))
        pred = model(input_seq).numpy().flatten()
        
    # Inverse transform and store
    actual_preds = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    
    # Determine how many steps to take from this block (handle last block)
    steps_to_keep = min(pred_len, test_size - len(all_forecasts))
    all_forecasts.extend(actual_preds[:steps_to_keep].tolist())
    
    # Update training set with ground truth for next block retraining
    if b < num_blocks - 1:
        start_idx = b * pred_len
        end_idx = (b + 1) * pred_len
        current_train_scaled.extend(test_scaled[start_idx:end_idx].tolist())

print(all_forecasts)