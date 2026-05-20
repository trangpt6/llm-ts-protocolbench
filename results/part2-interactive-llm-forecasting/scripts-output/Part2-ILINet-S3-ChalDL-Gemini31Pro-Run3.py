import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
df = df.set_index('DATE')

# Preprocessing: Handle missing timestamp and interpolate
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_range)
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# Define split
train_size = 1044
test_size = 261
target_col = '% WEIGHTED ILI'
data = df[target_col].values.astype(np.float32).reshape(-1, 1)

# Hyperparameters
input_size = 1
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# GRU Model Definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:(i + seq_len)]
        y = data[(i + seq_len):(i + seq_len + pred_len)]
        xs.append(x)
        ys.append(y.flatten())
    return np.array(xs), np.array(ys)

# Initial training data
train_data = data[:train_size]
device = torch.device('cpu')

# Forecasting with rolling update and retraining
forecasts = []
current_train_data = list(train_data.flatten())

for i in range(test_size):
    # Prepare training set from current available data
    temp_data = np.array(current_train_data).reshape(-1, 1)
    X_train, y_train = create_sequences(temp_data, seq_len, pred_len)
    
    X_train_t = torch.tensor(X_train).float()
    y_train_t = torch.tensor(y_train).float()
    
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Retrain model
    model.train()
    for epoch in range(epochs):
        for j in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[j:j+batch_size]
            batch_y = y_train_t[j:j+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Predict next steps
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(temp_data[-seq_len:]).float().unsqueeze(0)
        pred = model(input_seq).numpy().flatten()
        # Multi-step ahead prediction: we take the first step of the 4-step horizon for the rolling point
        forecasts.append(float(pred[0]))
        
    # Update training data with ground truth (rolling update)
    current_train_data.append(data[train_size + i][0])

print(forecasts)