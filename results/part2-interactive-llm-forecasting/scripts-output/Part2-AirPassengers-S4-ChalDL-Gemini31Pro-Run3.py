import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.reshape(-1, 1)

# Split data
train_size = int(0.8 * len(data))
train_data_raw = data[:train_size]
test_data_raw = data[train_size:]

# Hyperparameters
input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(data_array, seq_length, pred_length):
    X, Y = [], []
    for i in range(len(data_array) - seq_length - pred_length + 1):
        X.append(data_array[i : i + seq_length])
        Y.append(data_array[i + seq_length : i + seq_length + pred_length])
    return np.array(X), np.array(Y)

def train_model(train_array):
    X, Y = create_sequences(train_array, seq_len, pred_len)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            
    return model

history_raw = train_data_raw.copy()
test_idx = 0
forecasts = []

while test_idx < len(test_data_raw):
    scaler = MinMaxScaler()
    history_scaled = scaler.fit_transform(history_raw)
    
    model = train_model(history_scaled)
    model.eval()
    
    last_seq = history_scaled[-seq_len:]
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_scaled = model(last_seq_tensor).cpu().numpy()
        
    pred_scaled = pred_scaled.reshape(-1, 1)
    pred_raw = scaler.inverse_transform(pred_scaled).squeeze()
    
    if pred_raw.ndim == 0:
        pred_list = [float(pred_raw)]
    else:
        pred_list = pred_raw.tolist()
        
    steps_to_take = min(pred_len, len(test_data_raw) - test_idx)
    forecasts.extend(pred_list[:steps_to_take])
    
    history_raw = np.concatenate((history_raw, test_data_raw[test_idx : test_idx + steps_to_take]))
    test_idx += steps_to_take

print(forecasts)