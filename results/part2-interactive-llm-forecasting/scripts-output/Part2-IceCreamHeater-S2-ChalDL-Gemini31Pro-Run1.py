import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']
data = df[feature_cols].values.astype(float)

# Split parameters
train_size = 158
test_size = 40
seq_len = 6
input_size = 2
hidden_size = 16
num_layers = 1
lr = 0.01
epochs = 5
batch_size = 8

# Scaling
scaler = MinMaxScaler()
scaler.fit(data[:train_size])
scaled_data = scaler.transform(data)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len, 1]) # Target is Ice cream at index 1
    return np.array(xs), np.array(ys)

def train_model(model, train_x, train_y):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    dataset_size = len(train_x)
    for epoch in range(epochs):
        for i in range(0, dataset_size, batch_size):
            batch_x = torch.FloatTensor(train_x[i:i+batch_size])
            batch_y = torch.FloatTensor(train_y[i:i+batch_size]).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

forecasts = []
# Rolling update strategy: retrain at every step
for i in range(test_size):
    current_train_end = train_size + i
    train_data = scaled_data[:current_train_end]
    
    train_x, train_y = create_sequences(train_data, seq_len)
    
    model = GRUModel(input_size, hidden_size, num_layers)
    model = train_model(model, train_x, train_y)
    
    # Predict next step
    model.eval()
    input_seq = torch.FloatTensor(scaled_data[current_train_end - seq_len:current_train_end]).view(1, seq_len, input_size)
    with torch.no_grad():
        pred = model(input_seq).item()
    
    # Inverse transform prediction
    # Create dummy array for inverse scaling
    dummy = np.zeros((1, input_size))
    dummy[0, 1] = pred
    inv_pred = scaler.inverse_transform(dummy)[0, 1]
    forecasts.append(inv_pred)

print(forecasts)