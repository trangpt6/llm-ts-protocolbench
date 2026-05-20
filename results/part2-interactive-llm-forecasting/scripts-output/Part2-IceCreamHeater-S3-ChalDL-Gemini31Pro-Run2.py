import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Features and target
data = df[['Heater', 'Ice cream']].values
target_idx = 1

train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

seq_len = 6
pred_len = 12
input_size = 2
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Rolling update
forecasts = []
history = list(train_data)

for i in range(len(test_data)):
    # Prepare training data
    X_train = []
    y_train = []
    for j in range(len(history) - seq_len - pred_len + 1):
        X_train.append(history[j:j+seq_len])
        y_train.append(history[j+seq_len:j+seq_len+pred_len])
        
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_target = np.array(y_train)[:, :, target_idx]
    y_train_tensor = torch.tensor(y_train_target, dtype=torch.float32)
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Predict
    last_seq = torch.tensor(np.array([history[-seq_len:]]), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).numpy()[0]
        
    forecasts.extend(pred.tolist())
    
    # Advance by 1 step
    history.append(test_data[i])

print(forecasts)