import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values.astype(np.float32)

# Define split point
train_size = int(0.8 * len(target))
train_data = target[:train_size]
test_data = target[train_size:]

# Fixed Hyperparameters
seq_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_size_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

# Initial training data
current_train_stream = list(train_data)
predictions = []

# Rolling update with retraining
for i in range(len(test_data)):
    # Prepare sequences from current stream
    x_train, y_train = create_sequences(np.array(current_train_stream), seq_len)
    x_train_tensor = torch.from_numpy(x_train).unsqueeze(-1)
    y_train_tensor = torch.from_numpy(y_train).unsqueeze(-1)
    
    # Initialize model and optimizer
    model = GRUModel(1, hidden_size, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train model
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
    # Forecast next step
    model.eval()
    with torch.no_grad():
        x_input = torch.from_numpy(np.array(current_train_stream[-seq_len:])).view(1, seq_len, 1)
        pred = model(x_input).item()
        predictions.append(pred)
        
    # Update stream with ground truth
    current_train_stream.append(test_data[i])

print(predictions)