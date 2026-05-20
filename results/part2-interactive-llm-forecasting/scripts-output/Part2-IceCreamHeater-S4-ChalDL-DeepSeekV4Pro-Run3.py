import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()

# Columns: Heater and Ice cream
target_col = 'Ice cream'
exog_col = 'Heater'

# No preprocessing needed (Turn 1)

# Chronological split (Turn 0)
train_size = 158
test_size = 40
train_end_idx = train_size - 1  # index position 157
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:train_size+test_size]

# Fixed hyperparameters (Turn 2)
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

# TCN implementation
class Chomp1d(nn.Module):
    # Remove extra padding from the right side for causal conv
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1)*dilation_size,
                                     dropout=0.0)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)  # output_size = pred_len
    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> permute to (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        y = self.network(x)
        # take the last time step's hidden representations
        y = y[:, :, -1]
        y = self.linear(y)
        return y  # (batch, pred_len)

# Prepare model using num_layers, and hidden_size per channel
# Use hidden_size for all layers
num_channels = [hidden_size] * num_layers
model = TCN(input_size, pred_len, num_channels, kernel_size, dilations[:num_layers])

# Training function
def train_model(train_data, exog_col, target_col, seq_len, pred_len, model, epochs, batch_size, lr):
    # train_data: DataFrame with both columns, all available history before current block
    model.train()
    X, y = [], []
    values_exog = train_data[exog_col].values
    values_target = train_data[target_col].values
    n = len(values_target)
    # create sequences of (seq_len) inputs, and output of next pred_len target values
    for i in range(n - seq_len - pred_len + 1):
        X.append(np.column_stack([values_exog[i:i+seq_len], values_target[i:i+seq_len]]))
        y.append(values_target[i+seq_len:i+seq_len+pred_len])
    X = np.array(X)  # (samples, seq_len, 2)
    y = np.array(y)  # (samples, pred_len)
    if len(X) == 0:
        # not enough data, return model unchanged
        return model, 0.0
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model, 0.0

# Block-wise forecasting using ground truth and retraining
device = 'cpu'  # fallback
model.to(device)

forecasts = []
current_data = train_df.copy()  # holds all available history (initially train set)

# Calculate number of full blocks and remainder
block_size = 12
total_test = len(test_df)
num_full_blocks = total_test // block_size  # 40//12 = 3
remainder = total_test % block_size  # 4

test_indices = test_df.index.tolist()
block_start_idx = 0
for b in range(num_full_blocks):
    # Retrain on current_data
    model = TCN(input_size, pred_len, num_channels, kernel_size, dilations[:num_layers])
    model.to(device)
    model, _ = train_model(current_data, exog_col, target_col, seq_len, pred_len, model, epochs, batch_size, lr)
    # Prepare input sequence: last seq_len points from current_data
    last_seq_exog = current_data[exog_col].values[-seq_len:]
    last_seq_target = current_data[target_col].values[-seq_len:]
    input_seq = np.column_stack([last_seq_exog, last_seq_target])
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 2)
    model.eval()
    with torch.no_grad():
        preds = model(input_tensor).cpu().numpy().flatten()  # (pred_len,)
    # We forecast only the needed steps; block_size exactly matches pred_len, so take all
    forecast_block = preds[:block_size]
    forecasts.extend(forecast_block.tolist())
    # Get true values for this block from test set
    block_true = test_df.iloc[block_start_idx:block_start_idx+block_size]
    # Append true block to current_data
    current_data = pd.concat([current_data, block_true])
    block_start_idx += block_size

if remainder > 0:
    # Retrain on current_data (which includes all previous true blocks)
    model = TCN(input_size, pred_len, num_channels, kernel_size, dilations[:num_layers])
    model.to(device)
    model, _ = train_model(current_data, exog_col, target_col, seq_len, pred_len, model, epochs, batch_size, lr)
    last_seq_exog = current_data[exog_col].values[-seq_len:]
    last_seq_target = current_data[target_col].values[-seq_len:]
    input_seq = np.column_stack([last_seq_exog, last_seq_target])
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(input_tensor).cpu().numpy().flatten()
    # Only take first 'remainder' predictions
    forecast_block = preds[:remainder]
    forecasts.extend(forecast_block.tolist())

print(forecasts)