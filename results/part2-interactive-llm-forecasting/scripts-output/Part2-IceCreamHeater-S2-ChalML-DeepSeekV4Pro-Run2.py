import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Reproducibility seeds
np.random.seed(42)
random_state = 42

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Train/test split (chronological 80/20)
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# Combine all data for iterative expansion
all_data = train_df.copy()

# Hyperparameters
lags = 6
input_features = 2  # Heater and Ice cream

# Feature engineering function: create lag features for both columns
def create_lag_features(data, lags):
    """
    data: DataFrame with columns 'Heater' and 'Ice cream'
    Returns DataFrame with columns: Heater_lag1...Heater_lag6, Icecream_lag1...Icecream_lag6,
    and target Ice cream at time t.
    """
    df_feat = pd.DataFrame()
    for lag in range(1, lags + 1):
        df_feat[f'Heater_lag{lag}'] = data['Heater'].shift(lag)
        df_feat[f'Icecream_lag{lag}'] = data['Ice cream'].shift(lag)
    df_feat['target'] = data['Ice cream']
    # Drop rows where any lag is NaN
    df_feat = df_feat.dropna()
    return df_feat

# Prepare initial training features from start (using all available history up to pre-test)
# First, build features from all_data (which is only train_df initially)
feat_df = create_lag_features(all_data, lags)
X_train = feat_df.drop('target', axis=1)
y_train = feat_df['target']

# Train initial model
model = XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=random_state,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

forecasts = []

# Rolling one-step ahead forecasting with retraining after each step
for idx, test_row in test_df.iterrows():
    # Predict for current test month using the most recent 6 months in all_data
    # The all_data already includes up to the immediately prior month
    last_6 = all_data.tail(lags)
    # Build feature vector: Heater lags 1..6, Icecream lags 1..6
    features = []
    for lag in range(lags, 0, -1):
        features.append(last_6['Heater'].iloc[-lag])
    for lag in range(lags, 0, -1):
        features.append(last_6['Ice cream'].iloc[-lag])
    X_pred = np.array(features).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)

    # Append true observation of this test month to all_data
    all_data = pd.concat([all_data, test_row.to_frame().T], ignore_index=True)

    # Retrain using updated all_data (lag features from expanded data)
    feat_df = create_lag_features(all_data, lags)
    X_retrain = feat_df.drop('target', axis=1)
    y_retrain = feat_df['target']
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
        objective='reg:squarederror'
    )
    model.fit(X_retrain, y_retrain)

# Output the forecast list
print(forecasts)