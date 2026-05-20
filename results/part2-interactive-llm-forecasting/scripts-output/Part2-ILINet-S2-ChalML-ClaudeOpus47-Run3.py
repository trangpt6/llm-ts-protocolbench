import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'

total_rows = len(df)
train_size = int(0.8 * total_rows)
original_test_dates = df['DATE'].iloc[train_size:].tolist()
test_date_set = set(original_test_dates)

df_indexed = df.set_index('DATE')[target_col]
full_idx = pd.date_range(start=df_indexed.index.min(), end=df_indexed.index.max(), freq='7D')
y = df_indexed.reindex(full_idx).ffill()

LAGS = 13

all_dates = y.index.tolist()
predict_positions = [i for i, d in enumerate(all_dates) if d in test_date_set]

forecasts = []

for pos in predict_positions:
    history_values = y.iloc[:pos].values
    X_train = []
    y_train = []
    for i in range(LAGS, len(history_values)):
        X_train.append(history_values[i - LAGS:i])
        y_train.append(history_values[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    x_pred = y.iloc[pos - LAGS:pos].values.reshape(1, -1)
    yhat = model.predict(x_pred)[0]
    forecasts.append(float(yhat))

print(forecasts)