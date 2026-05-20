import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# reproducibility seed
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
heater = df['Heater'].values
ice = df['Ice cream'].values

lags = 6
train_end = 157  # index of last training point (2017-02)

# initial known series (will be extended with true values during rolling)
heater_all = heater.copy()
ice_all = ice.copy()

forecasts = []

for i in range(158, 198):
    # build features for training: use all known values up to i-1
    # X: lags of heater_all and ice_all, y: ice_all at time t
    # we can create X, y from indices lags to i-1
    X_train = []
    y_train = []
    for t in range(lags, i):
        feat = list(heater_all[t-lags:t]) + list(ice_all[t-lags:t])
        X_train.append(feat)
        y_train.append(ice_all[t])
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(np.array(X_train), np.array(y_train))

    # features for prediction at i: lags from i-lags to i-1
    feat_pred = np.array([list(heater_all[i-lags:i]) + list(ice_all[i-lags:i])])
    pred = model.predict(feat_pred)[0]
    forecasts.append(pred)

    # now true values for month i become known; we keep the original arrays
    # they are already in heater_all and ice_all because we started with full arrays
    # but we need to ensure we don't use future values; the arrays already contain true values for i,
    # so we should not access them before prediction. We set heater_all and ice_all as full arrays initially,
    # but we only use up to i-1 for training and prediction. The true value at i exists, so after prediction
    # we can treat it as known for next iterations. That's fine because we use indices up to i-1, not i.
    # However, if we allow the model to inadvertently have access to future, no, because we restrict t < i.
    # So we can keep full arrays.
    # No need to append, they are already there.

print(forecasts)