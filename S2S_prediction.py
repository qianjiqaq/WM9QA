"""
The code aims to implement a S2S model based on hpyerparameters searched from phase 1 to 4
and
figure ploting
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


DATA_PATH   = Path('/Users/zhangjiawei/Desktop/python_files/warwick_course/predictions/US_1_Retail_FINAL.csv')
LOOK_BACK   = 35
HORIZON     = 5
MAX_EPOCHS  = 100
TEST_RATIO  = 0.2
BATCH_SIZE  = 32
SEED        = 237

np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv(DATA_PATH)
df.rename(columns=lambda x: x.strip(), inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

df['dayofweek']  = df['Date'].dt.dayofweek
df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
df['month']      = df['Date'].dt.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

TARGET_COL      = 'Demand Forecast'
categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(TARGET_COL)
feature_cols = categorical_cols + numeric_cols

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes + 1
scaler = StandardScaler()
df[numeric_cols + [TARGET_COL]] = scaler.fit_transform(df[numeric_cols + [TARGET_COL]])

def make_sequences(frame, look_back, horizon):
    arr = frame[feature_cols + [TARGET_COL]].values
    X_h, X_f, Y = [], [], []
    for i in range(len(arr) - look_back - horizon + 1):
        X_h.append(arr[i : i+look_back, :-1])
        X_f.append(arr[i+look_back : i+look_back+horizon, :-1])
        Y.append(arr[i+look_back : i+look_back+horizon,  -1])
    return np.array(X_h), np.array(X_f), np.array(Y)

X_hist, X_fut, y = make_sequences(df, LOOK_BACK, HORIZON)

n_total = len(X_hist)
n_test  = int(TEST_RATIO * n_total)

X_h_train = X_hist[:-n_test]
X_f_train = X_fut[:-n_test]
y_train   = y[:-n_test]

X_h_test  = X_hist[-n_test:]
X_f_test  = X_fut[-n_test:]
y_test    = y[-n_test:]

d_in = X_h_train.shape[-1]
enc_inputs = tf.keras.Input(shape=(LOOK_BACK, d_in))
e1, h1, c1 = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)(enc_inputs)
e2, h2, c2 = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)(e1)
_,  h3, c3 = tf.keras.layers.LSTM(128, return_state=True)(e2)
dec_inputs = tf.keras.Input(shape=(HORIZON, d_in))

d1, _, _ = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)(
    dec_inputs, initial_state=[h1, c1]
)
d2, _, _ = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)(
    d1, initial_state=[h2, c2]
)
d3, _, _ = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)(
    d2, initial_state=[h3, c3]
)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(d3)
outputs = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

model = tf.keras.Model([enc_inputs, dec_inputs], outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mae',
    metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
             tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

model.fit(
    [X_h_train, X_f_train], y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        TqdmCallback(verbose=0)
    ],
    verbose=0
)

save_path = Path('/Users/zhangjiawei/Desktop/python_files/warwick_course/predictions/best_model.keras')
model.save(save_path)

eval_mae, eval_rmse = model.evaluate([X_h_test, X_f_test], y_test, batch_size=BATCH_SIZE, verbose=0)[1:]
y_pred = model.predict([X_h_test, X_f_test], batch_size=BATCH_SIZE, verbose=0)
r2 = r2_score(y_test.flatten(), y_pred.flatten())

orig_scaler = StandardScaler().fit(df[TARGET_COL].values.reshape(-1,1))
y_test_orig = orig_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
y_pred_orig = orig_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
dates_test = df['Date'].iloc[-n_test:].reset_index(drop=True)
times_vis  = dates_test + pd.Timedelta(days=LOOK_BACK)

print(f"MAE:  {eval_mae:.4f}")
print(f"RMSE: {eval_rmse:.4f}")
print(f"R²:   {r2:.4f}")
plt.figure(figsize=(12,6))
plt.plot(dates_test, y_test_orig[:, 0],  label='Ground truth', marker='o', markersize=3, linestyle='-')
plt.plot(dates_test, y_pred_orig[:, 0], label='Prediction', marker='s', markersize=3, linestyle='--')
plt.title('One-Step Forecast on Last 20% Test Period', pad=12)
plt.xlabel('Date')
plt.ylabel('Demand Forecast')
plt.grid(alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()