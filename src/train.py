import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.config import config
from src.predict_utils import *

# -----------------------------
# Config
# -----------------------------
TICKER = sys.argv[1]
LOOK_BACK = config.predict.look_back
HORIZON = config.predict.horizon
N_SPLITS = config.predict.n_splits
MIN_TRAIN_SIZE = config.predict.min_train_size
RANDOM_SEED = config.predict.random_seed
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

OUTDIR = "models"
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON = os.path.join(OUTDIR, f"{TICKER}_metrics.json")
OUT_MODEL = os.path.join(OUTDIR, f"{TICKER}_gru.h5")

# -----------------------------
# Train
# -----------------------------
df_raw = download_data(TICKER)
df = add_features(df_raw)
df = build_targets(df, HORIZON)

exclude = {"Open","High","Low","Close","Adj Close","log_close","log_ret"}
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns]
else:
    df.columns = [str(c) for c in df.columns]
feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("cum_logret_h")]
feature_cols = sorted(feature_cols)  # keep order stable

X_all, Y_all, last_obs_all = create_supervised_arrays(df, feature_cols, HORIZON, LOOK_BACK)
n_samples, _, n_features = X_all.shape

# train/test set
split_idx = int(0.8 * n_samples)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
Y_train, Y_test = Y_all[:split_idx], Y_all[split_idx:]
last_test = last_obs_all[split_idx:]

# Normalisation
scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
X_train_s = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

# GRU model
tf.keras.backend.clear_session()
gru = Sequential([
    GRU(64, return_sequences=False, input_shape=(LOOK_BACK, n_features)),
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dropout(0.15),
    Dense(HORIZON)
])
gru.compile(optimizer="adam", loss=tf.keras.losses.Huber())
es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, min_delta=1e-6)

gru.fit(
    X_train_s, Y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=0
)

# Évaluation
preds = gru.predict(X_test_s, verbose=0)
true_prices = inverse_cum_logret_to_price(last_test, Y_test)
pred_prices = inverse_cum_logret_to_price(last_test, preds)
metrics = per_horizon_metrics(true_prices, pred_prices)
metrics["directional_accuracy"] = directional_accuracy(true_prices, pred_prices, last_test)

# Sauvegarde
gru.save(OUT_MODEL)
with open(OUT_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

np.save(os.path.join(OUTDIR, f"{TICKER}_scaler_mean.npy"), scaler.mean_)
np.save(os.path.join(OUTDIR, f"{TICKER}_scaler_scale.npy"), scaler.scale_)

