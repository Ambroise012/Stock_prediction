import os
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.model import save_lstm_model, train_pytorch_lstm
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
torch.manual_seed(RANDOM_SEED)

OUTDIR = "models"
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON = os.path.join(OUTDIR, f"{TICKER}_metrics.json")
OUT_MODEL = os.path.join(OUTDIR, f"{TICKER}_lstm.pt")

# -----------------------------
# Train
# -----------------------------
# Download and preprocess data
df_raw = download_data(TICKER)
df = add_features(df_raw, TICKER, config.data.text_dir)

dataset_path = (
    Path(__file__).resolve().parent.parent
    / "stock_data"
    / f"{TICKER}_with_sentiment.csv"
)
dataset_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(dataset_path, index_label="Date")
print(f"Saved prepared dataset to {dataset_path}")

df = build_targets(df, HORIZON)

# Exclude unnecessary columns
exclude = {
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "log_close",
    "log_ret",
    "sentiment_text",
}
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns]
else:
    df.columns = [str(c) for c in df.columns]
feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("cum_logret_h")]
feature_cols = sorted(feature_cols)

# Create supervised arrays
X_all, Y_all, last_obs_all = create_supervised_arrays(df, feature_cols, HORIZON, LOOK_BACK)
n_samples, _, n_features = X_all.shape

# train/test split
split_idx = int(0.8 * n_samples)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
Y_train, Y_test = Y_all[:split_idx], Y_all[split_idx:]
last_test = last_obs_all[split_idx:]

# Normalisation
scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
X_train_s = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

# PyTorch LSTM model
model, _ = train_pytorch_lstm(
    X_train_s,
    Y_train,
    X_test_s,
    Y_test,
    horizon=HORIZON,
    hidden_size=64,
    epochs=60,
    batch_size=32,
    learning_rate=1e-3,
)

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(torch.from_numpy(X_test_s.astype(np.float32))).numpy()
true_prices = inverse_cum_logret_to_price(last_test, Y_test)
pred_prices = inverse_cum_logret_to_price(last_test, preds)
metrics = per_horizon_metrics(true_prices, pred_prices)
metrics["directional_accuracy"] = directional_accuracy(true_prices, pred_prices, last_test)

# Save model
OUT_MODEL = os.path.join(OUTDIR, f"{TICKER}_lstm.pt")
save_lstm_model(model, OUT_MODEL)
with open(OUT_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

np.save(os.path.join(OUTDIR, f"{TICKER}_scaler_mean.npy"), scaler.mean_)
np.save(os.path.join(OUTDIR, f"{TICKER}_scaler_scale.npy"), scaler.scale_)
print(f"Saved PyTorch LSTM model to {OUT_MODEL}")

