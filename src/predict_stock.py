import subprocess
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os, sys
import json

from src.config import config
from src.predict_utils import download_data, add_features

def inverse_cum_logret_to_price(last_price, cum_logret_preds):
    return last_price * np.exp(cum_logret_preds)

def predict_stock(TICKER: str):
    """Run GRU prediction for a given stock ticker."""
    LOOK_BACK = config.predict.look_back
    HORIZON = config.predict.horizon
    MODELDIR = "models"

    MODEL_PATH = os.path.join(MODELDIR, f"{TICKER}_gru.h5")
    METRICS_PATH = os.path.join(MODELDIR, f"{TICKER}_metrics.json")
    SCALER_MEAN = os.path.join(MODELDIR, f"{TICKER}_scaler_mean.npy")
    SCALER_SCALE = os.path.join(MODELDIR, f"{TICKER}_scaler_scale.npy")

    # ----------------------------
    # load model and scaler
    # ----------------------------
    if not os.path.exists(MODEL_PATH):
        subprocess.run([sys.executable, "-m", "src.train", TICKER], check=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = StandardScaler()
    scaler.mean_ = np.load(SCALER_MEAN)
    scaler.scale_ = np.load(SCALER_SCALE)

    # ----------------------------
    # download last days and prep features
    # ----------------------------
    df = download_data(TICKER)
    df = add_features(df)

    try:
        last_price = float(df["Close"].iloc[-1])
    except KeyError:
        close_col = [c for c in df.columns if "Close" in str(c)][0]
        last_price = float(df[close_col].iloc[-1])

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    exclude = {"Open", "High", "Low", "Close", "Adj Close", "log_close", "log_ret"}
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("cum_logret_h")]
    feature_cols = sorted(feature_cols)

    latest = df.iloc[-LOOK_BACK:].copy()
    arr = latest[feature_cols].values
    X_input_s = scaler.transform(arr).reshape(1, LOOK_BACK, len(feature_cols))

    # ----------------------------
    # predict
    # ----------------------------
    pred_cum_logret = model.predict(X_input_s, verbose=0)[0]
    pred_prices = inverse_cum_logret_to_price(last_price, pred_cum_logret)

    return df, pred_prices, last_price, METRICS_PATH


def display_results(TICKER, df, pred_prices, last_price, METRICS_PATH):
    """
    Display model predictions and metrics in Streamlit.

    Args:
        TICKER (str): stock TICKER symbol (e.g., "AAPL")
        df (pd.DataFrame): historical data with a "Close" column
        pred_prices (np.ndarray): predicted prices for the next few days
        last_price (float): last real closing price
        metrics_path (str): path to the metrics JSON file
    """
    st.header(f"📈 Prediction Results for {TICKER}")

    # --- Plot historical prices + predictions ---
    st.subheader("Price Evolution and Predictions")

    hist_days = 30  # number of past days to show
    if "Close" in df.columns:
        close_col = "Close"
    else:
        # cherche une colonne qui contient "Close" dans son nom
        candidates = [c for c in df.columns if "Close" in str(c)]
        if not candidates:
            raise KeyError("Aucune colonne contenant 'Close' trouvée dans le DataFrame.")
        close_col = candidates[0]

    hist_data = df[close_col].iloc[-hist_days:].copy()
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=len(pred_prices))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist_data.index, hist_data.values, label="Historical", linewidth=2)
    ax.plot(future_dates, pred_prices, "--o", color="orange", label="Predictions", linewidth=2)
    ax.axhline(y=last_price, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"Predicted Price for {TICKER}")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.pyplot(fig)

    # --- Prediction Table ---
    st.subheader("Short-Term Predictions")
    df_pred = pd.DataFrame({
        "Day": [f"+{i}" for i in range(1, len(pred_prices) + 1)],
        "Predicted Price (USD)": np.round(pred_prices, 2),
        "Change (%)": np.round((pred_prices - last_price) / last_price * 100, 2),
        "Direction": ["⬆️ Up" if p > last_price else "⬇️ Down" for p in pred_prices],
    })
    st.dataframe(df_pred, hide_index=True)

    # --- Model Metrics ---
    if os.path.exists(METRICS_PATH):
        st.subheader("📊 Model Metrics")
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        df_metrics = pd.DataFrame({
            "Horizon": [h for h in metrics if h.startswith("h")],
            "MSE": [metrics[h]["mse"] for h in metrics if h.startswith("h")],
            "MAE": [metrics[h]["mae"] for h in metrics if h.startswith("h")],
            "R²": [metrics[h]["r2"] for h in metrics if h.startswith("h")]
        })

        # 🔧 Convert columns to float before formatting
        for col in ["MSE", "MAE", "R²"]:
            df_metrics[col] = pd.to_numeric(df_metrics[col], errors="coerce")

        st.table(df_metrics.style.format({"MSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}))

    else:
        st.warning(f"Metrics file not found: {METRICS_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict_stock <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1]
    df, pred_prices, last_price, METRICS_PATH = predict_stock(ticker)