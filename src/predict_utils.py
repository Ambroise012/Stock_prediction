import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import sys
import time
import random
import json
import logging
from datetime import datetime, timedelta

import requests
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class MarketstackProvider:
    def __init__(self):
        self.api_key = os.getenv("MARKETSTACK_API_KEY")
        if not self.api_key:
            raise ValueError("MARKETSTACK_API_KEY environment variable not set")
        self.base_url = "http://api.marketstack.com/v1/"
        self.max_retries = 3
        self.min_delay = 1
        self.max_delay = 5

    def _make_request(self, endpoint, params=None):
        params = params or {}
        params.update({"access_key": self.api_key})

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = random.uniform(self.min_delay, self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise Exception(f"API request failed: {str(e)}")

    def get_stock_data(self, ticker, start_date=None, end_date=None):
        params = {
            "symbols": ticker,
            "limit": 1000,
            "sort": "ASC"
        }
        if start_date:
            params["date_from"] = start_date
        if end_date:
            params["date_to"] = end_date

        data = self._make_request("eod", params)

        if not data.get("data"):
            return None

        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.sort_index()

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        # Log the available dates
        self._log_available_dates(df, source="Marketstack", ticker=ticker)

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def resolve_ticker(self, yahoo_ticker: str):
        """
        Try to resolve a Yahoo Finance ticker to Marketstack's symbol.
        Example: AI.PA (Yahoo) -> AIR (Marketstack, Euronext Paris)
        """
        base_symbol = yahoo_ticker.split(".")[0]
        data = self._make_request("tickers", params={"search": base_symbol})
        candidates = data.get("data", [])

        if not candidates:
            logger.warning(f"No Marketstack match found for {yahoo_ticker}")
            return yahoo_ticker  # fallback

        if "." in yahoo_ticker:
            suffix = yahoo_ticker.split(".")[1]
            exchange_map = {
                "PA": "XPAR",  # Euronext Paris
                "DE": "XETR",  # Xetra (Germany)
                "L": "XLON",   # London Stock Exchange
                "MI": "XMIL",  # Milan
                "AS": "XAMS",  # Amsterdam
            }
            target_exchange = exchange_map.get(suffix)
            for c in candidates:
                if c.get("stock_exchange", {}).get("acronym") == target_exchange:
                    return c["symbol"]

        return candidates[0]["symbol"]

    def _log_available_dates(self, df: pd.DataFrame, source: str, ticker: str):
        """Helper to log the date range of a given DataFrame."""
        if df is None or df.empty:
            logger.info(f"{source}: No data available for {ticker}")
            return

        first_date = df.index.min().date()
        last_date = df.index.max().date()
        count = len(df)

        logger.info(
            f"{source}: Extracted {count} rows for {ticker} "
            f"from {first_date} to {last_date}"
        )

def get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName", ticker)
    except Exception:
        return ticker

def download_data(ticker, period="10y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    return df

def add_features(df):
    """
    Ajoute features utiles :
    - log_close, log_volume
    - lags of log_close & log_vol
    - rolling means/std
    - RSI (14), simple MACD approximation, momentum
    """
    df = df.copy()
    df["log_close"] = np.log(df["Close"])
    df["log_vol"] = np.log(df["Volume"].replace(0, np.nan)).fillna(0)

    # lags
    for lag in [1,2,3,5,10,21]:
        df[f"lag_logc_{lag}"] = df["log_close"].shift(lag)
        df[f"lag_logv_{lag}"] = df["log_vol"].shift(lag)

    # rolling stats
    for w in [5,10,21,60]:
        df[f"roll_mean_{w}"] = df["log_close"].rolling(window=w).mean()
        df[f"roll_std_{w}"] = df["log_close"].rolling(window=w).std()

    # momentum
    df["mom_5"] = df["log_close"] - df["log_close"].shift(5)
    df["mom_21"] = df["log_close"] - df["log_close"].shift(21)

    # simple MACD-like (ema diff)
    df["ema12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]

    # RSI (14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df = df.dropna().copy()
    return df

def inverse_cum_logret_to_price(last_price, cum_logret_preds):
    return last_price * np.exp(cum_logret_preds)


# -----------------------------
# Train only
# -----------------------------
def build_targets(df, horizon):
    """
    Calcule les rendements log cumulés futurs sur 1..horizon jours.
    Exemple : cum_logret_h3 = log_ret[t+1] + log_ret[t+2] + log_ret[t+3]
    """
    df = df.copy()
    df["log_ret"] = df["log_close"].diff()

    for h in range(1, horizon + 1):
        # Somme des rendements sur les h prochains jours (pas les passés)
        df[f"cum_logret_h{h}"] = df["log_ret"].shift(-1).rolling(window=h).sum().shift(-(h - 1))

    # Supprime les lignes avec NaN en fin de série (prévisions impossibles)
    df = df.dropna().reset_index(drop=True)
    return df

def create_supervised_arrays(df, feature_cols, horizon, look_back):
    X_list, Y_list, last_obs_price = [], [], []
    arr = df[feature_cols].values
    n = len(df)
    for end_idx in range(look_back - 1, n - horizon):
        start_idx = end_idx - (look_back - 1)
        X_seq = arr[start_idx:end_idx+1, :]
        y = [df.iloc[end_idx + h][f"cum_logret_h{h}"] for h in range(1, horizon + 1)]
        X_list.append(X_seq)
        Y_list.append(y)
        last_obs_price.append(np.exp(df.iloc[end_idx]["log_close"]))
    return np.array(X_list), np.array(Y_list), np.array(last_obs_price)

def flatten_tabular_from_seq(X):
    """
    Simple flattening for tabular model:
    - last row features concatenated with mean and std across lookback
    """
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    return np.concatenate([last, mean, std], axis=1)

def inverse_cum_logret_to_price(last_price, cum_logret_preds):
    """
    last_price: (n_samples,) price at t0
    cum_logret_preds: (n_samples, horizon) predicted cumulative log returns
    returns predicted prices shape (n_samples, horizon)
    """
    return last_price.reshape(-1,1) * np.exp(cum_logret_preds)

def per_horizon_metrics(true_prices, pred_prices):
    res = {}
    for h in range(true_prices.shape[1]):
        res[f"h{h+1}"] = {
            "mse": float(mean_squared_error(true_prices[:, h], pred_prices[:, h])),
            "mae": float(mean_absolute_error(true_prices[:, h], pred_prices[:, h])),
            "r2": float(r2_score(true_prices[:, h], pred_prices[:, h]))
        }
    return res

def directional_accuracy(true_prices, pred_prices, last_obs_prices):
    """
    Compute directional accuracy compared to last observed price:
    direction = sign(price_t+h - price_t0)
    return dict per horizon + overall
    """
    true_dir = np.sign(true_prices - last_obs_prices.reshape(-1,1))
    pred_dir = np.sign(pred_prices - last_obs_prices.reshape(-1,1))
    accs = {}
    for i in range(true_prices.shape[1]):
        accs[f"h{i+1}"] = float(np.mean(true_dir[:, i] == pred_dir[:, i]))
    accs["overall"] = float(np.mean(true_dir == pred_dir))
    return accs