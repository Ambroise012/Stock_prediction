
# Stock Price Prediction with LSTM

## Overview

This script fetches historical and up-to-date stock data for a given ticker symbol, preprocesses the data, trains an GRU model to predict future stock prices, and visualizes the results. The script uses both Yahoo Finance and Marketstack APIs to ensure comprehensive and current data.

Predict 5 days ahead

---

## Features

- **Data Fetching**: Combines historical data from Yahoo Finance and recent data from Marketstack.
- **Data Preprocessing**: Normalizes and prepares data for LSTM training.
- **LSTM Model**: Builds and trains an LSTM model for stock price prediction.
- **Visualization**: Plots the last 30 days of actual stock prices and the forecasted prices for the next N days.

---

## Requirements

Install the dependencies using:

```bash
poetry install
```

---

## Setup

1. **Environment Variables**:
   - Set your Marketstack API key in the `.env`:
     ```bash
     MARKETSTACK_API_KEY="your_api_key_here"
     ```

2. **Configuration**:
   - Edit the `config.py` file to set parameters of LSTM and predict :
     - `look_back`: Number of previous days to use for prediction.
     - `future_days`: Number of future days to forecast.
     - `epochs`: Number of training epochs.
     - `batch_size`: Batch size for training.

---

## Usage

Run the script with the stock ticker symbol as an argument:

```bash
poetry run streamlit run interface.py
```

Or in background:

```bash
nohup poetry run streamlit run interface.py > streamlit.log 2>&1 &
```

## Deploy

```bash
poetry lock
docker build -t predict:0.1.0 .
docker run -p 8501:8501 predict:0.1.0
```

`AAPL` with the ticker symbol of your choice.

E.g. of ticker:
- MSFT : Microsoft
- AAPL : Apple
- ENGI.PA : Engie
- SAN.PA : Sanofi
- GE : General Electric
- IBM : IBM
---



