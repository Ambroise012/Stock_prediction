
# Stock Price Prediction with LSTM

## Overview

This script fetches historical and up-to-date stock data for a given ticker symbol, preprocesses the data, trains an GRU model to predict future stock prices, and visualizes the results. The script uses both Yahoo Finance and Marketstack APIs to ensure comprehensive and current data.

Predict 5 days ahead


## Features

- **Data Fetching**: Combines historical data from Yahoo Finance and recent data from Marketstack.
- **Data Preprocessing**: Normalizes and prepares data for LSTM training.
- **LSTM Model**: Builds and trains an LSTM model for stock price prediction.
- **Visualization**: Plots the last 30 days of actual stock prices and the forecasted prices for the next N days.


## Requirements

Install `uv`, then install the dependencies with:

```bash
uv sync
```


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


## Usage

Train or use a model from the command line:

```bash
uv run python -m src.main train AAPL
uv run python -m src.main infer AAPL
```

The `infer` command automatically trains the model if the model file does not exist.

Run the script with the stock ticker symbol as an argument:

```bash
uv run streamlit run src/interface.py
```

Or in background:

```bash
nohup uv run streamlit run src/interface.py > streamlit.log 2>&1 &
```

## Deploy

```bash
uv lock
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

## Output

- The script saves a plot of the last 30 days of actual stock prices and the forecasted prices for the next N days in the `predict/` directory.
- The plot is saved as `{ticker}_forecast.png`.

---

## Sentiment Pipeline

The training and prediction flows now include two text features: FinBERT sentiment and daily publication volume. Text published on a given date is assigned to the next available trading day to prevent data leakage.

Create one CSV per ticker in `data/text/`, for example `data/text/AAPL.csv`:

```csv
timestamp,text
2025-01-02T14:30:00Z,"Apple reports stronger than expected results"
2025-01-03T09:15:00Z,"Investors remain cautious about demand"
```

The CSV must contain `timestamp` and either `text` or `title`. Install dependencies with `uv sync`; the first run downloads `ProsusAI/finbert`. If no text file is available, the model uses neutral sentiment and remains runnable. The source data can be produced by the official Reddit API (PRAW), Yahoo Finance or a press collector, but credentials and collection are intentionally kept outside the training code.

Training saves the prepared market and sentiment data to `stock_data/{ticker}_with_sentiment.csv`. The export includes the aligned text, signed sentiment score, sentiment volume, and engineered market features; the original price CSV is not overwritten. Raw text is retained for review but is not used as a numeric model input.
 
---



