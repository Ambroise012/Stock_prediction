import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import subprocess
from PIL import Image
import streamlit as st
import pandas as pd
import datetime

from src.predict_utils import get_company_name
from src.predict_stock import display_results, predict_stock

st.set_page_config(page_title="Stock Prediction", layout="wide")
st.title("📈 Stock Price Prediction")

if "ticker" not in st.session_state:
    st.session_state.ticker = ""

st.write("Quick examples:")
example_tickers = [
    "MSFT", "AI.PA", "SAN.PA", "TSLA", "AAPL", "SAF.PA", "IBM",
    "TTE.PA", "AIR.PA", "^FCHI", "^DJI", "MC.PA", "NVDA", "ENGI.PA",
    "AMZN", "GLE.PA", "VALD.PA", "LOR.MU", "639.SG"
]

default_ticker = ""

cols = st.columns(10)
for i, t in enumerate(example_tickers):
    if cols[i % 10].button(t):
        st.session_state.ticker = t

ticker = st.text_input(
    "Enter stock ticker (e.g., AAPL, AI.PA, MSFT):",
    value=st.session_state.ticker
)

st.session_state.ticker = ticker

if st.button("Predict"):
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner("Running prediction model... This may take a while ⏳"):
            proc = subprocess.run(
                [sys.executable, "-m", "src.predict_stock", ticker],
                capture_output=True,
                text=True
            )

        # --- Check if the subprocess ran successfully ---
        if proc.returncode != 0:
            st.error("❌ Prediction script failed.")
            st.text(proc.stderr)
            st.stop()

        # --- Get company name ---
        company = get_company_name(ticker)
        st.header(f"📊 Prediction for {company} ({ticker})")
        # --- train & predict ---
        df, preds, last_price, metrics_path = predict_stock(ticker)
        display_results(ticker, df, preds, last_price, metrics_path)