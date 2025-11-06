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

ticker = st.text_input("Enter stock ticker (e.g., AAPL, AI.PA, MSFT):")

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

        df, preds, last_price, metrics_path = predict_stock(ticker)
        display_results(ticker, df, preds, last_price, metrics_path)