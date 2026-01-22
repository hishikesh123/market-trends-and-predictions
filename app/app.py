import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


def load_css():
    css_path = Path("app/assets/style.css")
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

load_css()

logo_path = Path("app/assets/logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=120)

st.set_page_config(page_title="ASX Stock Price Predictor", layout="wide")

st.title("📈 ASX Adjusted Close Price Predictor")
st.markdown("""
This app demonstrates a machine learning model trained on historical  
ASX stock data (2015–2020) to predict **Adjusted Closing Price**.
""")

# Load trained pipeline (preprocessing + model)
import joblib
import xgboost as xgb

preprocessor = joblib.load("models/preprocessor.joblib")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgb_model.json")


st.subheader("Input stock features")

company = st.text_input("Company Ticker (e.g. CBA, WBC, BHP)", "CBA")
open_ = st.number_input("Open", value=10.0)
high = st.number_input("High", value=10.2)
low = st.number_input("Low", value=9.8)
close = st.number_input("Close", value=10.1)
volume = st.number_input("Volume", value=1_000_000.0)

input_df = pd.DataFrame([{
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
    "company": company
}])

expected_cols = ["open", "high", "low", "close", "volume", "company"]
missing = [c for c in expected_cols if c not in input_df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

if st.button("Predict Adjusted Close"):
    X = preprocessor.transform(input_df)
    prediction = xgb_model.predict(X)[0]
    st.success(f"Predicted Adjusted Close: **AUD {prediction:.2f}**")
