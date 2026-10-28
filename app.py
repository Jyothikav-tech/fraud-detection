import json
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")

samples = json.load(open("samples.json"))
kind = st.radio("Sample transaction type", ["fraud", "normal"], horizontal=True)
idx = st.slider("Sample number", 0, len(samples[kind]) - 1, 0)
tx = dict(samples[kind][idx])

st.subheader("Transaction details")
tx["Amount"] = st.number_input("Amount", min_value=0.0, value=float(tx["Amount"]))
tx["Time"] = st.number_input("Time (seconds)", min_value=0.0, value=float(tx["Time"]))
with st.expander("PCA features V1-V28"):
    st.json({k: v for k, v in tx.items() if k.startswith("V")})

if st.button("Check transaction"):
    try:
        r = requests.post(API_URL, json=tx, timeout=10)
        r.raise_for_status()
        res = r.json()
        st.metric("Fraud probability", f"{res['fraud_probability']:.2%}")
        if res["is_fraud"]:
            st.error("🚨 Flagged as FRAUD")
        else:
            st.success("✅ Looks legitimate")
    except Exception as e:
        st.error(f"API error: {e}")