import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# LOAD MODELS & ARTIFACTS
# ==============================

model = joblib.load("xgb_flow_model.pkl")
calibrator = joblib.load("calibrator_lr.pkl")

important_features = joblib.load("important_features.pkl")
labels = joblib.load("labels.pkl")

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="IDS System", layout="wide")
st.title("🚨 Network Intrusion Detection System")
st.markdown("Upload flow data (CICIDS format) to detect attacks.")

# ==============================
# FILE UPLOAD
# ==============================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ==============================
# PROCESS FILE
# ==============================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Uploaded Data")
    st.dataframe(df.head())

    try:
        # ==============================
        # CLEAN DATA
        # ==============================
        drop_cols = ['Flow ID','Src IP','Dst IP','Timestamp','Src Port']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        for col in df.columns:
            if col != "Label":
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # ==============================
        # FEATURE ENGINEERING
        # ==============================
        df['packet_ratio'] = df['Tot Fwd Pkts'] / (df['Tot Bwd Pkts'] + 1)
        df['byte_ratio'] = df['TotLen Fwd Pkts'] / (df['TotLen Bwd Pkts'] + 1)

        df['total_packets'] = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
        df['total_bytes'] = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']

        df['avg_pkt_size'] = df['total_bytes'] / (df['total_packets'] + 1)
        df['flow_intensity'] = df['total_packets'] / (df['Flow Duration'] + 1)
        df['burstiness'] = df['Pkt Len Std'] / (df['Pkt Len Mean'] + 1)

        # ==============================
        # FEATURE SELECTION
        # ==============================
        X = df[important_features].astype("float32")

        # ==============================
        # PREDICTION
        # ==============================
        probs = model.predict_proba(X)
        calibrated = calibrator.predict_proba(probs)

        preds = np.argmax(calibrated, axis=1)
        confidences = np.max(calibrated, axis=1)

        df["Prediction"] = [labels[p] for p in preds]
        df["Confidence"] = confidences

        # ==============================
        # OUTPUT
        # ==============================
        st.subheader("🔍 Prediction Results")
        st.dataframe(df[["Prediction", "Confidence"]])

        # Stats
        st.subheader("📊 Summary")
        st.write(df["Prediction"].value_counts())

    except Exception as e:
        st.error(f"Error processing file: {e}")