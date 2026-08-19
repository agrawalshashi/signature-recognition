import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add src folder to Python path
sys.path.append(
    os.path.join(os.path.dirname(__file__), "src")
)

from preprocessing import preprocess_signature
from feature_engineering import extract_features


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Signature Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Signature Recognition System")

st.write(
    "Upload a signature file to classify it as Genuine or Forged."
)


# -----------------------------
# Load Model
# -----------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "signature_random_forest.pkl"
)

model = joblib.load(MODEL_PATH)


# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Signature (.TXT)",
    type=["txt"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    try:

        # Read uploaded file
        uploaded_file.seek(0)

        # Read lines
        lines = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        ).splitlines()

        # Remove empty lines
        lines = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        # Extract valid 7-column rows
        data = []

        for line in lines:

            parts = line.split()

            if len(parts) == 7:

                try:
                    values = [
                        float(value)
                        for value in parts
                    ]

                    data.append(values)

                except ValueError:
                    continue

        # Check data
        if len(data) == 0:

            st.error(
                "No valid 7-column signature data found."
            )

            st.stop()

        # Convert to DataFrame
        df = pd.DataFrame(
            data,
            columns=[
                "x",
                "y",
                "timestamp",
                "pen_status",
                "azimuth",
                "altitude",
                "pressure"
            ]
        )

        # -----------------------------
        # Preprocessing
        # -----------------------------

        processed = preprocess_signature(df)

        # Limit sequence length
        processed = processed.iloc[:300]

        # -----------------------------
        # Feature Extraction
        # -----------------------------

        features = extract_features(processed)

        features = features.reshape(1, -1)

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        confidence = probabilities[prediction] * 100

        # -----------------------------
        # Display Result
        # -----------------------------

        if prediction == 0:

            st.success(
                "✅ Prediction: Genuine"
            )

        else:

            st.error(
                "⚠️ Prediction: Forged"
            )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # -----------------------------
        # Feature Information
        # -----------------------------

        st.write("### Signature Information")

        st.write(
            f"Sequence length: {len(processed)}"
        )

        st.write(
            f"Features extracted: {features.shape[1]}"
        )

    except Exception as e:

        st.error(
            "Unable to process this signature file."
        )

        st.exception(e)