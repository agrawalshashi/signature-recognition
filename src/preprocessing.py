import pandas as pd


def preprocess_signature(df):
    df = df.copy()

    # Convert absolute timestamp into elapsed time
    df["timestamp"] = (
        df["timestamp"] - df["timestamp"].iloc[0]
    )

    return df