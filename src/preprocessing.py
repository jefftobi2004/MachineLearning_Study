
import pandas as pd
import numpy as np


def prepare_dataset(df, engineer_features=True):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["rv1", "rv2"])
    df = df.sort_values("date").reset_index(drop=True)

    df["target"] = df["Appliances"].shift(-1)

    if engineer_features:
        
        lags = [1, 2, 3, 6, 144]

        for lag in lags:
            df[f"Appliances_lag_{lag}"] = df["Appliances"].shift(lag)

        
        df["Appliances_rolling_mean_1h"] = (
            df["Appliances"].rolling(window=6).mean()
        )

        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek
        
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df = df.drop(columns=["hour"])

        lag_columns = [f"Appliances_lag_{lag}" for lag in lags]

        df = df.dropna(
            subset=lag_columns + ["Appliances_rolling_mean_1h", "target"]
        ).reset_index(drop=True)

    else:
        df = df.iloc[144:].reset_index(drop=True)
        df = df.dropna(subset=["target"]).reset_index(drop=True)

    X = df.drop(columns=["date", "target"])
    y = df["target"]

    return X, y
