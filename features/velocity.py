import numpy as np
import pandas as pd
from typing import Dict, Any

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-based rolling velocity features for users and cards:
    - 1-hour, 24-hour, 7-day transaction counts
    - 1-hour, 24-hour transaction spend sums
    - Time elapsed since last user transaction
    - Amount relative to user's expanding mean and standard deviation (Z-score)
    """
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    # Time delta since last transaction (in seconds)
    df["prev_user_time"] = df.groupby("user_id")["timestamp"].shift(1)
    time_deltas = (df["timestamp"] - df["prev_user_time"]).dt.total_seconds().fillna(86400.0)
    df["time_since_last_tx_sec"] = time_deltas.clip(lower=0.0)
    df.drop(columns=["prev_user_time"], inplace=True)

    # Expanding user statistics (mean and std up to current tx to prevent data leakage)
    user_expanding = df.groupby("user_id")["amount"].expanding()
    exp_mean = user_expanding.mean().reset_index(level=0, drop=True)
    exp_std = user_expanding.std().reset_index(level=0, drop=True).fillna(1.0)
    # Replace zeros in std
    exp_std = exp_std.replace(0.0, 1.0)

    df["user_historical_mean"] = exp_mean
    df["user_amount_ratio"] = df["amount"] / (exp_mean + 1e-4)
    df["user_amount_zscore"] = (df["amount"] - exp_mean) / exp_std

    # Rolling window counts and sums using time offset
    # Set index to timestamp for rolling, grouped by user_id
    df_indexed = df.set_index("timestamp")

    user_1h_count = df_indexed.groupby("user_id")["amount"].rolling("1h", closed="left").count().reset_index()
    user_1h_sum = df_indexed.groupby("user_id")["amount"].rolling("1h", closed="left").sum().reset_index()

    user_24h_count = df_indexed.groupby("user_id")["amount"].rolling("24h", closed="left").count().reset_index()
    user_24h_sum = df_indexed.groupby("user_id")["amount"].rolling("24h", closed="left").sum().reset_index()

    user_7d_count = df_indexed.groupby("user_id")["amount"].rolling("7d", closed="left").count().reset_index()

    df["tx_count_1h"] = user_1h_count["amount"].fillna(0.0).values
    df["tx_sum_1h"] = user_1h_sum["amount"].fillna(0.0).values
    df["tx_count_24h"] = user_24h_count["amount"].fillna(0.0).values
    df["tx_sum_24h"] = user_24h_sum["amount"].fillna(0.0).values
    df["tx_count_7d"] = user_7d_count["amount"].fillna(0.0).values

    return df
