import math
import numpy as np
import pandas as pd
from typing import Tuple

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance between two points on the Earth's surface
    using the Haversine formula in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c


def compute_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates distance and speed between consecutive transactions for each user/card.
    Identifies impossible travel velocity anomalies.
    """
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    # Shifted values for consecutive transaction comparison
    df["prev_lat"] = df.groupby("user_id")["location_lat"].shift(1)
    df["prev_lon"] = df.groupby("user_id")["location_lon"].shift(1)
    df["prev_time"] = df.groupby("user_id")["timestamp"].shift(1)

    distances = []
    speeds = []
    impossible_flags = []

    for _, row in df.iterrows():
        if pd.isna(row["prev_lat"]) or pd.isna(row["prev_time"]):
            distances.append(0.0)
            speeds.append(0.0)
            impossible_flags.append(0)
            continue

        dist_km = haversine_distance_km(
            row["prev_lat"], row["prev_lon"],
            row["location_lat"], row["location_lon"]
        )
        time_diff_hours = (row["timestamp"] - row["prev_time"]).total_seconds() / 3600.0

        if time_diff_hours > 0.001:
            speed_kmh = dist_km / time_diff_hours
        else:
            speed_kmh = 0.0

        # Impossible travel: fast travel over meaningful distance (> 250 km and > 800 km/h)
        is_impossible = 1 if (speed_kmh > config.IMPOSSIBLE_TRAVEL_SPEED_KMH and dist_km > 250.0) else 0

        distances.append(round(dist_km, 2))
        speeds.append(round(speed_kmh, 2))
        impossible_flags.append(is_impossible)

    df["geo_distance_km"] = distances
    df["travel_speed_kmh"] = speeds
    df["is_impossible_travel"] = impossible_flags

    # Clean intermediate columns
    df.drop(columns=["prev_lat", "prev_lon", "prev_time"], inplace=True)
    return df
