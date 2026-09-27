import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from features.geospatial import compute_geospatial_features, haversine_distance_km
from features.velocity import compute_velocity_features
from features.graph_features import compute_graph_features, FraudGraphBuilder


# High-risk MCC mappings and categories
HIGH_RISK_MCCS = {"5732", "6051", "5944", "7995"}
MCC_RISK_MAP = {
    "5411": 0.05,  # Grocery Stores
    "5812": 0.08,  # Restaurants
    "5541": 0.12,  # Gas Stations
    "5311": 0.15,  # Department Stores
    "4814": 0.10,  # Telecom
    "4511": 0.35,  # Airlines
    "5732": 0.55,  # Electronics
    "5944": 0.65,  # Jewelry / Luxury
    "7995": 0.70,  # Gaming / Casinos
    "6051": 0.85,  # Crypto / Money Orders
}

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "is_night_hours",
    "day_of_week",
    "is_weekend",
    "mcc_risk_score",
    "is_high_risk_mcc",
    "is_channel_web",
    "is_channel_mobile",
    "is_channel_wire",
    "is_channel_pos",
    "is_aml_structuring_range",
    "geo_distance_km",
    "travel_speed_kmh",
    "is_impossible_travel",
    "time_since_last_tx_sec",
    "user_amount_ratio",
    "user_amount_zscore",
    "tx_count_1h",
    "tx_sum_1h",
    "tx_count_24h",
    "tx_sum_24h",
    "tx_count_7d",
    "device_user_count",
    "ip_user_count",
    "user_device_count",
    "user_card_count",
    "card_user_count",
    "is_shared_device",
    "is_shared_ip",
    "is_shared_card",
    "graph_risk_score",
]


class FeaturePipeline:
    """
    Transforms raw transaction records into analytical ML features for
    both offline batch model training and real-time streaming inference.
    """

    def __init__(self):
        self.user_state: Dict[str, Dict[str, Any]] = {}
        self.device_state: Dict[str, set] = {}
        self.ip_state: Dict[str, set] = {}
        self.card_state: Dict[str, set] = {}

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts diurnal and calendar signals."""
        df = df.copy()
        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["hour_of_day"] = df["timestamp"].dt.hour
        # Night hours (1 AM to 5 AM) often correlate with automated attacks / ATO
        df["is_night_hours"] = df["hour_of_day"].isin([1, 2, 3, 4, 5]).astype(int)
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        return df

    def extract_domain_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Domain heuristics: MCC risk weighting, channel encoding, structuring bounds."""
        df = df.copy()

        df["mcc_risk_score"] = df["mcc"].astype(str).map(MCC_RISK_MAP).fillna(0.15)
        df["is_high_risk_mcc"] = df["mcc"].astype(str).isin(HIGH_RISK_MCCS).astype(int)

        # One-hot encoded channels
        df["is_channel_web"] = (df["channel"] == "web").astype(int)
        df["is_channel_mobile"] = (df["channel"] == "mobile").astype(int)
        df["is_channel_wire"] = (df["channel"] == "wire_transfer").astype(int)
        df["is_channel_pos"] = (df["channel"] == "pos").astype(int)

        # Regulatory structuring test (just below $10,000 threshold)
        df["is_aml_structuring_range"] = (
            (df["amount"] >= config.AML_STRUCTURING_MIN) & (df["amount"] <= config.AML_STRUCTURING_MAX)
        ).astype(int)

        return df

    def process_batch(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes entire transaction dataset:
        1. Temporal extraction
        2. Domain heuristics
        3. Geospatial / Haversine velocities
        4. User & Card behavioral velocity rolling metrics
        5. Graph-based entity links
        Returns full enriched DataFrame and standard X feature matrix.
        """
        df = raw_df.copy()

        # Step 1: Temporal
        df = self.extract_temporal_features(df)

        # Step 2: Domain heuristics
        df = self.extract_domain_indicators(df)

        # Step 3: Geospatial
        df = compute_geospatial_features(df)

        # Step 4: Velocity
        df = compute_velocity_features(df)

        # Step 5: Graph features
        df, _ = compute_graph_features(df)

        # Populate state store for real-time inference
        self._build_state_cache(df)

        # Extract features matrix
        features_df = df[FEATURE_COLUMNS].fillna(0.0)

        return df, features_df

    def _build_state_cache(self, df: pd.DataFrame) -> None:
        """Populates online fast-lookup caches for streaming transaction scoring."""
        # User history cache: last tx time, last coords, rolling count & amounts
        for user_id, group in df.groupby("user_id"):
            last_row = group.iloc[-1]
            amounts = group["amount"].values
            self.user_state[user_id] = {
                "last_timestamp": last_row["timestamp"],
                "last_lat": float(last_row["location_lat"]),
                "last_lon": float(last_row["location_lon"]),
                "mean_amount": float(np.mean(amounts)),
                "std_amount": float(np.std(amounts)) if len(amounts) > 1 and np.std(amounts) > 0 else 25.0,
                "tx_history": list(zip(group["timestamp"].values, group["amount"].values))[-50:],
            }

        # Device & IP entity link caches
        for _, row in df.iterrows():
            dev = row["device_id"]
            ip = row["ip_address"]
            u = row["user_id"]
            c = row["card_id"]

            if dev not in self.device_state:
                self.device_state[dev] = set()
            self.device_state[dev].add(u)

            if ip not in self.ip_state:
                self.ip_state[ip] = set()
            self.ip_state[ip].add(u)

            if c not in self.card_state:
                self.card_state[c] = set()
            self.card_state[c].add(u)

    def extract_single_record(self, tx: Dict[str, Any]) -> pd.DataFrame:
        """
        Fast online feature extraction for a single incoming transaction.
        Designed for microsecond/millisecond production serving.
        """
        raw_ts = tx.get("timestamp")
        if raw_ts is None:
            ts = pd.Timestamp.now()
        else:
            ts = pd.to_datetime(raw_ts)
        user_id = tx.get("user_id", "USR_00000")
        card_id = tx.get("card_id", "CARD_00000")
        device_id = tx.get("device_id", "DEV_UNKNOWN")
        ip_address = tx.get("ip_address", "127.0.0.1")
        channel = tx.get("channel", "web")
        mcc = str(tx.get("mcc", "5411"))
        amount = float(tx.get("amount", 10.0))
        lat = float(tx.get("location_lat", 40.7128))
        lon = float(tx.get("location_lon", -74.0060))

        # Temporal
        hour = ts.hour
        is_night = int(hour in [1, 2, 3, 4, 5])
        dow = ts.dayofweek
        is_wknd = int(dow in [5, 6])

        # Domain
        mcc_risk = MCC_RISK_MAP.get(mcc, 0.15)
        is_high_mcc = int(mcc in HIGH_RISK_MCCS)
        is_web = int(channel == "web")
        is_mob = int(channel == "mobile")
        is_wire = int(channel == "wire_transfer")
        is_pos = int(channel == "pos")
        is_struct = int(config.AML_STRUCTURING_MIN <= amount <= config.AML_STRUCTURING_MAX)

        # User state lookup
        u_state = self.user_state.get(user_id)
        if u_state:
            last_ts = pd.to_datetime(u_state["last_timestamp"])
            time_since_sec = max(0.0, (ts - last_ts).total_seconds())

            # Haversine distance & velocity
            geo_dist = haversine_distance_km(u_state["last_lat"], u_state["last_lon"], lat, lon)
            time_hrs = time_since_sec / 3600.0
            travel_speed = (geo_dist / time_hrs) if time_hrs > 0.001 else 0.0
            is_impossible = int(travel_speed > config.IMPOSSIBLE_TRAVEL_SPEED_KMH and geo_dist > 250.0)

            # Velocity & amounts
            mean_amt = u_state["mean_amount"]
            std_amt = u_state["std_amount"]
            amt_ratio = amount / (mean_amt + 1e-4)
            amt_z = (amount - mean_amt) / (std_amt + 1e-4)

            # Rolling window counts from user tx_history
            now_np = pd.to_datetime(ts)
            hist = u_state["tx_history"]
            c_1h = sum(1 for t, a in hist if (now_np - pd.to_datetime(t)).total_seconds() <= 3600)
            s_1h = sum(a for t, a in hist if (now_np - pd.to_datetime(t)).total_seconds() <= 3600)
            c_24h = sum(1 for t, a in hist if (now_np - pd.to_datetime(t)).total_seconds() <= 86400)
            s_24h = sum(a for t, a in hist if (now_np - pd.to_datetime(t)).total_seconds() <= 86400)
            c_7d = sum(1 for t, a in hist if (now_np - pd.to_datetime(t)).total_seconds() <= 7 * 86400)
        else:
            time_since_sec = 86400.0
            geo_dist = 0.0
            travel_speed = 0.0
            is_impossible = 0
            amt_ratio = 1.0
            amt_z = 0.0
            c_1h, s_1h, c_24h, s_24h, c_7d = 0, 0.0, 0, 0.0, 0

        # Graph linkage from state
        dev_users = len(self.device_state.get(device_id, {user_id}))
        if device_id.startswith("POS_"):
            dev_users = 1
        ip_users = len(self.ip_state.get(ip_address, {user_id}))
        if ip_address.startswith("POS_"):
            ip_users = 1
        card_users = len(self.card_state.get(card_id, {user_id}))

        is_sh_dev = int(dev_users > 1)
        is_sh_ip = int(ip_users > 2)
        is_sh_card = int(card_users > 1)

        g_risk = min(1.0, max(0.0, ((dev_users - 1) * 0.45 + (ip_users - 1) * 0.25 + (card_users - 1) * 0.3) / 3.0))

        feat_dict = {
            "amount": amount,
            "hour_of_day": hour,
            "is_night_hours": is_night,
            "day_of_week": dow,
            "is_weekend": is_wknd,
            "mcc_risk_score": mcc_risk,
            "is_high_risk_mcc": is_high_mcc,
            "is_channel_web": is_web,
            "is_channel_mobile": is_mob,
            "is_channel_wire": is_wire,
            "is_channel_pos": is_pos,
            "is_aml_structuring_range": is_struct,
            "geo_distance_km": geo_dist,
            "travel_speed_kmh": travel_speed,
            "is_impossible_travel": is_impossible,
            "time_since_last_tx_sec": time_since_sec,
            "user_amount_ratio": amt_ratio,
            "user_amount_zscore": amt_z,
            "tx_count_1h": c_1h,
            "tx_sum_1h": s_1h,
            "tx_count_24h": c_24h,
            "tx_sum_24h": s_24h,
            "tx_count_7d": c_7d,
            "device_user_count": dev_users,
            "ip_user_count": ip_users,
            "user_device_count": 1,
            "user_card_count": 1,
            "card_user_count": card_users,
            "is_shared_device": is_sh_dev,
            "is_shared_ip": is_sh_ip,
            "is_shared_card": is_sh_card,
            "graph_risk_score": g_risk,
        }

        return pd.DataFrame([feat_dict])[FEATURE_COLUMNS]

    def save(self, filepath: str = None) -> None:
        """Persists pipeline state cache for online serving."""
        if filepath is None:
            filepath = config.MODELS_DIR / "feature_pipeline.joblib"
        joblib.dump(self, filepath)
        print(f"[FeaturePipeline] Saved feature pipeline to {filepath}")

    @classmethod
    def load(cls, filepath: str = None) -> "FeaturePipeline":
        if filepath is None:
            filepath = config.MODELS_DIR / "feature_pipeline.joblib"
        return joblib.load(filepath)
