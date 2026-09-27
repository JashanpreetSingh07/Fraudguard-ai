import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

FEATURE_DISPLAY_NAMES = {
    "amount": "Transaction Amount",
    "hour_of_day": "Hour of Day",
    "is_night_hours": "Late Night / Off-Hours Activity",
    "day_of_week": "Day of Week",
    "is_weekend": "Weekend Transaction",
    "mcc_risk_score": "Merchant Category Risk Weight",
    "is_high_risk_mcc": "High Risk Merchant (Crypto/Luxury/Gambling)",
    "is_channel_web": "Online Web Channel",
    "is_channel_mobile": "Mobile App Channel",
    "is_channel_wire": "Direct Wire Transfer",
    "is_channel_pos": "Physical POS Swiped",
    "is_aml_structuring_range": "Near CTR Threshold ($8,500 - $9,999)",
    "geo_distance_km": "Distance from Prior Location (km)",
    "travel_speed_kmh": "Travel Velocity (km/h)",
    "is_impossible_travel": "Impossible Travel Velocity (>800 km/h)",
    "time_since_last_tx_sec": "Time Since Last Transaction",
    "user_amount_ratio": "Ratio to Historical Average Spend",
    "user_amount_zscore": "Spend Statistical Z-Score",
    "tx_count_1h": "1-Hour Transaction Count",
    "tx_sum_1h": "1-Hour Cumulative Spend",
    "tx_count_24h": "24-Hour Transaction Count",
    "tx_sum_24h": "24-Hour Cumulative Spend",
    "tx_count_7d": "7-Day Transaction Count",
    "device_user_count": "Users Associated with Device",
    "ip_user_count": "Users Associated with IP",
    "user_device_count": "Devices Linked to User",
    "user_card_count": "Cards Linked to User",
    "card_user_count": "Users Linked to Card",
    "is_shared_device": "Shared Device Fingerprint",
    "is_shared_ip": "Shared IP Proxy / Cluster",
    "is_shared_card": "Shared Payment Card",
    "graph_risk_score": "Network Graph Risk Propagation Index",
}


class FraudExplainer:
    """
    Model Explainability using SHAP TreeExplainer.
    Provides local transaction-level risk attribution and human-readable narratives.
    """

    def __init__(self, model: Any = None):
        self.model = model
        self.explainer: Optional[shap.TreeExplainer] = None

    def fit(self, model: Any, background_sample: pd.DataFrame = None) -> "FraudExplainer":
        """Initializes TreeExplainer with background baseline data."""
        self.model = model
        # TreeExplainer is fast and exact for LightGBM/XGBoost trees
        if background_sample is not None and len(background_sample) > 200:
            sample = background_sample.sample(200, random_state=config.RANDOM_SEED)
            self.explainer = shap.TreeExplainer(self.model, data=sample)
        else:
            self.explainer = shap.TreeExplainer(self.model)
        return self

    def explain_record(self, X_row: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single transaction.
        Returns top risk-increasing and risk-decreasing factors with human readable labels.
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call fit() first.")

        shap_values = self.explainer(X_row)
        
        # Handle SHAP output format (binary classification values)
        if len(shap_values.values.shape) == 3:
            values = shap_values.values[0, :, 1]
            base_val = float(shap_values.base_values[0, 1])
        elif len(shap_values.values.shape) == 2:
            values = shap_values.values[0, :]
            base_val = float(shap_values.base_values[0]) if hasattr(shap_values.base_values, "__getitem__") else float(shap_values.base_values)
        else:
            values = shap_values.values
            base_val = 0.0

        feature_names = list(X_row.columns)
        raw_vals = X_row.iloc[0].values

        explanations = []
        for name, sh_val, raw_val in zip(feature_names, values, raw_vals):
            explanations.append({
                "feature": name,
                "display_name": FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title()),
                "shap_value": float(sh_val),
                "actual_value": float(raw_val),
                "is_risk_increasing": bool(sh_val > 0),
            })

        # Sort by magnitude of risk increase
        risk_increasing = sorted([e for e in explanations if e["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
        risk_mitigating = sorted([e for e in explanations if e["shap_value"] <= 0], key=lambda x: x["shap_value"])

        # Create natural language summary
        top_reasons = []
        for f in risk_increasing[:3]:
            top_reasons.append(f"{f['display_name']} (SHAP: +{f['shap_value']:.2f}, Val: {f['actual_value']:.1f})")

        narrative = "Elevated risk driven primarily by: " + "; ".join(top_reasons) if top_reasons else "Normal behavioral profile."

        return {
            "base_value": base_val,
            "top_risk_factors": risk_increasing[:top_k],
            "top_mitigating_factors": risk_mitigating[:top_k],
            "narrative": narrative,
        }

    def save(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = config.MODELS_DIR / "fraud_explainer.joblib"
        joblib.dump(self, filepath)
        print(f"[FraudExplainer] Saved explainer to {filepath}")

    @classmethod
    def load(cls, filepath: str = None) -> "FraudExplainer":
        if filepath is None:
            filepath = config.MODELS_DIR / "fraud_explainer.joblib"
        return joblib.load(filepath)
