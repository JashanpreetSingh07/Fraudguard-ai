import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Any, Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


ANOMALY_FEATURE_COLS = [
    "amount",
    "hour_of_day",
    "is_night_hours",
    "geo_distance_km",
    "travel_speed_kmh",
    "user_amount_ratio",
    "user_amount_zscore",
    "tx_count_1h",
    "tx_sum_1h",
    "device_user_count",
    "ip_user_count",
    "graph_risk_score",
]


class AnomalyDetector:
    """
    Unsupervised Isolation Forest anomaly detector trained on legitimate baseline
    patterns to detect novel, zero-day fraud attacks and unprecedented deviations.
    """

    def __init__(self, contamination: float = 0.035, random_state: int = config.RANDOM_SEED):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.score_min = -0.5
        self.score_max = 0.5

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AnomalyDetector":
        """
        Fits Isolation Forest preferentially on legitimate transactions (if labels available)
        to learn true baseline behavior.
        """
        cols = [c for c in ANOMALY_FEATURE_COLS if c in X.columns]
        X_sub = X[cols].fillna(0.0)

        if y is not None:
            # Fit primarily on legitimate records to establish normal baseline
            legit_mask = (y == 0)
            if legit_mask.sum() > 500:
                X_fit = X_sub[legit_mask]
            else:
                X_fit = X_sub
        else:
            X_fit = X_sub

        self.model.fit(X_fit)
        raw_scores = self.model.decision_function(X_sub)
        self.score_min = float(np.min(raw_scores))
        self.score_max = float(np.max(raw_scores))
        self.is_fitted = True
        return self

    def predict_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes calibrated anomaly score between 0.0 (completely normal)
        and 100.0 (extreme anomaly).
        """
        if not self.is_fitted:
            raise ValueError("AnomalyDetector must be fitted before predict_anomaly_score.")

        cols = [c for c in ANOMALY_FEATURE_COLS if c in X.columns]
        X_sub = X[cols].fillna(0.0)

        raw_scores = self.model.decision_function(X_sub)
        # Invert: lower raw score means more anomalous -> map to higher risk score
        # Normalize into [0, 100]
        denom = (self.score_max - self.score_min) if (self.score_max > self.score_min) else 1.0
        normalized = 1.0 - ((raw_scores - self.score_min) / denom)
        normalized_scores = np.clip(normalized * 100.0, 0.0, 100.0)
        return normalized_scores

    def save(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = config.MODELS_DIR / "isolation_forest.joblib"
        joblib.dump(self, filepath)
        print(f"[AnomalyDetector] Saved anomaly model to {filepath}")

    @classmethod
    def load(cls, filepath: str = None) -> "AnomalyDetector":
        if filepath is None:
            filepath = config.MODELS_DIR / "isolation_forest.joblib"
        return joblib.load(filepath)
