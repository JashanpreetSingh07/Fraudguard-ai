import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import lightgbm as lgb
import xgboost as xgb

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class LightGBMFraudModel:
    """Production LightGBM Gradient Boosting model optimized for fraud class imbalance."""

    def __init__(self, random_state: int = config.RANDOM_SEED):
        self.random_state = random_state
        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> "LightGBMFraudModel":
        neg_count = (y_train == 0).sum()
        pos_count = max(1, (y_train == 1).sum())
        scale_pos_weight = neg_count / pos_count

        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns fraud probability in [0.0, 1.0]."""
        return self.model.predict_proba(X)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated fraud risk score in [0.0, 100.0]."""
        return np.clip(self.predict_proba(X) * 100.0, 0.0, 100.0)

    def save(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = config.MODELS_DIR / "lightgbm_model.joblib"
        joblib.dump(self.model, filepath)
        print(f"[LightGBM] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = None) -> "LightGBMFraudModel":
        if filepath is None:
            filepath = config.MODELS_DIR / "lightgbm_model.joblib"
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance


class XGBoostFraudModel:
    """Production XGBoost model optimized for extreme fraud tabular data."""

    def __init__(self, random_state: int = config.RANDOM_SEED):
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> "XGBoostFraudModel":
        neg_count = (y_train == 0).sum()
        pos_count = max(1, (y_train == 1).sum())
        scale_pos_weight = neg_count / pos_count

        self.model = xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns fraud probability in [0.0, 1.0]."""
        return self.model.predict_proba(X)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated fraud risk score in [0.0, 100.0]."""
        return np.clip(self.predict_proba(X) * 100.0, 0.0, 100.0)

    def save(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = config.MODELS_DIR / "xgboost_model.joblib"
        joblib.dump(self.model, filepath)
        print(f"[XGBoost] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = None) -> "XGBoostFraudModel":
        if filepath is None:
            filepath = config.MODELS_DIR / "xgboost_model.joblib"
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
