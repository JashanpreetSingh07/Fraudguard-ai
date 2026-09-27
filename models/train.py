import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.rule_engine import RuleEngine
from models.anomaly_detector import AnomalyDetector
from models.supervised_models import LightGBMFraudModel, XGBoostFraudModel
from models.explainer import FraudExplainer
from models.ensemble import HybridFraudEnsemble


def run_training_pipeline() -> Tuple[HybridFraudEnsemble, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Executes end-to-end model training:
    1. Loads processed features and labels
    2. Splits into train, validation, and test sets
    3. Trains Isolation Forest, LightGBM, XGBoost
    4. Configures SHAP explainer
    5. Packages into HybridFraudEnsemble and saves all artifacts
    """
    print("==================================================")
    print("  AI-POWERED FRAUD DETECTION: TRAINING PIPELINE   ")
    print("==================================================")

    enriched_path = config.PROCESSED_DATA_DIR / "enriched_transactions.csv"
    features_path = config.PROCESSED_DATA_DIR / "features.csv"

    if not enriched_path.exists() or not features_path.exists():
        raise FileNotFoundError("Processed datasets not found. Run feature store first.")

    raw_df = pd.read_csv(enriched_path)
    X = pd.read_csv(features_path)
    y = raw_df["is_fraud"]

    print(f"Total Transactions: {len(X)}")
    print(f"Total Features:     {X.shape[1]}")
    print(f"Fraud Rate:         {y.mean()*100:.2f}% ({y.sum()} frauds, {(y==0).sum()} legitimate)")

    # Stratified Train / Val / Test Split (70% train, 15% val, 15% test)
    X_temp, X_test, y_temp, y_test, raw_temp, raw_test = train_test_split(
        X, y, raw_df, test_size=0.15, stratify=y, random_state=config.RANDOM_SEED
    )

    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
        X_temp, y_temp, raw_temp, test_size=val_ratio, stratify=y_temp, random_state=config.RANDOM_SEED
    )

    print(f"\nTrain set:      {X_train.shape[0]} samples ({y_train.sum()} frauds)")
    print(f"Validation set: {X_val.shape[0]} samples ({y_val.sum()} frauds)")
    print(f"Test set:       {X_test.shape[0]} samples ({y_test.sum()} frauds)")

    # 1. Train Anomaly Detector (Isolation Forest)
    print("\n[1/4] Training Unsupervised Isolation Forest Anomaly Detector...")
    anomaly_detector = AnomalyDetector(contamination=0.035)
    anomaly_detector.fit(X_train, y=y_train)
    anomaly_detector.save()

    # 2. Train LightGBM Classifier
    print("[2/4] Training LightGBM Gradient Boosted Classifier...")
    lgb_model = LightGBMFraudModel()
    lgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    lgb_model.save()

    # 3. Train XGBoost Classifier
    print("[3/4] Training XGBoost Gradient Boosted Classifier...")
    xgb_model = XGBoostFraudModel()
    xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    xgb_model.save()

    # 4. Train SHAP Explainer
    print("[4/4] Fitting SHAP TreeExplainer on LightGBM...")
    explainer = FraudExplainer()
    explainer.fit(lgb_model.model, background_sample=X_train)
    explainer.save()

    # 5. Build & Save Hybrid Ensemble
    print("\nPackaging Hybrid Multi-Layered Ensemble...")
    rule_engine = RuleEngine()
    ensemble = HybridFraudEnsemble(
        rule_engine=rule_engine,
        lightgbm_model=lgb_model,
        xgboost_model=xgb_model,
        anomaly_detector=anomaly_detector,
        explainer=explainer,
    )
    ensemble.save()

    # Save test partition for evaluation
    X_test.to_csv(config.PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_test.to_frame(name="is_fraud").to_csv(config.PROCESSED_DATA_DIR / "y_test.csv", index=False)
    raw_test.to_csv(config.PROCESSED_DATA_DIR / "raw_test.csv", index=False)

    print("\n[Success] Training pipeline completed successfully. All artifacts persisted.")
    return ensemble, X_test, raw_test, y_test, y_train


if __name__ == "__main__":
    run_training_pipeline()
