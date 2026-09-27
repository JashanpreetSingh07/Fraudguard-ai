import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

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


class HybridFraudEnsemble:
    """
    Multi-layered Hybrid Detection Engine combining:
    1. Deterministic Rule Engine (Expert policies)
    2. Supervised LightGBM Model (Tabular pattern classification)
    3. Supervised XGBoost Model (Model diversity & stability)
    4. Unsupervised Isolation Forest (Zero-day anomaly detection)
    5. Graph Risk Modifiers (Collusion & mule detection)
    """

    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        lightgbm_model: Optional[LightGBMFraudModel] = None,
        xgboost_model: Optional[XGBoostFraudModel] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        explainer: Optional[FraudExplainer] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.rule_engine = rule_engine or RuleEngine()
        self.lightgbm_model = lightgbm_model
        self.xgboost_model = xgboost_model
        self.anomaly_detector = anomaly_detector
        self.explainer = explainer
        self.weights = weights or config.ENSEMBLE_WEIGHTS

    def score_record(self, raw_tx: Dict[str, Any], feature_row: pd.DataFrame) -> Dict[str, Any]:
        """
        Scores a single transaction in real-time.
        Returns:
            - composite_risk_score (0-100)
            - risk_tier ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
            - recommended_action ('APPROVE', 'STEP_UP_AUTH', 'HOLD_FOR_INVESTIGATION', 'DECLINE_AND_FREEZE')
            - component_scores (rule, lgb, xgb, anomaly)
            - rules_triggered
            - xai_explanation (SHAP top drivers)
        """
        # 1. Rule Engine
        combined_rec = {**raw_tx, **feature_row.iloc[0].to_dict()}
        rule_score, rules_triggered = self.rule_engine.evaluate_record(combined_rec)

        # 2. Supervised LightGBM
        lgb_score = float(self.lightgbm_model.predict_score(feature_row)[0]) if self.lightgbm_model else rule_score

        # 3. Supervised XGBoost
        xgb_score = float(self.xgboost_model.predict_score(feature_row)[0]) if self.xgboost_model else rule_score

        # 4. Anomaly Detector
        iso_score = float(self.anomaly_detector.predict_anomaly_score(feature_row)[0]) if self.anomaly_detector else 30.0

        # Weighted blend
        w = self.weights
        composite = (
            w["rule_engine"] * rule_score
            + w["lightgbm"] * lgb_score
            + w["xgboost"] * xgb_score
            + w["isolation_forest"] * iso_score
        )

        # Graph risk boost (e.g. if part of multi-device or multi-user ring)
        graph_risk = feature_row.iloc[0].get("graph_risk_score", 0.0)
        composite += (graph_risk * 15.0)

        # Critical heuristic floor: if critical rules triggered (e.g. Impossible Travel), enforce minimum score
        has_critical_rule = any(r["severity"] == "CRITICAL" for r in rules_triggered)
        if has_critical_rule:
            composite = max(composite, 85.0)

        composite_risk_score = round(float(np.clip(composite, 0.0, 100.0)), 2)

        # Determine Tier and Action
        if composite_risk_score < config.THRESHOLD_LOW:
            risk_tier = "LOW"
            recommended_action = "APPROVE"
        elif composite_risk_score < config.THRESHOLD_MEDIUM:
            risk_tier = "MEDIUM"
            recommended_action = "STEP_UP_AUTH"  # Trigger 2FA / OTP challenge
        elif composite_risk_score < config.THRESHOLD_HIGH:
            risk_tier = "HIGH"
            recommended_action = "HOLD_FOR_INVESTIGATION"  # Queue into case management
        else:
            risk_tier = "CRITICAL"
            recommended_action = "DECLINE_AND_FREEZE"  # Immediate block and freeze

        # XAI Explanation
        xai_narrative = ""
        top_risk_factors = []
        if self.explainer:
            try:
                xai_res = self.explainer.explain_record(feature_row)
                top_risk_factors = xai_res["top_risk_factors"]
                xai_narrative = xai_res["narrative"]
            except Exception as e:
                xai_narrative = f"Explanation generation error: {str(e)}"

        return {
            "transaction_id": raw_tx.get("transaction_id", "TX_UNKNOWN"),
            "composite_risk_score": composite_risk_score,
            "risk_tier": risk_tier,
            "recommended_action": recommended_action,
            "component_scores": {
                "rule_engine": round(rule_score, 2),
                "lightgbm": round(lgb_score, 2),
                "xgboost": round(xgb_score, 2),
                "isolation_forest": round(iso_score, 2),
                "graph_risk_boost": round(graph_risk * 15.0, 2),
            },
            "rules_triggered": rules_triggered,
            "top_risk_factors": top_risk_factors,
            "explanation_narrative": xai_narrative,
        }

    def score_batch(self, raw_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Scores an entire batch of transactions efficiently."""
        rule_scores, all_triggered = self.rule_engine.evaluate_batch(raw_df)
        lgb_scores = self.lightgbm_model.predict_score(X) if self.lightgbm_model else rule_scores
        xgb_scores = self.xgboost_model.predict_score(X) if self.xgboost_model else rule_scores
        iso_scores = self.anomaly_detector.predict_anomaly_score(X) if self.anomaly_detector else np.full(len(X), 30.0)

        w = self.weights
        composite = (
            w["rule_engine"] * rule_scores
            + w["lightgbm"] * lgb_scores
            + w["xgboost"] * xgb_scores
            + w["isolation_forest"] * iso_scores
        )

        if "graph_risk_score" in X.columns:
            composite += (X["graph_risk_score"].values * 15.0)

        # Floor critical rules
        for i, trig in enumerate(all_triggered):
            if any(r["severity"] == "CRITICAL" for r in trig):
                composite[i] = max(composite[i], 85.0)

        composite = np.clip(composite, 0.0, 100.0)

        results_df = raw_df.copy()
        results_df["composite_risk_score"] = np.round(composite, 2)
        results_df["score_rule"] = np.round(rule_scores, 2)
        results_df["score_lgb"] = np.round(lgb_scores, 2)
        results_df["score_xgb"] = np.round(xgb_scores, 2)
        results_df["score_anomaly"] = np.round(iso_scores, 2)

        tiers = []
        actions = []
        for score in composite:
            if score < config.THRESHOLD_LOW:
                tiers.append("LOW")
                actions.append("APPROVE")
            elif score < config.THRESHOLD_MEDIUM:
                tiers.append("MEDIUM")
                actions.append("STEP_UP_AUTH")
            elif score < config.THRESHOLD_HIGH:
                tiers.append("HIGH")
                actions.append("HOLD_FOR_INVESTIGATION")
            else:
                tiers.append("CRITICAL")
                actions.append("DECLINE_AND_FREEZE")

        results_df["risk_tier"] = tiers
        results_df["recommended_action"] = actions
        results_df["rules_triggered_count"] = [len(t) for t in all_triggered]

        return results_df

    def save(self, filepath: str = None) -> None:
        if filepath is None:
            filepath = config.MODELS_DIR / "hybrid_ensemble.joblib"
        joblib.dump(self, filepath)
        print(f"[HybridEnsemble] Saved ensemble to {filepath}")

    @classmethod
    def load(cls, filepath: str = None) -> "HybridFraudEnsemble":
        if filepath is None:
            filepath = config.MODELS_DIR / "hybrid_ensemble.joblib"
        return joblib.load(filepath)
