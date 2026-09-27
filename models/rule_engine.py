import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class RuleEngine:
    """
    Expert Deterministic Fraud Rules Engine.
    Executes high-confidence business heuristics with severity scoring.
    """

    RULES = [
        {
            "id": "R01_IMPOSSIBLE_TRAVEL",
            "name": "Impossible Travel Velocity",
            "points": 45,
            "severity": "CRITICAL",
            "condition": lambda r: r.get("is_impossible_travel", 0) == 1 or r.get("travel_speed_kmh", 0) > config.IMPOSSIBLE_TRAVEL_SPEED_KMH,
            "description": "Physical travel between consecutive locations exceeds commercial airline velocity (>800 km/h).",
        },
        {
            "id": "R02_AML_STRUCTURING",
            "name": "AML Currency Transaction Structuring",
            "points": 40,
            "severity": "HIGH",
            "condition": lambda r: (config.AML_STRUCTURING_MIN <= r.get("amount", 0) <= config.AML_STRUCTURING_MAX),
            "description": "Transaction amount is deliberately clustered just below the $10,000 BSA regulatory threshold.",
        },
        {
            "id": "R03_RAPID_BURST_VELOCITY",
            "name": "High-Frequency Velocity Burst",
            "points": 35,
            "severity": "HIGH",
            "condition": lambda r: r.get("tx_count_1h", 0) >= 3 or r.get("tx_sum_1h", 0) > 3000.0,
            "description": "Abnormal velocity: 3+ transactions or >$3,000 spent within the rolling 1-hour window.",
        },
        {
            "id": "R04_ATO_PROFILE",
            "name": "Account Takeover Behavioral Anomaly",
            "points": 40,
            "severity": "HIGH",
            "condition": lambda r: (r.get("is_night_hours", 0) == 1 and r.get("user_amount_ratio", 1.0) > 4.0),
            "description": "Off-hours transaction (1 AM - 5 AM) with spending spike >4x user historical baseline.",
        },
        {
            "id": "R05_DEVICE_COLLUSION_FARM",
            "name": "Multi-Account Device Collusion",
            "points": 50,
            "severity": "CRITICAL",
            "condition": lambda r: r.get("device_user_count", 1) >= 3 and not str(r.get("device_id", "")).startswith("POS_"),
            "description": "Device hardware fingerprint shared across 3 or more distinct user accounts.",
        },
        {
            "id": "R06_SHARED_IP_CLUSTER",
            "name": "Shared IP Proxy / VPN Risk",
            "points": 30,
            "severity": "MEDIUM",
            "condition": lambda r: r.get("ip_user_count", 1) >= 4 and not str(r.get("ip_address", "")).startswith("POS_"),
            "description": "IP address accessed simultaneously by 4 or more distinct customer accounts.",
        },
        {
            "id": "R07_STATISTICAL_OUTLIER_AMOUNT",
            "name": "Extreme Amount Outlier (Z-Score > 4.5)",
            "points": 30,
            "severity": "MEDIUM",
            "condition": lambda r: r.get("user_amount_zscore", 0.0) > 4.5,
            "description": "Transaction amount is >4.5 standard deviations above the customer's typical spending distribution.",
        },
        {
            "id": "R08_HIGH_RISK_CRYPTO_WIRE",
            "name": "High-Risk Merchant Wire Transfer",
            "points": 35,
            "severity": "HIGH",
            "condition": lambda r: r.get("is_channel_wire", 0) == 1 and (r.get("is_high_risk_mcc", 0) == 1 or r.get("amount", 0) > 5000.0),
            "description": "Large wire transfer directed toward high-risk crypto, casino, or money service MCCs.",
        },
    ]

    def evaluate_record(self, record: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluates a single record against all deterministic rules.
        Returns:
            - rule_score: Clamped score from 0.0 to 100.0
            - triggered_rules: Detailed metadata of all rules triggered
        """
        triggered = []
        total_points = 0.0

        for rule in self.RULES:
            try:
                if rule["condition"](record):
                    triggered.append({
                        "rule_id": rule["id"],
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "points": rule["points"],
                        "description": rule["description"],
                    })
                    total_points += rule["points"]
            except Exception:
                continue

        # Clamp rule score to [0, 100]
        rule_score = min(100.0, total_points)
        return rule_score, triggered

    def evaluate_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        """Evaluates an entire DataFrame of transactions."""
        scores = []
        all_triggered = []

        records = df.to_dict(orient="records")
        for rec in records:
            score, trig = self.evaluate_record(rec)
            scores.append(score)
            all_triggered.append(trig)

        return np.array(scores), all_triggered
