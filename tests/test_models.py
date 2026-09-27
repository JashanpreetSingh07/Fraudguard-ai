import pytest
import pandas as pd
import numpy as np
from models.rule_engine import RuleEngine
from models.ensemble import HybridFraudEnsemble


def test_rule_engine_structuring():
    rule_engine = RuleEngine()
    # Transaction designed to trigger AML Structuring rule ($9,800)
    tx = {
        "amount": 9800.0,
        "channel": "wire_transfer",
        "is_aml_structuring_range": 1,
    }
    score, triggered = rule_engine.evaluate_record(tx)
    assert score >= 40.0
    rule_ids = [r["rule_id"] for r in triggered]
    assert "R02_AML_STRUCTURING" in rule_ids


def test_rule_engine_impossible_travel():
    rule_engine = RuleEngine()
    tx = {
        "amount": 500.0,
        "is_impossible_travel": 1,
        "travel_speed_kmh": 1200.0,
    }
    score, triggered = rule_engine.evaluate_record(tx)
    assert score >= 45.0
    rule_ids = [r["rule_id"] for r in triggered]
    assert "R01_IMPOSSIBLE_TRAVEL" in rule_ids


def test_hybrid_ensemble_scoring():
    ensemble = HybridFraudEnsemble.load()
    assert ensemble is not None

    raw_tx = {
        "transaction_id": "TX_TEST_ENSEMBLE",
        "user_id": "USR_00010",
        "card_id": "CARD_00010",
        "amount": 50.0,
        "channel": "pos",
        "device_id": "DEV_NORMAL",
        "ip_address": "192.168.1.1",
    }
    # Create valid dummy feature row
    from features.feature_store import FEATURE_COLUMNS
    feat_df = pd.DataFrame([{col: 0.0 for col in FEATURE_COLUMNS}])
    feat_df["amount"] = 50.0

    res = ensemble.score_record(raw_tx, feat_df)
    assert "composite_risk_score" in res
    assert "risk_tier" in res
    assert "recommended_action" in res
    assert 0.0 <= res["composite_risk_score"] <= 100.0
