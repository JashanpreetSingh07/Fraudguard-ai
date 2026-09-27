import uuid
import pytest
from investigation.case_manager import CaseManager
from investigation.sar_generator import SARGenerator


def test_case_lifecycle():
    cm = CaseManager()
    unique_tx = f"TX_TEST_LIFE_{uuid.uuid4().hex[:8]}"
    case_id = cm.create_case(
        transaction_id=unique_tx,
        user_id="USR_TEST_LIFE",
        card_id="CARD_TEST_LIFE",
        device_id="DEV_TEST_LIFE",
        ip_address="127.0.0.1",
        amount=1500.0,
        risk_score=88.5,
        risk_tier="CRITICAL",
        fraud_typology="Account Takeover",
        rules_triggered=[{"rule_id": "R04", "name": "ATO", "severity": "HIGH", "points": 40, "description": "Test"}],
        explanation_narrative="Test explanation",
    )

    assert case_id.startswith("CASE-")

    # Fetch case
    case_data = cm.get_case(case_id)
    assert case_data is not None
    assert case_data["status"] == "NEW"
    assert len(case_data["audit_history"]) >= 1

    # Transition status
    updated = cm.update_case_status(case_id, "UNDER_INVESTIGATION", actor="Lead Detective", notes="Checking IP logs")
    assert updated is True

    case_data2 = cm.get_case(case_id)
    assert case_data2["status"] == "UNDER_INVESTIGATION"
    assert len(case_data2["audit_history"]) == 2


def test_sar_generation():
    sar = SARGenerator()
    dummy_case = {
        "case_id": "CASE-TEST-SAR-001",
        "transaction_id": "TX_SAR_001",
        "user_id": "USR_SAR_VICTIM",
        "card_id": "CARD_SAR_001",
        "device_id": "DEV_SAR_ROGUE",
        "ip_address": "185.220.101.5",
        "amount": 9850.00,
        "risk_score": 96.5,
        "risk_tier": "CRITICAL",
        "fraud_typology": "aml_structuring",
        "rules_triggered": [{"rule_id": "R02", "name": "AML Structuring", "severity": "HIGH", "points": 40, "description": "Near 10k"}],
        "explanation_narrative": "Near CTR threshold with crypto wire transfer.",
        "assigned_investigator": "FinCEN Specialist",
    }

    markdown_narrative = sar.generate_narrative_text(dummy_case)
    assert "SUSPICIOUS ACTIVITY REPORT" in markdown_narrative
    assert "31 CFR § 1010.311" in markdown_narrative

    pdf_path = sar.export_pdf(dummy_case, filename="TEST_SAR.pdf")
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000
