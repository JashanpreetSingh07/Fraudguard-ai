import pytest
import pandas as pd
from data.generator import FinancialTransactionGenerator


def test_generator_output_structure():
    gen = FinancialTransactionGenerator(num_transactions=500, num_users=50, num_merchants=20)
    df = gen.generate()

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 450
    expected_cols = [
        "transaction_id",
        "timestamp",
        "user_id",
        "card_id",
        "merchant_id",
        "amount",
        "channel",
        "device_id",
        "ip_address",
        "location_lat",
        "location_lon",
        "is_fraud",
        "fraud_type",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_fraud_typologies_injected():
    gen = FinancialTransactionGenerator(num_transactions=1000, num_users=80, num_merchants=30)
    df = gen.generate()

    fraud_df = df[df["is_fraud"] == 1]
    assert len(fraud_df) > 0
    typologies = set(fraud_df["fraud_type"].unique())

    # Ensure key typologies were created
    assert "account_takeover" in typologies
    assert "card_not_present_burst" in typologies
    assert "aml_structuring" in typologies
