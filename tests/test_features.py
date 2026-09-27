import pytest
import pandas as pd
from features.geospatial import haversine_distance_km
from features.feature_store import FeaturePipeline, FEATURE_COLUMNS


def test_haversine_distance():
    # NYC (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5570 km
    dist = haversine_distance_km(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < dist < 5700

    # Same location should be zero
    zero_dist = haversine_distance_km(40.7128, -74.0060, 40.7128, -74.0060)
    assert zero_dist == pytest.approx(0.0, abs=1e-3)


def test_feature_pipeline_single_record():
    pipeline = FeaturePipeline()
    tx = {
        "transaction_id": "TX_TEST_01",
        "user_id": "USR_99999",
        "card_id": "CARD_99999",
        "amount": 125.50,
        "channel": "web",
        "mcc": "5732",
        "device_id": "DEV_TEST_01",
        "ip_address": "192.168.1.100",
        "location_lat": 40.7128,
        "location_lon": -74.0060,
    }

    feat_row = pipeline.extract_single_record(tx)
    assert isinstance(feat_row, pd.DataFrame)
    assert feat_row.shape[0] == 1
    assert feat_row.shape[1] == len(FEATURE_COLUMNS)

    for col in FEATURE_COLUMNS:
        assert col in feat_row.columns
        assert not feat_row[col].isna().any()
