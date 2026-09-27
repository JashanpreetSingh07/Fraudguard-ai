import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "artifacts" / "models"
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"
SAR_EXPORTS_DIR = BASE_DIR / "artifacts" / "sar_reports"
DATABASE_PATH = BASE_DIR / "fraud_investigation.db"

# Ensure directories exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SAR_EXPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Generation Settings
DEFAULT_NUM_TRANSACTIONS = 15000
DEFAULT_NUM_USERS = 800
DEFAULT_NUM_MERCHANTS = 250
DEFAULT_NUM_DEVICES = 1100
DEFAULT_NUM_IPS = 1300
RANDOM_SEED = 42

# Risk Thresholds
THRESHOLD_LOW = 30.0
THRESHOLD_MEDIUM = 65.0
THRESHOLD_HIGH = 85.0

# SAR Generation Threshold
SAR_MINIMUM_RISK_SCORE = 75.0

# Regulatory Structuring Thresholds (e.g. BSA / FinCEN $10,000 CTR limit)
AML_STRUCTURING_MIN = 8500.00
AML_STRUCTURING_MAX = 9999.99

# Impossible Travel Speed Threshold (km/h)
IMPOSSIBLE_TRAVEL_SPEED_KMH = 800.0

# Model Ensemble Weights
ENSEMBLE_WEIGHTS = {
    "rule_engine": 0.25,
    "lightgbm": 0.35,
    "xgboost": 0.25,
    "isolation_forest": 0.15,
}

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000

# Streamlit Configuration
DASHBOARD_PORT = 8501
