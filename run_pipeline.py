import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.generator import generate_and_save_data
from features.feature_store import FeaturePipeline
from models.train import run_training_pipeline
from models.evaluate import evaluate_models
from investigation.case_manager import CaseManager
import pandas as pd


def step_generate_data():
    print("\n>>> STEP 1: Generating Synthetic Transaction Stream with Fraud Typologies...")
    df = generate_and_save_data()
    return df


def step_process_features(df=None):
    print("\n>>> STEP 2: Running Feature Engineering Pipeline (Velocity, Geo, Graph)...")
    if df is None:
        raw_path = config.RAW_DATA_DIR / "transactions.csv"
        df = pd.read_csv(raw_path)
    pipeline = FeaturePipeline()
    enriched_df, X = pipeline.process_batch(df)
    pipeline.save()
    enriched_df.to_csv(config.PROCESSED_DATA_DIR / "enriched_transactions.csv", index=False)
    X.to_csv(config.PROCESSED_DATA_DIR / "features.csv", index=False)
    print(f"[Done] Enriched dataset: {enriched_df.shape}, Feature matrix: {X.shape}")
    return enriched_df, X


def step_train_models():
    print("\n>>> STEP 3: Training Detection Stack (LightGBM, XGBoost, Isolation Forest, SHAP)...")
    ensemble, X_test, raw_test, y_test, y_train = run_training_pipeline()
    return ensemble


def step_evaluate_models():
    print("\n>>> STEP 4: Running Evaluation Benchmark & Generating Visual Reports...")
    metrics = evaluate_models()
    return metrics


def step_seed_cases():
    print("\n>>> STEP 5: Seeding Case Management Database with High-Risk Alerts...")
    raw_test_path = config.PROCESSED_DATA_DIR / "raw_test.csv"
    X_test_path = config.PROCESSED_DATA_DIR / "X_test.csv"
    if not raw_test_path.exists():
        raw_test_path = config.PROCESSED_DATA_DIR / "enriched_transactions.csv"
        X_test_path = config.PROCESSED_DATA_DIR / "features.csv"

    from models.ensemble import HybridFraudEnsemble
    ensemble = HybridFraudEnsemble.load()
    raw_df = pd.read_csv(raw_test_path)
    X_df = pd.read_csv(X_test_path)

    scored_df = ensemble.score_batch(raw_df, X_df)
    cm = CaseManager()
    seeded = cm.seed_from_scored_dataset(scored_df, limit=100)
    print(f"[Done] Seeded {seeded} investigation cases.")


def run_all():
    print("===================================================================")
    print("   AI-POWERED FRAUD INVESTIGATION & DETECTION PLATFORM: PIPELINE   ")
    print("===================================================================")
    df = step_generate_data()
    step_process_features(df)
    step_train_models()
    step_evaluate_models()
    step_seed_cases()
    print("\n[COMPLETE] Platform pipeline executed successfully!")
    print("           Run API with:       python run_pipeline.py --serve-api")
    print("           Run Dashboard with: python run_pipeline.py --serve-dashboard")


def main():
    parser = argparse.ArgumentParser(description="AI Fraud Detection Platform Orchestrator")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline end-to-end")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic transaction stream")
    parser.add_argument("--process-features", action="store_true", help="Extract all features")
    parser.add_argument("--train", action="store_true", help="Train models and explainer")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate performance and save plots")
    parser.add_argument("--seed-cases", action="store_true", help="Seed case management database")
    parser.add_argument("--serve-api", action="store_true", help="Launch FastAPI REST server")
    parser.add_argument("--serve-dashboard", action="store_true", help="Launch Streamlit web dashboard")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.all:
        run_all()
        return

    if args.generate_data:
        step_generate_data()
    if args.process_features:
        step_process_features()
    if args.train:
        step_train_models()
    if args.evaluate:
        step_evaluate_models()
    if args.seed_cases:
        step_seed_cases()
    if args.serve_api:
        print(f"Launching FastAPI Server on http://{config.API_HOST}:{config.API_PORT} ...")
        subprocess.run([sys.executable, "-m", "uvicorn", "api.app:app", "--host", config.API_HOST, "--port", str(config.API_PORT), "--reload"])
    if args.serve_dashboard:
        print(f"Launching Streamlit Dashboard on port {config.DASHBOARD_PORT} ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(config.DASHBOARD_PORT)])


if __name__ == "__main__":
    main()
