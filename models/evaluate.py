import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless generation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.ensemble import HybridFraudEnsemble


def evaluate_models(output_dir: Path = config.REPORTS_DIR) -> Dict[str, Any]:
    """
    Evaluates the trained models and hybrid ensemble on the held-out test set:
    - ROC-AUC & PR-AUC curves
    - Precision @ Top K% (Investigator alert capacity)
    - Confusion Matrix
    - Financial Fraud Loss vs Prevention ROI Analysis
    - Persists charts and metrics to artifacts/reports/
    """
    print("==================================================")
    print("  AI-POWERED FRAUD DETECTION: BENCHMARK & EVAL    ")
    print("==================================================")

    output_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_test.csv")["is_fraud"].values
    raw_test = pd.read_csv(config.PROCESSED_DATA_DIR / "raw_test.csv")

    ensemble = HybridFraudEnsemble.load()

    # Predictions
    scored_test = ensemble.score_batch(raw_test, X_test)
    scores = scored_test["composite_risk_score"].values
    prob_preds = scores / 100.0

    # Individual model scores
    rule_probs = scored_test["score_rule"].values / 100.0
    lgb_probs = scored_test["score_lgb"].values / 100.0
    xgb_probs = scored_test["score_xgb"].values / 100.0
    iso_probs = scored_test["score_anomaly"].values / 100.0

    # Binary prediction at decision threshold (Threshold Medium = 65)
    pred_binary = (scores >= config.THRESHOLD_MEDIUM).astype(int)

    # 1. Core Metrics
    roc_auc = float(roc_auc_score(y_test, prob_preds))
    pr_auc = float(average_precision_score(y_test, prob_preds))

    cm = confusion_matrix(y_test, pred_binary)
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 2. Precision @ Top K%
    def precision_at_k(y_true, y_score, k_percent):
        n = len(y_true)
        k = max(1, int(n * (k_percent / 100.0)))
        top_indices = np.argsort(y_score)[::-1][:k]
        return float(np.mean(y_true[top_indices]))

    p_at_1 = precision_at_k(y_test, scores, 1.0)
    p_at_2 = precision_at_k(y_test, scores, 2.0)
    p_at_5 = precision_at_k(y_test, scores, 5.0)

    # 3. Financial Cost-Benefit ROI
    total_fraud_dollars = float(raw_test[y_test == 1]["amount"].sum())
    caught_fraud_dollars = float(raw_test[(y_test == 1) & (pred_binary == 1)]["amount"].sum())
    missed_fraud_dollars = total_fraud_dollars - caught_fraud_dollars
    investigation_cost_per_alert = 25.00  # Industry benchmark: $25 analyst review cost
    total_investigation_cost = (tp + fp) * investigation_cost_per_alert
    net_savings = caught_fraud_dollars - total_investigation_cost
    roi_percentage = (net_savings / total_investigation_cost * 100) if total_investigation_cost > 0 else 0.0

    metrics = {
        "dataset": {
            "test_samples": int(len(y_test)),
            "test_frauds": int(y_test.sum()),
            "fraud_prevalence_pct": round(float(y_test.mean() * 100), 2),
        },
        "performance": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "precision_at_top_1pct": round(p_at_1, 4),
            "precision_at_top_2pct": round(p_at_2, 4),
            "precision_at_top_5pct": round(p_at_5, 4),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "financial_impact": {
            "total_fraud_exposure_dollars": round(total_fraud_dollars, 2),
            "fraud_loss_prevented_dollars": round(caught_fraud_dollars, 2),
            "missed_fraud_dollars": round(missed_fraud_dollars, 2),
            "prevention_rate_pct": round((caught_fraud_dollars / total_fraud_dollars) * 100, 2) if total_fraud_dollars > 0 else 0.0,
            "investigation_overhead_dollars": round(total_investigation_cost, 2),
            "net_dollars_saved": round(net_savings, 2),
            "roi_percentage": round(roi_percentage, 1),
        },
    }

    # Save metrics JSON
    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n----- Evaluation Benchmark Results -----")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"PR-AUC:             {pr_auc:.4f}")
    print(f"Precision:          {precision*100:.2f}%")
    print(f"Recall:             {recall*100:.2f}%")
    print(f"F1-Score:           {f1*100:.2f}%")
    print(f"Precision @ Top 2%: {p_at_2*100:.2f}%")
    print(f"Fraud Loss Caught:  ${caught_fraud_dollars:,.2f} / ${total_fraud_dollars:,.2f} ({metrics['financial_impact']['prevention_rate_pct']}%)")
    print(f"Net ROI:            {roi_percentage:.1f}%")

    # 4. Generate Visual Charts
    _generate_evaluation_plots(y_test, prob_preds, rule_probs, lgb_probs, xgb_probs, iso_probs, cm, X_test, ensemble, output_dir)

    return metrics


def _generate_evaluation_plots(y_test, prob_preds, rule_probs, lgb_probs, xgb_probs, iso_probs, cm, X_test, ensemble, output_dir):
    """Generates ROC, PR-Curve, Confusion Matrix, and Feature Importance visuals."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for name, probs, color in [
        ("Hybrid Ensemble", prob_preds, "#10b981"),
        ("LightGBM", lgb_probs, "#3b82f6"),
        ("XGBoost", xgb_probs, "#8b5cf6"),
        ("Rule Engine", rule_probs, "#f59e0b"),
        ("Isolation Forest", iso_probs, "#ef4444"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curve.png")
    plt.close(fig)

    # Plot 2: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for name, probs, color in [
        ("Hybrid Ensemble", prob_preds, "#10b981"),
        ("LightGBM", lgb_probs, "#3b82f6"),
        ("XGBoost", xgb_probs, "#8b5cf6"),
        ("Rule Engine", rule_probs, "#f59e0b"),
    ]:
        p, r, _ = precision_recall_curve(y_test, probs)
        prauc = average_precision_score(y_test, probs)
        ax.plot(r, p, label=f"{name} (PR-AUC = {prauc:.3f})", color=color, linewidth=2)

    ax.set_title("Precision-Recall Curve (Extreme Imbalance)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Fraud Coverage)", fontsize=11)
    ax.set_ylabel("Precision (Investigator Accuracy)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "pr_curve.png")
    plt.close(fig)

    # Plot 3: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        cbar=False,
        ax=ax,
    )
    ax.set_title("Detection Confusion Matrix (Test Set)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Ground Truth", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    # Plot 4: Feature Importance (from LightGBM)
    if ensemble.lightgbm_model and hasattr(ensemble.lightgbm_model.model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        importances = ensemble.lightgbm_model.model.feature_importances_
        feature_names = X_test.columns
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=False).head(15)

        sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax)
        ax.set_title("Top 15 Feature Importances (Gradient Boosted Tree)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Split Importance Weight", fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "feature_importance.png")
        plt.close(fig)

    print(f"[Evaluation] All benchmark charts exported to {output_dir}")


if __name__ == "__main__":
    evaluate_models()
