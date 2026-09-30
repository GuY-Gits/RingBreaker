"""
IEEE-CIS External Validation Benchmark Evaluation Pipeline
Computes:
- ROC-AUC
- PR-AUC (Average Precision)
- Precision, Recall, F1
- False Positive Rate (FPR)
- Fraud Counts per split

Writes results to experiments/ieee_cis/results.json
"""

import json
import os
import pickle
from typing import Dict, Any
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> Dict[str, Any]:
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Evaluate at standard decision threshold (0.50)
    preds_50 = (y_prob >= 0.50).astype(int)
    p_50, r_50, f1_50, _ = precision_recall_fscore_support(y_true, preds_50, average="binary", zero_division=0)
    tn_50, fp_50, fn_50, tp_50 = confusion_matrix(y_true, preds_50).ravel()
    fpr_50 = float(fp_50 / (fp_50 + tn_50)) if (fp_50 + tn_50) > 0 else 0.0

    # Optimal F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    best_p = float(precisions[best_idx])
    best_r = float(recalls[best_idx])

    preds_opt = (y_prob >= best_thresh).astype(int)
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, preds_opt).ravel()
    fpr_opt = float(fp_opt / (fp_opt + tn_opt)) if (fp_opt + tn_opt) > 0 else 0.0

    return {
        "model_name": model_name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "at_threshold_0.50": {
            "precision": round(float(p_50), 4),
            "recall": round(float(r_50), 4),
            "f1": round(float(f1_50), 4),
            "fpr": round(fpr_50, 4),
        },
        "optimal_f1": {
            "threshold": round(best_thresh, 4),
            "precision": round(best_p, 4),
            "recall": round(best_r, 4),
            "f1": round(best_f1, 4),
            "fpr": round(fpr_opt, 4),
        }
    }


def run_evaluation(
    models_dir: str = "/Users/mokssha/Desktop/RingBreaker/RingBreaker/experiments/ieee_cis/models",
    output_path: str = "/Users/mokssha/Desktop/RingBreaker/RingBreaker/experiments/ieee_cis/results.json",
):
    test_data_path = os.path.join(models_dir, "test_data.pkl")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Missing test data artifact at {test_data_path}. Run train.py first.")

    with open(test_data_path, "rb") as f:
        data = pickle.load(f)

    X_test = data["X_test"]
    y_test = data["y_test"].values
    fraud_counts = data["fraud_counts"]

    results = {
        "benchmark": "IEEE-CIS Fraud Detection External Validation",
        "split_summary": fraud_counts,
        "features_evaluated_count": len(data["feature_cols"]),
        "features_evaluated": data["feature_cols"],
        "models": {}
    }

    # 1. Logistic Regression
    lr_path = os.path.join(models_dir, "baseline_logistic_regression.joblib")
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)
        lr_probs = lr_model.predict_proba(X_test)[:, 1]
        results["models"]["logistic_regression"] = compute_metrics(y_test, lr_probs, "Logistic Regression Baseline")

    # 2. XGBoost
    xgb_path = os.path.join(models_dir, "xgboost_model.json")
    if os.path.exists(xgb_path):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        results["models"]["xgboost"] = compute_metrics(y_test, xgb_probs, "XGBoost Classifier")

    print("\n=======================================================")
    print("        IEEE-CIS EXTERNAL BENCHMARK EVALUATION         ")
    print("=======================================================")
    print(f"Test Split Size: {len(y_test)} transactions, {fraud_counts['test_fraud']} frauds ({fraud_counts['test_fraud']/len(y_test)*100:.2f}% fraud rate)")
    print("-------------------------------------------------------")
    print(f"{'Model':<25} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'F1 (opt)':<8} | {'Recall':<8} | {'Precision':<9} | {'FPR':<6}")
    print("--------------------------------------------------------------------------------------------------")

    for k, m in results["models"].items():
        opt = m["optimal_f1"]
        print(f"{m['model_name']:<25} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<8.4f} | {opt['f1']:<8.4f} | {opt['recall']:<8.4f} | {opt['precision']:<9.4f} | {opt['fpr']:<6.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed evaluation results saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
