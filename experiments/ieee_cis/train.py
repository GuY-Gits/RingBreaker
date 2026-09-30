"""
IEEE-CIS External Validation Benchmark Training Pipeline
Trains:
1. Baseline Linear Model (Logistic Regression with class_weight='balanced')
2. Gradient Boosted Decision Tree (XGBoost with scale_pos_weight)

Saves models to experiments/ieee_cis/models/
"""

import argparse
import os
import pickle
import joblib
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_mapping import engineer_features, load_raw_ieee_data


def train_models(
    data_dir: str = "/Users/mokssha/Desktop/RingBreaker/ieee-fraud-detection",
    output_dir: str = "/Users/mokssha/Desktop/RingBreaker/RingBreaker/experiments/ieee_cis/models",
    max_rows: int = 150000,
):
    os.makedirs(output_dir, exist_ok=True)
    df = load_raw_ieee_data(data_dir=data_dir, max_rows=max_rows)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, fraud_counts = engineer_features(df)

    print("\n--- Split Statistics (Chronological) ---")
    print(f"Train: {fraud_counts['train_total']} rows, {fraud_counts['train_fraud']} frauds ({fraud_counts['train_fraud']/fraud_counts['train_total']*100:.2f}%)")
    print(f"Val:   {fraud_counts['val_total']} rows, {fraud_counts['val_fraud']} frauds ({fraud_counts['val_fraud']/fraud_counts['val_total']*100:.2f}%)")
    print(f"Test:  {fraud_counts['test_total']} rows, {fraud_counts['test_fraud']} frauds ({fraud_counts['test_fraud']/fraud_counts['test_total']*100:.2f}%)")

    # 1. Baseline: Logistic Regression Pipeline
    print("\n[1/2] Training Baseline Logistic Regression (balanced)...")
    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)

    lr_path = os.path.join(output_dir, "baseline_logistic_regression.joblib")
    joblib.dump(lr_pipeline, lr_path)
    print(f"Saved Logistic Regression model to {lr_path}")

    # 2. Gradient Boosted Tree: XGBoost
    print("\n[2/2] Training XGBoost Classifier...")
    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    xgb_model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        eval_metric="aucpr",
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    xgb_path = os.path.join(output_dir, "xgboost_model.json")
    xgb_model.save_model(xgb_path)
    print(f"Saved XGBoost model to {xgb_path}")

    # Save test dataset and metadata for standalone evaluate.py
    test_data_path = os.path.join(output_dir, "test_data.pkl")
    with open(test_data_path, "wb") as f:
        pickle.dump({
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "fraud_counts": fraud_counts,
        }, f)
    print(f"Saved test dataset and metadata to {test_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IEEE-CIS models")
    parser.add_argument("--max_rows", type=int, default=150000, help="Max rows to load for benchmark")
    args = parser.parse_args()
    train_models(max_rows=args.max_rows)
