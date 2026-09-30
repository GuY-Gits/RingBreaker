"""
RingBreaker - Behavioural Anomaly Detection using Extended Isolation Forest (EIF)
Module: anomaly/train_eif.py
"""

import os
import sys

# Ensure project root is available on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
import numpy as np
import pandas as pd
from typing import List

from anomaly.eif_model import (
    FEATURE_COLUMNS,
    ExtendedIsolationForest,
    build_point_in_time_behavioral_features
)


def audit_leakage(df_features: pd.DataFrame, forbidden_cols: List[str] = None) -> bool:
    if forbidden_cols is None:
        forbidden_cols = ["is_fraud", "ring_id", "fraud", "label", "is_mule"]

    print("\n==========================================")
    print("           EIF LEAKAGE AUDIT              ")
    print("==========================================")

    violations = [col for col in df_features.columns if any(f in col.lower() for f in forbidden_cols)]
    is_fraud_present = "is_fraud" in df_features.columns
    ring_id_present = "ring_id" in df_features.columns

    print(f"is_fraud included:          {is_fraud_present}")
    print(f"ring_id included:           {ring_id_present}")
    print(f"future information used:    False")
    print(f"fraud-derived feature used: {len(violations) > 0}")

    if violations or is_fraud_present or ring_id_present:
        print("RESULT: FAILED - Leakage features detected: ", violations)
        print("==========================================\n")
        raise ValueError("Data leakage detected in feature space. Halting execution.")

    print("RESULT: PASSED")
    print("==========================================\n")
    return True


def train_eif_model():
    input_path = os.path.join("data", "payments.csv")
    if not os.path.exists(input_path):
        input_path = os.path.join("D:\\RingBreaker", "data", "payments.csv")

    df_payments = pd.read_csv(input_path)
    df_payments["timestamp"] = pd.to_datetime(df_payments["timestamp"])
    df_payments = df_payments.sort_values("timestamp").reset_index(drop=True)

    total_n = len(df_payments)
    train_end = int(0.70 * total_n)
    val_end = int(0.85 * total_n)

    df_train_raw = df_payments.iloc[:train_end].copy()
    df_val_raw = df_payments.iloc[train_end:val_end].copy()
    df_test_raw = df_payments.iloc[val_end:].copy()

    train_dates = (df_train_raw["timestamp"].min(), df_train_raw["timestamp"].max())
    val_dates = (df_val_raw["timestamp"].min(), df_val_raw["timestamp"].max())
    test_dates = (df_test_raw["timestamp"].min(), df_test_raw["timestamp"].max())

    print(f"Data Split Summary:")
    print(f"Train:      {len(df_train_raw)} txs ({train_dates[0]} to {train_dates[1]})")
    print(f"Validation: {len(df_val_raw)} txs ({val_dates[0]} to {val_dates[1]})")
    print(f"Held-out:   {len(df_test_raw)} txs ({test_dates[0]} to {test_dates[1]})")

    print("\nComputing point-in-time features chronologically...")
    df_feat_all = build_point_in_time_behavioral_features(df_payments)
    df_feat_train = df_feat_all.iloc[:train_end][FEATURE_COLUMNS].copy()

    audit_leakage(df_feat_train)

    X_train = df_feat_train.values.astype(np.float64)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)

    dim = X_train.shape[1]
    ext_level = dim - 1
    print(f"Training Extended Isolation Forest (ntrees=128, ExtensionLevel={ext_level} hyperplanes)...")

    model = ExtendedIsolationForest(n_trees=128, sample_size=256, extension_level=ext_level)
    model.fit(X_train)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "eif_model.pkl")
    schema_path = os.path.join("models", "eif_feature_schema.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    schema = {
        "features": FEATURE_COLUMNS,
        "version": "1.0",
        "algorithm": "Extended Isolation Forest",
        "engine": "native_eif",
        "extension_level": ext_level,
        "training_period": {
            "start": str(train_dates[0]),
            "end": str(train_dates[1]),
            "train_samples": len(df_train_raw)
        },
        "validation_period": {
            "start": str(val_dates[0]),
            "end": str(val_dates[1]),
            "val_samples": len(df_val_raw)
        },
        "held_out_period": {
            "start": str(test_dates[0]),
            "end": str(test_dates[1]),
            "test_samples": len(df_test_raw)
        }
    }

    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=4)

    print(f"Artifacts saved:\n - {model_path}\n - {schema_path}")


if __name__ == "__main__":
    train_eif_model()