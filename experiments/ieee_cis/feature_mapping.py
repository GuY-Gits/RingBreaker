"""
IEEE-CIS Feature Mapping & Chronological Dataset Preparation
Part of RingBreaker External Validation Benchmark.

Ensures zero future-information leakage:
1. Strict chronological ordering via TransactionDT.
2. Train (70%) / Validation (15%) / Test (15%) temporal split.
3. Feature transformations and frequency encodings are strictly learned on the training split.
4. No synthetic P2P relationships or fake graph edges are fabricated.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def load_raw_ieee_data(
    data_dir: str = "/Users/mokssha/Desktop/RingBreaker/ieee-fraud-detection",
    max_rows: int = 150000,
) -> pd.DataFrame:
    """Load transaction and identity files from the IEEE-CIS directory.
    
    Loads a chronological slice up to max_rows (or full dataset if max_rows is None).
    """
    tx_file = os.path.join(data_dir, "train_transaction.csv")
    id_file = os.path.join(data_dir, "train_identity.csv")

    cols_to_load = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
    ] + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]

    print(f"Loading IEEE-CIS transaction data from {tx_file} (max_rows={max_rows})...")
    df = pd.read_csv(tx_file, usecols=cols_to_load, nrows=max_rows)

    if os.path.exists(id_file):
        print(f"Merging identity metadata from {id_file}...")
        df_id = pd.read_csv(
            id_file,
            usecols=["TransactionID", "DeviceType", "DeviceInfo"],
        )
        df = df.merge(df_id, on="TransactionID", how="left")
    else:
        df["DeviceType"] = np.nan
        df["DeviceInfo"] = np.nan

    # Sort strictly chronologically
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return df


def engineer_features(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    List[str],
    Dict[str, int]
]:
    """Engineer chronological features with zero leakage across temporal splits."""
    n = len(df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    df_clean = df.copy()

    # Derived amount features (equivalent to RingBreaker amount behaviors)
    df_clean["log_amount"] = np.log1p(df_clean["TransactionAmt"].fillna(0))
    df_clean["is_round_amount"] = (df_clean["TransactionAmt"] % 1 == 0).astype(int)
    df_clean["high_amount_flag"] = (df_clean["TransactionAmt"] > 500).astype(int)

    # Cyclical temporal features
    df_clean["hour_of_day"] = (df_clean["TransactionDT"] // 3600) % 24
    df_clean["day_of_week"] = (df_clean["TransactionDT"] // (3600 * 24)) % 7
    df_clean["has_identity"] = (~df_clean["DeviceType"].isna()).astype(int)

    # Base feature list
    feature_cols = [
        "TransactionAmt",
        "log_amount",
        "is_round_amount",
        "high_amount_flag",
        "hour_of_day",
        "day_of_week",
        "has_identity",
    ] + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]

    # Temporal split
    train_df = df_clean.iloc[:train_end].copy()
    val_df = df_clean.iloc[train_end:val_end].copy()
    test_df = df_clean.iloc[val_end:].copy()

    # Frequency encodings learned strictly on train_df
    cat_cols = ["card1", "card2", "addr1", "ProductCD", "DeviceType"]
    for col in cat_cols:
        freq = train_df[col].value_counts(normalize=True).to_dict()
        freq_col_name = f"{col}_freq"
        train_df[freq_col_name] = train_df[col].map(freq).fillna(0)
        val_df[freq_col_name] = val_df[col].map(freq).fillna(0)
        test_df[freq_col_name] = test_df[col].map(freq).fillna(0)
        feature_cols.append(freq_col_name)

    fraud_counts = {
        "train_total": len(train_df),
        "train_fraud": int(train_df["isFraud"].sum()),
        "val_total": len(val_df),
        "val_fraud": int(val_df["isFraud"].sum()),
        "test_total": len(test_df),
        "test_fraud": int(test_df["isFraud"].sum()),
    }

    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"]

    X_val = val_df[feature_cols]
    y_val = val_df["isFraud"]

    X_test = test_df[feature_cols]
    y_test = test_df["isFraud"]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, fraud_counts
