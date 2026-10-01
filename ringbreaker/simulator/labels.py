"""Label noise application and ring metadata serialization for RingBreaker Simulator v2."""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pandas as pd

from ringbreaker.simulator.rings.base import RingSpec


def apply_label_noise(
    payments_df: pd.DataFrame,
    fraud_unlabelled_rate: float = 0.10,
    normal_labelled_fraud_rate: float = 0.002,
    seed: int = 42,
) -> pd.DataFrame:
    """Applies realistic production label noise:

    - 10% of fraud payments are unlabelled (fraud never reported/discovered) -> label_observed = 0
    - 0.2% of normal payments are labelled fraud (disputed chargebacks/mistakes) -> label_observed = 1
    Ground truth `is_fraud` is untouched.
    """
    rng = np.random.default_rng(seed)
    labels_observed = np.array(payments_df["is_fraud"].values, dtype=int, copy=True)

    fraud_mask = (labels_observed == 1)
    normal_mask = (labels_observed == 0)

    # 1. False negatives: unlabel ~10% of fraud
    fraud_indices = np.where(fraud_mask)[0]
    unlabel_count = int(round(len(fraud_indices) * fraud_unlabelled_rate))
    if unlabel_count > 0:
        unlabelled_idx = rng.choice(fraud_indices, size=unlabel_count, replace=False)
        labels_observed[unlabelled_idx] = 0

    # 2. False positives: label ~0.2% of normal as fraud
    normal_indices = np.where(normal_mask)[0]
    fp_count = int(round(len(normal_indices) * normal_labelled_fraud_rate))
    if fp_count > 0:
        fp_idx = rng.choice(normal_indices, size=fp_count, replace=False)
        labels_observed[fp_idx] = 1

    payments_df["label_observed"] = labels_observed
    return payments_df


def rings_to_dataframe(specs: List[RingSpec]) -> pd.DataFrame:
    """Converts a list of RingSpec objects to the rings.csv DataFrame schema."""
    records = [s.to_dict() for s in specs]
    return pd.DataFrame(records)
