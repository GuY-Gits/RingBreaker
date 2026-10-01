"""Paths and tunables shared by the pipeline, engine and API.

Every path can be overridden with an environment variable so tests and demos
can point RingBreaker at a scratch directory without touching committed data.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DATA_DIR = Path(os.environ.get("RINGBREAKER_DATA_DIR", PACKAGE_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("RINGBREAKER_MODELS_DIR", PACKAGE_ROOT / "models"))
DASHBOARD_DIST = Path(os.environ.get("RINGBREAKER_DASHBOARD_DIST", REPO_ROOT / "dashboard" / "dist"))

USERS_CSV = DATA_DIR / "users.csv"
PAYMENTS_CSV = DATA_DIR / "payments.csv"
RINGS_CSV = DATA_DIR / "rings.csv"
BENIGN_GROUPS_CSV = DATA_DIR / "benign_groups.csv"
PAIR_FEATURES_CSV = DATA_DIR / "pair_features.csv"

PAIR_MODEL_PATH = MODELS_DIR / "pair_model.json"
RETRAINED_MODEL_PATH = MODELS_DIR / "pair_model_retrained.json"
EIF_MODEL_PATH = MODELS_DIR / "eif_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "metadata.json"

# PRD "Split": first 70% train, next 15% validation, last 15% replayed live.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
STREAM_START_FRACTION = TRAIN_FRACTION + VALIDATION_FRACTION

# Slow path (F9): recompute lockstep clusters and named patterns every N payments.
SLOW_PATH_EVERY = int(os.environ.get("RINGBREAKER_SLOW_PATH_EVERY", "50"))
# Under load, space slow-path runs so they use at most this share of wall time
# (0 disables the throttle; at demo rates the every-N rule is what triggers).
SLOW_PATH_BUDGET = float(os.environ.get("RINGBREAKER_SLOW_PATH_BUDGET", "0.1"))

# Allowed CORS origins for the dashboard dev server; the built dashboard is
# served from the API origin itself and needs none.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "RINGBREAKER_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
