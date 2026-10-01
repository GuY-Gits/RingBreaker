"""Loaded models used on the fast path: XGBoost pair model + calibrated EIF."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb

from ringbreaker import config
from ringbreaker.features.online import BEHAVIOUR_FEATURE_NAMES, PAIR_FEATURE_NAMES


class ModelBundle:
    def __init__(
        self,
        pair_path: Path = config.PAIR_MODEL_PATH,
        eif_path: Path = config.EIF_MODEL_PATH,
        version: str = "1.0.0",
    ) -> None:
        if not Path(pair_path).exists():
            raise FileNotFoundError(
                f"Pair model not found at {pair_path}. Run `python -m ringbreaker.pipeline` first."
            )
        self.booster = xgb.Booster()
        self.booster.load_model(str(pair_path))
        self.eif = None
        if Path(eif_path).exists():
            with open(eif_path, "rb") as fh:
                self.eif = pickle.load(fh)
        self.version = version
        self.pair_path = str(pair_path)
        # Decision thresholds fitted on the validation slice (see engine/risk.py).
        self.calibration: Optional[Dict] = None
        if config.MODEL_METADATA_PATH.exists():
            self.calibration = json.loads(config.MODEL_METADATA_PATH.read_text()).get("risk_calibration")

    def use_booster(self, booster: xgb.Booster, version: str, path: str,
                    calibration: Optional[Dict] = None) -> None:
        self.booster = booster
        self.version = version
        self.pair_path = path
        if calibration is not None:
            self.calibration = calibration

    def _matrix(self, rows: List[Dict[str, float]]) -> xgb.DMatrix:
        arr = np.array([[r[n] for n in PAIR_FEATURE_NAMES] for r in rows], dtype=np.float32)
        return xgb.DMatrix(arr, feature_names=PAIR_FEATURE_NAMES)

    def pair_predict(self, feats: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Fraud probability and per-feature SHAP contributions (log-odds)."""
        dm = self._matrix([feats])
        prob = float(self.booster.predict(dm)[0])
        contribs = self.booster.predict(dm, pred_contribs=True)[0]
        return prob, {n: float(c) for n, c in zip(PAIR_FEATURE_NAMES, contribs[:-1])}

    def pair_predict_many(self, rows: List[Dict[str, float]]) -> np.ndarray:
        if not rows:
            return np.zeros(0)
        return self.booster.predict(self._matrix(rows))

    def anomaly(self, behaviour: Dict[str, float]) -> Optional[float]:
        if self.eif is None:
            return None
        x = np.array([[behaviour[n] for n in BEHAVIOUR_FEATURE_NAMES]], dtype=np.float64)
        return float(self.eif.score(x)[0])
