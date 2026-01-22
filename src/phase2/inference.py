# src/phase2/inference.py
from __future__ import annotations

from typing import Dict, Any
import joblib
import pandas as pd

from src.config.settings import PIPELINE_PATH


def load_pipeline(path=PIPELINE_PATH):
    return joblib.load(path)


def predict_one(payload: Dict[str, Any], pipeline_path=PIPELINE_PATH) -> float:
    """
    payload example:
    {
      "open": 1.2, "high": 1.3, "low": 1.1, "close": 1.25, "volume": 10.5,
      "company": "CBA"
    }
    NOTE: must be on the same scale as training data (log1p if you enabled it in Phase1 preprocess).
    """
    pipe = load_pipeline(pipeline_path)
    X = pd.DataFrame([payload])
    pred = pipe.predict(X)[0]
    return float(pred)
