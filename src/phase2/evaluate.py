# src/phase2/evaluate.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config.settings import CLEANED_PATH, PIPELINE_PATH, MODEL_METADATA_PATH
from src.utils.io import write_json


def evaluate(data_path: Path = CLEANED_PATH, pipeline_path: Path = PIPELINE_PATH) -> dict:
    df = pd.read_csv(data_path)

    y = df["adj_close"]
    X = df.drop(columns=["adj_close", "month", "year"], errors="ignore")

    pipe = joblib.load(pipeline_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preds = pipe.predict(X_test)

    metrics = {
        "mse": float(mean_squared_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_rows": int(df.shape[0]),
        "n_features_input": int(X.shape[1]),
    }

    write_json(metrics, MODEL_METADATA_PATH)
    return metrics


if __name__ == "__main__":
    m = evaluate()
    print("Saved metrics ->", MODEL_METADATA_PATH)
    print(m)
