# src/phase2/train.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor

from src.config.settings import CLEANED_PATH, PIPELINE_PATH
from src.utils.io import ensure_dir


@dataclass
class TrainConfig:
    data_path: Path = CLEANED_PATH
    pipeline_path: Path = PIPELINE_PATH
    target: str = "adj_close"
    test_size: float = 0.2
    random_state: int = 42


def train(cfg: TrainConfig = TrainConfig()) -> Pipeline:
    df = pd.read_csv(cfg.data_path)

    y = df[cfg.target]
    X = df.drop(columns=[cfg.target, "month", "year"], errors="ignore")  # matches Phase2 :contentReference[oaicite:11]{index=11}

    cat_cols = ["company"] if "company" in X.columns else []
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    pipe.fit(X_train, y_train)

    ensure_dir(cfg.pipeline_path.parent)
    joblib.dump(pipe, cfg.pipeline_path)

    return pipe


if __name__ == "__main__":
    train()
    print(f"Saved pipeline -> {PIPELINE_PATH}")
