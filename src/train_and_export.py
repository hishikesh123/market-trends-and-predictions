import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = Path("data/processed/cleaned.csv")  # <-- change this
TARGET_COL = "adj_close"  # <-- change this

# 1) Load data
df = pd.read_csv(DATA_PATH)

# 2) Define features
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# 3) Identify numeric/categorical columns
num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# 4) Preprocessor only (sklearn-safe)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# 5) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) Fit preprocessor
X_train_tx = preprocessor.fit_transform(X_train)
X_test_tx = preprocessor.transform(X_test)

# 7) Train XGBoost (native model persistence)
model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_tx, y_train)

# 8) Evaluate
pred = model.predict(X_test_tx)
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

# 9) Save artifacts
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
model.save_model(str(MODELS_DIR / "xgb_model.json"))

metadata = {
    "mse": float(mse),
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "n_rows": int(df.shape[0]),
    "n_features_input": int(X.shape[1]),
    "numeric_features": num_cols,
    "categorical_features": cat_cols,
}

with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved:")
print("-", MODELS_DIR / "preprocessor.joblib")
print("-", MODELS_DIR / "xgb_model.json")
print("-", MODELS_DIR / "metadata.json")

