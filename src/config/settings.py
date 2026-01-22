# src/config/settings.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR = PROJECT_ROOT / "models"

# Canonical artifacts
CLEANED_PATH = PROCESSED_DIR / "cleaned.csv"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "metadata.json"
