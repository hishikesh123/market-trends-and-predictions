# src/phase1/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.config.settings import RAW_DIR, CLEANED_PATH
from src.utils.io import write_csv, ensure_dir


@dataclass
class PreprocessConfig:
    raw_dir: Path = RAW_DIR
    output_path: Path = CLEANED_PATH
    date_col: str = "Date"
    company_col: str = "Company"
    # Phase2 expects these cleaned names:
    # open, high, low, close, adj_close, volume, company, month, year
    log_transform_numeric: bool = True


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Map from raw dataset columns to cleaned schema
    # Phase1 shows: Date, Open, High, Low, Close, Adj Close, Volume, Company :contentReference[oaicite:4]{index=4}
    mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Company": "company",
        "Month": "month",
        "Year": "year",
    }
    # keep 'Date' if present; we'll drop after extracting month/year
    df = df.rename(columns=mapping)
    return df


def _extract_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["month"] = df[date_col].dt.month_name()
        df["year"] = df[date_col].dt.year
        df = df.drop(columns=[date_col])
    return df


def _impute_missing_by_company_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "company" not in df.columns:
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Fill numeric NaNs by company mean (your Phase 2 report mentions company imputation). :contentReference[oaicite:5]{index=5}
    df[numeric_cols] = df.groupby("company")[numeric_cols].transform(
        lambda s: s.fillna(s.mean())
    )
    return df


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in numeric_cols:
        if c in df.columns:
            # Safe log transform; handles zeros.
            df[c] = np.log1p(df[c].astype(float))
    return df


def load_raw_folder(raw_dir: Path) -> pd.DataFrame:
    """
    Loads all CSVs in raw_dir, appends company ticker from filename,
    and concatenates into a single DataFrame (Phase 1 approach). :contentReference[oaicite:6]{index=6}
    """
    csv_files = sorted([p for p in raw_dir.glob("*.csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in: {raw_dir}")

    dfs = []
    for p in csv_files:
        df = pd.read_csv(p)
        # Add company from filename (as your Phase 1 notebook does). :contentReference[oaicite:7]{index=7}
        df["Company"] = p.stem
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def preprocess(cfg: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    df = load_raw_folder(cfg.raw_dir)

    df = _standardize_columns(df)
    df = _extract_time_features(df, cfg.date_col)  # expects original 'Date' if present
    df = _standardize_columns(df)  # standardize again after Month/Year creation

    # Basic cleaning
    df = df.drop_duplicates()

    # Ensure types
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Missing handling (company-wise mean for numeric)
    df = _impute_missing_by_company_mean(df)

    # Optional: log transform numeric to reduce skew (Phase 2 mentions this). :contentReference[oaicite:8]{index=8}
    if cfg.log_transform_numeric:
        df = _log_transform(df)

    # Final schema guard
    expected = {"open", "high", "low", "close", "adj_close", "volume", "company", "month", "year"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Cleaned dataset missing columns: {sorted(missing)}")

    # Drop rows that still have critical nulls
    df = df.dropna(subset=["open", "high", "low", "close", "adj_close", "volume", "company", "year"])

    ensure_dir(cfg.output_path.parent)
    write_csv(df, cfg.output_path, index=False)
    return df


if __name__ == "__main__":
    preprocess()
    print(f"Saved cleaned dataset -> {CLEANED_PATH}")
