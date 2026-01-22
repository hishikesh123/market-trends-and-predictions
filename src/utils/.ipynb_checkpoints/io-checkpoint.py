# src/utils/io.py
import json
from pathlib import Path
import pandas as pd

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=index)

def write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))
