# src/phase1/eda.py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.config.settings import CLEANED_PATH, FIGURES_DIR
from src.utils.io import ensure_dir


def export_eda_figures(cleaned_path: Path = CLEANED_PATH, out_dir: Path = FIGURES_DIR) -> None:
    ensure_dir(out_dir)
    df = pd.read_csv(cleaned_path)

    # 1) Distribution of adj_close
    plt.figure()
    df["adj_close"].plot(kind="hist", bins=50)
    plt.title("Distribution: adj_close (log1p scaled if enabled)")
    plt.xlabel("adj_close")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_adj_close_hist.png", dpi=200)
    plt.close()

    # 2) Average adj_close by year
    plt.figure()
    df.groupby("year")["adj_close"].mean().plot(kind="line")
    plt.title("Mean adj_close by year")
    plt.xlabel("year")
    plt.ylabel("mean adj_close")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_adj_close_by_year.png", dpi=200)
    plt.close()

    # 3) Top-10 companies by mean volume
    top = (
        df.groupby("company")["volume"].mean()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure()
    top.plot(kind="bar")
    plt.title("Top 10 companies by mean volume")
    plt.xlabel("company")
    plt.ylabel("mean volume")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_top10_volume.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    export_eda_figures()
    print(f"Saved EDA figures -> {FIGURES_DIR}")
