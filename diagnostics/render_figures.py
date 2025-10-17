from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_equity_curve(ts_path: Path, output_path: Path) -> None:
    df = pd.read_parquet(ts_path)
    if df.empty:
        raise ValueError(f"No data found in {ts_path}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["date"],
        df["equity"],
        label="Strategy (XGBoost)",
        color="#1f77b4",
        linewidth=2.0,
    )
    if "benchmark_equity" in df.columns and df["benchmark_equity"].notna().any():
        plt.plot(
            df["date"],
            df["benchmark_equity"],
            label="S&P 500 Index",
            color="#ff7f0e",
            linewidth=1.8,
            linestyle="--",
        )
    plt.title("Strategy vs S&P 500 (Test Set)", fontsize=13, pad=12)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--docs_dir", type=Path, default=Path("docs/images"))
    args = parser.parse_args()

    shap_src = args.results_dir / "shap" / "shap_validation.png"
    if not shap_src.exists():
        raise FileNotFoundError(
            f"Expected SHAP visualization at {shap_src}. Run train_tree_pipeline.py first."
        )
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    shap_dst = args.docs_dir / "shap_summary.png"
    shutil.copyfile(shap_src, shap_dst)

    equity_src = args.results_dir / "markowitz_test_ts.parquet"
    equity_dst = args.docs_dir / "backtest_cumulative.png"
    generate_equity_curve(equity_src, equity_dst)

    print(f"Copied SHAP plot to {shap_dst} and regenerated equity curve at {equity_dst}")


if __name__ == "__main__":
    main()
