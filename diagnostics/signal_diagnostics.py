from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.model.backtest_portfolio import _load_cached_model, _predict, build_combined_dataset
from src.model.factor_workflow import compute_masks, load_config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics(df: pd.DataFrame, name: str) -> Dict[str, float]:
    spearman = df["prediction"].corr(df["y"], method="spearman")
    pearson = df["prediction"].corr(df["y"], method="pearson")
    mse = float(np.mean(np.square(df["prediction"] - df["y"])))
    return {
        "split": name,
        "spearman": float(spearman),
        "pearson": float(pearson),
        "mse": mse,
        "n_samples": int(len(df)),
    }


def bucket_summary(df: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["decile"] = pd.qcut(df["prediction"], buckets, labels=False, duplicates="drop")
    summary = (
        df.groupby("decile")
        .agg(
            count=("ticker", "count"),
            mean_prediction=("prediction", "mean"),
            mean_y=("y", "mean"),
        )
        .reset_index()
    )
    summary["cum_return"] = np.cumprod(1 + summary["mean_y"]) - 1
    return summary


def monthly_ic(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    rows: list[dict[str, float | str]] = []
    for month, grp in df.groupby("month"):
        rows.append(
            {
                "split": name,
                "month": month,
                "spearman": float(grp["prediction"].corr(grp["y"], method="spearman")),
                "pearson": float(grp["prediction"].corr(grp["y"], method="pearson")),
                "n": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    diagnostic_dir = Path("results") / "diagnostics"
    ensure_dir(diagnostic_dir)

    combined_df, groups, meta = build_combined_dataset(cfg)
    masks = compute_masks(meta, groups, cfg)

    model, model_type = _load_cached_model(cfg)
    top_features_path = Path("results") / "symbolic_transformer" / "top_features.json"
    if not top_features_path.exists():
        raise FileNotFoundError(f"Missing top_features.json at {top_features_path}")
    top_features = json.loads(top_features_path.read_text(encoding="utf-8"))
    if not top_features:
        raise ValueError("top_features.json is empty.")

    combined_df[top_features] = combined_df[top_features].fillna(0.0)
    feature_matrix = combined_df[top_features].to_numpy()
    combined_df["prediction"] = _predict(model, model_type, feature_matrix)

    splits = {
        "train": combined_df.loc[masks.train].copy(),
        "valid": combined_df.loc[masks.valid].copy(),
        "test": combined_df.loc[masks.test].copy(),
    }

    metrics_rows = []
    monthly_rows = []
    for name, df in splits.items():
        if df.empty:
            continue
        metrics_rows.append(compute_metrics(df, name))
        bucket_summary(df).to_csv(diagnostic_dir / f"bucket_{name}.csv", index=False)
        monthly_rows.append(monthly_ic(df, name))

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(diagnostic_dir / "metrics.csv", index=False)
    if monthly_rows:
        pd.concat(monthly_rows, ignore_index=True).to_csv(diagnostic_dir / "monthly_ic.csv", index=False)

    print("Signal diagnostics saved under results/diagnostics/")


if __name__ == "__main__":
    main()
