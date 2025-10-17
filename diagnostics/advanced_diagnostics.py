from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge

from src.model.backtest_portfolio import (
    _load_cached_model,
    _predict,
    build_combined_dataset,
)
from src.model.factor_workflow import compute_masks, load_config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_dataset(cfg: Dict) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    combined_df, groups, meta = build_combined_dataset(cfg)
    masks = compute_masks(meta, groups, cfg)

    top_features_path = Path("results") / "symbolic_transformer" / "top_features.json"
    top_features = json.loads(top_features_path.read_text(encoding="utf-8"))
    if not top_features:
        raise ValueError("top_features.json is empty.")

    model, model_type = _load_cached_model(cfg)
    combined_df[top_features] = combined_df[top_features].fillna(0.0)
    combined_df["prediction"] = _predict(model, model_type, combined_df[top_features].to_numpy())
    combined_df["date"] = pd.to_datetime(combined_df["date"])

    splits = {
        "train": combined_df.loc[masks.train].copy(),
        "valid": combined_df.loc[masks.valid].copy(),
        "test": combined_df.loc[masks.test].copy(),
    }
    return combined_df, splits, top_features


def compute_feature_drift(splits: Dict[str, pd.DataFrame], features: Iterable[str], out_dir: Path) -> None:
    train_df = splits.get("train")
    test_df = splits.get("test")
    if train_df is None or test_df is None or train_df.empty or test_df.empty:
        return

    records: list[dict[str, float]] = []
    for feature in features:
        train_vals = train_df[feature].dropna()
        test_vals = test_df[feature].dropna()
        if train_vals.empty or test_vals.empty:
            continue
        ks_stat, ks_pvalue = ks_2samp(train_vals, test_vals)
        records.append(
            {
                "feature": feature,
                "train_mean": float(train_vals.mean()),
                "train_std": float(train_vals.std(ddof=1)),
                "test_mean": float(test_vals.mean()),
                "test_std": float(test_vals.std(ddof=1)),
                "train_median": float(train_vals.median()),
                "test_median": float(test_vals.median()),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
            }
        )
    if records:
        pd.DataFrame(records).to_csv(out_dir / "feature_drift.csv", index=False)


def evaluate_model(pred: np.ndarray, target: pd.Series) -> Dict[str, float]:
    series_pred = pd.Series(pred, index=target.index)
    spearman = float(series_pred.corr(target, method="spearman"))
    pearson = float(series_pred.corr(target, method="pearson"))
    mse = float(np.mean(np.square(series_pred - target)))
    return {"spearman": spearman, "pearson": pearson, "mse": mse}


def compare_models(splits: Dict[str, pd.DataFrame], features: List[str], cfg: Dict, out_dir: Path) -> None:
    rows = []

    for split_name, df in splits.items():
        if df.empty:
            continue
        rows.append(
            {
                "model": "xgboost_cached",
                "split": split_name,
                **evaluate_model(df["prediction"].to_numpy(), df["y"]),
            }
        )

    ridge = Ridge(alpha=1.0)
    ridge.fit(splits["train"][features].fillna(0.0), splits["train"]["y"])
    for split_name, df in splits.items():
        if df.empty:
            continue
        preds = ridge.predict(df[features].fillna(0.0))
        rows.append({"model": "ridge", "split": split_name, **evaluate_model(preds, df["y"])})

    lgb_params = cfg.get("lightgbm", {}) or {}
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(splits["train"][features].fillna(0.0), splits["train"]["y"])
    for split_name, df in splits.items():
        if df.empty:
            continue
        preds = lgb_model.predict(df[features].fillna(0.0))
        rows.append({"model": "lightgbm", "split": split_name, **evaluate_model(preds, df["y"])})

    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)


def rolling_ridge_ic(combined_df: pd.DataFrame, features: List[str], out_dir: Path, window_months: int = 6) -> None:
    df = combined_df.copy()
    df["month"] = df["date"].dt.to_period("M")
    months = sorted(df["month"].unique())
    rows = []

    for idx in range(window_months, len(months)):
        train_months = months[idx - window_months : idx]
        eval_month = months[idx]
        train_df = df[df["month"].isin(train_months)]
        eval_df = df[df["month"] == eval_month]
        if train_df.empty or eval_df.empty:
            continue

        ridge = Ridge(alpha=1.0)
        ridge.fit(train_df[features].fillna(0.0), train_df["y"])
        preds = ridge.predict(eval_df[features].fillna(0.0))
        metrics = evaluate_model(preds, eval_df["y"])
        rows.append(
            {
                "eval_month": str(eval_month),
                "train_start": str(train_months[0]),
                "train_end": str(train_months[-1]),
                "n_eval": int(len(eval_df)),
                **metrics,
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "rolling_ridge_ic.csv", index=False)


def monthly_quantile_returns(df: pd.DataFrame, features: List[str], quantile: float, out_dir: Path) -> None:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    rows = []

    for month, grp in df.groupby("month"):
        if grp.empty:
            continue
        top_cut = grp["prediction"].quantile(1 - quantile)
        bottom_cut = grp["prediction"].quantile(quantile)
        top_grp = grp[grp["prediction"] >= top_cut]
        bottom_grp = grp[grp["prediction"] <= bottom_cut]
        rows.append(
            {
                "month": str(month),
                "top_mean_return": float(top_grp["y"].mean()) if not top_grp.empty else np.nan,
                "bottom_mean_return": float(bottom_grp["y"].mean()) if not bottom_grp.empty else np.nan,
                "spread": float(top_grp["y"].mean() - bottom_grp["y"].mean())
                if not top_grp.empty and not bottom_grp.empty
                else np.nan,
                "top_count": int(len(top_grp)),
                "bottom_count": int(len(bottom_grp)),
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "monthly_quantile_returns.csv", index=False)


def main() -> None:
    cfg = load_config()
    diagnostic_dir = Path("results") / "diagnostics"
    ensure_dir(diagnostic_dir)

    combined_df, splits, top_features = prepare_dataset(cfg)

    compute_feature_drift(splits, top_features, diagnostic_dir)
    compare_models(splits, top_features, cfg, diagnostic_dir)
    rolling_ridge_ic(combined_df, top_features, diagnostic_dir, window_months=6)
    monthly_quantile_returns(splits["test"], top_features, quantile=0.1, out_dir=diagnostic_dir)

    print("Advanced diagnostics saved under results/diagnostics/")


if __name__ == "__main__":
    main()
