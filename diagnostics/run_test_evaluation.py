from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from diagnostics.run_markowitz_backtest import (
    BacktestParams,
    load_benchmark_returns,
    load_panels,
    run_markowitz_backtest,
    summarise_backtest,
)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _run_backtest(split: str, params: BacktestParams, results_dir: Path, benchmark) -> dict:
    preds = _load_preds(results_dir / f"{split}_preds.parquet")
    ret_panel, vol_panel = load_panels(Path("data/features.parquet"))
    bt = run_markowitz_backtest(preds, ret_panel, vol_panel, params, benchmark_returns=benchmark)
    summary = summarise_backtest(bt)
    bt.to_parquet(results_dir / f"markowitz_{split}_ts.parquet", index=False)
    (results_dir / f"markowitz_{split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _copy_shap_artifacts(results_dir: Path) -> None:
    shap_dir = results_dir / "shap"
    if not shap_dir.exists():
        raise FileNotFoundError("Missing results/shap directory. Run train_tree_pipeline.py first.")
    shutil.copyfile(shap_dir / "shap_test.png", results_dir / "shap_summary_test.png")
    shutil.copyfile(shap_dir / "shap_importance_test.csv", results_dir / "shap_importance_test.csv")
    print("Copied SHAP artefacts from results/shap/ to root results/ directory.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    params = BacktestParams(
        top_quantile=cfg.get("top_quantile", 0.08),
        bottom_quantile=cfg.get("bottom_quantile", 0.08),
        lookback=63,
        ridge=1e-4,
        max_abs_weight=0.08,
        gross_leverage=1.0,
        cost_bps=float(cfg.get("cost_bps_oneway", 1.0)),
        min_bucket=cfg.get("min_bucket", 20),
    )

    ret_panel, _ = load_panels(Path("data/features.parquet"))
    benchmark = load_benchmark_returns(ret_panel.index.min(), ret_panel.index.max())

    val_summary = _run_backtest("valid", params, args.results_dir, benchmark)
    test_summary = _run_backtest("test", params, args.results_dir, benchmark)
    _copy_shap_artifacts(args.results_dir)

    print("Validation summary:", val_summary)
    print("Test summary:", test_summary)


if __name__ == "__main__":
    main()
