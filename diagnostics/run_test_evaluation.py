from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.model.backtest_portfolio import _load_cached_model, _predict, build_combined_dataset
from src.model.factor_workflow import compute_masks, load_config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = load_config()
    output_dir = Path("results") / "symbolic_transformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_df, groups, meta = build_combined_dataset(cfg)
    masks = compute_masks(meta, groups, cfg)

    eval_mask = masks.valid.astype(bool)
    if not np.any(eval_mask):
        raise ValueError("Validation mask is empty.")

    top_features_path = output_dir / "top_features.json"
    top_features = json.loads(top_features_path.read_text(encoding="utf-8"))
    if not top_features:
        raise ValueError("top_features.json is empty.")

    combined_df[top_features] = combined_df[top_features].fillna(0.0)
    model, model_type = _load_cached_model(cfg)
    combined_df["prediction"] = _predict(model, model_type, combined_df[top_features].to_numpy())

    eval_df = combined_df.loc[eval_mask].reset_index(drop=True)

    shap_cfg = cfg.get("shap", {}) or {}
    max_samples = int(shap_cfg.get("max_samples", 2000))
    plot_top = int(shap_cfg.get("plot_top", shap_cfg.get("top_k", 10)))
    if len(eval_df) > max_samples:
        sample_df = eval_df.sample(n=max_samples, random_state=42)
    else:
        sample_df = eval_df

    explainer_target = model.get_booster() if model_type == "xgboost" and hasattr(model, "get_booster") else model
    explainer = shap.TreeExplainer(explainer_target)
    shap_values = explainer.shap_values(sample_df[top_features], check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": top_features, "shap_importance": mean_abs}).sort_values(
        "shap_importance", ascending=False
    )
    importance.to_csv(output_dir / "shap_importance_valid.csv", index=False)

    bar_df = importance.head(plot_top).iloc[::-1]
    plt.figure(figsize=(8, max(4, plot_top * 0.3)))
    plt.barh(bar_df["feature"], bar_df["shap_importance"], color="steelblue")
    plt.xlabel("Mean |SHAP| value")
    plt.title("Feature importance via SHAP (Validation)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_valid.png", dpi=200)
    plt.close()

    shap.summary_plot(
        shap_values[:, : len(top_features)],
        sample_df[top_features],
        max_display=plot_top,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_valid.png", dpi=300)
    plt.close()

    from src.model.factor_workflow import run_markowitz_backtest

    run_markowitz_backtest(
        cfg,
        combined_df,
        masks,
        top_features,
        model,
        output_dir,
        test_mask_override=eval_mask.to_numpy(),
        output_suffix="_valid",
    )

    mapping = {
        "backtest_timeseries_valid.parquet": "backtest_timeseries.parquet",
        "backtest_cumulative_valid.png": "backtest_cumulative.png",
        "backtest_summary_valid.json": "backtest_summary.json",
        "shap_importance_valid.csv": "shap_importance.csv",
        "shap_importance_valid.png": "shap_importance.png",
        "shap_summary_valid.png": "shap_summary.png",
    }
    for src_name, dst_name in mapping.items():
        src = output_dir / src_name
        if src.exists():
            shutil.copyfile(src, output_dir / dst_name)

    print("Validation-only backtest and SHAP artefacts saved to results/symbolic_transformer/")


if __name__ == "__main__":
    main()
