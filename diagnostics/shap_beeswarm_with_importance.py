from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.model.backtest_portfolio import _load_cached_model
from src.model.factor_workflow import build_dataset, compute_masks, load_config


def main() -> None:
    cfg = load_config()
    output_dir = Path("results") / "symbolic_transformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cached data
    X, y, groups, meta = build_dataset(cfg)
    from src.model.run_gp import FEATURE_COLS as RUN_GP_FEATURES
    base_df = pd.DataFrame(X, columns=RUN_GP_FEATURES)
    st_path = output_dir / "symbolic_all.parquet"
    if not st_path.exists():
        raise FileNotFoundError("symbolic_all.parquet missing; run factor workflow first.")
    st_df = pd.read_parquet(st_path)
    if len(st_df) != len(base_df):
        raise ValueError("symbolic_all.parquet length mismatch with base features.")

    combined_df = pd.concat([meta.reset_index(drop=True), base_df, st_df.reset_index(drop=True)], axis=1)
    combined_df["y"] = y
    masks = compute_masks(meta, groups, cfg)
    eval_mask = masks.valid.astype(bool)
    if not np.any(eval_mask):
        raise ValueError("Validation mask empty; cannot produce SHAP visual.")

    top_features_path = output_dir / "top_features.json"
    top_features = json.loads(top_features_path.read_text(encoding="utf-8"))
    if not top_features:
        raise ValueError("top_features.json empty.")

    combined_df[top_features] = combined_df[top_features].fillna(0.0)
    eval_df = combined_df.loc[eval_mask].reset_index(drop=True)

    model, model_type = _load_cached_model(cfg)
    shap_cfg = cfg.get("shap", {}) or {}
    max_samples = int(shap_cfg.get("max_samples", 2000))
    plot_top = int(shap_cfg.get("plot_top", shap_cfg.get("top_k", 10)))

    if len(eval_df) > max_samples:
        sample_df = eval_df.sample(n=max_samples, random_state=42)
    else:
        sample_df = eval_df

    explainer_target = model
    if model_type == "xgboost" and hasattr(model, "get_booster"):
        explainer_target = model.get_booster()
    explainer = shap.TreeExplainer(explainer_target)
    shap_values = explainer.shap_values(sample_df[top_features], check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:plot_top]
    ranked_features = [top_features[i] for i in order]
    n_features = len(ranked_features)

    importance_df = pd.DataFrame(
        {"feature": ranked_features, "mean_abs_shap": mean_abs_shap[order]}
    )
    importance_df.to_csv(output_dir / "shap_importance.csv", index=False)

    shap.summary_plot(
        shap_values[:, order],
        sample_df[ranked_features],
        plot_type="dot",
        max_display=n_features,
        color_bar=True,
        show=False,
        sort=False,
    )

    fig = plt.gcf()
    ax_swarm = fig.axes[0]
    cbar_ax = fig.axes[1] if len(fig.axes) > 1 else None

    height = max(5.5, n_features * 0.48)
    fig.set_size_inches(13, height)
    fig.subplots_adjust(left=0.36, right=0.88, top=0.92, bottom=0.12)
    ax_swarm = fig.axes[0]
    if cbar_ax is not None:
        box = ax_swarm.get_position()
        cbar_ax.set_position([box.x1 + 0.02, box.y0, 0.02, box.height])

    ax2 = ax_swarm.twiny()

    y_positions = ax_swarm.get_yticks()
    y_labels = [tick.get_text() for tick in ax_swarm.get_yticklabels()]
    feature_to_mean = {feat: mean_abs_shap[top_features.index(feat)] for feat in ranked_features}
    max_val = max(feature_to_mean.values()) if feature_to_mean else 0.0

    for y, label in zip(y_positions, y_labels):
        val = feature_to_mean.get(label, 0.0)
        ax2.barh(
            y,
            val,
            height=0.6,
            color="lightsteelblue",
            alpha=0.35,
            zorder=1,
        )

    ax2.set_xlim(0, max_val * 1.1 if max_val > 0 else 1.0)
    ax2.set_xlabel("Mean |SHAP| Value (Feature Importance)", fontsize=12)
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()
    ax2.grid(axis="x", color="lightgrey", linestyle="--", alpha=0.4)

    ax_swarm.set_xlabel("Shapley Value Contribution (Bee Swarm)", fontsize=12)
    ax_swarm.set_ylabel("Features", fontsize=12)

    fig.suptitle("SHAP Beeswarm with Mean |SHAP| Overlay (Validation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    combo_path = output_dir / "shap_summary.png"
    fig.savefig(combo_path, dpi=300)
    plt.close(fig)

    print(f"Combined SHAP figure saved to {combo_path}")


if __name__ == "__main__":
    main()
