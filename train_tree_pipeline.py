import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml

from diagnostics.run_markowitz_backtest import (
    BacktestParams,
    load_benchmark_returns,
    load_panels,
    run_markowitz_backtest,
    summarise_backtest,
)
from run_pipeline import ensure_features, ensure_prices, ensure_tickers
from src.model.fitness import daily_rank_ic
from src.utils.time_split import train_valid_test_splits
from gplearn.genetic import SymbolicTransformer
from sklearn.base import BaseEstimator

def _ensure_splits(dates: np.ndarray, embargo: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr, va, te = train_valid_test_splits(dates, embargo_days=embargo)

    def _has(mask: np.ndarray) -> bool:
        return bool(mask.sum())

    if not (_has(tr) and _has(va) and _has(te)):
        tr, va, te = train_valid_test_splits(dates, embargo_days=0)
    if not (_has(tr) and _has(va) and _has(te)):
        uniq = np.unique(dates)
        n = len(uniq)
        if n < 3:
            raise RuntimeError("Too few unique dates for splitting.")
        k1, k2 = int(n * 0.6), int(n * 0.8)
        tr = np.isin(dates, uniq[:k1])
        va = np.isin(dates, uniq[k1:k2])
        te = np.isin(dates, uniq[k2:])
    if not (_has(tr) and _has(va) and _has(te)):
        raise RuntimeError("Failed to compute non-empty train/valid/test splits.")
    return tr, va, te


def _load_dataset(features_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_parquet(features_path)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in df.columns if c not in {"date", "ticker", "y_fwd"}]
    df = df.dropna(subset=feature_cols + ["y_fwd"]).reset_index(drop=True)
    feature_df = df[feature_cols].astype(np.float32)
    y = df["y_fwd"].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(df["date"])
    groups = pd.factorize(dates)[0].astype(np.float64)
    meta = df[["date", "ticker"]].copy()
    return feature_df, y, groups, meta


def _patch_symbolic_tags() -> None:
    def _tags(self):
        try:
            return BaseEstimator.__sklearn_tags__(self)
        except Exception:
            from sklearn.utils._tags import Tags
            return Tags()

    if getattr(SymbolicTransformer, "__patched_tags__", False):
        return
    SymbolicTransformer.__sklearn_tags__ = _tags
    for cls in SymbolicTransformer.mro():
        if cls.__name__ == "BaseSymbolic":
            cls.__sklearn_tags__ = _tags
            break
    SymbolicTransformer.__patched_tags__ = True


def _train_symbolic_transformer(
    base_df: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    _patch_symbolic_tags()
    st = SymbolicTransformer(
        n_components=n_components,
        generations=15,
        population_size=2000,
        hall_of_fame=400,
        stopping_criteria=0.0,
        const_range=(-0.5, 0.5),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sin", "cos", "log", "sqrt"),
        metric="spearman",
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        random_state=random_state,
    )
    X_train = base_df.to_numpy(dtype=np.float64, copy=False)
    st.fit(X_train[train_mask], y[train_mask])
    transformed = st.transform(X_train)
    sym_cols = [f"sym_{i}" for i in range(transformed.shape[1])]
    sym_df = pd.DataFrame(transformed, columns=sym_cols).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    expr_map = {}
    try:
        programs = getattr(st, "_best_programs", None)
        if programs is None:
            programs = getattr(st, "programs_", [])[ -len(sym_cols) :] if hasattr(st, "programs_") else []
        for col, prog in zip(sym_cols, programs or []):
            expr_map[col] = str(prog)
    except Exception:
        expr_map = {col: "" for col in sym_cols}
    return sym_df, expr_map


def _make_feature_frame(base_df: pd.DataFrame, sym_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([base_df.reset_index(drop=True), sym_df.reset_index(drop=True)], axis=1)
    combined = combined.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return combined


def _cross_sectional_zscore(features: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    means = features.groupby(dates).transform("mean")
    stds = features.groupby(dates).transform("std").replace(0.0, 1.0)
    zscored = (features - means) / stds
    return zscored.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _cross_sectional_rank(y: np.ndarray, dates: pd.Series) -> np.ndarray:
    df = pd.DataFrame({"date": dates, "y": y})
    df["rank"] = df.groupby("date")["y"].rank(pct=True, method="average") - 0.5
    return df["rank"].to_numpy(dtype=np.float32)


def _train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method=params.get("tree_method", "hist"),
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.5),
        reg_lambda=params.get("reg_lambda", 1.0),
        random_state=params.get("random_state", 42),
        eval_metric=params.get("eval_metric", "rmse"),
    )
    model.fit(X, y)
    return model


def _shap_rank_features(
    model: xgb.XGBRegressor,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    top_k: int = 10,
) -> Tuple[List[str], pd.DataFrame]:
    if len(val_df) > 2000:
        sample_df = val_df.sample(n=2000, random_state=42)
    else:
        sample_df = val_df
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_df[feature_cols].to_numpy(), check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranked = [feature_cols[i] for i in order]
    importance_df = pd.DataFrame(
        {"feature": ranked, "mean_abs_shap": mean_abs[order]}
    )
    top_features = ranked[:top_k]
    return top_features, importance_df


def _save_shap_plot(
    shap_values: np.ndarray,
    sample_df: pd.DataFrame,
    ranked_features: List[str],
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    shap.summary_plot(
        shap_values,
        sample_df[ranked_features],
        plot_type="dot",
        max_display=len(ranked_features),
        show=False,
        sort=False,
    )
    fig = plt.gcf()
    ax_swarm = fig.axes[0]
    cbar_ax = fig.axes[1] if len(fig.axes) > 1 else None

    fig.set_size_inches(13, max(5.5, len(ranked_features) * 0.55))
    fig.subplots_adjust(left=0.32, right=0.88, top=0.92, bottom=0.12)
    if cbar_ax is not None:
        box = ax_swarm.get_position()
        cbar_ax.set_position([box.x1 + 0.02, box.y0, 0.02, box.height])

    ax2 = ax_swarm.twiny()
    mean_map = dict(zip(importance_df["feature"], importance_df["mean_abs_shap"]))
    for y_val, tick in zip(ax_swarm.get_yticks(), ax_swarm.get_yticklabels()):
        label = tick.get_text()
        ax2.barh(
            y_val,
            mean_map.get(label, 0.0),
            height=0.6,
            color="lightsteelblue",
            alpha=0.35,
            zorder=1,
        )

    if mean_map:
        ax2.set_xlim(0, max(mean_map.values()) * 1.1)
    else:
        ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Mean |SHAP| Value (Feature Importance)", fontsize=12)
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()
    ax2.grid(axis="x", color="lightgrey", linestyle="--", alpha=0.4)

    ax_swarm.set_xlabel("Shapley Value Contribution (Bee Swarm)", fontsize=12)
    ax_swarm.set_ylabel("Features", fontsize=12)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_predictions(
    path: Path,
    meta: pd.DataFrame,
    mask: np.ndarray,
    preds: np.ndarray,
    y: np.ndarray,
) -> None:
    df = meta.loc[mask].reset_index(drop=True).copy()
    df["pred"] = preds
    df["y"] = y[mask]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _prepare_markowitz(
    preds_df: pd.DataFrame, results_dir: Path, suffix: str, params: BacktestParams, ret_panel, vol_panel, benchmark
) -> None:
    bt = run_markowitz_backtest(preds_df, ret_panel, vol_panel, params, benchmark_returns=benchmark)
    summary = summarise_backtest(bt)
    out_ts = results_dir / f"markowitz_{suffix}_ts.parquet"
    out_summary = results_dir / f"markowitz_{suffix}_summary.json"
    bt.to_parquet(out_ts, index=False)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    ensure_tickers(cfg)
    ensure_prices(cfg)
    ensure_features(cfg)

    feature_df_raw, y, groups, meta = _load_dataset(Path("data/features.parquet"))
    dates = pd.to_datetime(meta["date"]).to_numpy()
    tr_mask, va_mask, te_mask = _ensure_splits(dates, cfg.get("embargo_days", 5))

    print(f"[split] train={tr_mask.sum()} valid={va_mask.sum()} test={te_mask.sum()}")

    base_df = feature_df_raw.reset_index(drop=True)
    dates_series = pd.Series(pd.to_datetime(meta["date"]), name="date")
    y_rank = _cross_sectional_rank(y, dates_series)

    sym_components = int(cfg.get("symbolic", {}).get("n_components", 120))
    if sym_components > 0:
        sym_df, expr_map = _train_symbolic_transformer(
            base_df,
            y_rank,
            tr_mask,
            n_components=sym_components,
            random_state=cfg.get("symbolic", {}).get("random_state", 42),
        )
    else:
        sym_df = pd.DataFrame(index=base_df.index)
        expr_map = {}
    feature_df = _make_feature_frame(base_df, sym_df)
    feature_cols = feature_df.columns.tolist()
    feature_df = _cross_sectional_zscore(feature_df, dates_series)

    params = {
        "tree_method": "hist",
        "n_estimators": cfg.get("xgboost", {}).get("n_estimators", 400),
        "max_depth": cfg.get("xgboost", {}).get("max_depth", 6),
        "learning_rate": cfg.get("xgboost", {}).get("learning_rate", 0.05),
        "subsample": cfg.get("xgboost", {}).get("subsample", 0.8),
        "colsample_bytree": cfg.get("xgboost", {}).get("colsample_bytree", 0.5),
        "reg_lambda": cfg.get("xgboost", {}).get("reg_lambda", 1.0),
        "random_state": cfg.get("xgboost", {}).get("random_state", 42),
    }

    X_tr = feature_df.loc[tr_mask, feature_cols].to_numpy(dtype=np.float32)
    y_tr = y_rank[tr_mask]

    model_all = _train_xgb(X_tr, y_tr, params)

    X_va = feature_df.loc[va_mask, feature_cols].to_numpy(dtype=np.float32)
    y_va = y[va_mask]
    g_va = groups[va_mask]
    _ = model_all.predict(X_va)  # ensure prediction path for SHAP baseline

    val_df = feature_df.loc[va_mask].copy()
    top_features, importance_all = _shap_rank_features(model_all, val_df, feature_cols, top_k=10)

    X_tr_top = feature_df.loc[tr_mask, top_features].to_numpy(dtype=np.float32)
    X_va_top = feature_df.loc[va_mask, top_features].to_numpy(dtype=np.float32)
    model_top = _train_xgb(X_tr_top, y_rank[tr_mask], params)
    va_pred = model_top.predict(X_va_top)
    ic_valid = daily_rank_ic(y_va, va_pred, g_va, min_group_size=cfg.get("min_group_size", 30))
    print(f"[valid] IC={ic_valid:.5f}")

    shap_cfg = cfg.get("shap", {}) or {}
    max_samples = int(shap_cfg.get("max_samples", 800))

    val_sample = feature_df.loc[va_mask, top_features]
    if len(val_sample) > max_samples:
        val_sample = val_sample.sample(n=max_samples, random_state=42)
    val_sample = val_sample.reset_index(drop=True)
    explainer_val = shap.TreeExplainer(model_top)
    shap_val = explainer_val.shap_values(val_sample[top_features].to_numpy(), check_additivity=False)
    if isinstance(shap_val, list):
        shap_val = shap_val[0]
    mean_abs_val = np.abs(shap_val).mean(axis=0)
    order_val = np.argsort(mean_abs_val)[::-1]
    ranked_val = [top_features[i] for i in order_val]
    importance_val = pd.DataFrame({"feature": ranked_val, "mean_abs_shap": mean_abs_val[order_val]})

    final_mask = tr_mask | va_mask
    X_final = feature_df.loc[final_mask, top_features].to_numpy(dtype=np.float32)
    y_final = y_rank[final_mask]

    final_model = _train_xgb(X_final, y_final, params)

    X_te = feature_df.loc[te_mask, top_features].to_numpy(dtype=np.float32)
    y_te = y[te_mask]
    g_te = groups[te_mask]
    te_pred = final_model.predict(X_te)
    ic_test = daily_rank_ic(y_te, te_pred, g_te, min_group_size=cfg.get("min_group_size", 30))
    print(f"[test]  IC={ic_test:.5f}")

    test_sample = feature_df.loc[te_mask, top_features]
    if len(test_sample) > max_samples:
        test_sample = test_sample.sample(n=max_samples, random_state=43)
    test_sample = test_sample.reset_index(drop=True)
    explainer_test = shap.TreeExplainer(final_model)
    shap_test = explainer_test.shap_values(test_sample[top_features].to_numpy(), check_additivity=False)
    if isinstance(shap_test, list):
        shap_test = shap_test[0]
    mean_abs_test = np.abs(shap_test).mean(axis=0)
    order_test = np.argsort(mean_abs_test)[::-1]
    ranked_test = [top_features[i] for i in order_test]
    importance_test = pd.DataFrame({"feature": ranked_test, "mean_abs_shap": mean_abs_test[order_test]})

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    shap_dir = results_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    importance_all.to_csv(shap_dir / "importance_all_features.csv", index=False)
    importance_val.to_csv(shap_dir / "shap_importance_validation.csv", index=False)
    importance_test.to_csv(shap_dir / "shap_importance_test.csv", index=False)
    importance_test.to_csv(results_dir / "shap_importance_test.csv", index=False)

    _save_shap_plot(
        shap_val[:, order_val],
        val_sample[ranked_val],
        ranked_val,
        importance_val,
        shap_dir / "shap_validation.png",
        "SHAP Beeswarm with Mean |SHAP| Overlay (Validation)",
    )

    _save_shap_plot(
        shap_test[:, order_test],
        test_sample[ranked_test],
        ranked_test,
        importance_test,
        shap_dir / "shap_test.png",
        "SHAP Beeswarm with Mean |SHAP| Overlay (Test)",
    )

    final_model.save_model(str(results_dir / "final_xgb_model.json"))

    summary = {
        "model_type": "xgboost",
        "valid_ic": ic_valid,
        "test_ic": ic_test,
        "n_features_total": len(feature_cols),
        "top_features": top_features,
        "symbolic_components": sym_df.shape[1],
        "model_params": params,
        "label_transform": "cross_sectional_rank",
    }
    (results_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "symbolic_feature_map.json").write_text(json.dumps(expr_map, indent=2), encoding="utf-8")

    meta_df = meta.copy()
    meta_df["date"] = pd.to_datetime(meta_df["date"])

    valid_preds = meta_df.loc[va_mask].copy()
    valid_preds["pred"] = va_pred
    valid_preds["y"] = y_va
    valid_preds.to_parquet(results_dir / "valid_preds.parquet", index=False)

    test_preds = meta_df.loc[te_mask].copy()
    test_preds["pred"] = te_pred
    test_preds["y"] = y_te
    test_preds.to_parquet(results_dir / "test_preds.parquet", index=False)

    top_features_path = results_dir / "top_features.json"
    top_features_path.write_text(json.dumps(top_features, indent=2), encoding="utf-8")

    ret_panel, vol_panel = load_panels(Path("data/features.parquet"))
    benchmark = load_benchmark_returns(ret_panel.index.min(), ret_panel.index.max())
    backtest_params = BacktestParams(
        top_quantile=cfg.get("backtest", {}).get("top_quantile", 0.08),
        bottom_quantile=cfg.get("backtest", {}).get("bottom_quantile", 0.30),
        lookback=cfg.get("backtest", {}).get("lookback", 63),
        ridge=cfg.get("backtest", {}).get("ridge", 1e-4),
        max_abs_weight=cfg.get("backtest", {}).get("max_abs_weight", 0.08),
        gross_leverage=cfg.get("backtest", {}).get("gross_leverage", 1.0),
        cost_bps=float(cfg.get("cost_bps_oneway", 1.0)),
        min_bucket=cfg.get("backtest", {}).get("min_bucket", 10),
    )

    valid_bt_df = valid_preds[["date", "ticker", "pred", "y"]].copy()
    test_bt_df = test_preds[["date", "ticker", "pred", "y"]].copy()

    _prepare_markowitz(valid_bt_df, results_dir, "valid", backtest_params, ret_panel, vol_panel, benchmark)
    _prepare_markowitz(test_bt_df, results_dir, "test", backtest_params, ret_panel, vol_panel, benchmark)

    print("[done] Tree-based pipeline complete.")


if __name__ == "__main__":
    main()
