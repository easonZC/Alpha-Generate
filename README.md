# GP Factor Pipeline (US Equities, Daily)

This project builds an end-to-end research stack for cross-sectional equity alphas.  
It combines **gplearn SymbolicTransformer** (to expand factor space), **XGBoost** (for modelling and SHAP feature
selection), and a **covariance-aware Markowitz** backtest with benchmark comparison.

The current workflow lives in `train_tree_pipeline.py`.  
The original `run_pipeline.py` is kept only as a lightweight symbolic-regressor demo and no longer used for the main experiments.

## End-to-End Flow

1. **Data layer**
   - Universe: S&P 500 constituents (downloaded from Wikipedia).
  - Prices: daily OHLCV fetched via `yfinance`.
  - Cached under `data/` – no raw prices are committed.

2. **Feature construction**
   - Base technical features: returns, momentum, volatility, RSI, moving-average spreads, etc.
   - Symbolic features: 60 expressions per run from `gplearn.SymbolicTransformer`
     (configurable via `config.yaml:symbolic.*`). Expression map saved to
     `results/symbolic_feature_map.json`.
   - All features are cross-sectionally z-scored each trading day.

3. **Label preparation**
   - Target: 3-day forward return (`y_fwd` in `data/features.parquet`).
   - For modelling, the label is converted to daily cross-sectional percentile ranks
     (centred at 0) to stabilise rank-based metrics.

4. **Model training & SHAP pruning**
   - Model: `xgboost.XGBRegressor` (defaults: hist tree method, 400 estimators,
     max depth 6, learning rate 0.05, subsample 0.8, colsample 0.5, `random_state=42`).
   - Training set: 60 % earliest dates; validation: next 20 %; test: last 20 %, with
     embargo fallback logic to guarantee non-empty splits.
   - SHAP (`TreeExplainer`) runs on the validation slice, selecting the top 10 features
     (combining base + symbolic). Final XGBoost model is refit using train+validation data
     restricted to these 10 factors.
   - Artefacts:
     - `results/model_summary.json` (IC metrics, parameters, selected factors).
     - `results/top_features.json` (ordered list of top 10 factors).
     - `results/final_xgb_model.json` (serialised booster).
     - SHAP plots/importance: `results/shap/shap_validation.png`,
       `results/shap/shap_test.png`, `results/shap_importance_{validation,test}.csv`.

5. **Backtesting**
   - Script: `diagnostics.run_markowitz_backtest` (optionally invoked via
     `diagnostics/run_test_evaluation.py`).
   - Parameters (defaults, override in CLI or `config.yaml`):
     - `top_quantile`, `bottom_quantile` (currently both 0.08).
     - Rolling lookback 63 trading days.
     - Ridge regularisation `1e-4`.
     - Soft weight cap 0.08, gross leverage 1.0, min basket size 20.
     - Transaction cost 1 bp per side.
   - Produces daily long/short weights, turnover, and returns. Baseline benchmark is
     S&P 500 (^GSPC). Summaries land in:
     - `results/markowitz_valid_summary.json`
     - `results/markowitz_test_summary.json` (includes annual alpha vs benchmark)
   - Time-series parquet files include strategy and benchmark equity curves for plotting.

6. **Visuals**
   - `docs/images/backtest_cumulative.png`: strategy vs benchmark test-equity curve.
   - `docs/images/shap_summary.png`: validation SHAP beeswarm with overlaid mean |SHAP|.

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate      # or source .venv/bin/activate on Unix
pip install -r requirements.txt
python train_tree_pipeline.py
```

This single command:
1. Ensures data/feature caches.
2. Generates 60 symbolic factors (train-only fit).
3. Trains XGBoost, computes SHAP, keeps top 10 factors.
4. Predicts validation/test IC, exports SHAP graphics/tables.
5. Runs Markowitz backtests (validation + out-of-sample test).

The resulting files appear under `results/` and plots under `docs/images/`.

## Diagnostics Commands

| Purpose                                    | Command |
|--------------------------------------------|---------|
| Re-run Markowitz backtest only             | `python -m diagnostics.run_markowitz_backtest --top_quantile 0.08 --bottom_quantile 0.08 --max_abs_weight 0.08 --min_bucket 20` |
| Copy SHAP plots & backtest curve to docs   | `python -m diagnostics.render_figures` |
| Refresh summaries & copies after edits     | `python -m diagnostics.run_test_evaluation` |

(`run_pipeline.py` can be ignored unless you want the legacy single-factor demo.)

## Key Outputs

- `results/model_summary.json` – IC metrics, selected factors, model params.
- `results/top_features.json` – top 10 features (mix of base + symbolic).
- `results/shap_importance_test.csv` – SHAP ranking on the genuine test set.
- `results/markowitz_test_summary.json` – annualised return, Sharpe, max drawdown, info ratio, alpha.
- `docs/images/backtest_cumulative.png` – comparison chart (strategy vs S&P 500).
- `docs/images/shap_summary.png` – validation SHAP beeswarm + mean |SHAP|.

## Repository Layout

```
train_tree_pipeline.py        # Main end-to-end pipeline (SymbolicTransformer + XGBoost + Markowitz)
run_pipeline.py               # Legacy demo (SymbolicRegressor only)
diagnostics/
  run_markowitz_backtest.py   # Markowitz backtest engine
  run_test_evaluation.py      # Copies SHAP & reruns backtests with chosen params
  render_figures.py           # Copies SHAP plot & regenerates equity curve
data/                         # Cached prices/features (ignored by git)
docs/images/                  # Plots used in README/reporting
```

## Contributing & Notes

- Adjust `config.yaml` to change SymbolicTransformer population/generations or XGBoost hyperparameters.
- To experiment with LightGBM or alternative learners, swap out model training in `train_tree_pipeline.py`.
- `run_pipeline.py` remains for backwards compatibility but is not part of the production workflow.
- Large datasets/results stay out of version control; only lightweight JSON/plots are persisted.

---
Happy factor hunting! Feel free to open issues/PRs with improvements or bug reports.
