# Project Evolution Roadmap

## Goals
- Expand the alpha universe with richer, point-in-time data.
- Keep the factor discovery layer pluggable so models can evolve without rewrites.
- Build a rigorous training, validation, and monitoring loop that prevents overfitting and highlights true alpha.

## Phase 1: Data Foundation
- Integrate `qlib` (and similar data providers) to fetch Alpha101/Alpha158/Alpha360 style features, fundamentals, and point-in-time index membership.
- Version datasets by hashing configuration inputs (`horizon_days`, feature sets, universe) and regenerate features/labels whenever the hash changes.
- Fix the current `sample_weight` misuse in `SymbolicRegressor` so grouping is passed explicitly and later samples do not get higher implicit weights.

## Phase 2: Modular Alpha Discovery
- Build an interface (e.g., `alpha_generators/`) that accepts a standardized feature frame and returns candidate alpha expressions plus metadata.
- Provide interchangeable implementations: current `gplearn`, `tsfresh` auto features, GP successors (AlphaGen, AlphaQCM), and reinforcement-learning-based generators.
- Persist random seeds, hyperparameters, and run IDs for every generator to guarantee reproducibility.

## Phase 3: Predictive Modelling Pipeline
- Train LightGBM/XGBoost (and optional baselines) on candidate factors using purged or combinatorial cross-validation to avoid leakage.
- Enforce regularization and conservative hyperparameter grids (depth, learning rate, min data in leaf) and log all configurations.
- Support ensemble options (bagging, boosting, model blending) with consistent data splits.

## Phase 4: Interpretability & Factor Selection
- Apply SHAP, permutation importance, and tree-path analysis to measure marginal contributions and interaction patterns.
- Run single-factor diagnostics (RankIC, Sharpe, PSR/DSR) on the top-ranked signals to confirm they survive out-of-sample testing.
- Document the accepted factors, their interpretation, and guardrails (e.g., expected regimes, turnover bounds).

## Phase 5: Portfolio Construction & Evaluation
- Feed the refined factor set into downstream ML/DL models or strategy optimizers; support multi-model voting or stacking when resources allow.
- Execute rolling backtests with transaction-cost assumptions, benchmark comparisons (e.g., S&P 500, CSI 300), and stability metrics (turnover, drawdown).
- Promote a strategy only if it demonstrates persistent outperformance and passes risk checks.

## Governance & Experiment Tracking
- Use MLflow/W&B (or a lightweight logging layer) to track data hashes, configs, seeds, metrics, and artifacts.
- Schedule periodic retraining and decommission models that underperform in recent windows.
- Maintain documentation for each release so future model swaps remain low-friction.

## Immediate Next Steps
- Add the data-versioning + feature rebuild guardrails and patch the GP fitness grouping issue.
- Implement the modular alpha-generator interface with the current gplearn baseline as the first adapter.
- Integrate cross-validated LightGBM/XGBoost training and attach SHAP-based reporting for the existing factor set.
