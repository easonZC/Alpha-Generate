# src/model/fitness.py
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from gplearn.fitness import make_fitness

def _safe_spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Compatible with SciPy old and new versions, uniformly returns float correlation coefficient"""
    res: Any = spearmanr(x, y)

    # Old version: tuple (corr, pval)
    if isinstance(res, tuple):
        val: Any = res[0]
    # New version: StatsResult with .correlation
    elif hasattr(res, "correlation"):
        val = getattr(res, "correlation")
    else:
        val = res  # Fallback (sometimes numpy scalar)

    try:
        return float(val)
    except Exception:
        return float("nan")

# Key part of fitness.py
def daily_rank_ic(y, yhat, groups, min_group_size=30,
                  empty_value=-1e6, tie_break_eps=1e-12):
    df = pd.DataFrame({'g': groups, 'y': y, 'p': yhat}) \
           .replace([np.inf, -np.inf], np.nan).dropna()
    if tie_break_eps:
        rng = np.random.RandomState(42)
        df['p'] = df['p'] + tie_break_eps * rng.normal(size=len(df))
    ics = []
    for _, sub in df.groupby('g'):
        if len(sub) < min_group_size: 
            continue
        if sub['p'].nunique() < 2 or sub['y'].nunique() < 2:
            continue
        r = _safe_spearman_corr(sub['p'], sub['y'])
        if np.isfinite(r): ics.append(r)
    return float(np.nanmean(np.clip(ics, -0.5, 0.5))) if ics else float(empty_value)

def make_rankic_metric(min_group_size=30):
    def _metric(y, y_pred, sample_weight):
        # Training phase: strong penalty + minimal perturbation to break ties
        return daily_rank_ic(y, y_pred, sample_weight,
                             min_group_size=min_group_size,
                             empty_value=-1e6, tie_break_eps=1e-12)
    return make_fitness(function=_metric, greater_is_better=True)
