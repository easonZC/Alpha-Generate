
import json, os
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from .gp_functions import default_function_set
from .fitness import make_rankic_metric, daily_rank_ic
from typing import Any, cast

# Truth be told, random state doesn't matter much here
RANDOM_STATE = 42

FEATURE_COLS = [
    "ret_1d","ret_5d","ret_10d","mom_20d","vol_20d",
    "ma_gap","dist_ma20","hl_range20","rsi14","vol_z20","vol_chg_5d"
]

def make_dataset(features_path, horizon):
    df = pd.read_parquet(features_path, engine="fastparquet")
    df = df.sort_values(['date','ticker']).reset_index(drop=True)

    # First convert ±Inf to NaN
    import numpy as np
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)

    # Drop samples where any feature or label is NaN
    df = df.dropna(subset=FEATURE_COLS + ['y_fwd']).reset_index(drop=True)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df['y_fwd'].to_numpy(dtype=float)
    # Date -> integer group code (float), for sample_weight usage
    groups = pd.factorize(pd.to_datetime(df['date']))[0].astype(np.float64)
    meta = df[['date','ticker']].reset_index(drop=True)
    return X, y, groups, meta

def train_model(X, y, groups, cfg):
    metric = make_rankic_metric(min_group_size=cfg.get('min_group_size', 30))

    gens = int(cfg.get('generations', 12))
    stop = cfg.get('stopping_criteria', None)
    stop = np.inf if stop is None else float(stop)

    # Read four probabilities from config (can provide defaults)
    p_cross  = float(cfg.get('p_crossover',        0.65))
    p_sub    = float(cfg.get('p_subtree_mutation', 0.18))
    p_point  = float(cfg.get('p_point_mutation',   0.10))
    p_hoist  = float(cfg.get('p_hoist_mutation',   0.04))

    # Optional: print check & simple validation (not exceeding 1)
    total_p = p_cross + p_sub + p_point + p_hoist
    print(f"[train] gens={gens} stop={stop} "
          f"p_cx={p_cross} p_sub={p_sub} p_point={p_point} p_hoist={p_hoist} "
          f"p_repro={max(0.0, 1.0-total_p):.2f}")

    est = SymbolicRegressor(
        population_size=cfg.get('population_size', 600),
        generations=gens,
        tournament_size=cfg.get('tournament_size', 5),
        stopping_criteria=stop,
        const_range=tuple(cfg.get('const_range', [-0.2, 0.2])),
        init_depth=(2, 6),
        function_set=default_function_set(),
        metric=cast(Any, metric),
        parsimony_coefficient=cfg.get('parsimony_coefficient', 0.001),
        # ★ Use probabilities read from cfg above
        p_crossover=p_cross,
        p_subtree_mutation=p_sub,
        p_point_mutation=p_point,
        p_hoist_mutation=p_hoist,
        n_jobs=max(1, (os.cpu_count() or 1)//2),
        random_state=cfg.get('random_state', 42),
        verbose=1
    )
    est.fit(X, y, sample_weight=groups)
    return est

def evaluate_ic(est, X, y, groups, min_group_size=30):
    yhat = est.predict(X)
    rng = np.random.RandomState(0)
    yhat = yhat + 1e-12 * rng.normal(size=yhat.shape)   # Slightly break ties
    score = daily_rank_ic(y, yhat, groups,
                          min_group_size=min_group_size,
                          empty_value=0.0,       # Evaluation period: empty→0
                          tie_break_eps=1e-12)
    return score, yhat

def save_program(est, path):
    prog = est._program
    with open(path, "w") as f:
        f.write(str(prog))

if __name__=="__main__":
    import argparse, yaml
    from ..utils.time_split import train_valid_test_splits
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, default="data/features.parquet")
    p.add_argument("--config", type=str, default="config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    X, y, groups, meta = make_dataset(args.features, cfg['horizon_days'])

    tr, va, te = train_valid_test_splits(meta['date'].values, embargo_days=cfg.get('embargo_days',5))
    # ---- Size printing + fallback ----
    import numpy as _np, pandas as _pd

    def _sizes(msg, tr, va, te):
        print(f"[split] {msg} sizes:", int(tr.sum()), int(va.sum()), int(te.sum()))

    _sizes("orig", tr, va, te)

    # Fallback 1: remove embargo and split again
    if (tr.sum()==0) or (va.sum()==0) or (te.sum()==0):
        print("[split] fallback -> embargo_days=0")
        tr, va, te = train_valid_test_splits(meta['date'].values, embargo_days=0)
        _sizes("no-embargo", tr, va, te)

    # Fallback 2: if still empty, manually split by "trading days 60/20/20"
    if (tr.sum()==0) or (va.sum()==0) or (te.sum()==0):
        print("[split] fallback -> manual 60/20/20 by unique dates")
        days = _pd.to_datetime(meta['date']).unique()
        days.sort()
        n = len(days)
        if n < 3:
            raise RuntimeError(f"too few trading days: {n}")
        a, b = int(n*0.6), int(n*0.8)
        set_tr, set_va, set_te = set(days[:a]), set(days[a:b]), set(days[b:])
        tr = _pd.to_datetime(meta['date']).isin(set_tr).to_numpy()
        va = _pd.to_datetime(meta['date']).isin(set_va).to_numpy()
        te = _pd.to_datetime(meta['date']).isin(set_te).to_numpy()
        _sizes("manual", tr, va, te)
    # ------------------------
    # Train on train set
    est = train_model(X[tr], y[tr], groups[tr], cfg)

    # Validation IC
    va_ic, va_pred = evaluate_ic(est, X[va], y[va], groups[va], min_group_size=cfg.get('min_group_size',30))

    # Test IC
    te_ic, te_pred = evaluate_ic(est, X[te], y[te], groups[te], min_group_size=cfg.get('min_group_size',30))

    os.makedirs("results", exist_ok=True)
    # Save factor expressions & metrics
    out = {
        "program": str(est._program),
        "train_samples": int(tr.sum()),
        "valid_samples": int(va.sum()),
        "test_samples": int(te.sum()),
        "ic_valid": float(va_ic),
        "ic_test": float(te_ic),
    }
    with open("results/factors.json", "w") as f:
        json.dump(out, f, indent=2)
    # Save predictions with meta
    meta_va = meta[va].copy(); meta_va["pred"] = va_pred; meta_va["y"] = y[va]
    meta_te = meta[te].copy(); meta_te["pred"] = te_pred; meta_te["y"] = y[te]
    meta_va.to_parquet("results/valid_preds.parquet", index=False)
    meta_te.to_parquet("results/test_preds.parquet", index=False)
    print("Saved results/factors.json and preds parquet.")
