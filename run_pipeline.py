# run_pipeline.py
import os, json, yaml, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.data.download_symbols import get_sp500_symbols
from src.data.download_prices import main as dl_prices
from src.features.build_features import build_all
from src.model.run_gp import make_dataset, train_model, evaluate_ic
from src.utils.time_split import train_valid_test_splits

# ---------- helpers ----------
def ensure_tickers(cfg):
    if cfg.get("universe", "sp500") == "sp500":
        syms = get_sp500_symbols()
        os.makedirs("data", exist_ok=True)
        pd.Series(syms).to_csv("data/tickers.txt", index=False, header=False)
        print(f"[universe] saved {len(syms)} S&P500 tickers to data/tickers.txt")

def ensure_prices(cfg):
    if not os.path.exists("data/prices.parquet"):
        dl_prices(cfg["start_date"], cfg["end_date"], cfg["universe"], cfg["tickers_file"])

def ensure_features(cfg):
    if not os.path.exists("data/features.parquet"):
        df = pd.read_parquet("data/prices.parquet", engine="fastparquet")
        df = build_all(df)
        df = df.sort_values(["ticker", "date"])
        h = cfg["horizon_days"]
        df["y_fwd"] = df.groupby("ticker")["Close"].shift(-h) / df["Close"] - 1.0
        df.to_parquet("data/features.parquet", index=False, engine="fastparquet")
        print(f"[features] saved data/features.parquet {tuple(df.shape)}")

def _clean_subset(X, y, g, mask):
    Xi, yi, gi = X[mask], y[mask], g[mask]
    ok = np.isfinite(yi) & np.isfinite(Xi).all(axis=1)
    return Xi[ok], yi[ok], gi[ok], ok

# ---------- main ----------
def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

    # 1) Data preparation (only runs when files are missing)
    ensure_tickers(cfg)
    ensure_prices(cfg)
    ensure_features(cfg)

    # 2) Read features and do basic cleaning (make_dataset already does Inf->NaN / dropna / group encoding)
    X, y, groups, meta = make_dataset("data/features.parquet", cfg["horizon_days"])

    # 3) Time splitting (with two-level fallback)
    dates = pd.to_datetime(meta["date"]).to_numpy()
    tr, va, te = train_valid_test_splits(dates, embargo_days=cfg.get("embargo_days", 5))

    def _sizes(tag, tr, va, te):
        print(f"[split] {tag} sizes:", int(tr.sum()), int(va.sum()), int(te.sum()), flush=True)

    _sizes("orig", tr, va, te)

    if (tr.sum()==0) or (va.sum()==0) or (te.sum()==0):
        print("[split] fallback -> embargo_days=0", flush=True)
        tr, va, te = train_valid_test_splits(dates, embargo_days=0)
        _sizes("no-embargo", tr, va, te)

    if (tr.sum()==0) or (va.sum()==0) or (te.sum()==0):
        print("[split] fallback -> manual 60/20/20 by unique dates", flush=True)
        uniq = np.unique(dates)
        n = len(uniq)
        if n < 3:
            raise RuntimeError(f"too few trading days: {n}")
        k1, k2 = int(n*0.6), int(n*0.8)
        tr = np.isin(dates, uniq[:k1])
        va = np.isin(dates, uniq[k1:k2])
        te = np.isin(dates, uniq[k2:])
        _sizes("manual", tr, va, te)

    assert tr.any() and va.any() and te.any(), "split still empty"

    # 4) Subset cleaning (remove NaN/Inf rows)
    X_tr, y_tr, g_tr, ok_tr = _clean_subset(X, y, groups, tr)
    print("[train] before/after NaN:", int(tr.sum()), "->", len(y_tr))

    # 5) Training (train_model already handles None compatibility for stopping_criteria)
    est = train_model(X_tr, y_tr, g_tr, cfg)

    # 6) Evaluation: validation / test
    X_va, y_va, g_va, ok_va = _clean_subset(X, y, groups, va)
    X_te, y_te, g_te, ok_te = _clean_subset(X, y, groups, te)

    va_ic, va_pred = evaluate_ic(est, X_va, y_va, g_va, min_group_size=cfg.get("min_group_size", 30))
    te_ic, te_pred = evaluate_ic(est, X_te, y_te, g_te, min_group_size=cfg.get("min_group_size", 30))

    # 7) Save results
    os.makedirs("results", exist_ok=True)

    # 7.1 Factor expression + summary metrics
    out = {
        "program": str(est._program),
        "train_samples": int(len(y_tr)),
        "valid_samples": int(len(y_va)),
        "test_samples": int(len(y_te)),
        "ic_valid": float(va_ic),
        "ic_test": float(te_ic),
        "config": cfg,
    }
    with open("results/factors.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[save] results/factors.json written")

    # 7.2 Per-sample predictions (validation/test)
    meta_va = meta[va].reset_index(drop=True).iloc[ok_va].copy()
    meta_te = meta[te].reset_index(drop=True).iloc[ok_te].copy()
    meta_va["pred"] = va_pred; meta_va["y"] = y_va
    meta_te["pred"] = te_pred; meta_te["y"] = y_te
    meta_va.to_parquet("results/valid_preds.parquet", index=False, engine="fastparquet")
    meta_te.to_parquet("results/test_preds.parquet",  index=False, engine="fastparquet")
    print("[save] results/valid_preds.parquet & results/test_preds.parquet written")

    print("[done] program:", out["program"])
    print("[done] IC(valid) =", out["ic_valid"], " IC(test) =", out["ic_test"])

if __name__ == "__main__":
    main()


