from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import max_drawdown, sharpe_ann


@dataclass
class BacktestParams:
    top_quantile: float
    bottom_quantile: float
    lookback: int
    ridge: float
    max_abs_weight: float
    gross_leverage: float
    cost_bps: float
    min_bucket: int


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_panels(features_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["date", "ticker", "ret_1d", "vol_20d"]
    feats = pd.read_parquet(features_path, columns=cols)
    feats["date"] = pd.to_datetime(feats["date"])
    feats = feats.sort_values(["date", "ticker"]).dropna(subset=["ret_1d"])
    ret_panel = feats.pivot(index="date", columns="ticker", values="ret_1d")
    vol_panel = feats.pivot(index="date", columns="ticker", values="vol_20d")
    return ret_panel, vol_panel


def load_benchmark_returns(
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbol: str = "^GSPC",
    cache_path: Path | str = Path("data/sp500_index.parquet"),
) -> pd.Series:
    cache_path = Path(cache_path)
    bench = pd.DataFrame()
    if cache_path.exists():
        bench = pd.read_parquet(cache_path)
        if "date" not in bench.columns or "close" not in bench.columns:
            bench = pd.DataFrame()
    if bench.empty or bench["date"].min() > start or bench["date"].max() < end:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start - pd.Timedelta(days=10),
            end=end + pd.Timedelta(days=10),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise RuntimeError(f"Failed to download benchmark data for {symbol}.")
        close_series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        close_series = close_series.rename("close")
        bench = (
            close_series.reset_index()
            .rename(columns={"Date": "date"})
            .dropna(subset=["close"])
        )
        bench.to_parquet(cache_path, index=False)

    bench = bench.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench = (
        bench[bench["date"].between(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10))]
        .drop_duplicates(subset="date")
        .sort_values("date")
    )
    bench.set_index("date", inplace=True)
    returns = bench["close"].pct_change().dropna()
    if returns.empty:
        raise RuntimeError("Benchmark return series is empty.")
    return returns


def _solve_markowitz(mu: np.ndarray, cov: np.ndarray, ridge: float) -> np.ndarray:
    mat = cov + np.eye(cov.shape[0]) * ridge
    inv = np.linalg.pinv(mat)
    w = inv @ mu
    w = np.clip(w, 0.0, None)
    if np.allclose(w.sum(), 0.0):
        w = np.ones_like(mu) / max(len(mu), 1)
    else:
        w /= w.sum()
    return w


def _normalise_weights(weights: pd.Series, gross: float) -> pd.Series:
    pos = weights[weights > 0]
    neg = weights[weights < 0]
    if pos.sum() > 0:
        weights.loc[pos.index] = pos / pos.sum() * (gross / 2.0)
    if neg.sum() < 0:
        weights.loc[neg.index] = neg / (-neg.sum()) * (gross / 2.0)
    return weights


def _apply_soft_cap(weights: pd.Series, cap: float) -> pd.Series:
    if cap <= 0:
        return weights
    scaled = np.tanh(weights / cap) * cap
    return pd.Series(scaled, index=weights.index)


def run_markowitz_backtest(
    preds: pd.DataFrame,
    ret_panel: pd.DataFrame,
    vol_panel: pd.DataFrame,
    params: BacktestParams,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.sort_values(["date", "ticker"])

    ret_df = ret_panel.fillna(0.0)
    vol_df = vol_panel
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.reindex(ret_df.index).ffill().fillna(0.0)

    prev_weights = pd.Series(0.0, index=ret_df.columns)
    records = []

    for cur_date, day_df in preds.groupby("date"):
        try:
            idx = ret_df.index.get_loc(cur_date)
        except KeyError:
            prev_weights = prev_weights * 0.0
            continue

        if isinstance(idx, slice):
            idx = idx.start

        if idx is None or idx < params.lookback:
            prev_weights = prev_weights * 0.0
            continue

        hist_dates = ret_df.index[idx - params.lookback : idx]
        long_cut = day_df["pred"].quantile(1.0 - params.top_quantile)
        short_cut = day_df["pred"].quantile(params.bottom_quantile)
        longs = day_df[day_df["pred"] >= long_cut]
        shorts = day_df[day_df["pred"] <= short_cut]

        if len(longs) < params.min_bucket or len(shorts) < params.min_bucket:
            prev_weights = prev_weights * 0.0
            continue

        long_tickers = longs["ticker"].tolist()
        short_tickers = shorts["ticker"].tolist()
        need_tickers = long_tickers + short_tickers

        hist_returns = ret_df.loc[hist_dates, need_tickers]
        if hist_returns.isna().any().any():
            hist_returns = hist_returns.fillna(0.0)
        latest_vol = vol_df.loc[hist_dates[-1], need_tickers].fillna(vol_df.mean())

        if len(long_tickers) > 0:
            mu_long = longs.set_index("ticker")["pred"] / (np.abs(latest_vol[long_tickers]) + 1e-6)
            cov_long = hist_returns[long_tickers].cov().to_numpy()
            w_long = _solve_markowitz(mu_long.to_numpy(), cov_long, params.ridge)
        else:
            w_long = np.array([])

        if len(short_tickers) > 0:
            mu_short = (-shorts.set_index("ticker")["pred"]) / (np.abs(latest_vol[short_tickers]) + 1e-6)
            cov_short = hist_returns[short_tickers].cov().to_numpy()
            w_short = _solve_markowitz(mu_short.to_numpy(), cov_short, params.ridge)
        else:
            w_short = np.array([])

        weights = pd.Series(0.0, index=ret_df.columns)
        if len(long_tickers) > 0:
            weights.loc[long_tickers] = w_long
        if len(short_tickers) > 0:
            weights.loc[short_tickers] = -w_short

        weights = _apply_soft_cap(weights, params.max_abs_weight)
        weights = _normalise_weights(weights, params.gross_leverage)

        fut = day_df.set_index("ticker")["y"]
        perf = float((weights.loc[fut.index] * fut).sum())
        turnover = float((weights - prev_weights).abs().sum())
        cost = turnover * (params.cost_bps / 1e4)
        ret_net = perf - cost
        bench_ret = (
            float(benchmark_returns.loc[cur_date])
            if benchmark_returns is not None and cur_date in benchmark_returns.index
            else np.nan
        )

        records.append(
            {
                "date": cur_date,
                "ret": ret_net,
                "ret_gross": perf,
                "turnover": turnover,
                "cost": cost,
                "n_long": len(long_tickers),
                "n_short": len(short_tickers),
                "benchmark_ret": bench_ret,
            }
        )
        prev_weights = weights

    result = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if not result.empty:
        result["equity"] = (1.0 + result["ret"]).cumprod()
        if "benchmark_ret" in result.columns:
            result["benchmark_ret"] = result["benchmark_ret"].fillna(0.0)
            result["benchmark_equity"] = (1.0 + result["benchmark_ret"]).cumprod()
        else:
            result["benchmark_equity"] = np.nan
    return result


def information_ratio(strategy: pd.Series, benchmark: pd.Series, freq: int = 252) -> float:
    active = (strategy - benchmark).dropna()
    if len(active) < 2:
        return float("nan")
    mu = active.mean() * freq
    sd = active.std(ddof=1) * np.sqrt(freq)
    return float(mu / (sd + 1e-12))


def summarise_backtest(bt: pd.DataFrame) -> Dict[str, float]:
    if bt.empty:
        return {
            "n_periods": 0,
            "ann_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "info_ratio": float("nan"),
            "benchmark_ann_return": float("nan"),
        }
    n = len(bt)
    ann = (1.0 + bt["ret"]).prod() ** (252 / max(n, 1)) - 1.0
    sharpe = sharpe_ann(bt["ret"], freq=252)
    mdd = max_drawdown(bt["equity"])
    if "benchmark_ret" in bt.columns and bt["benchmark_ret"].notna().any():
        bench_series = bt["benchmark_ret"]
        bench_ann = (1.0 + bench_series).prod() ** (252 / max(len(bench_series), 1)) - 1.0
        ir = information_ratio(bt["ret"], bench_series, freq=252)
    else:
        bench_ann = float("nan")
        ir = float("nan")
    return {
        "n_periods": n,
        "ann_return": float(ann),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "info_ratio": float(ir),
        "benchmark_ann_return": float(bench_ann),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--features", type=Path, default=Path("data/features.parquet"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--top_quantile", type=float, default=0.08)
    parser.add_argument("--bottom_quantile", type=float, default=0.30)
    parser.add_argument("--lookback", type=int, default=63)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--max_abs_weight", type=float, default=0.08)
    parser.add_argument("--gross_leverage", type=float, default=1.0)
    parser.add_argument("--min_bucket", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cost = float(cfg.get("cost_bps_oneway", 1.0))
    params = BacktestParams(
        top_quantile=args.top_quantile,
        bottom_quantile=args.bottom_quantile,
        lookback=args.lookback,
        ridge=args.ridge,
        max_abs_weight=args.max_abs_weight,
        gross_leverage=args.gross_leverage,
        cost_bps=cost,
        min_bucket=args.min_bucket,
    )

    ret_panel, vol_panel = load_panels(args.features)
    benchmark = load_benchmark_returns(ret_panel.index.min(), ret_panel.index.max())
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for split in ("valid", "test"):
        pred_path = args.results_dir / f"{split}_preds.parquet"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions parquet at {pred_path}")
        preds = pd.read_parquet(pred_path)
        bt = run_markowitz_backtest(preds, ret_panel, vol_panel, params, benchmark_returns=benchmark)
        summary = summarise_backtest(bt)

        bt_path = args.results_dir / f"markowitz_{split}_ts.parquet"
        bt.to_parquet(bt_path, index=False)

        summary_path = args.results_dir / f"markowitz_{split}_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        print(f"[{split}] periods={summary['n_periods']} ann_return={summary['ann_return']:.4f} "
              f"sharpe={summary['sharpe']:.3f} max_dd={summary['max_drawdown']:.3f}")


if __name__ == "__main__":
    main()
