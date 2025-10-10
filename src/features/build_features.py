import pandas as pd
import numpy as np

def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    # ensure float for calculations, avoid object dtype
    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()

    # use clip to separate gains and losses, avoid delta > 0 / < 0 type warnings
    gain = delta.clip(lower=0.0)        # Keep delta>0, set <=0 to 0
    loss = (-delta).clip(lower=0.0)     # Convert delta<0 to positive, set >=0 to 0

    roll_up = gain.ewm(alpha=1.0/n, adjust=False).mean()
    roll_down = loss.ewm(alpha=1.0/n, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-12)  # Prevent division by zero
    return 100.0 - 100.0 / (1.0 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input columns must include: [date, ticker, Open, High, Low, Close, Volume]"""
    df = df.copy()

    # returns and momentum
    df['ret_1d']  = df.groupby('ticker')['Close'].pct_change(1,  fill_method=None)
    df['ret_5d']  = df.groupby('ticker')['Close'].pct_change(5,  fill_method=None)
    df['ret_10d'] = df.groupby('ticker')['Close'].pct_change(10, fill_method=None)
    df['mom_20d'] = df.groupby('ticker')['Close'].pct_change(20, fill_method=None)

    # volatility, moving averages, price distance
    df['vol_20d'] = df.groupby('ticker')['ret_1d'].rolling(20).std().reset_index(level=0, drop=True)
    df['ma10']    = df.groupby('ticker')['Close'].rolling(10).mean().reset_index(level=0, drop=True)
    df['ma20']    = df.groupby('ticker')['Close'].rolling(20).mean().reset_index(level=0, drop=True)
    df['ma_gap']  = (df['ma10'] - df['ma20']) / (df['ma20'] + 1e-12)
    df['dist_ma20'] = (df['Close'] - df['ma20']) / (df['ma20'] + 1e-12)
    df['hl_range20'] = (
        df.groupby('ticker')['High'].rolling(20).max().reset_index(level=0, drop=True)
        - df.groupby('ticker')['Low'].rolling(20).min().reset_index(level=0, drop=True)
    ) / (df['ma20'] + 1e-12)

    # RSI
    df['rsi14'] = df.groupby('ticker')['Close'].apply(_rsi, n=14).reset_index(level=0, drop=True)

    # volume related
    vol_ema20 = df.groupby('ticker')['Volume'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    vol_std20 = df.groupby('ticker')['Volume'].transform(lambda x: x.rolling(20).std())
    df['vol_ema20']  = vol_ema20
    df['vol_z20']    = (df['Volume'] - vol_ema20) / (vol_std20 + 1e-12)
    df['vol_chg_5d'] = df.groupby('ticker')['Volume'].pct_change(5, fill_method=None)

    return df

def cs_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply cross-sectional z-score to specified features by date (subtract mean/divide by std within same day)."""
    out = df.copy()
    for c in cols:
        mu = out.groupby("date")[c].transform("mean")
        sd = out.groupby("date")[c].transform("std")
        out[c] = (out[c] - mu) / (sd + 1e-12)   # avoid division by zero
    return out

def cs_winsor3(df, cols):
    out = df.copy()
    for c in cols:
        mu = out.groupby('date')[c].transform('mean')
        sd = out.groupby('date')[c].transform('std')
        lo, hi = mu - 3*sd, mu + 3*sd
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out

def build_all(df: pd.DataFrame) -> pd.DataFrame:
    FEATS = ["ret_1d","ret_5d","ret_10d","mom_20d","vol_20d",
         "ma_gap","dist_ma20","hl_range20","rsi14","vol_z20","vol_chg_5d"]
    df1 = add_features(df)
    df1 = cs_zscore(df1, FEATS)
    df1 = cs_winsor3(df1, FEATS)
    return df1

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--infile", type=str, default="data/prices.parquet")
    p.add_argument("--outfile", type=str, default="data/features.parquet")
    p.add_argument("--horizon", type=int, default=5)
    args = p.parse_args()

    # read, build features
    df = pd.read_parquet(args.infile, engine="fastparquet")
    df = build_all(df)

    # generate labels (future returns in next `horizon` days)
    df = df.sort_values(['ticker', 'date'])
    df['y_fwd'] = df.groupby('ticker')['Close'].shift(-args.horizon) / df['Close'] - 1.0

    # save
    df.to_parquet(args.outfile, index=False, engine="fastparquet")
    print("Saved", args.outfile, df.shape)
