
import numpy as np
import pandas as pd

def time_blocks(dates, n_blocks):
    u = np.array(sorted(pd.Series(dates).unique()))
    splits = np.array_split(u, n_blocks)
    return [ (blk[0], blk[-1]) for blk in splits ]

def embargo_mask(dates, train_end, embargo_days):
    # mask rows that fall within [train_end+1, train_end+embargo]
    if embargo_days <= 0:
        return np.zeros(len(dates), dtype=bool)
    u = np.array(sorted(pd.Series(dates).unique()))
    try:
        idx = np.where(u == train_end)[0][0]
    except IndexError:
        return np.zeros(len(dates), dtype=bool)
    banned = set(u[idx+1: idx+1+embargo_days])
    return np.array([d in banned for d in dates])

def train_valid_test_splits(dates, embargo_days=5, valid_blocks=1, test_blocks=1):
    """
    Simple blocked splits with embargo: [train][embargo][valid][embargo][test]
    Returns boolean masks for train/valid/test of length len(dates).
    """
    u = np.array(sorted(pd.Series(dates).unique()))
    n = len(u)
    
    # split 60/20/20 on unique dates
    t_end = int(0.6*n)
    v_end = int(0.8*n)
    train_days = set(u[:t_end])
    valid_days = set(u[t_end:v_end])
    test_days  = set(u[v_end:])

    train_mask = np.array([d in train_days for d in dates])
    valid_mask = np.array([d in valid_days for d in dates])
    test_mask  = np.array([d in test_days  for d in dates])

    # embargo
    if embargo_days > 0 and len(u) > 0:
        emb1 = embargo_mask(dates, u[t_end-1], embargo_days)
        emb2 = embargo_mask(dates, u[v_end-1], embargo_days)
        train_mask = train_mask & ~emb1 & ~emb2
        valid_mask = valid_mask & ~emb2

    return train_mask, valid_mask, test_mask
