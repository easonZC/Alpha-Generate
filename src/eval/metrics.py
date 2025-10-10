
import numpy as np
import pandas as pd
from scipy.stats import norm

def sharpe_ann(returns, freq=252):
    r = pd.Series(returns).dropna()
    if len(r)==0:
        return np.nan
    mu = r.mean() * freq
    sd = r.std(ddof=1) * np.sqrt(freq)
    return mu / (sd + 1e-12)

def max_drawdown(equity):
    x = pd.Series(equity).values
    peak = -np.inf
    run_max = np.maximum.accumulate(np.asarray(x, dtype=np.float64))
    dd = (x.astype(np.float64) / (run_max + 1e-12)) - 1.0
    return dd.min()

def probabilistic_sharpe_ratio(sr_hat, sr0=0.0, T=252, skew=0.0, kurt=3.0):
    # Bailey & Lopez de Prado (2012) PSR
    # sigma_sr ≈ sqrt((1 - skew*sr_hat + (kurt-1)/4 * sr_hat^2) / (T-1))
    sig = np.sqrt((1 - skew*sr_hat + ((kurt-1)/4.0)*(sr_hat**2)) / max(T-1,1))
    z = (sr_hat - sr0) / (sig + 1e-12)
    return norm.cdf(z)

def deflated_sharpe_ratio(sr_max, sr0=0.0, T=252, n_trials=100, var_sr=1.0):
    # Very rough DSR proxy using effective #trials; for proper DSR see Bailey et al.
    # Here we just shrink sr0 upward by multiple-testing penalty.
    adj = np.sqrt(2*np.log(max(n_trials,1))) * np.sqrt(var_sr/(T-1))
    sr_bar = sr0 + adj
    return probabilistic_sharpe_ratio(sr_max, sr_bar, T)
