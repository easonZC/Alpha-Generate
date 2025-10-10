
import numpy as np
import pandas as pd

def long_short_backtest(pred_df, top_q=0.2, bot_q=0.2, cost_bps_oneway=0.0):
    """
    pred_df: columns [date, ticker, pred, y] where y is future return (e.g., 5-day)
    We form daily long (top_q) and short (bot_q) and hold for 1 period aligned to y.
    """
    df = pred_df.dropna(subset=['pred','y']).copy()
    out = []
    for d, sub in df.groupby('date'):
        if len(sub) < 30:
            continue
        q_hi = sub['pred'].quantile(1-top_q)
        q_lo = sub['pred'].quantile(bot_q)
        long = sub[sub['pred']>=q_hi]
        short = sub[sub['pred']<=q_lo]
        n_long, n_short = len(long), len(short)
        if n_long==0 or n_short==0:
            continue
        ret = long['y'].mean() - short['y'].mean()
        # rough turnover proxy: assume full turnover each rebalance; costs on both sides
        cost = (cost_bps_oneway/1e4) * (1.0 + 1.0)
        ret_net = ret - cost
        out.append({"date": d, "ret": ret_net, "ret_gross": ret})
    res = pd.DataFrame(out).sort_values('date')
    res['equity'] = (1.0 + res['ret']).cumprod()
    return res
