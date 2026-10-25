"""Time-series momentum (trend following).

Rule: for each symbol, long if the last `lookback` days were up, short if
down. Size positions inversely to volatility so wild markets get less money.
Rebalance weekly to keep costs down.

This is the strategy family with the strongest published evidence, and the
design that passed the earlier 12-experiment Titan Quant cycle (on futures).
It FAILED on the cTrader CFD universe (EXP-005) because that universe lacks
bonds and commodities. Kept here as the reference implementation.
"""

import numpy as np
import pandas as pd

PARAMS = {"lookback": 252, "vol_window": 60, "weekly": True}

# small, pre-registered grid for walk-forward (3 values only, on purpose)
GRID = {"lookback": [63, 126, 252]}

# engine settings this strategy expects
BACKTEST_KWARGS = {"vol_target": 0.10, "vol_window": 60, "lev_cap": 3.0}


def positions(px: pd.DataFrame, sp: pd.DataFrame, lookback=252, vol_window=60, weekly=True):
    ret = px.pct_change(fill_method=None)
    sig = np.sign(px.pct_change(lookback, fill_method=None))
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    inv_vol = (1.0 / vol).where(sig.notna())
    w = inv_vol.div(inv_vol.abs().sum(axis=1), axis=0) * sig
    if weekly:
        monday = px.index.dayofweek == 0
        w = w.where(pd.Series(monday, index=px.index), np.nan).ffill()
    return w
