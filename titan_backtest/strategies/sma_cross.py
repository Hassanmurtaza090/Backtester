"""Classic moving-average cross. Long when fast SMA above slow SMA, short
when below. Equal weight across symbols. Tested and killed on gold alone
(EXP-004: it lost to simply holding gold). Kept as a reference strategy."""

import numpy as np
import pandas as pd

PARAMS = {"fast": 50, "slow": 200}
GRID = {"fast": [20, 50], "slow": [100, 200]}
BACKTEST_KWARGS = {}


def positions(px: pd.DataFrame, sp: pd.DataFrame, fast=50, slow=200):
    f = px.rolling(fast).mean()
    s = px.rolling(slow).mean()
    sig = pd.DataFrame(np.where(f > s, 1.0, -1.0), index=px.index, columns=px.columns)
    sig = sig.where(s.notna())
    n = sig.notna().sum(axis=1).replace(0, np.nan)
    return sig.div(n, axis=0)
