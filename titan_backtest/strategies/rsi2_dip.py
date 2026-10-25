"""Short-term dip buying with a trend filter (mean reversion).

Rule: only in an uptrend (close above 200-day average), buy when 2-day RSI
drops below `rsi_entry`; sell when RSI rises above `rsi_exit` or after
`max_hold` days. Long only.

Tested and killed on 11 index CFDs (EXP-006: +0.12% average trade, too thin
after costs, not better than random dip buying). Kept as a reference
implementation of a stateful entry/exit strategy."""

import numpy as np
import pandas as pd

PARAMS = {"rsi_entry": 10, "rsi_exit": 60, "max_hold": 5, "sma_filter": 200, "weight": 0.20}
GRID = {"rsi_entry": [5, 10, 15], "max_hold": [3, 5]}
BACKTEST_KWARGS = {}


def _rsi(series: pd.Series, n: int = 2) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd)


def positions(px: pd.DataFrame, sp: pd.DataFrame, rsi_entry=10, rsi_exit=60, max_hold=5, sma_filter=200, weight=0.20):
    out = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for col in px.columns:
        c = px[col].dropna()
        if len(c) < sma_filter + 10:
            continue
        r2 = _rsi(c, 2)
        sma = c.rolling(sma_filter).mean()
        enter = (r2 < rsi_entry) & (c > sma)
        exit_ = r2 > rsi_exit
        holding, days = 0, 0
        p = np.zeros(len(c))
        for i in range(1, len(c)):
            if holding == 0:
                if enter.iloc[i]:
                    holding, days = 1, 0
            else:
                days += 1
                if exit_.iloc[i] or days >= max_hold:
                    holding, days = 0, 0
            p[i] = holding
        out[col] = pd.Series(p, index=c.index).reindex(px.index).fillna(0)
    # 20% of equity per open position, scaled down if more than 5 at once
    concurrent = out.sum(axis=1)
    scale = (5.0 / concurrent.clip(lower=5)).fillna(1.0)
    return out.mul(scale, axis=0) * weight
