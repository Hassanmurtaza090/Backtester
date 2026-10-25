"""Performance metrics. All functions take a daily net-return Series."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0)).cumprod()


def cagr(r: pd.Series) -> float:
    eq = equity_curve(r)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return eq.iloc[-1] ** (1 / years) - 1


def max_drawdown(r: pd.Series) -> float:
    eq = equity_curve(r)
    return float((eq / eq.cummax() - 1).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = equity_curve(r)
    return eq / eq.cummax() - 1


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(TRADING_DAYS))


def summary(r: pd.Series, name: str = "") -> dict:
    """Full metric suite for a daily return series."""
    r = r.fillna(0)
    eq = equity_curve(r)
    dd = drawdown_series(r)
    c = cagr(r)
    mdd = float(dd.min())

    # drawdown duration and recovery
    in_dd = dd < 0
    dur, max_dur, cur = [], 0, 0
    for flag in in_dd.values:
        cur = cur + 1 if flag else 0
        max_dur = max(max_dur, cur)

    monthly = (1 + r).resample("ME").prod() - 1
    yearly = (1 + r).groupby(r.index.year).prod() - 1

    wins = r[r > 0]
    losses = r[r < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf

    out = {
        "name": name,
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "years": round((r.index[-1] - r.index[0]).days / 365.25, 1),
        "total_return_pct": round((eq.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(c * 100, 2),
        "vol_pct": round(float(r.std() * np.sqrt(TRADING_DAYS)) * 100, 1),
        "maxdd_pct": round(mdd * 100, 2),
        "avg_dd_pct": round(float(dd[dd < 0].mean() * 100) if (dd < 0).any() else 0, 1),
        "longest_dd_days": int(max_dur),
        "sharpe": round(sharpe(r), 2),
        "sortino": round(sortino(r), 2),
        "calmar": round(c / abs(mdd), 2) if mdd != 0 else 0.0,
        "profit_factor_daily": round(pf, 2),
        "pct_positive_days": round(float((r > 0).mean() * 100), 1),
        "skew": round(float(r.skew()), 2),
        "kurtosis": round(float(r.kurtosis()), 2),
        "var95_daily_pct": round(float(np.percentile(r, 5)) * 100, 2),
        "cvar95_daily_pct": round(float(r[r <= np.percentile(r, 5)].mean()) * 100, 2),
        "worst_month_pct": round(float(monthly.min()) * 100, 1),
        "best_month_pct": round(float(monthly.max()) * 100, 1),
        "worst_year_pct": round(float(yearly.min()) * 100, 1),
        "best_year_pct": round(float(yearly.max()) * 100, 1),
    }
    return out


def yearly_table(r: pd.Series) -> pd.Series:
    r = r.fillna(0)
    return ((1 + r).groupby(r.index.year).prod() - 1) * 100
