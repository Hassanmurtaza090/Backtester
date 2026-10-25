"""Validation tools: the tests that decide if a strategy lives or dies.

Kill criteria (fixed, from research_plan.md):
  1. Must beat buy-and-hold of its own market on a risk-adjusted basis.
  2. Must beat random entries with the same trading frequency (p < 0.05).
  3. Must stay positive at 3x costs.
  4. Needs enough trades (>=100) and history (>=5 years).
  5. Top 5 trades must not carry >30% of profit; no symbol >40% of profit.
  6. Walk-forward: >=55% of folds profitable.
"""

import itertools

import numpy as np
import pandas as pd

from . import metrics
from .backtest import run_backtest


# ---------------------------------------------------------------- benchmarks
def buy_and_hold(px: pd.DataFrame, start=None) -> pd.Series:
    """Equal-weight buy-and-hold of the same symbols (daily returns)."""
    r = px.pct_change(fill_method=None).mean(axis=1).fillna(0)
    return r[start:] if start else r


# ------------------------------------------------------------ random control
def random_control_pvalue(
    px: pd.DataFrame,
    result,
    n_sims: int = 300,
    start=None,
    seed: int = 42,
) -> float:
    """Compare the strategy CAGR to random strategies that trade with the
    same gross exposure profile but random direction, flipping sign at the
    same frequency. Returns p-value = share of random runs that did better.
    """
    rng = np.random.default_rng(seed)
    ret = px.pct_change(fill_method=None)
    pos = result.positions
    if start:
        ret, pos = ret[start:], pos[start:]
    strat_r = result.returns[start:] if start else result.returns
    strat_cagr = metrics.cagr(strat_r)

    sign = np.sign(pos)
    flips_per_col = (sign.diff().abs() > 0).sum()
    magnitude = pos.abs()

    sims = np.zeros(n_sims)
    n = len(ret)
    for i in range(n_sims):
        rand_sign = pd.DataFrame(1.0, index=ret.index, columns=ret.columns)
        for col in ret.columns:
            k = int(flips_per_col.get(col, 0))
            if k <= 0 or k >= n:
                continue
            flips = np.sort(rng.choice(n - 1, size=k, replace=False)) + 1
            s = np.ones(n)
            cur = rng.choice([-1.0, 1.0])
            j = 0
            for t in range(n):
                if j < len(flips) and t == flips[j]:
                    cur *= -1
                    j += 1
                s[t] = cur
            rand_sign[col] = s
        rand_r = (magnitude * rand_sign * ret).sum(axis=1).fillna(0)
        sims[i] = metrics.cagr(rand_r)
    return float((sims >= strat_cagr).mean())


# ---------------------------------------------------------------- stress test
def cost_stress(run_fn, multipliers=(1, 2, 3, 5), start=None) -> pd.DataFrame:
    """run_fn(cost_mult) must return a BacktestResult. Reports CAGR/DD per level."""
    rows = []
    for m in multipliers:
        res = run_fn(m)
        r = res.returns[start:] if start else res.returns
        rows.append(
            {
                "cost_mult": f"{m}x",
                "cagr_pct": round(metrics.cagr(r) * 100, 1),
                "maxdd_pct": round(metrics.max_drawdown(r) * 100, 1),
                "sharpe": round(metrics.sharpe(r), 2),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- halves
def split_halves(r: pd.Series) -> pd.DataFrame:
    r = r.fillna(0)
    half = len(r) // 2
    rows = []
    for tag, rr in [("first_half", r.iloc[:half]), ("second_half", r.iloc[half:])]:
        rows.append(
            {
                "period": tag,
                "start": str(rr.index[0].date()),
                "end": str(rr.index[-1].date()),
                "cagr_pct": round(metrics.cagr(rr) * 100, 1),
                "maxdd_pct": round(metrics.max_drawdown(rr) * 100, 1),
                "sharpe": round(metrics.sharpe(rr), 2),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------- profit concentration
def concentration(result, px: pd.DataFrame) -> dict:
    """Share of total profit from the top-5 days and from the best symbol."""
    ret = px.pct_change(fill_method=None)
    by_symbol = (result.positions * ret).sum().sort_values(ascending=False)
    total = by_symbol.sum()
    top_symbol_share = float(by_symbol.iloc[0] / total) if total > 0 else np.nan
    daily = result.returns
    pos_days = daily[daily > 0].sort_values(ascending=False)
    top5_share = (
        float(pos_days.head(5).sum() / daily.sum()) if daily.sum() > 0 else np.nan
    )
    return {
        "best_symbol": by_symbol.index[0],
        "best_symbol_profit_share": round(top_symbol_share, 2)
        if total > 0
        else None,
        "top5_days_profit_share": round(top5_share, 2) if daily.sum() > 0 else None,
    }


# --------------------------------------------------------------- walk-forward
def walk_forward(
    px: pd.DataFrame,
    sp: pd.DataFrame,
    strategy_fn,
    param_grid: dict,
    train_days: int = 756,     # ~3 years
    test_days: int = 252,      # ~1 year
    objective: str = "sortino",
    backtest_kwargs: dict | None = None,
):
    """Classic rolling walk-forward (Pardo style).

    strategy_fn(px, sp, **params) -> target position DataFrame.
    For each fold: pick the param set with the best `objective` on the train
    window, then run it on the UNSEEN test window. Stitch all test windows
    into one out-of-sample return series.
    """
    backtest_kwargs = backtest_kwargs or {}
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    obj_fn = {"sortino": metrics.sortino, "sharpe": metrics.sharpe, "cagr": metrics.cagr}[objective]

    idx = px.index
    folds = []
    oos_parts = []
    start = train_days
    fold_no = 0
    while start + test_days <= len(idx):
        tr_px = px.iloc[start - train_days : start]
        tr_sp = sp.iloc[start - train_days : start]
        # pick best params on the train window only
        best, best_score = None, -np.inf
        for params in combos:
            pos = strategy_fn(tr_px, tr_sp, **params)
            res = run_backtest(tr_px, tr_sp, pos, **backtest_kwargs)
            score = obj_fn(res.returns)
            if score > best_score:
                best, best_score = params, score
        # evaluate on the unseen test window (signal needs history -> pass
        # train+test data but keep only test-window returns)
        te_px = px.iloc[start - train_days : start + test_days]
        te_sp = sp.iloc[start - train_days : start + test_days]
        pos = strategy_fn(te_px, te_sp, **best)
        res = run_backtest(te_px, te_sp, pos, **backtest_kwargs)
        test_r = res.returns.iloc[-test_days:]
        oos_parts.append(test_r)
        fold_no += 1
        folds.append(
            {
                "fold": fold_no,
                "test_start": str(test_r.index[0].date()),
                "test_end": str(test_r.index[-1].date()),
                "chosen_params": str(best),
                "train_score": round(best_score, 2),
                "test_cagr_pct": round(metrics.cagr(test_r) * 100, 1),
                "test_sharpe": round(metrics.sharpe(test_r), 2),
                "profitable": bool((1 + test_r).prod() > 1),
            }
        )
        start += test_days

    folds_df = pd.DataFrame(folds)
    oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    pct_profitable = float(folds_df["profitable"].mean()) if len(folds_df) else np.nan
    return folds_df, oos, pct_profitable
