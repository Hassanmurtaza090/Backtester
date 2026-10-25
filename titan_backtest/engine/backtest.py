"""Core backtest engine.

Honesty rules enforced here (not in the strategy):
  1. No look-ahead: desired positions computed from data up to day t are
     only applied from day t+1 (exec_lag=1).
  2. Costs: every change in position pays half the measured spread per side.
  3. Optional overnight financing/swap charge on gross exposure.
  4. Optional portfolio volatility targeting with a leverage cap.

A "strategy" is any function that returns a DataFrame of desired positions
(signed fractions of equity, one column per symbol). The engine does the rest.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series          # daily net returns
    gross_returns: pd.Series    # before costs
    positions: pd.DataFrame     # actual positions held (after lag + vol target)
    costs: pd.Series            # daily spread cost
    financing: pd.Series        # daily financing cost
    n_trades: int = 0
    avg_gross_exposure: float = 0.0
    avg_leverage: float = 1.0
    info: dict = field(default_factory=dict)


def run_backtest(
    px: pd.DataFrame,
    sp: pd.DataFrame,
    target_pos: pd.DataFrame,
    exec_lag: int = 1,
    cost_mult: float = 1.0,
    financing_annual: float = 0.0,
    vol_target: float | None = None,
    vol_window: int = 60,
    lev_cap: float = 3.0,
) -> BacktestResult:
    """Run a position-based backtest.

    px, sp      : price and spread panels (same shape, daily index)
    target_pos  : desired signed position per symbol, as fraction of equity,
                  computed with information available at that day's close
    exec_lag    : days between signal and execution (1 = trade next day)
    cost_mult   : multiply spread costs (for stress tests: 2x, 3x, 5x)
    financing_annual : yearly overnight cost on gross exposure (e.g. 0.03 = 3%)
    vol_target  : if set (e.g. 0.10), scale the whole portfolio so trailing
                  volatility matches the target, with leverage capped at lev_cap
    """
    if isinstance(px, pd.Series):
        px = px.to_frame()
    if isinstance(sp, pd.Series):
        sp = sp.to_frame()
    if isinstance(target_pos, pd.Series):
        target_pos = target_pos.to_frame()
    target_pos = target_pos.reindex(px.index).reindex(columns=px.columns)

    ret = px.pct_change(fill_method=None)

    # 1) execution lag: today's signal trades tomorrow
    pos = target_pos.shift(exec_lag)

    # 2) optional portfolio-level volatility targeting
    lev = pd.Series(1.0, index=px.index)
    if vol_target is not None:
        r_unlev = (pos * ret).sum(axis=1)
        realized = r_unlev.rolling(vol_window).std() * np.sqrt(252)
        lev = (vol_target / realized).clip(upper=lev_cap).shift(1)
        pos = pos.mul(lev, axis=0)

    pos = pos.fillna(0.0)

    # 3) P&L
    gross = (pos * ret).sum(axis=1)

    # 4) spread cost: half spread per side on every position change
    frac_spread = (sp / px).fillna(0)
    turnover = pos.diff().abs().fillna(0)
    cost = (turnover * frac_spread * 0.5).sum(axis=1) * cost_mult

    # 5) financing on gross exposure
    gross_exp = pos.abs().sum(axis=1)
    fin = gross_exp * (financing_annual / 252.0)

    net = (gross - cost - fin).fillna(0)

    n_trades = int((np.sign(pos).diff().abs() > 0).sum().sum())
    return BacktestResult(
        returns=net,
        gross_returns=gross.fillna(0),
        positions=pos,
        costs=cost,
        financing=fin,
        n_trades=n_trades,
        avg_gross_exposure=float(gross_exp.mean()),
        avg_leverage=float(lev.mean()) if vol_target else 1.0,
        info={"cost_mult": cost_mult, "financing_annual": financing_annual},
    )
