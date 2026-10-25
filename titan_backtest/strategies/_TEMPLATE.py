"""TEMPLATE for a new strategy. Copy this file, rename it, fill it in,
then register it in strategies/__init__.py.

Rules you must follow (they keep the research honest):
  1. Write the rules BEFORE running the backtest. Do not change them after
     seeing results. If you want to change them, that is a NEW experiment.
  2. Use only information available at that day's close. The engine delays
     execution by one day automatically; do not use .shift(-1) or future data.
  3. Keep the number of adjustable parameters small (3 or fewer).
  4. GRID is the complete list of parameter values you are allowed to try.
     Write it now, keep it small. Every extra combination weakens the
     statistical value of a good result.
"""

import pandas as pd

# Default parameter values, decided before testing.
PARAMS = {"example_param": 20}

# The ONLY parameter values walk-forward testing may choose from.
GRID = {"example_param": [10, 20, 40]}

# Engine settings (volatility targeting etc.). Empty dict = plain backtest.
BACKTEST_KWARGS = {}


def positions(px: pd.DataFrame, sp: pd.DataFrame, example_param=20) -> pd.DataFrame:
    """Return desired positions: one column per symbol.

    +0.5 means: hold 50% of account equity long in that symbol.
    -0.5 means: hold 50% short.  0 means flat.
    The engine handles execution delay, costs, and financing.
    """
    # --- your rules here, computed only from past data ---
    pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    return pos
