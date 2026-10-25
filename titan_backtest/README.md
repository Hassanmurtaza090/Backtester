# Titan Backtest

The backtesting engine used in our strategy research. It tests any rule-based
strategy on your H1 parquet data with realistic costs, then runs the
validation checks that decide if the strategy has a real edge or not.

## 1. Setup (Windows, PowerShell)

```powershell
cd titan_backtest
pip install -r requirements.txt
```

Needs Python 3.10 or newer.

## 2. Data

Point the engine at your data folder. Expected layout (this matches
`D:\bars\h1`):

```
D:\bars\h1\
    XAUUSD\
        year=2014.parquet
        year=2015.parquet
        ...
    EURUSD\
        ...
```

Each file needs columns: `h` (epoch ms), `o, hi, lo, c`, `spread_mean`.
The engine builds daily bars from H1 automatically and uses the real
measured spread of each symbol as the trading cost.

## 3. Run a backtest

```powershell
# trend following on gold only
python run_backtest.py --data D:\bars\h1 --symbols XAUUSD --strategy tsmom

# trend following on every symbol, with 1.5%/yr overnight financing cost
python run_backtest.py --data D:\bars\h1 --symbols ALL --strategy tsmom --financing 0.015

# dip-buying on indices, plus walk-forward validation
python run_backtest.py --data D:\bars\h1 --symbols US500,NAS100,GER40 --strategy rsi2_dip --wfa

# see what strategies exist
python run_backtest.py --list
```

## 4. What the report gives you

For every run you get, automatically:

1. Full metrics: CAGR, max drawdown, Calmar, Sharpe, Sortino, worst
   month/year, drawdown length, tail risk.
2. Year-by-year returns.
3. Benchmark: did it beat simply buying and holding the same symbols?
   (Kill test 1)
4. First half vs second half: does it work in both, or was it one lucky
   period?
5. Profit concentration: is the profit carried by a few days or one symbol?
6. Cost stress: results at 1x, 2x, 3x, 5x costs. A real edge survives 3x.
7. Random control: p-value against random trading with the same activity.
   (Kill test 2, must be below 0.05)
8. Walk-forward (`--wfa`): parameters are re-chosen each year using only
   past data, then tested on the next unseen year. At least 55% of the
   years must be profitable.

Every run is also appended to `results/run_log.csv`. Never delete that file:
the number of things we tried is part of the evidence. Trying 100 variants
and showing the best one is how fake backtests are made.

## 5. How the engine stays honest

- A signal computed today is only traded tomorrow (no look-ahead).
- Every position change pays half the real measured spread per side.
- Overnight financing/swap can be charged on everything you hold
  (`--financing 0.03` = 3% per year on exposure).
- Optional volatility targeting with a hard leverage cap.

## 6. Add your own strategy

1. Copy `strategies/_TEMPLATE.py` to `strategies/my_strategy.py`.
2. Fill in the `positions()` function: given prices, return how much of the
   account to hold in each symbol (+ long, - short, 0 flat).
3. Write `PARAMS` (defaults) and `GRID` (the only values walk-forward may
   try) BEFORE running anything. Keep 3 parameters or fewer.
4. Register it in `strategies/__init__.py`:

```python
from . import my_strategy
REGISTRY = { ..., "my_strategy": my_strategy }
```

5. Run it:

```powershell
python run_backtest.py --data D:\bars\h1 --symbols ALL --strategy my_strategy --wfa
```

## 7. The rules of the research (do not skip)

1. Define the rules first, test second. Never tune the rules after seeing
   the result and call it the same strategy.
2. A strategy must beat buy-and-hold AND random trading, after costs.
3. A result that only exists at one magic parameter value is noise.
4. If it dies at 3x costs, it is too fragile to trade.
5. A beautiful backtest is not evidence. Surviving ALL the checks above,
   plus unseen data, is evidence.

## Project layout

```
titan_backtest/
    run_backtest.py          command-line runner
    engine/
        data.py              parquet loading, H1 -> daily
        backtest.py          execution, costs, financing, vol targeting
        metrics.py           all performance statistics
        validate.py          benchmarks, random control, stress, walk-forward
        report.py            printing + run_log.csv
    strategies/
        _TEMPLATE.py         copy this to add a strategy
        tsmom.py             trend following (reference)
        sma_cross.py         MA cross (reference, killed on gold)
        rsi2_dip.py          dip buying (reference, killed on indices)
    results/
        run_log.csv          every run ever made (the honesty counter)
```
