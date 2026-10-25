"""Data loading for Titan Backtest.

Expected layout (same as D:\\bars\\h1 on your machine):

    DATA_ROOT/
        XAUUSD/
            year=2014.parquet
            year=2015.parquet
            ...
        EURUSD/
            year=2014.parquet
            ...

Each parquet file has columns:
    h (epoch ms), o, hi, lo, c, bid_c, ask_c, n_ticks, spread_mean

A flat folder with year=YYYY.parquet files (single symbol) also works.
"""

import glob
import os

import pandas as pd


def load_symbol_h1(folder: str) -> pd.DataFrame:
    """Load every year=*.parquet (or year_*.parquet) file in one symbol folder.

    Returns an H1 DataFrame indexed by UTC timestamp.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "year=*.parquet"))
        + glob.glob(os.path.join(folder, "year_*.parquet"))
        + glob.glob(os.path.join(folder, "year*.parquet"))
    )
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No year*.parquet files found in {folder}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df.sort_values("h").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["h"], unit="ms", utc=True)
    df = df.set_index("dt")
    df = df[~df.index.duplicated()]
    return df


def h1_to_daily(h1: pd.DataFrame) -> pd.DataFrame:
    """Resample H1 bars to daily bars (UTC calendar days, weekends dropped)."""
    daily = h1.resample("1D").agg(
        o=("o", "first"),
        hi=("hi", "max"),
        lo=("lo", "min"),
        c=("c", "last"),
        spread=("spread_mean", "mean"),
    )
    daily = daily.dropna(subset=["c"])
    daily = daily[daily.index.dayofweek < 5]
    return daily


def load_universe(root: str, symbols=None, timeframe: str = "daily"):
    """Load many symbols into aligned price and spread panels.

    root     : folder that contains one sub-folder per symbol
    symbols  : list of symbol names, or None = every sub-folder
    timeframe: 'daily' (default) or 'h1'

    Returns (px, sp): two DataFrames, columns = symbols.
    """
    if symbols is None:
        symbols = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
    closes, spreads = {}, {}
    for s in symbols:
        h1 = load_symbol_h1(os.path.join(root, s))
        if timeframe == "daily":
            d = h1_to_daily(h1)
            closes[s] = d["c"]
            spreads[s] = d["spread"]
        else:
            closes[s] = h1["c"]
            spreads[s] = h1["spread_mean"]
    px = pd.DataFrame(closes).sort_index()
    sp = pd.DataFrame(spreads).reindex(px.index)
    if timeframe == "daily":
        px = px[px.index.dayofweek < 5]
        sp = sp.reindex(px.index)
    return px, sp


def load_single(path: str, timeframe: str = "daily"):
    """Load one symbol folder. Returns (px, sp) as 1-column DataFrames."""
    h1 = load_symbol_h1(path)
    name = os.path.basename(os.path.normpath(path)) or "SYMBOL"
    if timeframe == "daily":
        d = h1_to_daily(h1)
        px = d[["c"]].rename(columns={"c": name})
        sp = d[["spread"]].rename(columns={"spread": name})
    else:
        px = h1[["c"]].rename(columns={"c": name})
        sp = h1[["spread_mean"]].rename(columns={"spread_mean": name})
    return px, sp
