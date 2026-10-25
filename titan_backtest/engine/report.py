"""Printing and logging of results. Every run is appended to results/run_log.csv
so the total number of experiments is always known (needed for honest
overfitting statistics: the more things you try, the stronger the evidence
must be)."""

import csv
import os
from datetime import datetime, timezone

import pandas as pd

LOG_COLUMNS = [
    "timestamp",
    "strategy",
    "symbols",
    "params",
    "start",
    "end",
    "cagr_pct",
    "maxdd_pct",
    "sharpe",
    "calmar",
    "n_trades",
    "cost_mult",
    "financing_annual",
    "notes",
]


def log_run(results_dir: str, row: dict):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "run_log.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
        w.writerow(row)


def hline(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def print_summary(s: dict):
    order = [
        ("Period", f"{s['start']} -> {s['end']} ({s['years']} years)"),
        ("Total return", f"{s['total_return_pct']}%"),
        ("CAGR", f"{s['cagr_pct']}%"),
        ("Max drawdown", f"{s['maxdd_pct']}%"),
        ("Calmar", s["calmar"]),
        ("Sharpe", s["sharpe"]),
        ("Sortino", s["sortino"]),
        ("Volatility", f"{s['vol_pct']}%"),
        ("Longest drawdown", f"{s['longest_dd_days']} days"),
        ("Worst month", f"{s['worst_month_pct']}%"),
        ("Worst year", f"{s['worst_year_pct']}%"),
        ("Best year", f"{s['best_year_pct']}%"),
        ("Positive days", f"{s['pct_positive_days']}%"),
        ("Skew / Kurtosis", f"{s['skew']} / {s['kurtosis']}"),
        ("Daily VaR95 / CVaR95", f"{s['var95_daily_pct']}% / {s['cvar95_daily_pct']}%"),
    ]
    for k, v in order:
        print(f"  {k:<22} {v}")


def print_yearly(y: pd.Series, label="Year-by-year returns (%)"):
    print(f"\n{label}")
    for year, v in y.items():
        bar = "#" * min(40, int(abs(v)))
        sign = "+" if v >= 0 else "-"
        print(f"  {year}  {v:7.1f}  {sign}{bar}")
