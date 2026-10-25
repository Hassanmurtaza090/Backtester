"""Titan Backtest runner.

Examples (PowerShell):

  # one symbol
  python run_backtest.py --data D:\\bars\\h1 --symbols XAUUSD --strategy tsmom

  # several symbols
  python run_backtest.py --data D:\\bars\\h1 --symbols XAUUSD,EURUSD,US500 --strategy tsmom

  # every symbol folder under --data
  python run_backtest.py --data D:\\bars\\h1 --symbols ALL --strategy tsmom

  # add financing cost and run walk-forward validation too
  python run_backtest.py --data D:\\bars\\h1 --symbols ALL --strategy tsmom --financing 0.015 --wfa

  # list strategies
  python run_backtest.py --list
"""

import argparse
import os
import sys

import pandas as pd

from engine import data as dataio
from engine import metrics, report, validate
from engine.backtest import run_backtest
from strategies import REGISTRY


def main():
    ap = argparse.ArgumentParser(description="Titan Quant backtesting engine")
    ap.add_argument("--data", help="root folder with one sub-folder per symbol")
    ap.add_argument("--symbols", default="ALL", help="comma list, or ALL")
    ap.add_argument("--strategy", help="strategy name (see --list)")
    ap.add_argument("--start", default=None, help="evaluation start, e.g. 2015-06-01")
    ap.add_argument("--financing", type=float, default=0.0, help="yearly overnight cost on exposure, e.g. 0.015")
    ap.add_argument("--cost-mult", type=float, default=1.0, help="spread cost multiplier")
    ap.add_argument("--wfa", action="store_true", help="also run walk-forward validation")
    ap.add_argument("--no-stress", action="store_true", help="skip the cost stress table")
    ap.add_argument("--no-random", action="store_true", help="skip the random control test")
    ap.add_argument("--sims", type=int, default=300, help="random-control simulations")
    ap.add_argument("--list", action="store_true", help="list available strategies")
    args = ap.parse_args()

    if args.list or not args.strategy:
        print("Available strategies:")
        for name, mod in REGISTRY.items():
            doc = (mod.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<12} {doc}")
            print(f"  {'':<12} default params: {mod.PARAMS}")
        if not args.strategy:
            sys.exit(0)

    if args.strategy not in REGISTRY:
        sys.exit(f"Unknown strategy '{args.strategy}'. Use --list.")
    if not args.data:
        sys.exit("Please pass --data <folder>.")

    strat = REGISTRY[args.strategy]
    bt_kwargs = dict(getattr(strat, "BACKTEST_KWARGS", {}))
    bt_kwargs["financing_annual"] = args.financing
    bt_kwargs["cost_mult"] = args.cost_mult

    # ---------------- data ----------------
    symbols = None if args.symbols.upper() == "ALL" else [s.strip() for s in args.symbols.split(",")]
    print("Loading data ...")
    px, sp = dataio.load_universe(args.data, symbols=symbols)
    print(f"Loaded {px.shape[1]} symbols, {px.shape[0]} daily bars, "
          f"{px.index.min().date()} -> {px.index.max().date()}")

    # pick a default evaluation start: after warm-up
    start = args.start or str((px.index.min() + pd.Timedelta(days=420)).date())

    # ---------------- main run ----------------
    pos = strat.positions(px, sp, **strat.PARAMS)
    res = run_backtest(px, sp, pos, **bt_kwargs)
    r = res.returns[start:]

    report.hline(f"STRATEGY: {args.strategy}   params={strat.PARAMS}   "
                 f"financing={args.financing*100:.1f}%/yr  cost x{args.cost_mult}")
    s = metrics.summary(r, args.strategy)
    report.print_summary(s)
    print(f"  Trades (direction changes): {res.n_trades}")
    print(f"  Avg gross exposure: {res.avg_gross_exposure:.2f}x   avg leverage: {res.avg_leverage:.2f}x")
    report.print_yearly(metrics.yearly_table(r))

    # ---------------- benchmark ----------------
    report.hline("BENCHMARK: equal-weight buy & hold of the same symbols")
    bh = validate.buy_and_hold(px, start=start)
    report.print_summary(metrics.summary(bh, "buy_and_hold"))
    verdict_bh = "PASS" if s["sharpe"] > metrics.sharpe(bh) else "FAIL"
    print(f"\n  Kill test 1 (beats buy & hold, risk-adjusted): {verdict_bh}")

    # ---------------- halves ----------------
    report.hline("STABILITY: first half vs second half")
    print(validate.split_halves(r).to_string(index=False))

    # ---------------- concentration ----------------
    conc = validate.concentration(res, px)
    print(f"\n  Profit concentration: {conc}")

    # ---------------- cost stress ----------------
    if not args.no_stress:
        report.hline("COST STRESS (same strategy, higher costs)")
        def rerun(m):
            kw = dict(bt_kwargs)
            kw["cost_mult"] = m
            return run_backtest(px, sp, pos, **kw)
        print(validate.cost_stress(rerun, start=start).to_string(index=False))

    # ---------------- random control ----------------
    if not args.no_random:
        report.hline("RANDOM CONTROL")
        p = validate.random_control_pvalue(px, res, n_sims=args.sims, start=start)
        verdict = "PASS" if p < 0.05 else "FAIL"
        print(f"  p-value vs random trading: {p:.3f}  ({verdict}; must be < 0.05)")

    # ---------------- walk-forward ----------------
    if args.wfa:
        report.hline("WALK-FORWARD VALIDATION (params re-chosen on past data only)")
        folds, oos, pct = validate.walk_forward(
            px, sp, strat.positions, strat.GRID, backtest_kwargs=bt_kwargs
        )
        print(folds.to_string(index=False))
        if len(oos):
            print(f"\n  Folds profitable: {pct*100:.0f}%  (kill test: must be >= 55%)")
            os_ = metrics.summary(oos, "walk_forward_oos")
            print(f"  Stitched out-of-sample: CAGR {os_['cagr_pct']}%  "
                  f"MaxDD {os_['maxdd_pct']}%  Sharpe {os_['sharpe']}")

    # ---------------- log ----------------
    here = os.path.dirname(os.path.abspath(__file__))
    report.log_run(
        os.path.join(here, "results"),
        {
            "strategy": args.strategy,
            "symbols": args.symbols,
            "params": str(strat.PARAMS),
            "start": s["start"],
            "end": s["end"],
            "cagr_pct": s["cagr_pct"],
            "maxdd_pct": s["maxdd_pct"],
            "sharpe": s["sharpe"],
            "calmar": s["calmar"],
            "n_trades": res.n_trades,
            "cost_mult": args.cost_mult,
            "financing_annual": args.financing,
            "notes": "",
        },
    )
    print("\nRun appended to results/run_log.csv (this trial counter keeps us honest).")


if __name__ == "__main__":
    main()
