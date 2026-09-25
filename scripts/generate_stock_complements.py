"""
Weekly: complements for every stock on the Stock Lookup page, using complement_engine (the Complement Finder
notebook's method; up to 5 per stock, drawn from the stock's own index: TSX candidates for Canadian stocks, S&P 500
candidates for US stocks, including Nasdaq 100 names outside the S&P 500).

Reads the covered stocks from outputs/stock-lookup/data/*.json and writes outputs/stock-lookup/complements.json,
which the daily generate_stock_lookup_web.py build folds into each stock's data file. Complements barely move day to
day and this is the heaviest part of the page (every stock screened against ~600 candidates), so it runs weekly.
"""

import glob
import json
import os
import time
from datetime import datetime

import complement_engine as ce
from common_screening import load_sp500_symbols
from generate_stock_screener import download_prices

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOKUP_DIR = os.path.join(REPO_ROOT, "outputs", "stock-lookup")
OUT_PATH = os.path.join(LOOKUP_DIR, "complements.json")


def pct(v, d=4):
    return round(float(v), d)


def main():
    t0 = time.time()
    # Each stock's market ("cdn" / "us") decides which index its complements come from
    market = {}
    for f in glob.glob(os.path.join(LOOKUP_DIR, "data", "*.json")):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        market[d["t"]] = d["m"]
    anchors = sorted(market)
    sp500 = {s.replace(".", "-") for s in load_sp500_symbols()}
    universe = ce.load_universe(sp500)
    print(f"{len(anchors)} stocks to screen; {len(universe)} candidate complements "
          f"({(universe['Home'] == 'CDN').sum()} TSX, {(universe['Home'] == 'US').sum()} S&P 500)")

    years = (datetime.now() - datetime.strptime(ce.PRICE_START, "%Y-%m-%d")).days / 365.25 + 0.1
    symbols = sorted(set(anchors) | set(universe.index) | set(ce.HOME_BENCH.values()) | {ce.FX})
    print(f"Downloading {len(symbols)} price series since {ce.PRICE_START}...")
    raw = download_prices(symbols, years)
    px = ce.to_cad(raw)
    print(f"Prices through {px.index[-1]:%Y-%m-%d} ({time.time() - t0:.0f}s)")

    results, reasons = {}, {}
    for i, a in enumerate(anchors, 1):
        try:
            home = "CDN" if market[a] == "cdn" else "US"
            res, why = ce.analyze(a, px, universe[universe["Home"] == home])
        except Exception as e:                      # one bad series shouldn't sink the whole run
            res, why = None, f"error: {e}"
        if res is None:
            results[a] = {"reason": why}
            key = "under 5 years of history" if why.startswith("only") else why
            reasons[key] = reasons.get(key, 0) + 1
            continue
        s = res["a_stats"]
        results[a] = {
            "start": res["start"].strftime("%Y-%m-%d"), "end": res["end"].strftime("%Y-%m-%d"), "stress_tests": res["n_eps"],
            "alone": [pct(s["CAGR"]), pct(s["Vol"]), pct(s["Max DD"])],
            "top": [{"t": c, "n": ce.short_name(str(r["Name"])), "sector": r["Sector"], "home": r["Home"],
                     "corr": round(float(r["Worst sub-period corr"]), 2), "held": int(r["Held up"]),
                     "blend": [pct(r["Blend CAGR"]), pct(r["Blend vol"]), pct(r["Blend max DD"])]}
                    for c, r in res["top"].iterrows()],
            "paths": ce.blend_paths(a, list(res["top"].index), px),   # for the growth / drawdown chart
        }
        if i % 100 == 0:
            print(f"  {i}/{len(anchors)} ({time.time() - t0:.0f}s)")

    found = [r for r in results.values() if "top" in r]
    sizes = [len(r["top"]) for r in found]
    print(f"Complements found for {len(found)}/{len(anchors)} stocks "
          f"(5: {sizes.count(5)}, 3-4: {sum(3 <= n <= 4 for n in sizes)}, 1-2: {sum(n <= 2 for n in sizes)}); "
          f"none: {reasons}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"asof": px.index[-1].strftime("%Y-%m-%d"), "results": results}, f, separators=(",", ":"))
    print(f"Saved {OUT_PATH} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
