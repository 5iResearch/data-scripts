"""
Leader Selloff study - the research companion to generate_peak_fear_report.py.

Tests the simplified buy signal from fear_engine.py: a historic market leader
whose weekly AND monthly RSI are at statistically low points vs its own
history, within 20 trading days of a stock-level volatility spike (its own
equivalent of VIX jumping to 30+).

  1. Pipeline check: forward SPY returns by market gauge / VIX regime.
  2. Event study: every signal (max one per ticker per 63 trading days),
     forward excess return vs SPY at 1/3/6/12 months with entry at the NEXT
     day's close, hit rate and max adverse excursion - against baselines:
       A  leaders on any day
       B  the same RSI + volatility conditions without the leadership filter
       D  the whole universe on any day
     plus ablations that drop one condition at a time, to show what each adds.
  3. Walk-forward: thresholds chosen on 2005-2014 only, judged on 2015-present.

Survivorship caveat: yfinance only has today's S&P 500 members, so failed
former leaders are missing and absolute returns are optimistic. Signal-vs-
baseline comparisons (same biased universe on both sides) are more
trustworthy than the levels.

Usage:
  python scripts/fear_study.py                    # uses data/cache if present
  python scripts/fear_study.py --refresh          # re-download prices
  python scripts/fear_study.py --check-lookahead  # point-in-time assertion, then exit
  python scripts/fear_study.py --write-config     # freeze chosen thresholds + base rates
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fear_engine as fe

OUTPUT_DIR = os.path.join(fe.REPO_ROOT, "outputs", "fear-study")

EVENT_START = pd.Timestamp("2005-06-01")  # 5y leadership + 5y of monthly bars need warmup from 2000
IS_END = pd.Timestamp("2014-12-31")
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
MAE_DAYS = 63
GRID_W = [0.05, 0.10, 0.20]
GRID_M = [0.10, 0.20, 0.30]
GRID_VIX = [25, 30, 35]
MIN_IS_EVENTS = 100
N_BOOT = 1000
PERIODS = {"Full 2005+": None, "In-sample 2005-14": "IS", "Out-of-sample 2015+": "OOS"}

DGRAY, MGRAY, LGRAY = "#1C1C1E", "#2C2C2E", "#3A3A3C"
TEXT, SUBTEXT = "#E5E5EA", "#8E8E93"
GREEN, RED, ORANGE, BLUE, YELLOW, PURPLE = "#2ECC71", "#E74C3C", "#C67A29", "#1F79BE", "#F4D03F", "#9B59B6"


# ── Forward returns ───────────────────────────────────────────────────────────
def forward_frames(close, bench, ppy=252):
    """Excess return vs bench from the NEXT day's close (the signal needs day
    t's close, so t+1 is the first realistic fill), plus max adverse excursion.
    Horizons are trading days; `ppy` rescales them for calendar-day assets."""
    k = ppy / 252
    b = bench.reindex(close.index).ffill()  # crypto weekends carry SPY's last close
    entry, b_entry = close.shift(-1), b.shift(-1)
    fwd = {}
    for lbl, h in HORIZONS.items():
        h = round(h * k)
        fwd[lbl] = (close.shift(-1 - h) / entry - 1).sub(b.shift(-1 - h) / b_entry - 1, axis=0)
    path_min = close.shift(-2).iloc[::-1].rolling(round(MAE_DAYS * k), min_periods=1).min().iloc[::-1]
    mae = (path_min / entry - 1).clip(upper=0)
    return fwd, mae


def universe_ctx(name, s, bench, ppy=252):
    fwd, mae = forward_frames(s["close"], bench, ppy)
    return dict(name=name, s=s, fwd=fwd, mae=mae)


def period_mask(dates, period):
    keep = dates >= EVENT_START
    if period == "IS":
        keep &= dates <= IS_END
    elif period == "OOS":
        keep &= dates > IS_END
    return keep


def in_period(ev, period):
    if not period:
        return ev
    return ev[ev["period"].str.startswith("In" if period == "IS" else "Out")]


# ── Events ────────────────────────────────────────────────────────────────────
def build_events(ctx, signal, label, gauge=None, cooldown=63, enrich=True):
    ev = fe.extract_events(signal, cooldown)
    ev = ev[ev["date"] >= EVENT_START].reset_index(drop=True)
    ev["signal"], ev["universe"] = label, ctx["name"]
    s = ctx["s"]
    i = ev["i"].to_numpy()
    j = s["close"].columns.get_indexer(ev["ticker"])

    def pick(frame):
        return frame.to_numpy()[i, j] if len(ev) else np.array([])

    for lbl in HORIZONS:
        ev[f"xs_{lbl}"] = pick(ctx["fwd"][lbl])
    ev["mae"] = pick(ctx["mae"])
    ev["period"] = np.where(ev["date"] <= IS_END, "In-sample 2005-14", "Out-of-sample 2015+")
    if not enrich or ev.empty:
        return ev

    for k in ("rel5", "w_rsi", "m_rsi", "rv", "drawdown", "move21", "mkt_part", "sec_part", "spec_part"):
        ev[k] = pick(s[k])
    ev["driver"] = fe.driver_labels(ev["mkt_part"], ev["sec_part"], ev["spec_part"], ev["move21"], s["has_sector"])
    if gauge is not None:
        gr = gauge.reindex(ev["date"])
        ev["mkt_gauge"] = gr["gauge"].to_numpy()
        ev["regime"] = gr["regime"].to_numpy()
        ev["credit_stress"] = gr["credit_stress"].to_numpy()
    return ev


# ── Statistics ────────────────────────────────────────────────────────────────
def boot_ci(ev, col):
    """90% CI of the mean, resampling whole calendar months - events cluster
    in crises (2008, 2020), so resampling single events would overstate precision."""
    x = ev[[col, "date"]].dropna()
    if len(x) < 20:
        return (np.nan, np.nan)
    g = x.groupby(x["date"].dt.to_period("M"))[col].agg(["sum", "count"])
    idx = np.random.default_rng(0).integers(0, len(g), size=(N_BOOT, len(g)))
    means = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return tuple(np.percentile(means, [5, 95]))


def summarize(ev, label, period=None):
    ev = in_period(ev, period)
    row = {"Signal": label, "Basis": "events", "N": len(ev), "Dates": ev["date"].nunique() if len(ev) else 0}
    for lbl in HORIZONS:
        x = ev[f"xs_{lbl}"].dropna() if len(ev) else pd.Series(dtype=float)
        row[f"{lbl} mean"] = x.mean() if len(x) else np.nan
        row[f"{lbl} hit"] = (x > 0).mean() if len(x) else np.nan
    row["6m median"] = ev["xs_6m"].median() if len(ev) else np.nan
    row["6m CI"] = boot_ci(ev, "xs_6m") if len(ev) else (np.nan, np.nan)
    row["MAE"] = ev["mae"].mean() if len(ev) else np.nan
    return row


def cell_summary(ctx, mask, label, period=None):
    """Base rate over every (day, ticker) cell in `mask` - no cooldown, no CI."""
    dates = ctx["s"]["close"].index
    m = mask.to_numpy(bool) & period_mask(dates, period)[:, None]
    row = {"Signal": label, "Basis": "stock-days", "N": int(m.sum()), "Dates": None}
    for lbl in HORIZONS:
        v = ctx["fwd"][lbl].to_numpy()[m]
        v = v[np.isfinite(v)]
        row[f"{lbl} mean"] = v.mean() if len(v) else np.nan
        row[f"{lbl} hit"] = (v > 0).mean() if len(v) else np.nan
        if lbl == "6m":
            row["6m median"] = np.median(v) if len(v) else np.nan
    row["6m CI"] = (np.nan, np.nan)
    mae = ctx["mae"].to_numpy()[m]
    row["MAE"] = np.nanmean(mae) if len(mae) else np.nan
    return row


def split_rows(ev, col, order=None):
    groups = order or sorted(ev[col].dropna().unique())
    return [summarize(ev[ev[col] == g], str(g)) for g in groups if (ev[col] == g).any()]


# ── Pipeline check: market gauge vs forward SPY ───────────────────────────────
def gauge_study(gauge, spy):
    spy = spy.reindex(gauge.index)
    df = pd.DataFrame({lbl: spy.shift(-h) / spy - 1 for lbl, h in HORIZONS.items()})
    df["gauge"] = gauge["gauge"]
    df = df.dropna(subset=["gauge"])
    df["bucket"] = pd.cut(df["gauge"], np.arange(0, 101, 10), labels=[f"{a}-{a + 10}" for a in range(0, 100, 10)], include_lowest=True)
    rows = []

    def add(label, sub):
        r = {"Condition": label, "Days": len(sub)}
        for lbl in HORIZONS:
            x = sub[lbl].dropna()
            r[f"SPY {lbl}"] = x.mean() if len(x) else np.nan
        x = sub["6m"].dropna()
        r["6m hit"] = (x > 0).mean() if len(x) else np.nan
        rows.append(r)

    add("All days", df)
    for b in df["bucket"].cat.categories:
        add(f"Gauge {b}", df[df["bucket"] == b])
    reg = gauge["regime"].reindex(df.index)
    cs = gauge["credit_stress"].reindex(df.index).fillna(False).astype(bool)
    add("Elevated (gauge >= 70, not inverted)", df[reg == "Elevated"])
    add("Panic (VIX > VIX3M), credit calm", df[(reg == "Panic") & ~cs])
    add("Panic (VIX > VIX3M) + credit stress", df[(reg == "Panic") & cs])
    return pd.DataFrame(rows)


# ── Look-ahead guard ──────────────────────────────────────────────────────────
def check_lookahead(stocks, spy, sec_close, cfg, vol_pct, n_tickers=40, n_dates=5):
    """Recompute everything on data truncated at t and require the value AT t
    to match the full-history run exactly. Any future leakage breaks equality."""
    rng = np.random.default_rng(1)
    counts = stocks["Close"].notna().sum()
    pool = counts[counts > 4000].index.tolist()
    sub = sorted(rng.choice(pool, size=min(n_tickers, len(pool)), replace=False).tolist())
    sp = fe.subset_panel(stocks, sub)
    sec = sec_close[sub]
    full = fe.compute_signal(sp, spy, sec, cfg)
    full_c = fe.conditions(full, cfg, vol_pct)
    idx = sp["Close"].index
    positions = sorted(rng.integers(1800, len(idx) - 5, size=n_dates))
    failures = []
    for pos in positions:
        t = idx[pos]
        trunc = {f: sp[f].loc[:t] for f in fe.FIELDS}
        part = fe.compute_signal(trunc, spy.loc[:t], sec.loc[:t], cfg)
        part_c = fe.conditions(part, cfg, vol_pct)
        pairs = [(k, full[k], part[k]) for k in ("w_rsi", "m_rsi", "rv", "rel5", "leader", "market_driven")]
        pairs += [(k, full_c[k], part_c[k]) for k in ("w_thr", "m_thr", "v_thr", "weekly", "monthly", "vol_spike", "buy")]
        for name, a, b in pairs:
            va, vb = a.loc[t].to_numpy(dtype=float), b.loc[t].to_numpy(dtype=float)
            close = np.isclose(va, vb, rtol=1e-9, atol=1e-9, equal_nan=True)
            if not close.all():
                failures.append(f"{t.date()} {name}: {[sub[k] for k in np.flatnonzero(~close)][:5]}")
        print(f"  {t.date()}: checked {len(pairs)} fields x {len(sub)} tickers")
    if failures:
        print("LOOK-AHEAD CHECK FAILED:")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)
    print(f"LOOK-AHEAD CHECK PASSED ({len(positions)} dates x {len(sub)} tickers)")


# ── HTML helpers ──────────────────────────────────────────────────────────────
def colored(x, digits=1):
    if x is None or pd.isna(x):
        return "&ndash;"
    return f'<span class="{"pos" if x > 0 else "neg"}">{x * 100:+.{digits}f}%</span>'


def plain_pct(x, signed=False):
    if x is None or pd.isna(x):
        return "&ndash;"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.0f}%"


def summary_table(rows):
    out = []
    for r in rows:
        o = {"Signal": r["Signal"], "Basis": r["Basis"], "N": f"{r['N']:,}",
             "Dates": f"{r['Dates']:,}" if r["Dates"] is not None else "&ndash;"}
        for lbl in HORIZONS:
            o[f"{lbl} excess"] = colored(r[f"{lbl} mean"])
            o[f"{lbl} hit"] = plain_pct(r[f"{lbl} hit"])
        o["6m median"] = colored(r["6m median"])
        lo, hi = r["6m CI"]
        o["6m mean 90% CI"] = "&ndash;" if pd.isna(lo) else f"{lo * 100:+.1f}% to {hi * 100:+.1f}%"
        o["Avg MAE (3m)"] = plain_pct(r["MAE"], signed=True)
        out.append(o)
    return '<div class="wrap">' + pd.DataFrame(out).to_html(escape=False, index=False, classes="tbl", border=0) + "</div>"


def fig_to_div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def note(text):
    return f'<div class="note">{text}</div>'


def base_layout(fig, title, height=460):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=height, margin=dict(l=70, r=40, t=70, b=60),
        legend=dict(bgcolor=LGRAY, bordercolor=LGRAY, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY, zeroline=True, zerolinecolor=SUBTEXT)
    return fig


def chart_gauge(table):
    buckets = table[table["Condition"].str.startswith("Gauge")]
    fig = go.Figure()
    for lbl, color in zip(["1m", "3m", "6m", "12m"], [BLUE, ORANGE, GREEN, PURPLE]):
        fig.add_trace(go.Bar(x=buckets["Condition"].str.replace("Gauge ", ""), y=buckets[f"SPY {lbl}"] * 100,
                             name=f"SPY {lbl} fwd", marker_color=color,
                             hovertemplate=f"%{{x}}: %{{y:.1f}}%<extra>{lbl}</extra>"))
    all_days = table[table["Condition"] == "All days"].iloc[0]
    fig.add_hline(y=all_days["SPY 6m"] * 100, line_dash="dot", line_color=GREEN,
                  annotation_text="all-days 6m avg", annotation_font_color=GREEN)
    base_layout(fig, "Forward SPY return by Market Fear Gauge bucket (0 = calm, 100 = panic)")
    fig.update_layout(barmode="group", xaxis_title="Market Fear Gauge", yaxis_title="Avg forward return (%)")
    return fig


def chart_by_year(ev, title):
    ev = ev.dropna(subset=["date"]).copy()
    ev["year"] = ev["date"].dt.year
    g = ev.groupby("year").agg(n=("ticker", "size"), xs=("xs_6m", "mean")).reset_index()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                        subplot_titles=["Signals per year", "Avg 6m excess return of that year's signals"])
    fig.add_trace(go.Bar(x=g["year"], y=g["n"], marker_color=BLUE, name="signals",
                         hovertemplate="%{x}: %{y} signals<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=g["year"], y=g["xs"] * 100, name="avg 6m excess",
                         marker_color=[GREEN if v > 0 else RED for v in g["xs"].fillna(0)],
                         hovertemplate="%{x}: %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_vline(x=IS_END.year + 0.5, line_dash="dash", line_color=YELLOW)
    base_layout(fig, title, height=520)
    fig.update_layout(showlegend=False)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    return fig


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Leader Selloff Study</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E5E5EA; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E5E5EA; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin-top: 6px; max-width: 1050px; line-height: 1.5; }}
  h3 {{ margin: 18px 32px 4px; font-size: 15px; color: #C67A29; }}
  .note {{ color: #AEAEB2; font-size: 13px; margin: 8px 32px; max-width: 1050px; line-height: 1.55; }}
  .wrap {{ overflow-x: auto; margin: 8px 32px; }}
  table.tbl {{ border-collapse: collapse; font-size: 12.5px; white-space: nowrap; }}
  .tbl th {{ background: #2C2C2E; color: #E5E5EA; padding: 6px 10px; text-align: right; border-bottom: 1px solid #3A3A3C; }}
  .tbl td {{ padding: 5px 10px; text-align: right; border-bottom: 1px solid #2C2C2E; }}
  .tbl td:first-child, .tbl th:first-child {{ text-align: left; }}
  .tbl tr.chosen td {{ background: #3a3220; }}
  .pos {{ color: #2ECC71; }} .neg {{ color: #E74C3C; }}
  .verdict {{ margin: 20px 32px; padding: 14px 18px; border-radius: 6px; border: 1px solid; max-width: 1050px; line-height: 1.55; }}
  .verdict.pass {{ border-color: #2ECC71; background: rgba(46,204,113,0.08); }}
  .verdict.fail {{ border-color: #E74C3C; background: rgba(231,76,60,0.08); }}
  @media (max-width: 600px) {{ .section, header {{ padding-left: 16px; padding-right: 16px; }}
    .note, .wrap, h3, .verdict {{ margin-left: 16px; margin-right: 16px; }} }}
</style>
</head>
<body>
<header>
  <h1>Leader Selloff &mdash; Event Study</h1>
  <div class="meta">Generated {date_str} &middot; S&amp;P 500 + Nasdaq-100 members, BTC/ETH, sector/theme ETFs &middot; prices since 2000, signals since mid-2005 &middot; excess returns vs SPY from next-day close</div>
</header>
{body}
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download prices instead of using data/cache")
    ap.add_argument("--check-lookahead", action="store_true", help="run the point-in-time check and exit")
    ap.add_argument("--write-config", action="store_true", help="write chosen thresholds + base rates to data/fear_model_config.json")
    args = ap.parse_args()
    t0 = time.time()

    cfg = dict(fe.DEFAULT_CONFIG)  # the study always starts from defaults, never from a frozen config
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading universes...")
    sp_tickers, sp_names, sp_sectors = fe.us_stock_universe()  # S&P 500 + Nasdaq-100
    etf_tickers, etf_names, etf_groups = fe.etf_universe()
    crypto_tickers, crypto_names, _ = fe.crypto_universe()
    sector_etfs = sorted(set(fe.SECTOR_ETF_BY_GICS.values()))
    panel = fe.load_panel(sp_tickers + etf_tickers + sector_etfs + [fe.BENCH], fe.HISTORY_START,
                          cache_name="fear_study", refresh=args.refresh)
    # separate panel: crypto's 7-day calendar must not add weekend rows to the stocks
    crypto = fe.load_panel(crypto_tickers, fe.HISTORY_START, cache_name="fear_study_crypto", refresh=args.refresh)
    spy = panel["Close"][fe.BENCH]
    stocks = fe.subset_panel(panel, sp_tickers)
    sec_close = fe.sector_close_frame(stocks["Close"].columns, sp_sectors,
                                      panel["Close"][[e for e in sector_etfs if e in panel["Close"].columns]])

    vol_pcts = {lvl: fe.vol_percentile(lvl) for lvl in GRID_VIX}
    for lvl, p in vol_pcts.items():
        print(f"  VIX closed below {lvl} on {p * 100:.1f}% of days since 1990 -> stock 'VIX {lvl}' = its own {p * 100:.1f}th pct vol")

    if args.check_lookahead:
        check_lookahead(stocks, spy, sec_close, cfg, vol_pcts[cfg["vix_equiv_level"]])
        return

    print(f"Computing inputs for {stocks['Close'].shape[1]} stocks...")
    s_s = fe.compute_signal(stocks, spy, sec_close, cfg)
    etfs = fe.subset_panel(panel, etf_tickers)
    print(f"Computing inputs for {etfs['Close'].shape[1]} ETFs...")
    s_e = fe.compute_signal(etfs, spy, None, cfg)
    print("Building market fear gauge...")
    gauge = fe.market_gauge(stocks["Close"], fe.HISTORY_START)
    print("Computing inputs for crypto...")
    s_c = fe.compute_signal(crypto, spy, None, fe.crypto_config(cfg))
    ctx_s = universe_ctx("US stocks (S&P 500 + Nasdaq-100)", s_s, spy)
    ctx_c = universe_ctx("Crypto (BTC, ETH)", s_c, spy, fe.CRYPTO_PPY)
    ctx_e = universe_ctx("Sector/theme ETFs", s_e, spy)
    ctxs = {c["name"]: c for c in (ctx_s, ctx_c, ctx_e)}
    print(f"  ready in {time.time() - t0:.0f}s")

    # ── walk-forward grid (stocks), thresholds picked on in-sample only ──
    print("Walk-forward grid...")
    idx = s_s["close"].index
    wk = {q: s_s["w_rsi"] <= fe.own_quantile(s_s["w_bars"], s_s["w_key"], idx, q, fe.W_MIN_BARS) for q in GRID_W}
    mo = {q: s_s["m_rsi"] <= fe.own_quantile(s_s["m_bars"], s_s["m_key"], idx, q, fe.M_MIN_BARS) for q in GRID_M}
    vs = {}
    for lvl in GRID_VIX:
        thr = s_s["rv"].expanding(min_periods=fe.VOL_MIN_DAYS).quantile(vol_pcts[lvl])
        vs[lvl] = (s_s["rv"] >= thr).astype(float).rolling(cfg["vol_lookback"], min_periods=1).max() > 0
    grid = []
    for w in GRID_W:
        for m in GRID_M:
            for lvl in GRID_VIX:
                ev = build_events(ctx_s, s_s["leader"] & wk[w] & mo[m] & vs[lvl], "", cooldown=cfg["cooldown"], enrich=False)
                r = {"weekly_pct": w, "monthly_pct": m, "vix_equiv": lvl}
                for p, key in (("IS", "is"), ("OOS", "oos")):
                    s = summarize(ev, "", p)
                    r[f"{key}_n"] = s["N"]
                    for lbl in ("3m", "6m", "12m"):
                        r[f"{key}_{lbl}"] = s[f"{lbl} mean"]
                    r[f"{key}_6m_hit"] = s["6m hit"]
                r["is_obj"] = np.nanmean([r["is_3m"], r["is_6m"]])
                grid.append(r)
    grid = pd.DataFrame(grid)
    eligible = grid[grid["is_n"] >= MIN_IS_EVENTS]
    best = (eligible if len(eligible) else grid).sort_values("is_obj", ascending=False).iloc[0]
    chosen = dict(cfg, w_rsi_pct=float(best["weekly_pct"]), m_rsi_pct=float(best["monthly_pct"]),
                  vix_equiv_level=int(best["vix_equiv"]))
    vol_star = vol_pcts[chosen["vix_equiv_level"]]
    print(f"  chosen on 2005-14: weekly <= {chosen['w_rsi_pct']:.0%} pct, monthly <= {chosen['m_rsi_pct']:.0%} pct, "
          f"vol spike = VIX {chosen['vix_equiv_level']} equivalent")
    del wk, mo, vs

    # ── event sets at the chosen thresholds, with ablations ──
    print("Building event sets...")
    main_label = "Buy signal: all four conditions (chosen)"
    b_label = "B: RSI + vol conditions, no leader filter"
    events, headline = [], {}
    for ctx in (ctx_s, ctx_c, ctx_e):
        s = ctx["s"]
        ucfg = fe.crypto_config(chosen) if ctx is ctx_c else chosen
        c = fe.conditions(s, ucfg, vol_star)
        L, W, M, V = c["leader"], c["weekly"], c["monthly"], c["vol_spike"]
        sets = {
            main_label: L & W & M & V,
            "Without vol spike": L & W & M,
            "Without monthly RSI": L & W & V,
            "Without weekly RSI": L & M & V,
            b_label: W & M & V,
        }
        ev_sets = {k: build_events(ctx, v, k, gauge, ucfg["cooldown"]) for k, v in sets.items()}
        events.extend(ev_sets.values())
        live = s["close"].notna()
        headline[ctx["name"]] = {"sets": ev_sets, "A": L & live, "D": live}

    gtable = gauge_study(gauge, spy)

    def headline_rows(name, period):
        h, ctx = headline[name], ctxs[name]
        rows = [summarize(ev, k, period) for k, ev in h["sets"].items()]
        rows += [cell_summary(ctx, h["A"], "A: leaders, any day", period),
                 cell_summary(ctx, h["D"], "D: whole universe, any day", period)]
        return rows

    # ── verdict: chosen signal vs baselines A and B, out-of-sample, stocks ──
    parts = []
    oos = {r["Signal"]: r for r in headline_rows(ctx_s["name"], "OOS")}
    m, A, B = oos[main_label], oos["A: leaders, any day"], oos[b_label]
    checks = [(h, m[f"{h} mean"], A[f"{h} mean"], B[f"{h} mean"]) for h in ("3m", "6m")]
    passed = all(pd.notna(x) and x > a and x > b for _, x, a, b in checks)
    detail = "; ".join(f"{h}: signal {x * 100:+.1f}% vs A {a * 100:+.1f}% / B {b * 100:+.1f}%" for h, x, a, b in checks)
    lo, hi = m["6m CI"]
    parts.append(
        f'<div class="verdict {"pass" if passed else "fail"}"><b>{"PASS" if passed else "NOT PROVEN"}</b> &mdash; '
        f"out-of-sample (2015+), US stocks, {m['N']:,} signals on {m['Dates']:,} dates. The chosen signal "
        f"(weekly RSI &le; own {chosen['w_rsi_pct']:.0%} pct, monthly RSI &le; own {chosen['m_rsi_pct']:.0%} pct, "
        f"vol spike = VIX {chosen['vix_equiv_level']} equivalent, historic leader) "
        f"{'beats' if passed else 'does not beat'} both baselines (A: leaders on any day, B: same conditions without "
        f"the leader filter) on 3m and 6m excess return. {detail}. 6m mean 90% CI "
        f"{'n/a' if pd.isna(lo) else f'{lo * 100:+.1f}% to {hi * 100:+.1f}%'}.</div>"
    )
    parts.append(note(
        "<b>How to read this.</b> Every number is an average <i>excess</i> return vs SPY, entered at the close the day "
        "<i>after</i> the signal. Signals are capped at one per ticker per 63 trading days. \"Hit\" = share that beat SPY. "
        "MAE = average worst drawdown from entry within 3 months (how much pain came first). The 90% CI resamples whole "
        "months, because signals cluster in crises. <b>Survivorship caveat:</b> the universe is today's S&amp;P 500, so "
        "former leaders that failed are missing; absolute levels are optimistic, and the comparisons against baselines "
        "are the part to trust."))

    parts.append(section_header("1. Market regime check: forward SPY returns",
                                "Context for the stock signal &mdash; Panic = VIX term structure inverted (VIX above VIX3M)"))
    parts.append(fig_to_div(chart_gauge(gtable)))
    gt = gtable.copy()
    for lbl in HORIZONS:
        gt[f"SPY {lbl}"] = gt[f"SPY {lbl}"].map(colored)
    gt["6m hit"] = gt["6m hit"].map(plain_pct)
    gt["Days"] = gt["Days"].map(lambda v: f"{v:,}")
    parts.append('<div class="wrap">' + gt.to_html(escape=False, index=False, classes="tbl", border=0) + "</div>")

    for name in (ctx_s["name"], ctx_c["name"], ctx_e["name"]):
        if name == ctx_c["name"]:
            parts.append(section_header(f"2. Signal vs baselines &mdash; {name}",
                                        "Only two coins with short histories (BTC 2014+, ETH 2017+; 5 years of monthly "
                                        "bars needed first) &mdash; a handful of signals, illustrative rather than proof. "
                                        "Same rules on a 7-day calendar; leader = beat SPY over 5 years."))
            for pname, p in PERIODS.items():
                parts.append(f"<h3>{pname}</h3>")
                parts.append(summary_table(headline_rows(name, p)))
            continue
        parts.append(section_header(f"2. Signal vs baselines &mdash; {name}",
                                    "The \"Without ...\" rows drop one condition at a time: if a row does as well as the full "
                                    "signal, that condition isn't pulling its weight"))
        for pname, p in PERIODS.items():
            parts.append(f"<h3>{pname}</h3>")
            parts.append(summary_table(headline_rows(name, p)))

    parts.append(section_header("3. Walk-forward threshold grid (US stocks)",
                                f"Chosen on 2005-14 only (highest avg of 3m and 6m excess, min {MIN_IS_EVENTS} signals); "
                                f"2015+ shown untouched. Percentiles are vs each stock's own history."))
    g = grid.copy()
    g["weekly_pct"] = g["weekly_pct"].map(lambda v: f"&le; {v:.0%}")
    g["monthly_pct"] = g["monthly_pct"].map(lambda v: f"&le; {v:.0%}")
    g["vix_equiv"] = g["vix_equiv"].map(lambda v: f"VIX {v} ({vol_pcts[v] * 100:.0f}th pct)")
    for col in [col for col in g.columns if col.endswith(("_3m", "_6m", "_12m")) or col == "is_obj"]:
        g[col] = g[col].map(colored)
    for col in ("is_6m_hit", "oos_6m_hit"):
        g[col] = g[col].map(plain_pct)
    g = g.rename(columns={"weekly_pct": "Weekly RSI", "monthly_pct": "Monthly RSI", "vix_equiv": "Vol spike",
                          "is_n": "IS signals", "is_3m": "IS 3m", "is_6m": "IS 6m", "is_12m": "IS 12m", "is_6m_hit": "IS 6m hit",
                          "oos_n": "OOS signals", "oos_3m": "OOS 3m", "oos_6m": "OOS 6m", "oos_12m": "OOS 12m",
                          "oos_6m_hit": "OOS 6m hit", "is_obj": "IS objective"})
    head, body = g.to_html(escape=False, index=False, classes="tbl", border=0).split("<tbody>")
    rows_html = body.split("<tr>")  # rows_html[k + 1] is data row k
    k = int(best.name)
    body = "<tr>".join(rows_html[: k + 1]) + '<tr class="chosen">' + "<tr>".join(rows_html[k + 1 :])
    parts.append('<div class="wrap">' + head + "<tbody>" + body + "</div>")
    parts.append(note("Highlighted row = chosen. A robust signal shows a broad plateau of similar results across nearby "
                      "thresholds; one isolated winner is a sign of overfitting."))

    ev_main = headline[ctx_s["name"]]["sets"][main_label].copy()
    parts.append(section_header("4. Diagnostics (US stocks, chosen signal)"))
    parts.append("<h3>What drove the 21-day drop? (market / sector / stock-specific, betas from the prior year)</h3>")
    for pname, p in PERIODS.items():
        parts.append(f'<div class="note"><b>{pname}</b></div>')
        parts.append(summary_table(split_rows(in_period(ev_main, p), "driver", ["Market", "Sector", "Specific"])))
    parts.append("<h3>Market regime on the signal date</h3>")
    ev_main["regime_x_credit"] = ev_main["regime"].astype(str) + np.where(ev_main["credit_stress"] == True, " + credit stress", "")
    order = ["Calm", "Elevated", "Panic", "Calm + credit stress", "Elevated + credit stress", "Panic + credit stress"]
    parts.append(summary_table(split_rows(ev_main, "regime_x_credit", order)))

    parts.append(section_header("5. When did the signals happen?", "Clustering check &mdash; yellow line = in-sample / out-of-sample split"))
    parts.append(fig_to_div(chart_by_year(ev_main, "Buy signal &mdash; US stocks")))

    parts.append(note(
        f"Run time {time.time() - t0:.0f}s. Definitions: weekly and monthly RSI(14), live (current bar = today's close), "
        f"compared with every completed bar in the name's history (min {fe.W_MIN_BARS} weeks / {fe.M_MIN_BARS} months). "
        f"Vol spike = 20-day Garman-Klass volatility at or above the percentile of its own history (min 3 years) that the "
        f"chosen VIX level is of VIX's history since 1990 ({vol_star * 100:.1f}th), within the last {cfg['vol_lookback']} "
        f"trading days. Leader = beat SPY over the {cfg['lead_years']} years ending {cfg['lead_lag']} trading days ago and in "
        f"the top {cfg['lead_top_pct']:.0%} of its universe on that measure. Research only &mdash; not a recommendation."))

    today = datetime.now()
    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body="\n".join(parts))
    out_html = os.path.join(OUTPUT_DIR, "Fear_Study_Report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_html}")

    all_ev = pd.concat([e for e in events if len(e)], ignore_index=True).drop(columns=["i"])
    all_ev.insert(2, "name", all_ev["ticker"].map({**sp_names, **etf_names, **crypto_names}))
    out_csv = os.path.join(OUTPUT_DIR, "fear_events.csv")
    all_ev.to_csv(out_csv, index=False, float_format="%.5f")
    print(f"Saved: {out_csv} ({len(all_ev):,} events)")
    grid.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_grid.csv"), index=False, float_format="%.5f")
    print(f"  verdict: {'PASS' if passed else 'NOT PROVEN'} - {detail}")

    if args.write_config:
        def rate(ev):
            r = summarize(ev, "")
            return {"n": int(r["N"]), "3m_mean": r["3m mean"], "6m_mean": r["6m mean"],
                    "3m_hit": r["3m hit"], "6m_hit": r["6m hit"], "mae": r["MAE"]}

        base_rates = {key: {"signal": rate(headline[name]["sets"][main_label])}
                      for key, name in (("stocks", ctx_s["name"]), ("crypto", ctx_c["name"]), ("etfs", ctx_e["name"]))}
        base_rates["stocks"]["by_driver"] = {
            d: rate(ev_main[ev_main["driver"] == d]) for d in ("Market", "Sector", "Specific") if (ev_main["driver"] == d).sum() >= 20
        }
        frozen = {k: chosen[k] for k in fe.DEFAULT_CONFIG}
        frozen.update({"study_date": today.strftime("%Y-%m-%d"), "study_passed_oos": bool(passed),
                       "vol_percentile": vol_star, "base_rates": base_rates})
        frozen = json.loads(json.dumps(frozen, default=float).replace("NaN", "null"))
        with open(fe.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(frozen, f, indent=2)
        print(f"Saved: {fe.CONFIG_PATH}")
    else:
        print("(config not written - rerun with --write-config to freeze these thresholds for the daily report)")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
