"""
Market Musings — Channel Position Backtest Study.

Answers the article's core question: does buying/holding a stock in a
clean, statistically tight relative-strength uptrend actually pay off
better at the bottom, middle, or top of its own channel? This is a genuine
walk-forward backtest, not in-sample: at each monthly sample date, using
ONLY the trailing ~10 years of data available as of that date (no future
prices), the stock's ratio-regression channel is fit fresh, that date is
classified Bottom/Middle/Top against its own trailing-only fit, and the
stock's real forward return is measured 6M/12M/3Y/5Y later.

Standalone from market_musings_ratio_channel_study.py (the recurring "who
qualifies today" screen) — this is the expensive deep-dive backtest behind
the article's claim: 25y of history x the full universe x a monthly rolling
regression per name, plus ticker-level bootstrap significance testing. Not
on a daily schedule; run manually via workflow_dispatch or locally:
python scripts/market_musings_channel_backtest_study.py
"""

import base64
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from scipy.stats import linregress

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_ratio_channel_screener import (
    BENCH_LABELS,
    BENCH_QQQ,
    BENCH_SPY,
    BENCH_XIC,
    CHUNK_SIZE,
    MIN_TRADING_DAYS,
    REPO_ROOT,
    batch_download_closes,
    close_series_single,
    display_ticker,
    load_universe,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "market-musings")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")

SCREEN_MIN_R2 = 0.50  # same qualifying bar as the recurring screen — no estimates gate, this is a
                       # bigger sample meant to answer the general question, not describe today's picks
Z_THRESHOLD = 1.5  # |Z| >= this counts as "at a channel extreme"
BACKTEST_LOOKBACK_PERIOD = "25y"  # download window; needs runway for a 2520-day trailing fit PLUS a
                                   # 1260-day (5Y) forward horizon on top of that, with room for rolling
                                   # windows deep in the past
WINDOW_TRADING_DAYS = 2520  # ~10 trading years — the trailing window used for every rolling fit
RESAMPLE_STEP_DAYS = 21  # ~monthly; avoids overweighting autocorrelated day-to-day Z-score persistence
FORWARD_HORIZONS = {"6M": 126, "12M": 252, "3Y": 756, "5Y": 1260}  # trading days
MOMENTUM_LOOKBACK_DAYS = 63  # ~3 months; how far back each Middle-band observation looks to judge
                             # whether its Z-score has been rising (recovering off the bottom) or
                             # falling (fading off the top), using that same date's own trailing fit
N_BOOTSTRAP = 5000
N_EXAMPLE_CHARTS = 5

ORANGE = "#C67A29"
BLUE = "#1F79BE"
GREEN = "#44A660"
RED = "#A22A2A"
DGREY = "#363636"
LGREY = "#4A4A4A"
TEXTCLR = "#E8E8E8"
GREY_LINE = "#8E8E93"

BAND_ORDER = ["Bottom", "Middle", "Top"]
BAND_PAIRS = [("Bottom", "Middle"), ("Bottom", "Top"), ("Middle", "Top")]

with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()


def build_aligned(close, bench_close):
    return pd.DataFrame({"stock": close, "bench": bench_close}).dropna()


def band_for_z(z_val):
    if z_val <= -Z_THRESHOLD:
        return "Bottom"
    if z_val >= Z_THRESHOLD:
        return "Top"
    return "Middle"


def collect_rolling_backtest_observations(ticker, aligned_full, bench_label):
    """Walk-forward: at each monthly sample date, fit the regression using ONLY the trailing
    WINDOW_TRADING_DAYS of history up to and including that date (no future data), classify that
    date's band against that trailing-only fit, then measure the real forward return from there.
    Also records each qualifying date's own trailing channel bounds (Upper/Lower/Center) — used later
    to draw a genuinely rolling (not static) channel behind the example charts."""
    ratio_vals = (aligned_full["stock"] / aligned_full["bench"]).values
    log_ratio_vals = np.log(ratio_vals)
    stock_vals = aligned_full["stock"].values
    bench_vals = aligned_full["bench"].values
    dates = aligned_full.index
    n = len(aligned_full)
    max_horizon = max(FORWARD_HORIZONS.values())

    start_i = WINDOW_TRADING_DAYS - 1
    end_i = n - max_horizon
    obs = []
    band_seq = []
    for i in range(start_i, end_i, RESAMPLE_STEP_DAYS):
        w_start = i - WINDOW_TRADING_DAYS + 1
        y = log_ratio_vals[w_start:i + 1]
        x = np.arange(len(y))
        slope, intercept, r_value, _, _ = linregress(x, y)
        r2 = r_value**2
        if slope <= 0 or r2 < SCREEN_MIN_R2:
            continue

        window_ratio = ratio_vals[w_start:i + 1]
        rel_strength = window_ratio[-1] / window_ratio[0] - 1
        if rel_strength <= 0:
            continue

        resid = y - (intercept + slope * x)
        std = resid.std()
        if std == 0:
            continue
        z_i = resid[-1] / std
        band = band_for_z(z_i)
        fitted_end = intercept + slope * x[-1]
        band_seq.append({
            "Ticker": ticker, "Date": dates[i], "Band": band,
            "Center": np.exp(fitted_end), "Upper": np.exp(fitted_end + Z_THRESHOLD * std),
            "Lower": np.exp(fitted_end - Z_THRESHOLD * std),
        })

        z_prev = resid[-1 - MOMENTUM_LOOKBACK_DAYS] / std
        if z_i > z_prev:
            momentum = "Rising"
        elif z_i < z_prev:
            momentum = "Falling"
        else:
            momentum = "Flat"

        for horizon_label, horizon_days in FORWARD_HORIZONS.items():
            j = i + horizon_days
            if j >= n:
                continue
            stock_ret = stock_vals[j] / stock_vals[i] - 1
            bench_ret = bench_vals[j] / bench_vals[i] - 1
            obs.append({
                "Ticker": ticker, "Bench": bench_label, "Band": band, "Momentum": momentum,
                "Horizon": horizon_label, "Date": dates[i],
                "Fwd_Stock_Return": stock_ret, "Fwd_Relative_Return": stock_ret - bench_ret,
            })
    return obs, band_seq


def build_horizon_chart(summary_df, horizon, group_col="Band", group_order=None, title=None):
    group_order = group_order or BAND_ORDER
    sub = summary_df[summary_df["Horizon"] == horizon].set_index(group_col).reindex(group_order)
    fig = go.Figure()
    series = [
        ("Mean_Stock_Return", "Mean Stock Return", BLUE, 0.90),
        ("Median_Stock_Return", "Median Stock Return", BLUE, 0.55),
        ("Mean_Relative_Return", "Mean vs. Benchmark", GREY_LINE, 0.90),
        ("Median_Relative_Return", "Median vs. Benchmark", GREY_LINE, 0.55),
    ]
    for col, label, color, opacity in series:
        vals = sub[col] * 100
        fig.add_trace(go.Bar(
            x=group_order, y=vals, name=label,
            marker_color=color, opacity=opacity,
            text=[f"{v:+.1f}%" for v in vals], textposition="outside",
            hovertemplate="%{x}<br>" + label + ": %{y:+.1f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        height=460, width=950,
        paper_bgcolor=DGREY, plot_bgcolor=LGREY,
        font=dict(family="Arial, sans-serif", color=TEXTCLR, size=12),
        title=dict(text=title or f"{horizon} Forward Return by Channel Band", font=dict(size=15, color=TEXTCLR)),
        yaxis=dict(title="Forward Return (%)", ticksuffix="%", gridcolor="#555", zeroline=True, zerolinecolor="#888"),
        xaxis=dict(gridcolor="#555"),
        legend=dict(orientation="h", y=1.16, x=0.5, xanchor="center", font=dict(size=10)),
        margin=dict(t=90, b=60, l=60, r=40),
        uniformtext=dict(mode="hide", minsize=9),
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def build_horizon_table(summary_df, horizon, group_col="Band", group_order=None):
    group_order = group_order or BAND_ORDER
    sub = summary_df[summary_df["Horizon"] == horizon].set_index(group_col).reindex(group_order)
    cols = [group_col, "N", "Mean Return", "Median Return", "Mean vs. Bench", "Median vs. Bench", "Hit Rate (Beat Bench)"]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for group_val in group_order:
        r = sub.loc[group_val]
        if pd.isna(r.get("N")):
            html.append(f'<tr><td>{group_val}</td><td colspan="6">No observations</td></tr>')
            continue
        html.append(
            "<tr>"
            f"<td>{group_val}</td>"
            f"<td>{int(r['N'])}</td>"
            f"<td>{r['Mean_Stock_Return']*100:+.1f}%</td>"
            f"<td>{r['Median_Stock_Return']*100:+.1f}%</td>"
            f"<td>{r['Mean_Relative_Return']*100:+.1f}%</td>"
            f"<td>{r['Median_Relative_Return']*100:+.1f}%</td>"
            f"<td>{r['Hit_Rate']*100:.0f}%</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    return "".join(html)


def build_example_observations(backtest_df, n_per_band=5):
    sub = backtest_df[backtest_df["Horizon"] == "12M"]
    parts = ['<h3>Example Observations (12M horizon)</h3>']
    parts.append(
        '<div class="section-sub">A random sample of the underlying (ticker, date) observations behind the '
        "12M stats above — the date is when that name was sampled at that band, not today.</div>"
    )
    cols = ["Band", "Ticker", "Date", "Fwd Stock Return", "Fwd Return vs. Bench"]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for band in BAND_ORDER:
        band_rows = sub[sub["Band"] == band]
        sample = band_rows.sample(n=min(n_per_band, len(band_rows)), random_state=42).sort_values("Date")
        for _, r in sample.iterrows():
            html.append(
                "<tr>"
                f"<td>{r['Band']}</td>"
                f"<td>{display_ticker(r['Ticker'])}</td>"
                f"<td>{r['Date'].strftime('%Y-%m')}</td>"
                f"<td>{r['Fwd_Stock_Return']*100:+.1f}%</td>"
                f"<td>{r['Fwd_Relative_Return']*100:+.1f}%</td>"
                "</tr>"
            )
    html.append("</tbody></table>")
    parts.append("".join(html))
    return "\n".join(parts)


def build_takeaway(summary_df):
    def get(band, horizon, col):
        row = summary_df[(summary_df["Band"] == band) & (summary_df["Horizon"] == horizon)]
        return row[col].iloc[0] if not row.empty else None

    horizons = list(FORWARD_HORIZONS.keys())
    short_h, long_h = horizons[0], horizons[-1]

    rel = {h: {b: get(b, h, "Median_Relative_Return") for b in BAND_ORDER} for h in horizons}
    if any(v is None for h in horizons for v in rel[h].values()):
        return "Not enough historical observations to draw a comparison."

    n_by_band = {b: int(get(b, short_h, "N")) for b in BAND_ORDER}

    best_counts = {b: 0 for b in BAND_ORDER}
    for h in horizons:
        best_counts[max(rel[h], key=rel[h].get)] += 1
    best_band = max(best_counts, key=best_counts.get)

    short_str = ", ".join(f"{rel[short_h][b] * 100:+.1f}% ({b})" for b in BAND_ORDER)
    long_str = ", ".join(f"{rel[long_h][b] * 100:+.1f}% ({b})" for b in BAND_ORDER)

    erodes = rel[long_h][best_band] < rel[short_h][best_band]
    trend_clause = (
        "that edge erodes as the holding period lengthens" if erodes
        else "that edge holds up or widens as the holding period lengthens"
    )
    if best_band == "Middle":
        band_clause = "neither channel extreme reliably beat sitting in the middle of an established trend"
    else:
        band_clause = f"the {best_band.lower()} of the channel — not the middle — was where the edge showed up most consistently"

    return (
        f"Across {n_by_band['Bottom']} bottom-of-channel, {n_by_band['Middle']} middle-of-channel, and "
        f"{n_by_band['Top']} top-of-channel walk-forward observations (monthly-sampled, up to 25y lookback, "
        f"R²&ge;{SCREEN_MIN_R2:.2f} names only, no look-ahead), {best_band}-of-channel had the best (or "
        f"least-negative) median forward return relative to benchmark in {best_counts[best_band]} of "
        f"{len(horizons)} horizons measured. At {short_h}: {short_str}. By {long_h}: {long_str} — {trend_clause}, "
        f"and {band_clause}."
    )


def build_momentum_takeaway(momentum_summary_df):
    def get(momentum, horizon, col):
        row = momentum_summary_df[
            (momentum_summary_df["Momentum"] == momentum) & (momentum_summary_df["Horizon"] == horizon)
        ]
        return row[col].iloc[0] if not row.empty else None

    horizons = list(FORWARD_HORIZONS.keys())
    short_h, long_h = horizons[0], horizons[-1]

    rising = {h: get("Rising", h, "Median_Relative_Return") for h in horizons}
    falling = {h: get("Falling", h, "Median_Relative_Return") for h in horizons}
    if any(v is None for v in list(rising.values()) + list(falling.values())):
        return "Not enough historical observations to draw a comparison."

    n_rising = int(get("Rising", short_h, "N"))
    n_falling = int(get("Falling", short_h, "N"))

    wins = sum(1 for h in horizons if rising[h] > falling[h])
    if wins > len(horizons) / 2:
        verdict = "recent positive Z-score momentum (recovering off the bottom) tended to carry through into better forward relative returns than recent negative momentum (fading off the top)"
    else:
        verdict = "recent Z-score momentum direction didn't reliably predict which forward relative return was better — mean reversion showed up about as often as continuation"

    short_str = f"{rising[short_h] * 100:+.1f}% (Rising) vs. {falling[short_h] * 100:+.1f}% (Falling)"
    long_str = f"{rising[long_h] * 100:+.1f}% (Rising) vs. {falling[long_h] * 100:+.1f}% (Falling)"

    return (
        f"Restricting to the {n_rising + n_falling} middle-of-channel observations at the {short_h} horizon "
        f"({n_rising} with rising Z-score momentum over the trailing {MOMENTUM_LOOKBACK_DAYS} trading days, "
        f"{n_falling} with falling), median forward return relative to benchmark at {short_h} was {short_str}; "
        f"by {long_h} it was {long_str} — suggesting {verdict}."
    )


def ticker_level_paired_diff(sub, band_a, band_b, metric_col="Fwd_Relative_Return"):
    """Collapses each ticker's many overlapping monthly observations down to ONE number per ticker
    per band (its mean relative return in that band), then keeps only tickers that have both bands
    present, and returns each such ticker's paired difference — the unit that should actually be
    treated as one independent sample, since the raw observation count is inflated by autocorrelated,
    overlapping rolling windows drawn from the same underlying price path."""
    ticker_means = sub.groupby(["Ticker", "Band"])[metric_col].mean().unstack("Band")
    paired = ticker_means.dropna(subset=[band_a, band_b])
    return (paired[band_a] - paired[band_b]).values


def bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, seed=42):
    if len(diffs) < 10:
        return None
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"n": n, "mean_diff": diffs.mean(), "median_diff": np.median(diffs), "ci_lo": lo, "ci_hi": hi,
            "significant": lo > 0 or hi < 0}


def build_significance_section(backtest_df):
    rows = []
    for horizon in FORWARD_HORIZONS:
        sub = backtest_df[backtest_df["Horizon"] == horizon]
        for band_a, band_b in BAND_PAIRS:
            diffs = ticker_level_paired_diff(sub, band_a, band_b)
            result = bootstrap_ci(diffs)
            if result is not None:
                rows.append({"Horizon": horizon, "Comparison": f"{band_a} vs. {band_b}", **result})

    if not rows:
        return '<div class="section-sub">Not enough paired tickers to test significance.</div>', 0, 0

    n_significant = sum(1 for r in rows if r["significant"])
    parts = [
        f'<div class="section-sub">Ticker-level paired bootstrap (n={N_BOOTSTRAP} resamples), not a test on the '
        "raw observation count: each ticker's many overlapping monthly samples are first collapsed to one mean "
        "relative return per band, so a ticker with both bands present contributes exactly one paired difference "
        "— the true independent sample size is the ticker count below, not the far larger raw row count in the "
        "Study section. \"Significant\" means the 95% bootstrap confidence interval on the mean paired difference "
        "excludes zero.</div>"
    ]
    cols = ["Horizon", "Comparison", "Paired Tickers (n)", "Mean Diff", "Median Diff", "95% CI", "Significant?"]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for r in rows:
        html.append(
            "<tr>"
            f"<td>{r['Horizon']}</td>"
            f"<td>{r['Comparison']}</td>"
            f"<td>{r['n']}</td>"
            f"<td>{r['mean_diff']*100:+.1f}%</td>"
            f"<td>{r['median_diff']*100:+.1f}%</td>"
            f"<td>[{r['ci_lo']*100:+.1f}%, {r['ci_hi']*100:+.1f}%]</td>"
            f"<td>{'Yes' if r['significant'] else 'No'}</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    parts.append("".join(html))
    return "\n".join(parts), n_significant, len(rows)


def compute_dwell_times(band_seq_df):
    """How many consecutive monthly samples (~months, given RESAMPLE_STEP_DAYS is ~1 month) a ticker
    typically stays in the same band before its rolling fit reclassifies it. Each ticker's first and
    last run is dropped (left/right-censored — we don't know how long it had already been in that band
    before the sample window starts, or after it ends)."""
    rows = []
    for ticker, grp in band_seq_df.sort_values("Date").groupby("Ticker"):
        bands = grp["Band"].values
        if len(bands) < 3:
            continue
        run_id = np.concatenate([[0], np.cumsum(bands[1:] != bands[:-1])])
        run_bands = pd.Series(bands).groupby(run_id).first()
        run_lengths = pd.Series(bands).groupby(run_id).size()
        if len(run_bands) < 3:
            continue
        for rb, rl in zip(run_bands.iloc[1:-1], run_lengths.iloc[1:-1]):
            rows.append({"Ticker": ticker, "Band": rb, "Months": int(rl)})
    return pd.DataFrame(rows)


def build_dwell_time_section(band_seq_df):
    dwell_df = compute_dwell_times(band_seq_df)
    parts = ['<div class="section"><h2>How Long Do Stocks Stay in Each Band?</h2>']
    parts.append(
        '<div class="section-sub">Consecutive monthly rolling-classification samples a name stayed in the same '
        "band before its own trailing fit reclassified it. Each ticker's first and last run is dropped "
        "(left/right-censored), so this only counts fully-observed runs.</div>"
    )
    if dwell_df.empty:
        parts.append('<div class="section-sub">Not enough fully-observed runs.</div>')
        parts.append("</div>")
        return "\n".join(parts)
    summary = dwell_df.groupby("Band")["Months"].agg(N="size", Mean="mean", Median="median", Max="max").reindex(BAND_ORDER)
    cols = ["Band", "N Runs", "Mean Months", "Median Months", "Longest Observed (Months)"]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for band in BAND_ORDER:
        if band not in summary.index or pd.isna(summary.loc[band, "N"]):
            html.append(f'<tr><td>{band}</td><td colspan="4">No fully-observed runs</td></tr>')
            continue
        r = summary.loc[band]
        html.append(
            "<tr>"
            f"<td>{band}</td>"
            f"<td>{int(r['N'])}</td>"
            f"<td>{r['Mean']:.1f}</td>"
            f"<td>{r['Median']:.1f}</td>"
            f"<td>{int(r['Max'])}</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    parts.append("".join(html))
    parts.append("</div>")
    return "\n".join(parts)


def select_example_tickers_for_charts(backtest_df, horizon="12M", n=N_EXAMPLE_CHARTS, min_n_per_band=3):
    """The tickers with the largest Bottom-vs-Top gap in their own mean forward return vs. benchmark —
    the most striking illustrations of the effect, not necessarily typical ones. Requires a minimum
    observation count per band so the comparison isn't a fluke of one or two samples."""
    sub = backtest_df[backtest_df["Horizon"] == horizon]
    grouped = sub.groupby(["Ticker", "Band"]).agg(
        Mean_Rel=("Fwd_Relative_Return", "mean"), N=("Fwd_Relative_Return", "size"),
    )
    pivot_rel = grouped["Mean_Rel"].unstack("Band")
    pivot_n = grouped["N"].unstack("Band")
    if "Bottom" not in pivot_rel or "Top" not in pivot_rel:
        return []
    both = pivot_rel.dropna(subset=["Bottom", "Top"])
    both = both[(pivot_n.loc[both.index, "Bottom"] >= min_n_per_band) & (pivot_n.loc[both.index, "Top"] >= min_n_per_band)]
    if both.empty:
        return []
    gap = (both["Bottom"] - both["Top"]).sort_values(ascending=False)
    return gap.head(n).index.tolist()


def build_example_ticker_band_chart(ticker, aligned_full, ticker_bands, company_name, bench_label):
    ratio = aligned_full["stock"] / aligned_full["bench"]
    color_map = {"Bottom": GREEN, "Middle": GREY_LINE, "Top": RED}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values, mode="lines", line=dict(color=BLUE, width=1.2),
        name="Price / Bench", showlegend=False,
    ))

    # Local regression channel bounds at each qualifying sample date — a genuine walk-forward (rolling)
    # fit, a different regression at every point in time, so these bounds are what actually classified
    # each date, not a single static "as of today" line drawn over history.
    band_pts = ticker_bands.sort_values("Date")
    if not band_pts.empty:
        fig.add_trace(go.Scatter(
            x=band_pts["Date"], y=band_pts["Upper"], mode="lines",
            line=dict(color=RED, width=1, dash="dot"), name=f"+{Z_THRESHOLD:.1f}σ (rolling)",
            hovertemplate="%{x|%Y-%m}<br>Top threshold: %{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=band_pts["Date"], y=band_pts["Lower"], mode="lines",
            line=dict(color=GREEN, width=1, dash="dot"), name=f"-{Z_THRESHOLD:.1f}σ (rolling)",
            hovertemplate="%{x|%Y-%m}<br>Bottom threshold: %{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=band_pts["Date"], y=band_pts["Center"], mode="lines",
            line=dict(color=ORANGE, width=1.3, dash="dash"), name="Regression (rolling)",
            hovertemplate="%{x|%Y-%m}<br>Trend: %{y:.3f}<extra></extra>",
        ))

    for band in BAND_ORDER:
        band_dates = ticker_bands.loc[ticker_bands["Band"] == band, "Date"]
        if band_dates.empty:
            continue
        y_vals = ratio.reindex(band_dates)
        fig.add_trace(go.Scatter(
            x=band_dates, y=y_vals.values, mode="markers",
            marker=dict(color=color_map[band], size=7, line=dict(color=TEXTCLR, width=0.5)),
            name=band, hovertemplate="%{x|%Y-%m}<br>" + band + "<extra></extra>",
        ))
    fig.update_layout(
        height=380, width=950,
        paper_bgcolor=DGREY, plot_bgcolor=LGREY,
        font=dict(family="Arial, sans-serif", color=TEXTCLR, size=12),
        title=dict(
            text=f"<b>{display_ticker(ticker)}</b>  ·  {company_name}  ·  vs {bench_label}  ·  Rolling Channel Position History",
            font=dict(size=14, color=TEXTCLR), x=0.03, y=0.95,
        ),
        yaxis=dict(title="Price / Bench (log)", type="log", gridcolor="#555", zeroline=False),
        xaxis=dict(gridcolor="#555"),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=10)),
        margin=dict(t=65, b=40, l=60, r=40),
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Market Musings: Channel Position Backtest Study</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E8E8E8; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section h3 {{ margin: 20px 0 4px; font-size: 15px; color: #C67A29; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin: 6px 0 16px; }}
  .callout {{ background: #2A2D3A; border-left: 3px solid #C67A29; padding: 14px 18px; margin: 8px 0 20px;
    font-size: 14px; line-height: 1.5; color: #E8E8E8; border-radius: 0 4px 4px 0; }}
  .chart-wrap {{ padding: 8px 12px; }}
  table.summary {{ border-collapse: separate; border-spacing: 0; width: 100%; font-size: 13px; margin-bottom: 20px;
    border: 1px solid #2D3148; border-radius: 8px; overflow: hidden; }}
  table.summary th, table.summary td {{ padding: 9px 12px; text-align: center; border-bottom: 1px solid #2D3148;
    border-right: 1px solid #2D3148; }}
  table.summary th:last-child, table.summary td:last-child {{ border-right: none; }}
  table.summary tbody tr:last-child td {{ border-bottom: none; }}
  table.summary th {{ background: #1F79BE; color: #FFFFFF; font-weight: 700; letter-spacing: 0.2px;
    border-right-color: rgba(255,255,255,0.25); }}
  table.summary tbody tr:nth-child(even) {{ background: rgba(31,121,190,0.10); }}
  table.summary tbody tr:hover {{ background: rgba(198,122,41,0.14); }}
</style>
</head>
<body>
<header>
  <h1>Market Musings: Channel Position Backtest Study</h1>
  <div class="meta">Generated {date_str} &middot; Manual/on-demand deep-dive backtest, not part of daily automation &middot; walk-forward, 25y lookback, {window_years}y trailing regression window per sample &middot; SPY / QQQ / XIC benchmarks</div>
</header>
{body}
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()

    ticker_info, name_map = load_universe()
    print(f"Universe: {len(ticker_info)} unique tickers")

    bench_closes = {}
    for bench_ticker in (BENCH_SPY, BENCH_QQQ, BENCH_XIC):
        print(f"Downloading {bench_ticker} ({BACKTEST_LOOKBACK_PERIOD})...")
        bench_raw = yf.download(bench_ticker, period=BACKTEST_LOOKBACK_PERIOD, auto_adjust=True, progress=False)
        bench_close = close_series_single(bench_ticker, bench_raw)
        if bench_close is None:
            raise RuntimeError(f"Could not download {bench_ticker} close prices")
        bench_closes[bench_ticker] = bench_close

    all_tickers = sorted(ticker_info.keys())
    print(f"Downloading {len(all_tickers)} tickers ({BACKTEST_LOOKBACK_PERIOD})...")
    close_map = batch_download_closes(all_tickers, BACKTEST_LOOKBACK_PERIOD, CHUNK_SIZE)
    print(f"Got price history for {len(close_map)} tickers")

    backtest_obs = []
    band_seq_all = []
    for ticker, close in close_map.items():
        try:
            info = ticker_info[ticker]
            bench_ticker = info["bench"]
            bench_label = BENCH_LABELS[bench_ticker]
            close = close.rename(ticker)
            aligned_full = build_aligned(close, bench_closes[bench_ticker])
            if len(aligned_full) < MIN_TRADING_DAYS:
                continue
            obs, band_seq = collect_rolling_backtest_observations(ticker, aligned_full, bench_label)
            backtest_obs.extend(obs)
            band_seq_all.extend(band_seq)
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")

    print(f"Backtest pool: {len({o['Ticker'] for o in backtest_obs})} names, {len(backtest_obs)} observations")

    backtest_df = pd.DataFrame(backtest_obs)
    band_seq_df = pd.DataFrame(band_seq_all)

    parts = []

    if backtest_df.empty:
        parts.append('<div class="section"><div class="section-sub">No qualifying observations.</div></div>')
    else:
        summary_df = (
            backtest_df.groupby(["Band", "Horizon"])
            .agg(
                N=("Fwd_Stock_Return", "size"),
                Mean_Stock_Return=("Fwd_Stock_Return", "mean"),
                Median_Stock_Return=("Fwd_Stock_Return", "median"),
                Mean_Relative_Return=("Fwd_Relative_Return", "mean"),
                Median_Relative_Return=("Fwd_Relative_Return", "median"),
                Hit_Rate=("Fwd_Relative_Return", lambda s: (s > 0).mean()),
            )
            .reset_index()
        )
        band_order = {b: i for i, b in enumerate(BAND_ORDER)}
        horizon_order = {h: i for i, h in enumerate(FORWARD_HORIZONS)}
        summary_df["_b"] = summary_df["Band"].map(band_order)
        summary_df["_h"] = summary_df["Horizon"].map(horizon_order)
        summary_df = summary_df.sort_values(["_b", "_h"]).drop(columns=["_b", "_h"]).reset_index(drop=True)

        momentum_df = backtest_df[backtest_df["Band"] == "Middle"]
        momentum_summary_df = (
            momentum_df[momentum_df["Momentum"].isin(["Rising", "Falling"])]
            .groupby(["Momentum", "Horizon"])
            .agg(
                N=("Fwd_Stock_Return", "size"),
                Mean_Stock_Return=("Fwd_Stock_Return", "mean"),
                Median_Stock_Return=("Fwd_Stock_Return", "median"),
                Mean_Relative_Return=("Fwd_Relative_Return", "mean"),
                Median_Relative_Return=("Fwd_Relative_Return", "median"),
                Hit_Rate=("Fwd_Relative_Return", lambda s: (s > 0).mean()),
            )
            .reset_index()
        )
        momentum_order = {"Rising": 0, "Falling": 1}
        momentum_summary_df["_m"] = momentum_summary_df["Momentum"].map(momentum_order)
        momentum_summary_df["_h"] = momentum_summary_df["Horizon"].map(horizon_order)
        momentum_summary_df = (
            momentum_summary_df.sort_values(["_m", "_h"]).drop(columns=["_m", "_h"]).reset_index(drop=True)
        )

        study_parts = ['<div class="section"><h2>Study: Forward Returns From Channel Position</h2>']
        study_parts.append(
            '<div class="section-sub">Walk-forward backtest, not in-sample: at each monthly sample date (up to '
            "25 years of history per name), the regression is fit using only the trailing ~10 years of data "
            "available as of that date — no future prices — and R² &ge; 0.50, a positive slope, and positive "
            "trailing relative strength are all required at that date to count at all. That date is then "
            "classified Bottom (Z&le;-1.5) / Middle / Top (Z&ge;+1.5) against its own trailing-only fit, and the "
            "name's real forward return is measured 6M/12M/3Y/5Y later. Because classification never sees the "
            "future, this is a fair walk-forward read of whether the setup has actually worked historically.</div>"
        )
        study_parts.append(f'<div class="callout">{build_takeaway(summary_df)}</div>')
        for horizon in FORWARD_HORIZONS:
            study_parts.append(f'<h3>{horizon} Forward Return</h3>')
            study_parts.append(build_horizon_chart(summary_df, horizon))
            study_parts.append(build_horizon_table(summary_df, horizon))
        study_parts.append(build_example_observations(backtest_df))
        study_parts.append("</div>")
        parts.append("\n".join(study_parts))

        momentum_parts = ['<div class="section"><h2>Momentum Within the Middle</h2>']
        momentum_parts.append(
            '<div class="section-sub">Restricted to Middle-band observations from the study above, split by '
            "whether that name's Z-score was Rising (recovering off the bottom) or Falling (fading off the top) "
            f"over the trailing {MOMENTUM_LOOKBACK_DAYS} trading days as of the sample date, using that same "
            "date's own trailing regression fit — no additional look-ahead. Answers: does a middle-of-channel "
            "name's recent direction of travel predict its forward return better than its static position does?</div>"
        )
        if not momentum_summary_df.empty:
            momentum_parts.append(f'<div class="callout">{build_momentum_takeaway(momentum_summary_df)}</div>')
            for horizon in FORWARD_HORIZONS:
                momentum_parts.append(f'<h3>{horizon} Forward Return</h3>')
                momentum_parts.append(build_horizon_chart(
                    momentum_summary_df, horizon, group_col="Momentum", group_order=["Rising", "Falling"],
                    title=f"{horizon} Forward Return: Middle-of-Channel by Recent Momentum",
                ))
                momentum_parts.append(build_horizon_table(
                    momentum_summary_df, horizon, group_col="Momentum", group_order=["Rising", "Falling"],
                ))
        else:
            momentum_parts.append('<div class="section-sub">No qualifying observations.</div>')
        momentum_parts.append("</div>")
        parts.append("\n".join(momentum_parts))

        sig_parts = ['<div class="section"><h2>Bottom vs. Middle vs. Top: Is It Real, or Noise?</h2>']
        sig_html, n_significant, n_tests = build_significance_section(backtest_df)
        if n_tests == 0:
            verdict = "Not enough paired tickers to test significance."
        elif n_significant >= n_tests / 2:
            verdict = (f"{n_significant} of {n_tests} pairwise band comparisons (Bottom vs. Middle, Bottom vs. "
                       "Top, Middle vs. Top, across all four horizons) came back statistically significant (95% "
                       "bootstrap CI excludes zero) at the ticker level — take the band ordering from the Study "
                       "section seriously, it survives clustering by ticker.")
        else:
            verdict = (f"Only {n_significant} of {n_tests} pairwise band comparisons (Bottom vs. Middle, Bottom "
                       "vs. Top, Middle vs. Top, across all four horizons) came back statistically significant "
                       "(95% bootstrap CI excludes zero) at the ticker level — most of the band ordering from the "
                       "Study section does not survive clustering by ticker; treat it as directional at best, "
                       "not a real, actionable signal.")
        sig_parts.append(f'<div class="callout">{verdict}</div>')
        sig_parts.append(sig_html)
        sig_parts.append("</div>")
        parts.append("\n".join(sig_parts))

        parts.append(build_dwell_time_section(band_seq_df))

        example_parts = ['<div class="section"><h2>Example Charts: Rolling Channel Position Over Time</h2>']
        example_parts.append(
            '<div class="section-sub">The names with the largest Bottom-vs-Top gap in their own mean 12M forward '
            "return vs. benchmark — the most striking illustrations of the effect, not necessarily typical ones. "
            "Each dot is a monthly rolling-classification sample: green = Bottom, gray = Middle, red = Top, using "
            "only that sample date's own trailing-only regression fit (no look-ahead).</div>"
        )
        chart_tickers = select_example_tickers_for_charts(backtest_df)
        for ticker in chart_tickers:
            try:
                info = ticker_info[ticker]
                bench_ticker = info["bench"]
                bench_label = BENCH_LABELS[bench_ticker]
                close = close_map[ticker].rename(ticker)
                aligned_full = build_aligned(close, bench_closes[bench_ticker])
                ticker_bands = band_seq_df[band_seq_df["Ticker"] == ticker]
                company_name = name_map.get(ticker, display_ticker(ticker))
                chart_html = build_example_ticker_band_chart(ticker, aligned_full, ticker_bands, company_name, bench_label)
                example_parts.append(f'<div class="chart-wrap">{chart_html}</div>')
            except Exception as exc:
                print(f"  example chart error {ticker}: {exc}")
        example_parts.append("</div>")
        parts.append("\n".join(example_parts))

    html = PAGE_TEMPLATE.format(
        date_str=today.strftime("%B %d, %Y"), body="\n".join(parts),
        window_years=WINDOW_TRADING_DAYS // 252,
    )
    out_path = os.path.join(OUTPUT_DIR, "Ratio_Channel_Backtest_Study.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
