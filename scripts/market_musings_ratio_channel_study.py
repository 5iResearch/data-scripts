"""
ONE-TIME STUDY for a Market Musings article — not wired into the daily
GitHub Actions automation. Run manually: python scripts/market_musings_ratio_channel_study.py

Reuses the universe/loader/download plumbing from generate_ratio_channel_screener.py
but is otherwise a self-contained analysis with two parts:

Part 1 — Screen: for each benchmark (SPY/S&P 500, QQQ/Nasdaq-100, XIC/TSX),
the 15 cleanest (highest R^2) names currently at the bottom of their 10y
price/benchmark ratio channel, and separately the 15 cleanest at the top.
Every name must have: R^2 >= SCREEN_MIN_R2, a positive ratio-regression slope
and positive cumulative 10Y relative strength (i.e. it has actually
outperformed its benchmark, not just a statistically tight line), sit at a
channel extreme (|Z| >= Z_THRESHOLD), and have rising FY1E revenue estimates
(positive revision over both the 1M and 3M windows).

Part 2 — Study: "what happened next" for the broad pool of every name that
EVER had a clean, positive ratio trend (same R^2/slope bar as Part 1, but not
gated on today's Z-score or estimate revisions — this is a bigger sample
meant to answer the general question, not just describe today's picks). This
is a genuine walk-forward/rolling backtest, not an in-sample one: at each
monthly sample date, the regression is fit using ONLY the trailing
WINDOW_TRADING_DAYS (~10y) of history available as of that date — no future
data — then that date is classified Bottom/Middle/Top against that
trailing-only fit, and the name's own forward return (absolute and vs. its
benchmark) is measured 6M/12M/3Y/5Y later using real subsequent prices.
Requires up to BACKTEST_LOOKBACK_PERIOD (25y) of downloaded history per name
so there's enough runway for the 5Y horizon on top of the trailing fit, with
room for rolling windows deep in the past. Aggregated
mean/median/hit-rate by band x horizon is the "does buying the dip within a
clean uptrend actually pay off" evidence for the article — and because
classification never sees the future, this is a fair walk-forward read, not
a circular one.
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
    build_channel_chart,
    close_series_single,
    display_ticker,
    fig_to_div,
    has_rising_estimates,
    load_cap_map,
    load_revision_map,
    load_sector_map,
    load_universe,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "market-musings")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")

SCREEN_MIN_R2 = 0.50  # looser than the daily screener's 0.60 — this is an editorial piece, not the live screen
Z_THRESHOLD = 1.5  # |Z| >= this counts as "at a channel extreme" (same bar as the daily screener)
TOP_N_PER_TABLE = 15

BACKTEST_LOOKBACK_PERIOD = "25y"  # download window for the rolling backtest (Part 2); Part 1's screen uses
                                  # a tail-slice of the same download, so there's only one download per ticker.
                                  # Needs enough runway for a 2520-day trailing fit PLUS a 1260-day (5Y) forward
                                  # horizon on top of that (~15y minimum), with room to spare for sample size.
WINDOW_TRADING_DAYS = 2520  # ~10 trading years — the trailing window used for every regression fit, both
                            # Part 1's "as of today" fit and each rolling-window fit in Part 2
RESAMPLE_STEP_DAYS = 21  # ~monthly; avoids overweighting autocorrelated day-to-day Z-score persistence
FORWARD_HORIZONS = {"6M": 126, "12M": 252, "3Y": 756, "5Y": 1260}  # trading days
MOMENTUM_LOOKBACK_DAYS = 63  # ~3 months; how far back each Middle-band observation looks to judge whether
                             # its Z-score has been rising (recovering off the bottom) or falling (fading off
                             # the top), using that SAME sample date's own trailing regression fit — so this
                             # doesn't introduce any new look-ahead bias

ORANGE = "#C67A29"
BLUE = "#1F79BE"
DGREY = "#363636"
LGREY = "#4A4A4A"
TEXTCLR = "#E8E8E8"
GREY_LINE = "#8E8E93"

with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()


def build_aligned(close, bench_close):
    return pd.DataFrame({"stock": close, "bench": bench_close}).dropna()


def fit_ratio_regression(aligned):
    """Fits a single log(stock/bench) regression over the given aligned frame
    as-is (no trailing/rolling logic here — that's the caller's job). Used
    for Part 1's "as of today" snapshot, on a tail-slice of the full
    download. Returns None if the fit doesn't qualify."""
    if len(aligned) < MIN_TRADING_DAYS:
        return None

    ratio = aligned["stock"] / aligned["bench"]
    x = np.arange(len(ratio))
    y = np.log(ratio.values)
    slope, intercept, r_value, _, _ = linregress(x, y)
    r2 = r_value**2
    if slope <= 0 or r2 < SCREEN_MIN_R2:
        return None

    rel_strength_pct = (ratio.iloc[-1] / ratio.iloc[0] - 1) * 100
    if rel_strength_pct <= 0:
        return None  # must have actually outperformed cumulatively, not just a positive-slope fit

    resid = y - (intercept + slope * x)
    std = resid.std()
    if std == 0:
        return None
    z_last = resid[-1] / std

    stock_return_pct = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[0] - 1) * 100
    bench_return_pct = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1) * 100
    annual_trend_pct = (np.exp(slope * 252) - 1) * 100

    return {
        "R2": r2,
        "Z_Score": z_last,
        "10Y_Return_%": stock_return_pct,
        "10Y_Bench_Return_%": bench_return_pct,
        "Relative_Strength_10Y_%": rel_strength_pct,
        "Annual_Trend_%": annual_trend_pct,
        # underscore-prefixed: only consumed by build_channel_chart (imported from
        # generate_ratio_channel_screener), not by the summary tables
        "_ratio": ratio,
        "_x": x,
        "_slope": slope,
        "_intercept": intercept,
        "_std": std,
    }


def band_for_z(z_val):
    if z_val <= -Z_THRESHOLD:
        return "Bottom"
    if z_val >= Z_THRESHOLD:
        return "Top"
    return "Middle"


def collect_rolling_backtest_observations(ticker, aligned_full, bench_label):
    """Walk-forward: at each monthly sample date, fit the regression using
    ONLY the trailing WINDOW_TRADING_DAYS of history up to and including that
    date (no future data), classify that date's band against that
    trailing-only fit, then measure the real forward return from there. Every
    rolling window is independently required to pass the same qualifying
    bar as Part 1 (R^2, positive slope, positive trailing relative strength)."""
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
    for i in range(start_i, end_i, RESAMPLE_STEP_DAYS):
        w_start = i - WINDOW_TRADING_DAYS + 1
        y = log_ratio_vals[w_start : i + 1]
        x = np.arange(len(y))
        slope, intercept, r_value, _, _ = linregress(x, y)
        r2 = r_value**2
        if slope <= 0 or r2 < SCREEN_MIN_R2:
            continue

        window_ratio = ratio_vals[w_start : i + 1]
        rel_strength = window_ratio[-1] / window_ratio[0] - 1
        if rel_strength <= 0:
            continue

        resid = y - (intercept + slope * x)
        std = resid.std()
        if std == 0:
            continue
        z_i = resid[-1] / std
        band = band_for_z(z_i)

        # Momentum: compare today's Z to Z at MOMENTUM_LOOKBACK_DAYS ago, using this SAME
        # trailing fit's own residuals (no extra look-ahead) — was this name recently closer
        # to the bottom (Rising into the middle) or closer to the top (Falling into the middle)?
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
                "Ticker": ticker,
                "Bench": bench_label,
                "Band": band,
                "Momentum": momentum,
                "Horizon": horizon_label,
                "Date": dates[i],
                "Fwd_Stock_Return": stock_ret,
                "Fwd_Relative_Return": stock_ret - bench_ret,
            })
    return obs


def format_cap(mktcap_millions):
    if pd.isna(mktcap_millions):
        return "—"
    if mktcap_millions >= 1000:
        return f"${mktcap_millions / 1000:.1f}B"
    return f"${mktcap_millions:.0f}M"


def build_screen_table(rows, name_map, cap_map, sector_map):
    rows = sorted(rows, key=lambda r: r["R2"], reverse=True)[:TOP_N_PER_TABLE]
    if not rows:
        return rows, '<div class="section-sub">No names passed all filters.</div>'

    cols = ["Ticker", "Name", "Sector", "Industry", "Market Cap", "10Y Return %", "10Y Bench Return %",
            "Relative Strength 10Y %", "Annual Trend %", "R²", "Z-Score"]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for r in rows:
        disp = display_ticker(r["Ticker"])
        name = name_map.get(r["Ticker"], disp)
        sector, industry = sector_map.get(disp, ("—", "—"))
        cap = format_cap(cap_map.get(r["Ticker"]))
        html.append(
            "<tr>"
            f"<td>{disp}</td>"
            f"<td>{name}</td>"
            f"<td>{sector or '—'}</td>"
            f"<td>{industry or '—'}</td>"
            f"<td>{cap}</td>"
            f"<td>{r['10Y_Return_%']:.0f}%</td>"
            f"<td>{r['10Y_Bench_Return_%']:.0f}%</td>"
            f"<td>{r['Relative_Strength_10Y_%']:+.0f}%</td>"
            f"<td>{r['Annual_Trend_%']:.1f}%</td>"
            f"<td>{r['R2']:.2f}</td>"
            f"<td>{r['Z_Score']:.2f}</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    return rows, "".join(html)


def build_screen_charts(rows, name_map, revision_map, sector_map):
    parts = []
    for row in rows:
        disp = display_ticker(row["Ticker"])
        company_name = name_map.get(row["Ticker"], disp)
        revision_row = revision_map.get(row["Ticker"])
        sector, industry = sector_map.get(disp, ("", ""))
        sector_industry = " | ".join(s for s in (sector, industry) if s)
        try:
            fig = build_channel_chart(row, company_name, revision_row, sector_industry)
            parts.append(f'<div class="chart-wrap">{fig_to_div(fig)}</div>')
        except Exception as exc:
            print(f"  chart error {row['Ticker']}: {exc}")
    return "\n".join(parts)


def build_screen_section(position, all_rows, name_map, cap_map, sector_map, revision_map):
    title = f"{position} of Channel"
    parts = [f'<div class="section"><h2>{title}</h2>']
    parts.append(
        '<div class="section-sub">R² &ge; 0.50, positive ratio-regression slope, positive cumulative 10Y '
        f"relative strength, |Z| &ge; {Z_THRESHOLD}, and rising FY1E revenue estimates (1M &amp; 3M). "
        "Ranked by R² (cleanest trend first), top 15 shown.</div>"
    )
    for bench_ticker in (BENCH_SPY, BENCH_QQQ, BENCH_XIC):
        bench_label = BENCH_LABELS[bench_ticker]
        rows = [r for r in all_rows if r["Bench"] == bench_label and r["Position"] == position]
        kept, table_html = build_screen_table(rows, name_map, cap_map, sector_map)
        parts.append(f'<h3>vs {bench_label}</h3>')
        parts.append(f'<div class="section-sub">{len(kept)} of {len(rows)} qualifying names shown.</div>')
        parts.append(table_html)
        parts.append(build_screen_charts(kept, name_map, revision_map, sector_map))
    parts.append("</div>")
    return "\n".join(parts)


BAND_ORDER = ["Bottom", "Middle", "Top"]


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


def build_example_observations(backtest_df, n_per_band=5):
    """A handful of concrete (ticker, date) examples per band, for citing
    real instances in the article rather than only aggregate stats."""
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


N_BOOTSTRAP = 5000


def ticker_level_paired_diff(sub, band_a, band_b):
    """Collapses each ticker's many overlapping monthly observations down to
    ONE number per ticker per band (its mean relative return in that band),
    then keeps only tickers that have both bands present, and returns each
    such ticker's paired difference (band_a - band_b). This is the unit that
    should actually be treated as one independent sample — the raw
    observation count is inflated by autocorrelated, overlapping windows
    from the same underlying price path and would radically overstate
    confidence if tested directly."""
    ticker_means = sub.groupby(["Ticker", "Band"])["Fwd_Relative_Return"].mean().unstack("Band")
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
    """Ticker-level paired bootstrap: is Middle's edge over Bottom/Top (from
    the Study section above) distinguishable from noise, or does it wash out
    once autocorrelated overlapping observations are collapsed to one
    independent sample per ticker?"""
    rows = []
    for horizon in FORWARD_HORIZONS:
        sub = backtest_df[backtest_df["Horizon"] == horizon]
        for other_band in ("Bottom", "Top"):
            diffs = ticker_level_paired_diff(sub, "Middle", other_band)
            result = bootstrap_ci(diffs)
            if result is not None:
                rows.append({"Horizon": horizon, "Comparison": f"Middle vs. {other_band}", **result})

    if not rows:
        return '<div class="section-sub">Not enough paired tickers to test significance.</div>', 0, 0

    n_significant = sum(1 for r in rows if r["significant"])
    parts = []
    parts.append(
        '<div class="section-sub">Ticker-level paired bootstrap (n=%d resamples), not a test on the raw '
        "observation count: each ticker's many overlapping monthly samples are first collapsed to one mean "
        "relative return per band, so a ticker with both a Middle-band and Bottom/Top-band average contributes "
        "exactly one paired difference — the true independent sample size is the ticker count below, not the "
        "tens of thousands of raw rows in the Study section. \"Significant\" means the 95%% bootstrap confidence "
        "interval on the mean paired difference excludes zero.</div>" % N_BOOTSTRAP
    )
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


def build_takeaway(summary_df):
    """Templates the callout text from the actual computed numbers rather
    than asserting a conclusion up front. Walks all horizons (not just one)
    to describe both "which band wins" and "does the edge hold up or decay
    as the horizon lengthens" — the two things that actually showed up in
    the walk-forward results."""
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
    """Same templating approach as build_takeaway, but for the Rising-vs-Falling
    split within the Middle band only."""
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()

    ticker_info, name_map = load_universe()
    revision_map = load_revision_map()
    cap_map = load_cap_map()
    sector_map = load_sector_map()
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

    screen_rows = []
    backtest_obs = []
    for ticker, close in close_map.items():
        try:
            info = ticker_info[ticker]
            bench_ticker = info["bench"]
            bench_label = BENCH_LABELS[bench_ticker]
            close = close.rename(ticker)
            aligned_full = build_aligned(close, bench_closes[bench_ticker])
            if len(aligned_full) < MIN_TRADING_DAYS:
                continue

            backtest_obs.extend(collect_rolling_backtest_observations(ticker, aligned_full, bench_label))

            aligned_recent = aligned_full.tail(WINDOW_TRADING_DAYS)
            fit = fit_ratio_regression(aligned_recent)
            if fit is None:
                continue

            band = band_for_z(fit["Z_Score"])
            if band == "Middle":
                continue
            if not has_rising_estimates(revision_map.get(ticker)):
                continue
            row = dict(fit)
            row["Ticker"] = ticker
            row["Bench"] = bench_label
            row["Position"] = band
            screen_rows.append(row)
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")

    print(f"Screen: {len(screen_rows)} names at a channel extreme with rising estimates")
    print(f"Backtest pool: {len({o['Ticker'] for o in backtest_obs})} names, {len(backtest_obs)} observations")

    parts = []
    parts.append(build_screen_section("Bottom", screen_rows, name_map, cap_map, sector_map, revision_map))
    parts.append(build_screen_section("Top", screen_rows, name_map, cap_map, sector_map, revision_map))

    backtest_df = pd.DataFrame(backtest_obs)
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

    study_parts = ['<div class="section"><h2>Study: Forward Returns From Channel Extremes</h2>']
    study_parts.append(
        '<div class="section-sub">Walk-forward backtest, not in-sample: at each monthly sample date (up to 25 years '
        "of history per name), the regression is fit using only the trailing ~10 years of data available as of that "
        "date — no future prices — and R² &ge; 0.50, a positive slope, and positive trailing relative strength are "
        "all required at that date to count at all (not gated on today's estimates or Z-score). That date is then "
        "classified Bottom (Z&le;-1.5) / Middle / Top (Z&ge;+1.5) against its own trailing-only fit, and the name's "
        "real forward return is measured 6M/12M/3Y/5Y later. Because classification never sees the future, this "
        "is a fair walk-forward read of whether the setup has actually worked historically.</div>"
    )
    if not summary_df.empty:
        study_parts.append(f'<div class="callout">{build_takeaway(summary_df)}</div>')
        for horizon in FORWARD_HORIZONS:
            study_parts.append(f'<h3>{horizon} Forward Return</h3>')
            study_parts.append(build_horizon_chart(summary_df, horizon))
            study_parts.append(build_horizon_table(summary_df, horizon))
        study_parts.append(build_example_observations(backtest_df))
    else:
        study_parts.append('<div class="section-sub">No qualifying observations.</div>')
    study_parts.append("</div>")
    parts.append("\n".join(study_parts))

    momentum_parts = ['<div class="section"><h2>Momentum Within the Middle</h2>']
    momentum_parts.append(
        '<div class="section-sub">Restricted to Middle-band observations from the study above, split by whether '
        f"that name's Z-score was Rising (recovering off the bottom) or Falling (fading off the top) over the "
        f"trailing {MOMENTUM_LOOKBACK_DAYS} trading days as of the sample date, using that same date's own "
        "trailing regression fit — no additional look-ahead. Answers: does a middle-of-channel name's recent "
        "direction of travel predict its forward return better than its static position does?</div>"
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

    sig_parts = ['<div class="section"><h2>Is the Middle-Band Edge Real, or Noise?</h2>']
    if not backtest_df.empty:
        sig_html, n_significant, n_tests = build_significance_section(backtest_df)
        if n_tests == 0:
            verdict = "Not enough paired tickers to test significance."
        elif n_significant >= n_tests / 2:
            verdict = (f"{n_significant} of {n_tests} Middle-vs-Bottom / Middle-vs-Top comparisons came back "
                       "statistically significant (95% bootstrap CI excludes zero) at the ticker level — take the "
                       "Middle-band edge from the Study section seriously, it survives clustering by ticker.")
        else:
            verdict = (f"Only {n_significant} of {n_tests} Middle-vs-Bottom / Middle-vs-Top comparisons came back "
                       "statistically significant (95% bootstrap CI excludes zero) at the ticker level — most of "
                       "the Middle-band edge from the Study section does not survive clustering by ticker; treat "
                       "it as directional at best, not a real, actionable signal.")
        sig_parts.append(f'<div class="callout">{verdict}</div>')
        sig_parts.append(sig_html)
    else:
        sig_parts.append('<div class="section-sub">No observations to test.</div>')
    sig_parts.append("</div>")
    parts.append("\n".join(sig_parts))

    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "Ratio_Channel_Market_Musings_Study.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Market Musings: Relative-Strength Channel Study</title>
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
  table.summary {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 20px; }}
  table.summary th, table.summary td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #3A3A3C; }}
  table.summary th:first-child, table.summary td:first-child,
  table.summary th:nth-child(2), table.summary td:nth-child(2),
  table.summary th:nth-child(3), table.summary td:nth-child(3),
  table.summary th:nth-child(4), table.summary td:nth-child(4) {{ text-align: left; }}
  table.summary th {{ color: #C67A29; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>Market Musings: Relative-Strength Channel Study</h1>
  <div class="meta">Generated {date_str} &middot; One-time study, not part of daily automation &middot; 10y price/benchmark ratio regression (SPY / QQQ / XIC)</div>
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
