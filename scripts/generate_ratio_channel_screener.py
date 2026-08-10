"""
10-Year Relative-Strength (Ratio) Channel Screener.

Unlike generate_channel_screener.py / generate_channel_screener_cdn.py, which
fit the regression channel on raw log-price and separately check whether the
stock beat its benchmark start-to-end, this screener fits the regression
channel directly on the price/benchmark RATIO (log(stock/bench)). A positive
channel slope *is* the outperformance signal — the ratio has been trending up
for 10 years — and the Z-score of the current residual says whether that
relative-strength trend is currently stretched to the top or pulled back to
the bottom of its own channel:

  - Bottom of Channel: an outperformer's ratio has pulled back to the low end
    of its own relative-strength trend — a relative-strength dip, not
    necessarily a price decline.
  - Top of Channel: the ratio is stretched to the high end of its trend —
    relative-strength extension, potential mean-reversion vs the benchmark.

Both lists require R^2 of the 10y log-ratio regression >= MIN_R2 (a relatively
clean, consistent relative-strength trend, not a noisy one) — applies equally
to Bottom, Top, and (since it's a filtered subset of the two) Rising.

A third section, Rising Analyst Estimates, is the intersection of either list
with FY1E revenue estimate revisions that are positive over both the 1-month
and 3-month windows — names combining a relative-strength channel extreme
with improving underlying fundamentals.

Benchmark is assigned per name: QQQ for Nasdaq-100 members, SPY for other
S&P 500 / US revenue-revision-screener names, XIC.TO for the Canadian (TSX)
universe.

Universe: S&P 500 + Nasdaq-100 + data/us_1w_rev_est_screener.csv (US, vs SPY
or QQQ) union TSX universe + data/cdn_1w_rev_est_screener.csv (Canada, vs
XIC).
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
from plotly.subplots import make_subplots
from scipy.stats import linregress

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_screening import (
    CAP_FILTER_CONTROL_HTML,
    CAP_FILTER_CSS,
    CAP_FILTER_JS,
    cap_tier,
    load_nasdaq100_symbols,
    load_sp500_symbols,
    load_tsx_symbols,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "ratio-channel-screener")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")
US_REV_PATH = os.path.join(REPO_ROOT, "data", "us_1w_rev_est_screener.csv")
CDN_REV_PATH = os.path.join(REPO_ROOT, "data", "cdn_1w_rev_est_screener.csv")
KOYFIN_US_PATH = os.path.join(REPO_ROOT, "data", "koyfin_us.csv")
KOYFIN_CDN_PATH = os.path.join(REPO_ROOT, "data", "koyfin_cdn.csv")

BENCH_SPY = "SPY"
BENCH_QQQ = "QQQ"
BENCH_XIC = "XIC.TO"
BENCH_LABELS = {BENCH_SPY: "SPY", BENCH_QQQ: "QQQ", BENCH_XIC: "XIC"}

LOOKBACK_PERIOD = "10y"
MIN_TRADING_DAYS = 2400  # ~95% of the ~2520 trading days in 10y; excludes recent IPOs
CHANNEL_SIGMA = 2.0  # width of the plotted upper/lower channel bands
BOTTOM_Z_THRESHOLD = -1.5  # current residual must be at/below this to count as "at the bottom"
TOP_Z_THRESHOLD = 1.5  # current residual must be at/above this to count as "at the top"
MIN_R2 = 0.60  # 10y log-ratio regression must be a relatively clean fit to even qualify, same bar as
               # generate_uptrend_channel_screener.py's MIN_R2; applies to Bottom/Top/Rising alike since
               # Rising is a filtered subset of the other two
TOP_R2_FRACTION = 1.0  # fraction of R^2-ranked survivors to keep (1.0 = keep all)
CHUNK_SIZE = 250  # yfinance batch download size
RISING_EST_WINDOWS = ("fy1_1m", "fy1_3m")  # FY1E revenue-revision windows that must both be positive
RISING_CHART_CAP = 50  # cap on charts re-rendered in the Rising Analyst Estimates section (by R^2);
                       # the full list is always in that section's table regardless of this cap.
                       # Keeps the combined US+Canada report safely under GitHub's 100MB file limit
                       # even though every one of these names' charts is also rendered once already
                       # in Bottom/Top of Channel.

ORANGE = "#C67A29"
BLUE = "#1F79BE"
DBLUE = "#4B8EA9"
DGREY = "#363636"
LGREY = "#4A4A4A"
GREEN = "#44A660"
RED = "#A22A2A"
TEXTCLR = "#E8E8E8"
GREY_LINE = "#8E8E93"

REV_COL_MAP = {
    "Ticker": "ticker",
    "Revenues Est Avg Rev % (FY1E - 1W)": "fy1_1w", "Revenues Est Avg Rev % (FY1E - 1M)": "fy1_1m",
    "Revenues Est Avg Rev % (FY1E - 3M)": "fy1_3m", "Revenues Est Avg Rev % (FY1E - 6M)": "fy1_6m",
    "Revenues Est Avg Rev % (FY1E - 1Y)": "fy1_1y",
    "Revenues Est Avg Rev % (FY2E - 1W)": "fy2_1w", "Revenues Est Avg Rev % (FY2E - 1M)": "fy2_1m",
    "Revenues Est Avg Rev % (FY2E - 3M)": "fy2_3m", "Revenues Est Avg Rev % (FY2E - 6M)": "fy2_6m",
    "Revenues Est Avg Rev % (FY2E - 1Y)": "fy2_1y",
    "Revenues Est Avg Rev % (FY3E - 1W)": "fy3_1w", "Revenues Est Avg Rev % (FY3E - 1M)": "fy3_1m",
    "Revenues Est Avg Rev % (FY3E - 3M)": "fy3_3m", "Revenues Est Avg Rev % (FY3E - 6M)": "fy3_6m",
    "Revenues Est Avg Rev % (FY3E - 1Y)": "fy3_1y",
}
REV_WIN_KEYS = ["1w", "1m", "3m", "6m", "1y"]
REV_WIN_LABELS = ["1W", "1M", "3M", "6M", "1Y"]
FY_COLORS = [BLUE, DBLUE, ORANGE]

with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()


def normalize_us_ticker(t):
    return t.strip().upper().replace(".", "-")


def to_cdn_ticker(t):
    return t.strip().upper().replace(".", "-") + ".TO"


def load_universe():
    """Returns (ticker_info, name_map). ticker_info maps yfinance ticker ->
    {"bench": bench_ticker, "cdn": bool}. Nasdaq-100 membership wins the bench
    assignment over plain S&P 500 membership (QQQ over SPY) since it's the
    more specific index; everything else in the US revenue-revision CSV that
    isn't in either index defaults to SPY."""
    sp500 = {normalize_us_ticker(t) for t in load_sp500_symbols()}
    nasdaq100 = {normalize_us_ticker(t) for t in load_nasdaq100_symbols()}
    us_rev_df = pd.read_csv(US_REV_PATH).dropna(subset=["Ticker"])
    us_rev_tickers = {normalize_us_ticker(t) for t in us_rev_df["Ticker"].astype(str)}
    us_name_map = {normalize_us_ticker(t): n for t, n in zip(us_rev_df["Ticker"].astype(str), us_rev_df["Name"].astype(str))}

    tsx_symbols = set(load_tsx_symbols())  # already ".TO"-suffixed
    cdn_rev_df = pd.read_csv(CDN_REV_PATH).dropna(subset=["Ticker"])
    cdn_rev_tickers = {to_cdn_ticker(t) for t in cdn_rev_df["Ticker"].astype(str)}
    cdn_name_map = {to_cdn_ticker(t): n for t, n in zip(cdn_rev_df["Ticker"].astype(str), cdn_rev_df["Name"].astype(str))}

    us_universe = sp500 | nasdaq100 | us_rev_tickers
    cdn_universe = tsx_symbols | cdn_rev_tickers

    ticker_info = {}
    for t in us_universe:
        bench = BENCH_QQQ if t in nasdaq100 else BENCH_SPY
        ticker_info[t] = {"bench": bench, "cdn": False}
    for t in cdn_universe:
        ticker_info[t] = {"bench": BENCH_XIC, "cdn": True}

    name_map = {**us_name_map, **cdn_name_map}
    return ticker_info, name_map


def load_revision_map():
    """Ticker -> dict of FY1E/FY2E/FY3E revenue-estimate revision %s (1W/1M/3M/6M/1Y),
    for the analyst revision cascade panel and the Rising Analyst Estimates
    filter. Merges the US and CDN rev-screener CSVs; CDN tickers keep their
    ".TO" suffix to match the yfinance ticker used elsewhere."""
    frames = []
    for path, ticker_fn in ((US_REV_PATH, normalize_us_ticker), (CDN_REV_PATH, to_cdn_ticker)):
        raw = pd.read_csv(path).dropna(subset=["Ticker"])
        df = raw.rename(columns=REV_COL_MAP)
        keep = [c for c in REV_COL_MAP.values() if c in df.columns]
        df = df[keep].copy()
        for c in keep:
            if c != "ticker":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ticker"] = df["ticker"].astype(str).map(ticker_fn)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index("ticker").to_dict("index")


def load_cap_map():
    """Ticker -> Market Cap ($M), for the cap-tier filter. Merges the US and
    CDN rev-screener CSVs; names outside both fall back to "Unknown"."""
    cap_map = {}
    for path, ticker_fn in ((US_REV_PATH, normalize_us_ticker), (CDN_REV_PATH, to_cdn_ticker)):
        raw = pd.read_csv(path).dropna(subset=["Ticker"])
        tickers = raw["Ticker"].astype(str).map(ticker_fn)
        mktcap = pd.to_numeric(raw["Market Cap"], errors="coerce")
        cap_map.update(dict(zip(tickers, mktcap)))
    return cap_map


def load_sector_map():
    """Display-ticker (no ".TO" suffix) -> (Sector, Industry), merged from the
    US and CDN Koyfin exports. Sector/industry labels are a nice-to-have
    annotation, not a screening input, so missing files just yield blanks."""
    sector_map = {}
    for path in (KOYFIN_US_PATH, KOYFIN_CDN_PATH):
        if not os.path.exists(path):
            print(f"  no Koyfin export at {path}, sector/industry labels will be blank for that universe")
            continue
        df = pd.read_csv(path)
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        sector_map.update({
            row["Ticker"]: (row.get("Sector", "") or "", row.get("Industry", "") or "")
            for _, row in df.iterrows()
        })
    return sector_map


def display_ticker(ticker):
    return ticker[:-3] if ticker.endswith(".TO") else ticker


def close_series_single(ticker, raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty or "Close" not in raw:
        return None
    close = raw["Close"].dropna()
    return close if not close.empty else None


def batch_download_closes(tickers, period, chunk_size):
    close_map = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"  downloading {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        try:
            raw = yf.download(
                chunk, period=period, group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
        except Exception as exc:
            print(f"  batch error: {exc}")
            continue
        for t in chunk:
            try:
                close = raw[t]["Close"].dropna()
                if not close.empty:
                    close_map[t] = close
            except (KeyError, TypeError):
                pass
    return close_map


def score_ticker(close, bench_close, bench_ticker):
    aligned = pd.DataFrame({"stock": close, "bench": bench_close}).dropna()
    if len(aligned) < MIN_TRADING_DAYS:
        return None

    ratio = aligned["stock"] / aligned["bench"]
    x = np.arange(len(ratio))
    y = np.log(ratio.values)
    slope, intercept, r_value, _, _ = linregress(x, y)
    if slope <= 0:
        return None  # ratio must have a sustained relative-strength uptrend
    if r_value**2 < MIN_R2:
        return None  # ratio trend must be a relatively clean regression fit, not noisy

    fitted = intercept + slope * x
    resid = y - fitted
    std = resid.std()
    if std == 0:
        return None
    z_last = resid[-1] / std
    if z_last <= BOTTOM_Z_THRESHOLD:
        position = "Bottom"
    elif z_last >= TOP_Z_THRESHOLD:
        position = "Top"
    else:
        return None

    stock_return_10y = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[0] - 1) * 100
    bench_return_10y = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1) * 100
    rel_strength_10y_pct = (ratio.iloc[-1] / ratio.iloc[0] - 1) * 100
    annual_trend_return_pct = (np.exp(slope * 252) - 1) * 100  # annualized ratio-trend rate

    return {
        "Ticker": close.name,
        "Bench": BENCH_LABELS[bench_ticker],
        "Position": position,
        "R2": r_value**2,
        "Z_Score": z_last,
        "Annual_Trend_Return_%": annual_trend_return_pct,
        "10Y_Return_%": stock_return_10y,
        "10Y_Bench_Return_%": bench_return_10y,
        "Relative_Strength_10Y_%": rel_strength_10y_pct,
        "_x": x,
        "_ratio": ratio,
        "_slope": slope,
        "_intercept": intercept,
        "_std": std,
        "_close": close,
    }


def has_rising_estimates(revision_row):
    if not revision_row:
        return False
    vals = [revision_row.get(w) for w in RISING_EST_WINDOWS]
    return all(pd.notna(v) and v > 0 for v in vals)


def build_channel_chart(row, company_name, revision_row=None, sector_industry=None):
    ratio = row["_ratio"]
    x, slope, intercept, std = row["_x"], row["_slope"], row["_intercept"], row["_std"]
    center = np.exp(intercept + slope * x)
    upper = np.exp(intercept + slope * x + CHANNEL_SIGMA * std)
    lower = np.exp(intercept + slope * x - CHANNEL_SIGMA * std)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        subplot_titles=[f"10Y Ratio Channel vs {row['Bench']}", "Analyst Revenue Revision Trend"],
        horizontal_spacing=0.09,
    )

    fig.add_trace(go.Scatter(x=ratio.index, y=upper, mode="lines", line=dict(color=GREY_LINE, width=1, dash="dot"), name=f"+{CHANNEL_SIGMA:.0f}σ", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio.index, y=lower, mode="lines", line=dict(color=GREY_LINE, width=1, dash="dot"), name=f"-{CHANNEL_SIGMA:.0f}σ", fill="tonexty", fillcolor="rgba(255,255,255,0.05)", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio.index, y=center, mode="lines", line=dict(color=ORANGE, width=1.5, dash="dash"), name="Regression", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode="lines", line=dict(color=BLUE, width=1.8), name="Ratio", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[ratio.index[-1]], y=[ratio.values[-1]], mode="markers",
        marker=dict(color=RED, size=9, line=dict(color=TEXTCLR, width=1)),
        name="Last", showlegend=False, hovertemplate=f"Z={row['Z_Score']:.2f}σ<extra></extra>",
    ), row=1, col=1)

    if revision_row:
        for fy_idx, fy in enumerate(["fy1", "fy2", "fy3"]):
            vals = [revision_row.get(f"{fy}_{w}", np.nan) for w in REV_WIN_KEYS]
            vals = [v * 100 if pd.notna(v) else None for v in vals]
            fig.add_trace(go.Bar(
                x=REV_WIN_LABELS, y=vals, name=f"FY{fy_idx + 1}E", marker_color=FY_COLORS[fy_idx],
                opacity=0.85, hovertemplate="%{x}: %{y:+.2f}%<extra>FY" + str(fy_idx + 1) + "E</extra>",
            ), row=1, col=2)
    else:
        fig.add_annotation(
            x=0.81, y=0.5, xref="paper", yref="paper", text="No revision data",
            showarrow=False, font=dict(size=12, color=GREY_LINE),
        )

    ticker_disp = display_ticker(row["Ticker"])
    sector_line = f"<br><span style=\"font-size:12px;color:{GREY_LINE}\">{sector_industry}</span>" if sector_industry else ""
    fig.update_layout(
        height=520, width=1600,
        paper_bgcolor=DGREY, plot_bgcolor=LGREY,
        font=dict(family="Arial, sans-serif", color=TEXTCLR, size=12),
        title=dict(
            text=(
                f"<b>{ticker_disp}</b>  ·  {company_name}  ·  "
                f"R²={row['R2']:.2f}  ·  Z={row['Z_Score']:.2f}σ  ·  "
                f"10Y Relative Strength vs {row['Bench']}: {row['Relative_Strength_10Y_%']:+.0f}%{sector_line}"
            ),
            font=dict(size=15, color=TEXTCLR), x=0.03, y=0.97,
        ),
        margin=dict(t=105 if sector_industry else 90, b=40, l=60, r=40),
        hovermode="x unified",
        barmode="group",
        legend=dict(orientation="h", y=1.13, x=0.64, font=dict(size=10)),
    )
    fig.update_yaxes(title_text=f"Price / {row['Bench']} (log)", type="log", gridcolor="#555", gridwidth=0.5, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Revenue Revision %", ticksuffix="%", gridcolor="#555", gridwidth=0.5, zeroline=True, zerolinecolor="#555", row=1, col=2)
    fig.update_xaxes(gridcolor="#555", gridwidth=0.4)
    for ann in fig["layout"]["annotations"]:
        if ann.text in (f"10Y Ratio Channel vs {row['Bench']}", "Analyst Revenue Revision Trend"):
            ann.font.size = 12
            ann.font.color = TEXTCLR

    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=1.0, y=1.16,
        sizex=0.10, sizey=0.10, xanchor="right", yanchor="bottom",
        opacity=0.90, layer="above",
    ))
    return fig


def fig_to_div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def chart_anchor_id(ticker, prefix="chart"):
    return f"{prefix}-" + ticker.replace(".", "-")


def build_summary_table(rows, link_anchors=False):
    df = pd.DataFrame(rows).drop(columns=[c for c in rows[0] if c.startswith("_")])
    df = df.sort_values("R2", ascending=False).reset_index(drop=True)
    anchor_ids = df["Ticker"].map(chart_anchor_id)
    df["Ticker"] = df["Ticker"].map(display_ticker)
    cols = ["Ticker", "Bench", "Cap", "R2", "Z_Score", "10Y_Return_%", "10Y_Bench_Return_%", "Relative_Strength_10Y_%", "Annual_Trend_Return_%", "Rising_Est"]
    df = df[cols]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for (_, r), anchor_id in zip(df.iterrows(), anchor_ids):
        ticker_cell = f'<a href="#{anchor_id}">{r["Ticker"]}</a>' if link_anchors else r["Ticker"]
        html.append(
            "<tr>"
            f"<td>{ticker_cell}</td>"
            f"<td>{r['Bench']}</td>"
            f"<td>{r['Cap']}</td>"
            f"<td>{r['R2']:.2f}</td>"
            f"<td>{r['Z_Score']:.2f}</td>"
            f"<td>{r['10Y_Return_%']:.0f}%</td>"
            f"<td>{r['10Y_Bench_Return_%']:.0f}%</td>"
            f"<td>{r['Relative_Strength_10Y_%']:+.0f}%</td>"
            f"<td>{r['Annual_Trend_Return_%']:.1f}%</td>"
            f"<td>{'Yes' if r['Rising_Est'] else '—'}</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    return "".join(html)


def build_section(rows, title, subtitle, name_map, revision_map, sector_map):
    rows = sorted(rows, key=lambda r: r["R2"], reverse=True)
    keep_n = max(1, int(np.ceil(len(rows) * TOP_R2_FRACTION))) if rows else 0
    survivors = rows[:keep_n]
    print(f"[{title}] {len(rows)} candidates, keeping top {len(survivors)} by R^2 (top {TOP_R2_FRACTION:.0%})")

    parts = [f'<div class="section"><h2>{title}</h2>']
    if survivors:
        parts.append(f'<div class="section-sub">{subtitle}</div>{build_summary_table(survivors)}')
    else:
        parts.append('<div class="section-sub">No tickers passed all filters today.</div>')
    parts.append("</div>")

    for row in survivors:
        company_name = name_map.get(row["Ticker"], display_ticker(row["Ticker"]))
        revision_row = revision_map.get(row["Ticker"])
        sector, industry = sector_map.get(display_ticker(row["Ticker"]), ("", ""))
        sector_industry = " | ".join(s for s in (sector, industry) if s)
        try:
            fig = build_channel_chart(row, company_name, revision_row, sector_industry)
            anchor_id = chart_anchor_id(row["Ticker"])
            parts.append(f'<div class="chart-wrap" id="{anchor_id}" data-cap="{row["Cap"]}">{fig_to_div(fig)}</div>')
        except Exception as exc:
            print(f"  chart error {row['Ticker']}: {exc}")

    return "\n".join(parts)


def build_rising_section(rows, title, subtitle, name_map, revision_map, sector_map):
    """Every row here is also already charted once in Bottom/Top of Channel,
    so re-rendering all of them here a second time is what blew the combined
    US+Canada report past GitHub's 100MB file-size limit. The summary table
    always lists every row and links to its original chart; only the top
    RISING_CHART_CAP (by R^2) get a second, inline chart render here."""
    rows = sorted(rows, key=lambda r: r["R2"], reverse=True)
    charted = rows[:RISING_CHART_CAP]
    print(f"[{title}] {len(rows)} names, rendering charts for top {len(charted)} (cap={RISING_CHART_CAP})")

    parts = [f'<div class="section"><h2>{title}</h2>']
    if rows:
        parts.append(f'<div class="section-sub">{subtitle}</div>{build_summary_table(rows, link_anchors=True)}')
        if len(rows) > len(charted):
            parts.append(
                f'<div class="section-sub">Charts below are the top {len(charted)} of {len(rows)} by R² '
                "&mdash; use the table links above to jump to the rest.</div>"
            )
    else:
        parts.append('<div class="section-sub">No tickers passed all filters today.</div>')
        parts.append("</div>")
        return "\n".join(parts)
    parts.append("</div>")

    for row in charted:
        company_name = name_map.get(row["Ticker"], display_ticker(row["Ticker"]))
        revision_row = revision_map.get(row["Ticker"])
        sector, industry = sector_map.get(display_ticker(row["Ticker"]), ("", ""))
        sector_industry = " | ".join(s for s in (sector, industry) if s)
        try:
            fig = build_channel_chart(row, company_name, revision_row, sector_industry)
            anchor_id = chart_anchor_id(row["Ticker"], prefix="rising-chart")
            parts.append(f'<div class="chart-wrap" id="{anchor_id}" data-cap="{row["Cap"]}">{fig_to_div(fig)}</div>')
        except Exception as exc:
            print(f"  chart error {row['Ticker']}: {exc}")

    return "\n".join(parts)


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
        print(f"Downloading {bench_ticker} ({LOOKBACK_PERIOD})...")
        bench_raw = yf.download(bench_ticker, period=LOOKBACK_PERIOD, auto_adjust=True, progress=False)
        bench_close = close_series_single(bench_ticker, bench_raw)
        if bench_close is None:
            raise RuntimeError(f"Could not download {bench_ticker} close prices")
        bench_closes[bench_ticker] = bench_close

    all_tickers = sorted(ticker_info.keys())
    print(f"Downloading {len(all_tickers)} tickers ({LOOKBACK_PERIOD})...")
    close_map = batch_download_closes(all_tickers, LOOKBACK_PERIOD, CHUNK_SIZE)
    print(f"Got price history for {len(close_map)} tickers")

    rows = []
    for ticker, close in close_map.items():
        try:
            info = ticker_info[ticker]
            bench_ticker = info["bench"]
            close = close.rename(ticker)
            row = score_ticker(close, bench_closes[bench_ticker], bench_ticker)
            if row is not None:
                row["Cap"] = cap_tier(cap_map.get(ticker))
                row["Rising_Est"] = has_rising_estimates(revision_map.get(ticker))
                rows.append(row)
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")

    print(f"{len(rows)} tickers sit at a relative-strength channel extreme vs their benchmark")

    bottom_rows = [r for r in rows if r["Position"] == "Bottom"]
    top_rows = [r for r in rows if r["Position"] == "Top"]
    rising_rows = [r for r in rows if r["Rising_Est"]]

    parts = []
    parts.append(build_section(
        bottom_rows, "Bottom of Channel",
        f"{len(bottom_rows)} names, ranked by R² (10y log-ratio regression fit), highest first. "
        "A relative-strength pullback within an established outperformance trend vs benchmark.",
        name_map, revision_map, sector_map,
    ))
    parts.append(build_section(
        top_rows, "Top of Channel",
        f"{len(top_rows)} names, ranked by R² (10y log-ratio regression fit), highest first. "
        "Relative strength stretched to the top of an established outperformance trend vs benchmark.",
        name_map, revision_map, sector_map,
    ))
    parts.append(build_rising_section(
        rising_rows, "Rising Analyst Estimates",
        f"{len(rising_rows)} names from the Bottom/Top lists above where FY1E revenue estimates were also "
        "revised up over both the 1M and 3M windows &mdash; relative-strength extreme plus improving fundamentals.",
        name_map, revision_map, sector_map,
    ))

    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body="\n".join(parts), min_r2=MIN_R2,
                                 cap_css=CAP_FILTER_CSS, cap_js=CAP_FILTER_JS, cap_control=CAP_FILTER_CONTROL_HTML)
    out_path = os.path.join(OUTPUT_DIR, "Ratio_Channel_Screener_latest.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>10-Year Ratio Channel Screener</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E8E8E8; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin: 6px 0 16px; }}
  .chart-wrap {{ padding: 8px 12px; }}
  table.summary {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.summary th, table.summary td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #3A3A3C; }}
  table.summary th:first-child, table.summary td:first-child {{ text-align: left; }}
  table.summary th {{ color: #C67A29; font-weight: 600; }}
  table.summary a {{ color: #1F79BE; text-decoration: none; }}
  table.summary a:hover {{ text-decoration: underline; }}
{cap_css}
</style>
{cap_js}
</head>
<body>
<header>
  <h1>10-Year Ratio Channel Screener</h1>
  <div class="meta">Generated {date_str} &middot; Regression channel fit on price/benchmark ratio (SPY for S&amp;P 500, QQQ for Nasdaq-100, XIC for TSX) &middot; sustained relative-strength uptrend (R&sup2; &ge; {min_r2}) currently at a channel extreme &middot; Universe: S&amp;P 500 + Nasdaq-100 + US revenue-revision screener + TSX universe + Cdn revenue-revision screener</div>
  {cap_control}
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
