"""
Daily "Revenue Revision Screener" report — the pure analyst-estimate-revision
half of generate_rev_revision_benchmark_beaters.py, with the 6-month
relative-high-vs-benchmark price gate removed entirely. No price data is
downloaded; ranking is driven only by the revenue estimate revision CSVs.

Three sections, same universes as the benchmark-beaters report:
  1. US S&P 500 (revision CSV filtered to current S&P 500 constituents)
  2. All-US (~2,000-name CSV universe)
  3. Canada (CDN revision CSV)

Depends on the same two manually-refreshed CSVs:
  data/us_1w_rev_est_screener.csv
  data/cdn_1w_rev_est_screener.csv
"""

import os
import warnings
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "rev-revision-screener")
US_CSV_PATH = os.path.join(REPO_ROOT, "data", "us_1w_rev_est_screener.csv")
CDN_CSV_PATH = os.path.join(REPO_ROOT, "data", "cdn_1w_rev_est_screener.csv")

ORANGE, BLUE, DBLUE, GREEN, RED = "#C67A29", "#1F79BE", "#4B8EA9", "#44A660", "#A22A2A"
BG, PANEL, GRID, TEXT, SUBTEXT = "#0f1117", "#1a1d27", "#2d3148", "#e2e5f0", "#8b90a4"

TIER_COLORS = {
    "Structural Cascade": ORANGE, "Full Cascade": BLUE, "Strong": DBLUE,
    "Partial": "#297ABC", "Weak": SUBTEXT,
}

TODAY = datetime.today().strftime("%Y-%m-%d")

SPOTLIGHT_TOP_N = 40

COL_MAP = {
    "Ticker": "ticker", "Name": "name",
    "Last Price": "price", "Market Cap": "mktcap",
    "Trading Region": "region", "Trading Country": "country",
    "Below 52W High %": "below_52wh",
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

WINDOWS = ["1m", "3m", "6m", "1y"]
WINDOW_X = [1, 3, 6, 12]
FY_PERIODS = ["fy1", "fy2", "fy3"]
ALL_WIN_KEYS = ["1w", "1m", "3m", "6m", "1y"]
ALL_WIN_LABELS = ["1W", "1M", "3M", "6M", "1Y"]
FY_COLORS_P = [BLUE, DBLUE, ORANGE]

DISPLAY_COLS = [
    ("rank", "Rank"), ("ticker", "Ticker"), ("name", "Name"), ("tier", "Tier"),
    ("cascade_score", "Cascade (0-9)"), ("avg_1w", "Avg 1W Rev%"),
    ("fy1_1w", "FY1E 1W"), ("fy2_1w", "FY2E 1W"), ("fy3_1w", "FY3E 1W"),
    ("avg_magnitude_1y", "Avg 1Y Rev%"), ("all_fy_pos_1w", "All FY 1W+"),
]


def load_rev_csv(path):
    raw = pd.read_csv(path)
    df = raw.rename(columns=COL_MAP)
    df = df[[c for c in COL_MAP.values() if c in df.columns]].copy()
    for c in [c for c in df.columns if c.startswith("fy")] + ["price", "mktcap", "below_52wh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fy1_1w", "fy2_1w"], how="all").reset_index(drop=True)
    return df


def score_row(row):
    res = {}
    cascade_total = 0
    magnitude_1y, momentum_1w = [], []
    all_fy_pos_1w = []

    for fy in FY_PERIODS:
        v1w = row.get(f"{fy}_1w", np.nan)
        vals = [row.get(f"{fy}_{w}", np.nan) for w in WINDOWS]
        v1m, v3m, v6m, v1y = vals

        c1 = int(pd.notna(v1y) and pd.notna(v6m) and v1y > v6m)
        c2 = int(pd.notna(v6m) and pd.notna(v3m) and v6m > v3m)
        c3 = int(pd.notna(v3m) and pd.notna(v1m) and v3m > v1m)
        cascade_fy = c1 + c2 + c3
        cascade_total += cascade_fy
        res[f"{fy}_cascade"] = cascade_fy

        if pd.notna(v1y): magnitude_1y.append(v1y)
        if pd.notna(v1w): momentum_1w.append(v1w)
        all_fy_pos_1w.append(pd.notna(v1w) and v1w > 0)

    res["cascade_score"] = cascade_total
    res["avg_magnitude_1y"] = np.nanmean(magnitude_1y) if magnitude_1y else np.nan
    res["avg_1w"] = np.nanmean(momentum_1w) if momentum_1w else np.nan
    res["all_fy_pos_1w"] = all(all_fy_pos_1w)
    return pd.Series(res)


def winsorize_zscore(series, pct=2):
    s = series.dropna()
    if len(s) < 5:
        return series * 0
    lo, hi = np.nanpercentile(s, pct), np.nanpercentile(s, 100 - pct)
    clipped = series.clip(lo, hi)
    mean, std = np.nanmean(clipped), np.nanstd(clipped)
    return (clipped - mean) / std if std > 0 else clipped * 0


def assign_tier(row):
    cs = row["cascade_score"]
    mag = row["avg_magnitude_1y"]
    if cs == 9 and pd.notna(mag) and mag > 0.10:
        return "Structural Cascade"
    elif cs == 9:
        return "Full Cascade"
    elif cs >= 7:
        return "Strong"
    elif cs >= 5:
        return "Partial"
    return "Weak"


def rank_by_revisions(rev_df, extra_gate=None):
    joined = rev_df.dropna(subset=["fy1_1w"]).reset_index(drop=True)
    joined = joined[(joined["fy1_1w"] > 0) & (joined["fy2_1w"] > 0)]
    if extra_gate is not None:
        joined = joined[extra_gate(joined)]
    joined = joined.reset_index(drop=True)
    if joined.empty:
        return joined

    s = joined.copy()
    s["z_1w"] = winsorize_zscore(s["avg_1w"])
    s["z_cascade"] = winsorize_zscore(s["cascade_score"])
    s["z_mag1y"] = winsorize_zscore(s["avg_magnitude_1y"])
    # No price data here, so the composite drops the rel-6M and streak terms
    # from the benchmark-beaters report and reweights across what's left:
    # 1W revision momentum, cascade confirmation, and 1Y revision magnitude.
    s["combined"] = s["z_1w"] * 0.45 + s["z_cascade"] * 0.35 + s["z_mag1y"] * 0.20
    s["tier"] = s.apply(assign_tier, axis=1)
    ranked = s.sort_values("combined", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def pct(val, d=2):
    return f"{val * 100:+.{d}f}%" if pd.notna(val) else "—"


def style_tier(val):
    return f"color: {TIER_COLORS.get(val, TEXT)}; font-weight: bold"


def build_table_html(ranked, caption_label):
    cols = DISPLAY_COLS
    top40 = ranked.head(40)[[c for c, _ in cols]].rename(columns=dict(cols)).copy()

    for col in ["Avg 1W Rev%", "FY1E 1W", "FY2E 1W", "FY3E 1W", "Avg 1Y Rev%"]:
        top40[col] = top40[col].apply(pct)
    top40["Cascade (0-9)"] = top40["Cascade (0-9)"].apply(lambda v: f"{int(v)}/9" if pd.notna(v) else "—")
    top40["All FY 1W+"] = top40["All FY 1W+"].map({True: "✓", False: ""})

    styler = (
        top40.style
        .map(style_tier, subset=["Tier"])
        .set_properties(**{"background-color": PANEL, "color": TEXT, "border": f"1px solid {GRID}"})
        .set_table_styles([{"selector": "th", "props": [
            ("background-color", "#252840"), ("color", "#e2e5f0"),
            ("font-weight", "bold"), ("border", f"1px solid {GRID}")]}])
        .set_caption(f"Top 40 — {caption_label} | {TODAY} | "
                     "1W rev (45%) + cascade (35%) + avg 1Y rev magnitude (20%)")
        .hide(axis="index")
    )
    return styler.to_html()


def make_spotlight(row):
    ticker, name, tier = row["ticker"], row["name"], row["tier"]
    tc = TIER_COLORS.get(tier, SUBTEXT)
    rank, cs = int(row["rank"]), int(row["cascade_score"])
    avg1w = row["avg_1w"]
    w1_str = f"{avg1w * 100:+.2f}%" if pd.notna(avg1w) else "—"

    fig = go.Figure()
    for fy_idx, fy in enumerate(["fy1", "fy2", "fy3"]):
        vals = [row.get(f"{fy}_{w}", np.nan) * 100 for w in ALL_WIN_KEYS]
        fig.add_trace(go.Bar(x=ALL_WIN_LABELS, y=vals, name=f"FY{fy_idx + 1}E", marker_color=FY_COLORS_P[fy_idx],
            opacity=0.85, hovertemplate="%{x}: %{y:+.2f}%<extra>FY" + str(fy_idx + 1) + "E</extra>"))

    subtitle = f"Rank #{rank} | {tier} | Cascade {cs}/9 | Avg 1W rev: {w1_str}"

    fig.update_layout(
        title=dict(text=(f'<b><span style="font-size:26px;color:{tc}">{ticker}</span>'
                         f'<span style="font-size:18px;color:#e2e5f0">  {name}</span></b><br>'
                         f'<span style="font-size:12px;color:#8b90a4">{subtitle}</span>'),
                   x=0.0, xanchor="left", y=0.93, yanchor="top", pad=dict(l=10, t=10)),
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TEXT, family="DejaVu Sans, Arial"),
        height=340, margin=dict(l=60, r=40, t=100, b=50), barmode="group", showlegend=True,
        legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1, x=1.0, y=1.0, xanchor="right", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=GRID, title_text="Revenue Revision %", ticksuffix="%")
    return fig


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def build_section(ranked, table_caption, section_title, section_sub):
    parts = [section_header(section_title, section_sub)]
    if ranked.empty:
        parts.append('<div class="section" style="color:#8b90a4;">No names passed the screen today.</div>')
        return parts
    parts.append(f'<div class="table-wrap">{build_table_html(ranked, table_caption)}</div>')

    for _, row in ranked.head(SPOTLIGHT_TOP_N).iterrows():
        fig = make_spotlight(row)
        parts.append(fig_to_div(fig))
    return parts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parts = []

    # ── Section 1: US S&P 500 ────────────────────────────────────────────────
    print("=== US S&P 500 ===")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    us_symbols = set(pd.read_html(StringIO(resp.text))[0]["Symbol"].str.replace(".", "-", regex=False).tolist())

    us_rev_df = load_rev_csv(US_CSV_PATH)
    us_scores = us_rev_df.apply(score_row, axis=1)
    us_rev_df = pd.concat([us_rev_df, us_scores], axis=1)

    sp500_rev_df = us_rev_df[us_rev_df["ticker"].isin(us_symbols)].reset_index(drop=True)
    sp500_ranked = rank_by_revisions(sp500_rev_df)
    parts += build_section(sp500_ranked, "US S&P 500 Revenue Revision Screener",
                            "Section 1 - US S&P 500", "Gated on FY1E+FY2E 1W revisions positive, no price/trend filter")

    # ── Section 2: All-US ~2,000 names ───────────────────────────────────────
    print("=== All-US ~2,000 names ===")
    all_us_ranked = rank_by_revisions(us_rev_df)
    parts += build_section(all_us_ranked, "All-US Revenue Revision Screener (Full CSV ~2k)",
                            "Section 2 - All-US (~2,000-Name CSV Universe)",
                            "Same screen, full revision-CSV universe rather than just S&P 500")

    # ── Section 3: Canada ────────────────────────────────────────────────────
    print("=== Canada ===")
    cdn_rev_df = load_rev_csv(CDN_CSV_PATH)
    cdn_scores = cdn_rev_df.apply(score_row, axis=1)
    cdn_rev_df = pd.concat([cdn_rev_df, cdn_scores], axis=1)

    def cdn_gate(df):
        return (df["fy1_1m"] > 0) & (df["fy2_1m"] > 0)

    cdn_ranked = rank_by_revisions(cdn_rev_df, extra_gate=cdn_gate)
    parts += build_section(cdn_ranked, "CDN Revenue Revision Screener",
                            "Section 3 - Canada",
                            "Gated on FY1E+FY2E 1W AND 1M revisions positive, no price/trend filter")

    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "Rev_Revision_Screener.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Revenue Revision Screener</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #0f1117; color: #c8ccd8; font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #2d3148; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; color: #e2e5f0; }}
  header .meta {{ color: #8b90a4; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #e2e5f0; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8b90a4; font-size: 13px; margin-top: 6px; }}
  .table-wrap {{ padding: 12px 32px 8px; overflow-x: auto; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  caption {{ color: #e2e5f0; text-align: left; padding-bottom: 8px; font-size: 13px; }}
</style>
</head>
<body>
<header>
  <h1>Revenue Revision Screener</h1>
  <div class="meta">Generated {date_str} &middot; Stocks with confirming analyst revenue estimate revisions, no price/relative-high filter &middot; Top 40 per section</div>
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
