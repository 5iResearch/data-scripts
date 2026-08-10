"""
Daily Regime Monitors report, adapted from the "Regime Cross-Asset Monitor",
"Regime Sector Industry Monitor", and "Regime Themes Monitor" notebooks.

Only the ranked-snapshot bar charts and 20-day displacement quadrant charts
are reproduced here (not the signal-status tables or the dozens of
per-ticker price/regime charts those notebooks also generate). Bundles
everything into a single self-contained HTML report:
  Cross-Asset            (bar, quadrant)
  Sector / Industry       (sectors bar+quadrant, industries bar+quadrant)
  Themes                  (bar, quadrant) -- includes Crypto & Blockchain group
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "regime-monitors")

DGRAY = "#1C1C1E"
MGRAY = "#2C2C2E"
LGRAY = "#3A3A3C"
TEXT = "#E5E5EA"
SUBTEXT = "#8E8E93"
GREEN = "#2ECC71"
RED = "#E74C3C"

Z_WINDOW = 252
LOOKBACKS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
COMPARE_DAYS = 20


def hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# ── Universe definitions ──────────────────────────────────────────────────────
CROSS_ASSET_CLR = {
    "US Equity": "#1F79BE", "Intl Equity": "#5BA3D9", "EM Equity": "#3AB5E6",
    "Rates": "#2ECC71", "Credit": "#27AE60", "EM Debt": "#1ABC9C",
    "Commodity": "#C67A29", "Real Assets": "#E67E22", "FX": "#9B59B6",
}
CROSS_ASSET_UNIVERSE = [
    ("US Large Cap", "SPY", "US Equity"), ("US Tech", "QQQ", "US Equity"),
    ("US Small Cap", "IWM", "US Equity"), ("US Value", "VTV", "US Equity"),
    ("Dev. Intl", "EFA", "Intl Equity"), ("Europe", "VGK", "Intl Equity"),
    ("Japan", "EWJ", "Intl Equity"), ("EM Equities", "EEM", "EM Equity"),
    ("China", "MCHI", "EM Equity"), ("Long Treasuries", "TLT", "Rates"),
    ("Mid Treasuries", "IEF", "Rates"), ("Short Treasuries", "SHY", "Rates"),
    ("TIPS", "TIP", "Rates"), ("IG Corp", "LQD", "Credit"),
    ("High Yield", "HYG", "Credit"), ("EM Bonds", "EMB", "EM Debt"),
    ("Gold", "GLD", "Commodity"), ("Silver", "SLV", "Commodity"),
    ("Oil (WTI)", "USO", "Commodity"), ("Broad Comm.", "DBC", "Commodity"),
    ("REITs", "VNQ", "Real Assets"), ("Gold Miners", "GDX", "Real Assets"),
    ("USD Index", "UUP", "FX"),
]

SECTOR_CLR = {
    "Technology": "#1F79BE", "Health Care": "#16A34A", "Financials": "#CA8A04",
    "Energy": "#EA580C", "Industrials": "#78716C", "Cons. Discret.": "#DB2777",
    "Cons. Staples": "#0D9488", "Utilities": "#7C3AED", "Real Estate": "#92400E",
    "Materials": "#65A30D", "Comm. Services": "#0891B2", "Crypto": "#F7931A",
}
SECTOR_UNIVERSE = [
    ("Technology", "XLK", "Technology"), ("Semiconductors", "SOXX", "Technology"),
    ("Software", "IGV", "Technology"), ("Cybersecurity", "CIBR", "Technology"),
    ("Cloud Computing", "SKYY", "Technology"),
    ("Health Care", "XLV", "Health Care"), ("Biotech", "XBI", "Health Care"),
    ("Pharmaceuticals", "PJP", "Health Care"), ("Medical Devices", "IHI", "Health Care"),
    ("HC Providers", "IHF", "Health Care"),
    ("Financials", "XLF", "Financials"), ("Banks", "KBE", "Financials"),
    ("Regional Banks", "KRE", "Financials"), ("Insurance", "KIE", "Financials"),
    ("Broker-Dealers", "IAI", "Financials"),
    ("Energy", "XLE", "Energy"), ("Oil & Gas E&P", "XOP", "Energy"),
    ("Oil Services", "OIH", "Energy"), ("Midstream/MLPs", "AMLP", "Energy"),
    ("Industrials", "XLI", "Industrials"), ("Aerospace & Def.", "ITA", "Industrials"),
    ("Transportation", "IYT", "Industrials"), ("Infrastructure", "PAVE", "Industrials"),
    ("Cons. Discret.", "XLY", "Cons. Discret."), ("Retail", "XRT", "Cons. Discret."),
    ("Homebuilders", "XHB", "Cons. Discret."), ("Autos & EV", "CARZ", "Cons. Discret."),
    ("Cons. Staples", "XLP", "Cons. Staples"), ("Food & Beverage", "PBJ", "Cons. Staples"),
    ("Utilities", "XLU", "Utilities"),
    ("Real Estate", "XLRE", "Real Estate"), ("REITs", "VNQ", "Real Estate"),
    ("Residential REITs", "REZ", "Real Estate"),
    ("Materials", "XLB", "Materials"), ("Gold Miners", "GDX", "Materials"),
    ("Metals & Mining", "XME", "Materials"), ("Agriculture", "MOO", "Materials"),
    ("Comm. Services", "XLC", "Comm. Services"), ("Telecom", "IYZ", "Comm. Services"),
    ("Social Media", "SOCL", "Comm. Services"), ("Gaming & Esports", "HERO", "Comm. Services"),
    ("Bitcoin (IBIT)", "IBIT", "Crypto"), ("Ethereum (ETHA)", "ETHA", "Crypto"),
]
SECTOR_BENCHMARK = "SPY"

THEME_CLR = {
    "ARK": "#7C3AED", "AI & Robotics": "#1F79BE", "Cybersecurity": "#0D9488",
    "Defense & Space": "#78716C", "Quantum/Deep Tech": "#A855F7",
    "Cloud & Software": "#3B82F6", "Clean Energy": "#16A34A", "Genomics": "#65A30D",
    "Fintech & Payments": "#CA8A04", "Crypto & Blockchain": "#EA580C",
    "EV & Auto Tech": "#0891B2", "Digital & Media": "#DB2777",
    "Infrastructure": "#92400E", "Cannabis": "#15803D",
}
THEME_UNIVERSE = [
    ("ARK Innovation", "ARKK", "ARK"), ("ARK Next Gen Internet", "ARKW", "ARK"),
    ("ARK Fintech", "ARKF", "ARK"), ("ARK Genomics", "ARKG", "ARK"),
    ("ARK Robotics/Auto", "ARKQ", "ARK"), ("ARK Space", "ARKX", "ARK"),
    ("Robotics & AI", "BOTZ", "AI & Robotics"), ("AI & Big Data", "AIQ", "AI & Robotics"),
    ("Generative AI", "CHAT", "AI & Robotics"), ("ROBO Global Robotics", "ROBO", "AI & Robotics"),
    ("Semiconductor", "SMH", "AI & Robotics"),
    ("Cybersecurity", "HACK", "Cybersecurity"), ("NASDAQ Cyber", "CIBR", "Cybersecurity"),
    ("Global X Cyber", "BUG", "Cybersecurity"),
    ("Aerospace & Defense", "ITA", "Defense & Space"), ("Space Exploration ETF", "UFO", "Defense & Space"),
    ("Capital Group Defense", "JEDI", "Defense & Space"), ("Drone Economy", "DRNZ", "Defense & Space"),
    ("Quantum Computing", "QTUM", "Quantum/Deep Tech"), ("Kensho Future/Security", "XKST", "Quantum/Deep Tech"),
    ("Cloud Computing", "SKYY", "Cloud & Software"), ("WisdomTree Cloud", "WCLD", "Cloud & Software"),
    ("Software IGV", "IGV", "Cloud & Software"),
    ("Global Clean Energy", "ICLN", "Clean Energy"), ("Solar", "TAN", "Clean Energy"),
    ("Wind", "FAN", "Clean Energy"), ("Clean Edge Energy", "QCLN", "Clean Energy"),
    ("Uranium", "URA", "Clean Energy"),
    ("Genomics & Biotech", "GNOM", "Genomics"), ("Genomics iShares", "IDNA", "Genomics"),
    ("Biotech XBI", "XBI", "Genomics"),
    ("Global Fintech", "FINX", "Fintech & Payments"), ("Mobile Payments", "IPAY", "Fintech & Payments"),
    ("Bitcoin Spot ETF", "IBIT", "Crypto & Blockchain"), ("Blockchain", "BLOK", "Crypto & Blockchain"),
    ("Crypto Equity", "WGMI", "Crypto & Blockchain"),
    ("Autonomous & EV", "DRIV", "EV & Auto Tech"), ("EV Global", "KARS", "EV & Auto Tech"),
    ("Self-Driving EV", "IDRV", "EV & Auto Tech"),
    ("Social Media", "SOCL", "Digital & Media"), ("Video Games & Esports", "HERO", "Digital & Media"),
    ("Magnificent 7", "MAGS", "Digital & Media"),
    ("US Infrastructure", "PAVE", "Infrastructure"),
    ("Global Cannabis", "MJ", "Cannabis"), ("US Cannabis", "MSOS", "Cannabis"),
]


def momentum_zscore(prices, lookback_days, z_window=Z_WINDOW):
    ret = prices.pct_change(lookback_days)
    mu = ret.rolling(z_window, min_periods=z_window // 2).mean()
    sd = ret.rolling(z_window, min_periods=z_window // 2).std()
    return (ret - mu) / sd


def rel_momentum_zscore(prices, benchmark, lookback_days, z_window=Z_WINDOW):
    asset_ret = prices.pct_change(lookback_days)
    bench_ret = benchmark.pct_change(lookback_days)
    excess = asset_ret.subtract(bench_ret, axis=0)
    mu = excess.rolling(z_window, min_periods=z_window // 2).mean()
    sd = excess.rolling(z_window, min_periods=z_window // 2).std()
    return (excess - mu) / sd


def compute_universe(universe, start, min_history=126, benchmark=None):
    tickers = [u[1] for u in universe]
    names = {u[1]: u[0] for u in universe}
    groups = {u[1]: u[2] for u in universe}
    labels = {t: f"{names[t]} [{t}]" for t in tickers}

    dl_tickers = tickers + ([benchmark] if benchmark else [])
    print(f"Downloading {len(dl_tickers)} tickers...")
    raw = yf.download(dl_tickers, start=start, auto_adjust=True, progress=False)
    px_all = raw["Close"].ffill()

    valid_count = px_all.reindex(columns=tickers).notna().sum()
    available = [t for t in tickers if valid_count.get(t, 0) >= min_history]
    px = px_all[available]

    if benchmark:
        bench = px_all[benchmark]
        z_frames = {lb: rel_momentum_zscore(px, bench, d, Z_WINDOW) for lb, d in LOOKBACKS.items()}
    else:
        z_frames = {lb: momentum_zscore(px[available], d, Z_WINDOW) for lb, d in LOOKBACKS.items()}

    z_comp = pd.concat(z_frames.values()).groupby(level=0).mean().sort_index()
    z_d20 = z_comp.diff(20)
    today = z_comp.index[-1]
    z_d20_norm = pd.DataFrame({
        t: (z_d20[t] - z_d20[t].rolling(252, min_periods=63).mean()) / z_d20[t].rolling(252, min_periods=63).std()
        for t in available
    })

    return dict(
        available=available, names=names, groups=groups, labels=labels,
        z_comp=z_comp, z_d20=z_d20, z_d20_norm=z_d20_norm, today=today,
    )


def build_bar_snapshot(ctx, ticker_list, group_colors, title_label, benchmark=None):
    z_comp, z_d20_norm, today, labels, groups = (
        ctx["z_comp"], ctx["z_d20_norm"], ctx["today"], ctx["labels"], ctx["groups"]
    )
    sorted_tickers = z_comp.loc[today, ticker_list].sort_values(ascending=True).index.tolist()
    sorted_labels = [labels[t] for t in sorted_tickers]
    z_label = f"z (vs {benchmark})" if benchmark else "z"

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.08)

    for t in sorted_tickers:
        clr = group_colors[groups[t]]
        zv = float(z_comp.loc[today, t]) if pd.notna(z_comp.loc[today, t]) else 0.0
        dv = float(z_d20_norm.loc[today, t]) if pd.notna(z_d20_norm.loc[today, t]) else 0.0
        dclr = GREEN if dv > 0 else RED

        fig.add_trace(go.Bar(
            x=[zv], y=[labels[t]], orientation="h", marker_color=clr,
            text=[f"{zv:+.2f}"], textposition="outside", textfont=dict(size=9, color=TEXT),
            legendgroup=t, name=labels[t], showlegend=True,
            hovertemplate=f"<b>{labels[t]}</b><br>{z_label}: {zv:+.2f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=[dv], y=[labels[t]], orientation="h", marker_color=dclr,
            text=[f"{dv:+.2f}"], textposition="outside", textfont=dict(size=9, color=TEXT),
            legendgroup=t, showlegend=False,
            hovertemplate=f"<b>{labels[t]}</b><br>d(norm): {dv:+.2f}<extra></extra>",
        ), row=1, col=2)

    fig.add_vline(x=0, line_color=SUBTEXT, line_width=1, row=1, col=1)
    fig.add_vline(x=1, line_color=GREEN, line_width=0.8, line_dash="dot", row=1, col=1)
    fig.add_vline(x=-1, line_color=RED, line_width=0.8, line_dash="dot", row=1, col=1)
    fig.add_vline(x=0, line_color=SUBTEXT, line_width=1, row=1, col=2)

    all_zv = [float(z_comp.loc[today, t]) for t in ticker_list if pd.notna(z_comp.loc[today, t])]
    all_dv = [float(z_d20_norm.loc[today, t]) for t in ticker_list if pd.notna(z_d20_norm.loc[today, t])]

    bench_note = f"  &middot;  z-scores relative to {benchmark}" if benchmark else ""
    fig.update_layout(
        title=dict(
            text=f"<b>{title_label} Momentum Ranked Snapshot</b>  "
                 f'<span style="font-size:12px;color:{SUBTEXT}">{today.strftime("%b %d, %Y")}{bench_note}'
                 f"  &middot;  colour = group</span>",
            font=dict(size=15, color=TEXT), x=0.01,
        ),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY,
        font=dict(color=TEXT, family="monospace"),
        height=max(500, len(ticker_list) * 34 + 120),
        bargap=0.25, showlegend=False,
        margin=dict(l=220, r=160, t=70, b=50),
    )
    fig.add_annotation(x=0.01, y=1.04, xref="paper", yref="paper",
        text=f'<b>Composite {"Relative " if benchmark else ""}Z-Score</b>  '
             f'<span style="color:{SUBTEXT}">sorted weakest &rarr; strongest</span>',
        font=dict(color=TEXT, size=11), showarrow=False, xanchor="left", yanchor="bottom")
    fig.add_annotation(x=0.58, y=1.04, xref="paper", yref="paper",
        text=f'<b>20-Day Delta</b>  <span style="color:{SUBTEXT}">normalized &middot; green = gaining / red = losing</span>',
        font=dict(color=TEXT, size=11), showarrow=False, xanchor="left", yanchor="bottom")

    fig.update_xaxes(range=[min(all_zv) - 0.8, max(all_zv) + 0.8], gridcolor=LGRAY, zeroline=False,
                      tickfont=dict(color=TEXT), row=1, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=sorted_labels, gridcolor=LGRAY,
                      tickfont=dict(color=TEXT, size=9), row=1, col=1)
    fig.update_xaxes(range=[min(all_dv) - 0.8, max(all_dv) + 0.8], gridcolor=LGRAY, zeroline=False,
                      tickfont=dict(color=TEXT), row=1, col=2)
    fig.update_yaxes(categoryorder="array", categoryarray=sorted_labels, showticklabels=False,
                      gridcolor=LGRAY, row=1, col=2)
    return fig


def build_quadrant(ctx, ticker_list, group_colors, chart_title, benchmark=None):
    z_comp, z_d20, today, labels, groups = ctx["z_comp"], ctx["z_d20"], ctx["today"], ctx["labels"], ctx["groups"]
    past_loc = z_comp.index.get_loc(today) - COMPARE_DAYS
    past_idx = z_comp.index[past_loc]

    snap_z_d, past_z_d = z_comp.loc[today], z_comp.loc[past_idx]
    snap_d20, past_d20 = z_d20.loc[today], z_d20.loc[past_idx]

    disp_valid = [t for t in ticker_list if all(
        pd.notna(v) for v in [snap_z_d.get(t), past_z_d.get(t), snap_d20.get(t), past_d20.get(t)]
    )]
    if not disp_valid:
        return None

    def arrow_color(t):
        score = (1 if snap_z_d[t] > past_z_d[t] else -1) + (1 if snap_d20[t] > past_d20[t] else -1)
        return GREEN if score > 0 else (RED if score < 0 else "#F4D03F")

    all_xs = [snap_z_d[t] for t in disp_valid] + [past_z_d[t] for t in disp_valid]
    all_ys = [snap_d20[t] for t in disp_valid] + [past_d20[t] for t in disp_valid]
    xpad = (max(all_xs) - min(all_xs)) * 0.22
    ypad = (max(all_ys) - min(all_ys)) * 0.22
    x_rng = [min(all_xs) - xpad, max(all_xs) + xpad]
    y_rng = [min(all_ys) - ypad, max(all_ys) + ypad]

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=y_rng[1] + 1, fillcolor="rgba(46,204,113,0.05)", line_width=0)
    fig.add_hrect(y0=y_rng[0] - 1, y1=0, fillcolor="rgba(231,76,60,0.05)", line_width=0)
    fig.add_hline(y=0, line_color=SUBTEXT, line_width=1)
    fig.add_vline(x=0, line_color=SUBTEXT, line_width=1)

    xb = x_rng[1] * 0.78
    yb = y_rng[1] * 0.75
    for lbl, xp, yp, clr in [
        ("LEADING", xb, yb, "#2ECC71"), ("WEAKENING", xb, -yb, "#F4D03F"),
        ("IMPROVING", -xb, yb, "#1F79BE"), ("LAGGING", -xb, -yb, "#E74C3C"),
    ]:
        fig.add_annotation(x=xp, y=yp, text=f"<b>{lbl}</b>", font=dict(color=clr, size=13),
                            showarrow=False, opacity=0.5)

    for t in disp_valid:
        clr = group_colors[groups[t]]
        aclr = arrow_color(t)
        xn, yn = float(snap_z_d[t]), float(snap_d20[t])
        xw, yw = float(past_z_d[t]), float(past_d20[t])

        fig.add_trace(go.Scatter(x=[xw, xn], y=[yw, yn], mode="lines",
            line=dict(color=hex_rgba(aclr, 0.20), width=1.5), showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=xn, y=yn, ax=xw, ay=yw, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=2.2, arrowcolor=aclr)
        fig.add_trace(go.Scatter(x=[xw], y=[yw], mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=9, line=dict(color=hex_rgba(clr, 0.4), width=1.5)),
            showlegend=False,
            hovertemplate=f'<b>{labels[t]}</b> ({past_idx.strftime("%b %d")})'
                          f"<br>Z:{xw:.2f}  &Delta;:{yw:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[xn], y=[yn], mode="markers+text",
            marker=dict(color=clr, size=11, line=dict(color=TEXT, width=0.8)),
            text=[f"  {labels[t]}"], textposition="middle right", textfont=dict(size=8, color=TEXT),
            showlegend=False,
            hovertemplate=f"<b>{labels[t]}</b><br>Z:{xn:.2f} (was {xw:.2f})"
                          f"<br>&Delta;:{yn:.2f} (was {yw:.2f})<extra></extra>"))

    seen_groups = set(groups[t] for t in disp_valid)
    for grp, clr in group_colors.items():
        if grp in seen_groups:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                      marker=dict(color=clr, size=10, symbol="circle"), name=grp))

    fig.add_annotation(x=0.01, y=0.01, xref="paper", yref="paper",
        text=f'&#9675; = {past_idx.strftime("%b %d")}   &#9679; = {today.strftime("%b %d")}',
        font=dict(color=SUBTEXT, size=10), showarrow=False, xanchor="left", yanchor="bottom")

    bench_note = f"  &middot;  z-scores relative to {benchmark}" if benchmark else ""
    x_title = f"Composite {'Relative ' if benchmark else ''}Z-Score" + (f" (vs {benchmark})" if benchmark else "")
    fig.update_layout(
        title=dict(
            text=f"<b>{chart_title} &mdash; 20-Day Displacement</b>  "
                 f'<span style="font-size:11px;color:{SUBTEXT}">'
                 f'{past_idx.strftime("%b %d")} &rarr; {today.strftime("%b %d, %Y")}{bench_note}'
                 f"  &middot;  arrow = direction of travel</span>",
            font=dict(size=16, color=TEXT), x=0.01,
        ),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY,
        font=dict(color=TEXT, family="monospace"),
        height=900,
        margin=dict(l=60, r=60, t=90, b=120),
        xaxis=dict(title=x_title, title_font=dict(color=SUBTEXT), range=x_rng, gridcolor=LGRAY, zeroline=False),
        yaxis=dict(title="20-Day &Delta; Z", title_font=dict(color=SUBTEXT), range=y_rng, gridcolor=LGRAY, zeroline=False),
        hovermode="closest",
        legend=dict(bgcolor="rgba(28,28,30,0.85)", font=dict(color=TEXT, size=9), orientation="h", x=0, y=-0.18),
    )
    return fig


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def build_report() -> str:
    parts = []

    print("=== Cross-Asset ===")
    ctx_ca = compute_universe(CROSS_ASSET_UNIVERSE, start="2012-01-01")
    parts.append(section_header("Cross-Asset Monitor", "Composite momentum z-score across major asset classes"))
    parts.append(fig_to_div(build_bar_snapshot(ctx_ca, ctx_ca["available"], CROSS_ASSET_CLR, "Cross-Asset")))
    fig = build_quadrant(ctx_ca, ctx_ca["available"], CROSS_ASSET_CLR, "Cross-Asset")
    if fig:
        parts.append(fig_to_div(fig))

    print("=== Sector / Industry ===")
    ctx_si = compute_universe(SECTOR_UNIVERSE, start="2015-01-01", benchmark=SECTOR_BENCHMARK)
    sector_etfs = {u[1] for u in SECTOR_UNIVERSE if u[0] == u[2]}
    available_s = [t for t in ctx_si["available"] if t in sector_etfs]
    available_i = [t for t in ctx_si["available"] if t not in sector_etfs]

    parts.append(section_header("Sector Monitor", f"Momentum z-score relative to {SECTOR_BENCHMARK}"))
    parts.append(fig_to_div(build_bar_snapshot(ctx_si, available_s, SECTOR_CLR, "Sector", benchmark=SECTOR_BENCHMARK)))
    fig = build_quadrant(ctx_si, available_s, SECTOR_CLR, "Sectors", benchmark=SECTOR_BENCHMARK)
    if fig:
        parts.append(fig_to_div(fig))

    parts.append(section_header("Industry Monitor", f"Momentum z-score relative to {SECTOR_BENCHMARK}"))
    parts.append(fig_to_div(build_bar_snapshot(ctx_si, available_i, SECTOR_CLR, "Industry", benchmark=SECTOR_BENCHMARK)))
    fig = build_quadrant(ctx_si, available_i, SECTOR_CLR, "Industries", benchmark=SECTOR_BENCHMARK)
    if fig:
        parts.append(fig_to_div(fig))

    print("=== Themes ===")
    ctx_th = compute_universe(THEME_UNIVERSE, start="2017-01-01")
    parts.append(section_header("Themes Monitor", "Composite momentum z-score across thematic ETFs (includes Crypto & Blockchain)"))
    parts.append(fig_to_div(build_bar_snapshot(ctx_th, ctx_th["available"], THEME_CLR, "Theme")))
    fig = build_quadrant(ctx_th, ctx_th["available"], THEME_CLR, "Themes")
    if fig:
        parts.append(fig_to_div(fig))

    return "\n".join(parts)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Regime Monitors Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E5E5EA; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E5E5EA; border-bottom: 2px solid #1F79BE; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin-top: 6px; }}
</style>
</head>
<body>
<header>
  <h1>Regime Monitors Report</h1>
  <div class="meta">Generated {date_str} &middot; Cross-Asset, Sector/Industry, and Themes momentum regimes &middot; Ranked snapshot + 20-day displacement quadrant</div>
</header>
{body}
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()
    body = build_report()
    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body=body)

    out_path = os.path.join(OUTPUT_DIR, "Regime_Monitors_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
