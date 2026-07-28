"""
Daily VIX Structure report, adapted from the "VIX Structure" notebook.

Reproduces a specific subset of that notebook's charts (not the full
notebook, which also includes a VIX-VIX3M predictor and weekly-RSI section
not requested here):
  1. Term Structure - Curve Shifts Over Time
  2. Term Structure - 5-Year History
  3. Term Structure Heatmap - 90 Days
  4. SPY Forward Returns by VIX Regime
  5. SPY Forward Returns - VIX9D Analysis
  6. VIX Bottom Signal Dashboard
  7. VIX - VIX3M Spread vs S&P 500
  8. SPY Forward Returns by VIX-VIX3M Spread Level
  9. VIX Spike -> All-Clear: episode timeline, forward returns, per-episode heatmap
  10. Long-history level analysis: VIX/VIX3M level -> SPX/TSX forward returns
      (8-bucket heatmaps, time-spent bars, VIX x VIX3M combo heatmaps across
      1m/3m/6m/1y, and 1-point-interval bar charts)
"""

import base64
import os
import warnings
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "vix-structure")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")

DGRAY, MGRAY, LGRAY = "#1C1C1E", "#2C2C2E", "#3A3A3C"
TEXT, SUBTEXT = "#E5E5EA", "#8E8E93"
GREEN, RED, ORANGE, BLUE, YELLOW, PURPLE = "#2ECC71", "#E74C3C", "#C67A29", "#1F79BE", "#F4D03F", "#9B59B6"

TODAY = datetime.today()
START_LONG = "2004-01-01"
START_ANIM = (TODAY - timedelta(days=90)).strftime("%Y-%m-%d")
END_DATE = (TODAY + timedelta(days=1)).strftime("%Y-%m-%d")

VIX_TICKERS = {"^VIX9D": 9, "^VIX": 30, "^VIX3M": 93, "^VIX6M": 180}

REGIME_COLORS = {
    "Deep Contango": "#1a6b3c", "Contango": "#2ECC71", "Flat": "#F4D03F",
    "Mild Backwardation": "#C67A29", "Backwardation": "#E74C3C", "Severe Backwardation": "#8B0000",
}

SPIKE_THRESH = 30
ALLCLEAR_THRESH = 20

with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()


def add_logo(fig, x=0.99, y=0.99, sizex=0.10, sizey=0.10, opacity=0.45):
    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=x, y=y, sizex=sizex, sizey=sizey,
        xanchor="right", yanchor="top", opacity=opacity, layer="above",
    ))


# ── Base data ──────────────────────────────────────────────────────────────────
def load_term_structure():
    tickers = list(VIX_TICKERS.keys())
    raw = yf.download(tickers, start=START_LONG, end=END_DATE, auto_adjust=True, progress=False)
    closes = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].copy()

    all_labels = ["VIX9D", "VIX", "VIX3M", "VIX6M"]
    rename_map = {t: l for t, l in zip(tickers, all_labels) if t in closes.columns}
    closes = closes[list(rename_map.keys())].rename(columns=rename_map)
    labels = closes.columns.tolist()
    maturity_labels = ["9d", "30d", "93d", "180d"][:len(labels)]

    vix = closes.ffill().dropna(how="all")
    vix.index = pd.to_datetime(vix.index)

    spy = yf.download("SPY", start=START_LONG, end=END_DATE, auto_adjust=True, progress=False)["Close"].squeeze()
    spy.index = pd.to_datetime(spy.index)
    spy = spy.reindex(vix.index).ffill()

    long_end = labels[-1]
    vix["Contango_Ratio"] = vix["VIX9D"] / vix["VIX3M"]
    vix["Slope_Full"] = vix[long_end] - vix["VIX9D"]

    def classify(row):
        r = row["Contango_Ratio"]
        if r < 0.85: return "Deep Contango"
        elif r < 1.0: return "Contango"
        elif r < 1.10: return "Flat"
        elif r < 1.20: return "Mild Backwardation"
        elif r < 1.40: return "Backwardation"
        else: return "Severe Backwardation"

    vix["Regime"] = vix[["Contango_Ratio", "Slope_Full", "VIX"]].apply(classify, axis=1)
    vix["Spread"] = vix["VIX"] - vix["VIX3M"]
    return vix, spy, labels, maturity_labels, long_end


# ── 1. Curve Shifts Over Time ─────────────────────────────────────────────────
def chart_curve_shifts(vix, labels, maturity_labels):
    snapshots = {
        "Today": vix.iloc[-1],
        "1 Week Ago": vix.iloc[-6] if len(vix) >= 6 else vix.iloc[0],
        "1 Month Ago": vix.iloc[-22] if len(vix) >= 22 else vix.iloc[0],
        "3 Months Ago": vix.iloc[-63] if len(vix) >= 63 else vix.iloc[0],
    }
    snap_colors = [GREEN, BLUE, ORANGE, PURPLE]

    fig = go.Figure()
    for (label, row), color in zip(snapshots.items(), snap_colors):
        vals = [row[l] for l in labels]
        dash = "solid" if label == "Today" else "dot"
        width = 3 if label == "Today" else 1.8
        fig.add_trace(go.Scatter(
            x=maturity_labels, y=vals, mode="lines+markers",
            line=dict(color=color, width=width, dash=dash),
            marker=dict(size=9 if label == "Today" else 6, color=color),
            name=f'{label}  (ratio: {row["Contango_Ratio"]:.2f} | {row["Regime"]})',
            hovertemplate=f"{label} &mdash; %{{x}}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="<b>VIX Term Structure &mdash; Curve Shifts Over Time</b>", font=dict(size=18, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=480, hovermode="x unified",
        xaxis=dict(title="Maturity", gridcolor=LGRAY, linecolor=LGRAY, tickfont=dict(size=13)),
        yaxis=dict(title="Implied Volatility", gridcolor=LGRAY, linecolor=LGRAY),
        legend=dict(bgcolor=LGRAY, bordercolor=LGRAY, font=dict(size=11)),
        margin=dict(l=70, r=70, t=70, b=60),
    )
    return fig


# ── 2. 5-Year History Dashboard ───────────────────────────────────────────────
def chart_history_dashboard(vix, spy, labels):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, row_heights=[0.30, 0.25, 0.25, 0.20], vertical_spacing=0.04,
        subplot_titles=["All VIX Tenors", "Contango Ratio (VIX9D / VIX3M)", "Full Slope (VIX1Y - VIX9D)", "SPY"],
    )
    tenor_colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
    for col, color in zip(labels, tenor_colors):
        fig.add_trace(go.Scatter(x=vix.index, y=vix[col], mode="lines", line=dict(color=color, width=1.4), name=col,
                                  hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>"), row=1, col=1)

    fig.add_hline(y=1.0, line_color=SUBTEXT, line_width=1.2, line_dash="dash", row=2, col=1)
    fig.add_hline(y=1.20, line_color=RED, line_width=0.7, line_dash="dot", row=2, col=1)
    fig.add_hline(y=0.85, line_color=GREEN, line_width=0.7, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix["Contango_Ratio"].clip(upper=1.0), fill="tozeroy",
        fillcolor="rgba(46,204,113,0.12)", line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix["Contango_Ratio"].clip(lower=1.0), fill="tozeroy",
        fillcolor="rgba(231,76,60,0.12)", line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix["Contango_Ratio"], mode="lines", line=dict(color=ORANGE, width=1.8),
                              name="Contango Ratio", hovertemplate="Ratio: %{y:.3f}<extra></extra>"), row=2, col=1)

    fig.add_hline(y=0, line_color=SUBTEXT, line_width=1, line_dash="dash", row=3, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix["Slope_Full"].clip(lower=0), fill="tozeroy",
        fillcolor="rgba(46,204,113,0.15)", line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=3, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix["Slope_Full"].clip(upper=0), fill="tozeroy",
        fillcolor="rgba(231,76,60,0.15)", line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=3, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix["Slope_Full"], mode="lines", line=dict(color=BLUE, width=1.8),
                              name="Full Slope", hovertemplate="Slope: %{y:+.2f}<extra></extra>"), row=3, col=1)

    fig.add_trace(go.Scatter(x=spy.index, y=spy, mode="lines", line=dict(color=ORANGE, width=1.5), name="SPY",
                              hovertemplate="SPY: $%{y:.0f}<extra></extra>"), row=4, col=1)

    fig.update_layout(
        title=dict(text="<b>VIX Term Structure &mdash; 5-Year History</b>", font=dict(size=20, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        hovermode="x unified", height=850,
        legend=dict(bgcolor=LGRAY, bordercolor=LGRAY, font=dict(size=10)),
        margin=dict(l=65, r=65, t=80, b=40),
    )
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY, showspikes=True, spikecolor=SUBTEXT, spikethickness=1)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY, zeroline=False)
    fig.update_yaxes(title_text="VIX Level", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Points", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=4, col=1)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    return fig


# ── 3. 90-Day Heatmap ─────────────────────────────────────────────────────────
def chart_heatmap_90d(vix, labels, maturity_labels):
    vix_90 = vix[vix.index >= START_ANIM][labels]
    fig = go.Figure(go.Heatmap(
        x=vix_90.index, y=maturity_labels[::-1], z=vix_90[labels[::-1]].T.values,
        colorscale="RdYlGn_r", hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x|%b %d, %Y}<br>VIX: %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text="VIX", font=dict(color=TEXT)), tickfont=dict(color=TEXT), bgcolor=MGRAY, bordercolor=LGRAY),
    ))
    fig.update_layout(
        title=dict(text="<b>VIX Term Structure Heatmap &mdash; 90 Days</b>  "
                        f'<span style="font-size:12px; color:{SUBTEXT}">Red = High Vol | Green = Low Vol</span>',
                    font=dict(size=18, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=320, margin=dict(l=80, r=80, t=70, b=50),
        xaxis=dict(gridcolor=LGRAY, linecolor=LGRAY), yaxis=dict(gridcolor=LGRAY, linecolor=LGRAY),
    )
    return fig


# ── 4. SPY Forward Returns by VIX Regime ─────────────────────────────────────
def chart_regime_returns(vix, spy):
    combined = pd.DataFrame({"Regime": vix["Regime"], "VIX": vix["VIX"], "SPY": spy}).dropna()
    combined["SPY_1m_fwd"] = combined["SPY"].pct_change(21).shift(-21) * 100
    combined["SPY_3m_fwd"] = combined["SPY"].pct_change(63).shift(-63) * 100
    combined["SPY_6m_fwd"] = combined["SPY"].pct_change(126).shift(-126) * 100

    regime_stats = combined.groupby("Regime").agg(
        SPY_1m_avg=("SPY_1m_fwd", "mean"), SPY_3m_avg=("SPY_3m_fwd", "mean"), SPY_6m_avg=("SPY_6m_fwd", "mean"),
    ).round(2)
    order = ["Deep Contango", "Contango", "Flat", "Mild Backwardation", "Backwardation", "Severe Backwardation"]
    regime_stats = regime_stats.reindex([r for r in order if r in regime_stats.index])
    current_regime = vix["Regime"].iloc[-1]

    fig = make_subplots(rows=1, cols=3, subplot_titles=["1-Month Fwd Return", "3-Month Fwd Return", "6-Month Fwd Return"])
    for col_idx, ret_col in enumerate(["SPY_1m_avg", "SPY_3m_avg", "SPY_6m_avg"], 1):
        bar_colors = [YELLOW if r == current_regime else (GREEN if v >= 0 else RED)
                      for r, v in zip(regime_stats.index, regime_stats[ret_col])]
        fig.add_trace(go.Bar(
            x=regime_stats.index, y=regime_stats[ret_col], marker_color=bar_colors,
            text=[f"{v:+.1f}%" for v in regime_stats[ret_col]], textposition="outside",
            textfont=dict(color=TEXT, size=10), showlegend=False,
            hovertemplate="%{x}<br>%{y:+.2f}%<extra></extra>",
        ), row=1, col=col_idx)
        fig.add_hline(y=0, line_color=SUBTEXT, line_width=1, row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="<b>SPY Forward Returns by VIX Regime</b>  "
                        f'<span style="font-size:12px; color:{YELLOW}">Yellow = Current ({current_regime})</span>',
                    font=dict(size=17, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=500, margin=dict(l=50, r=50, t=80, b=130),
    )
    fig.update_xaxes(tickangle=30, tickfont=dict(size=9), gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY, ticksuffix="%")
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    return fig


# ── 5. SPY Forward Returns - VIX9D Analysis ──────────────────────────────────
def chart_vix9d_analysis(vix, spy):
    ana = pd.DataFrame({"VIX9D": vix["VIX9D"], "SPY": spy}).dropna()
    ana["VIX9D_1m_chg"] = ana["VIX9D"].pct_change(21) * 100
    ana["SPY_1m_fwd"] = ana["SPY"].pct_change(21).shift(-21) * 100
    ana["SPY_3m_fwd"] = ana["SPY"].pct_change(63).shift(-63) * 100
    ana["SPY_6m_fwd"] = ana["SPY"].pct_change(126).shift(-126) * 100
    ana["SPY_12m_fwd"] = ana["SPY"].pct_change(252).shift(-252) * 100

    bins_level = [0, 15, 20, 25, 30, 40, 999]
    labels_level = ["<15 (calm)", "15-20", "20-25", "25-30", "30-40", ">40 (panic)"]
    ana["VIX9D_bucket"] = pd.cut(ana["VIX9D"], bins=bins_level, labels=labels_level)

    level_stats = ana.groupby("VIX9D_bucket", observed=True).agg(
        fwd_1m_avg=("SPY_1m_fwd", "mean"), fwd_1m_win=("SPY_1m_fwd", lambda x: (x > 0).mean() * 100),
        fwd_3m_avg=("SPY_3m_fwd", "mean"), fwd_3m_win=("SPY_3m_fwd", lambda x: (x > 0).mean() * 100),
        fwd_6m_avg=("SPY_6m_fwd", "mean"), fwd_6m_win=("SPY_6m_fwd", lambda x: (x > 0).mean() * 100),
        fwd_12m_avg=("SPY_12m_fwd", "mean"), fwd_12m_win=("SPY_12m_fwd", lambda x: (x > 0).mean() * 100),
    ).round(2)

    bins_spike = [-999, -30, -15, 0, 15, 30, 999]
    labels_spike = ["Crashed >30%", "Fell 15-30%", "Fell 0-15%", "Rose 0-15%", "Rose 15-30%", "Spiked >30%"]
    ana["Spike_bucket"] = pd.cut(ana["VIX9D_1m_chg"], bins=bins_spike, labels=labels_spike)

    spike_stats = ana.groupby("Spike_bucket", observed=True).agg(
        fwd_1m_avg=("SPY_1m_fwd", "mean"), fwd_1m_win=("SPY_1m_fwd", lambda x: (x > 0).mean() * 100),
        fwd_3m_avg=("SPY_3m_fwd", "mean"), fwd_3m_win=("SPY_3m_fwd", lambda x: (x > 0).mean() * 100),
        fwd_6m_avg=("SPY_6m_fwd", "mean"), fwd_6m_win=("SPY_6m_fwd", lambda x: (x > 0).mean() * 100),
        fwd_12m_avg=("SPY_12m_fwd", "mean"), fwd_12m_win=("SPY_12m_fwd", lambda x: (x > 0).mean() * 100),
    ).round(2)

    current_vix9d = vix["VIX9D"].iloc[-1]
    current_spike = ana["VIX9D_1m_chg"].iloc[-1]
    current_lvl_bucket = pd.cut([current_vix9d], bins=bins_level, labels=labels_level)[0]
    current_spk_bucket = pd.cut([current_spike], bins=bins_spike, labels=labels_spike)[0]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Avg SPY Return by VIX9D Level", "Win Rate (%) by VIX9D Level",
                        "Avg SPY Return by 1-Month VIX9D Spike", "Win Rate (%) by 1-Month VIX9D Spike"],
        vertical_spacing=0.18, horizontal_spacing=0.10,
    )
    horizons = ["fwd_1m_avg", "fwd_3m_avg", "fwd_6m_avg", "fwd_12m_avg"]
    horizon_wins = ["fwd_1m_win", "fwd_3m_win", "fwd_6m_win", "fwd_12m_win"]
    horizon_labels = ["1m", "3m", "6m", "12m"]

    for h, hl in zip(horizons, horizon_labels):
        bar_colors = [YELLOW if str(b) == str(current_lvl_bucket) else (GREEN if v >= 0 else RED)
                      for b, v in zip(level_stats.index, level_stats[h])]
        fig.add_trace(go.Bar(x=level_stats.index.astype(str), y=level_stats[h], name=hl, marker_color=bar_colors,
            text=[f"{v:+.1f}%" for v in level_stats[h]], textposition="outside", textfont=dict(color=TEXT, size=9),
            showlegend=True, hovertemplate="%{x}<br>%{y:+.2f}%<extra>" + hl + "</extra>"), row=1, col=1)
    for h, hl in zip(horizon_wins, horizon_labels):
        fig.add_trace(go.Scatter(x=level_stats.index.astype(str), y=level_stats[h], mode="lines+markers", name=hl,
            showlegend=False, hovertemplate="%{x}<br>Win rate: %{y:.0f}%<extra>" + hl + "</extra>"), row=1, col=2)
    fig.add_hline(y=50, line_color=SUBTEXT, line_width=1, line_dash="dash", row=1, col=2)

    for h, hl in zip(horizons, horizon_labels):
        bar_colors = [YELLOW if str(b) == str(current_spk_bucket) else (GREEN if v >= 0 else RED)
                      for b, v in zip(spike_stats.index, spike_stats[h])]
        fig.add_trace(go.Bar(x=spike_stats.index.astype(str), y=spike_stats[h], name=hl, marker_color=bar_colors,
            text=[f"{v:+.1f}%" for v in spike_stats[h]], textposition="outside", textfont=dict(color=TEXT, size=9),
            showlegend=False, hovertemplate="%{x}<br>%{y:+.2f}%<extra>" + hl + "</extra>"), row=2, col=1)
    for h, hl in zip(horizon_wins, horizon_labels):
        fig.add_trace(go.Scatter(x=spike_stats.index.astype(str), y=spike_stats[h], mode="lines+markers", name=hl,
            showlegend=False, hovertemplate="%{x}<br>Win rate: %{y:.0f}%<extra>" + hl + "</extra>"), row=2, col=2)
    fig.add_hline(y=50, line_color=SUBTEXT, line_width=1, line_dash="dash", row=2, col=2)

    fig.update_layout(
        title=dict(text="<b>SPY Forward Returns &mdash; VIX9D Analysis</b>  "
                        f'<span style="font-size:12px; color:{YELLOW}">Yellow = Current | '
                        f"VIX9D: {current_vix9d:.1f} | 1mo spike: {current_spike:+.1f}%</span>",
                    font=dict(size=17, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=780, margin=dict(l=55, r=55, t=90, b=90),
    )
    fig.update_xaxes(tickangle=25, tickfont=dict(size=9), gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    return fig


# ── 6. VIX Bottom Signal Dashboard ───────────────────────────────────────────
def chart_bottom_signals(vix, spy):
    vvix_raw = yf.download("^VVIX", start=START_LONG, end=END_DATE, auto_adjust=True, progress=False)["Close"].squeeze()
    vvix_raw.index = pd.to_datetime(vvix_raw.index).tz_localize(None)
    vvix = vvix_raw.reindex(vix.index, method="ffill")

    spy_ret = spy.pct_change().dropna()
    rvol = (spy_ret.rolling(20).std() * np.sqrt(252) * 100).reindex(vix.index, method="ffill")
    vix_premium = vix["VIX"] - rvol

    pc = None
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PC_History.csv"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        pc_raw = pd.read_csv(StringIO(resp.text), skiprows=2, header=None)
        pc_raw.columns = ["date", "pc_ratio"]
        pc_raw["date"] = pd.to_datetime(pc_raw["date"], errors="coerce")
        pc_raw = pc_raw.dropna(subset=["date"]).set_index("date").sort_index()
        pc_raw.index = pc_raw.index.tz_localize(None)
        pc_raw["pc_ratio"] = pd.to_numeric(pc_raw["pc_ratio"], errors="coerce")
        pc = pc_raw["pc_ratio"].reindex(vix.index, method="ffill")
    except Exception as e:
        print(f"P/C fetch failed (dashboard will omit that panel): {e}")

    cr = vix["Contango_Ratio"]
    score = pd.DataFrame(index=vix.index)
    score["S1_ratio_rolling"] = ((cr > 1.0) & (cr.diff(3) < 0)).astype(int)
    score["S2_vvix_leads"] = ((vvix.pct_change(3) < -0.02) & (vix["VIX"] > 20)).astype(int)
    score["S3_vix_stalling"] = ((vix["VIX"] < vix["VIX"].rolling(5).max()) & (vix["VIX"] > 20)).astype(int)
    score["S4_premium_falling"] = ((vix_premium.diff(5) < -2) & (vix["VIX"] > 18)).astype(int)
    if pc is not None:
        score["S5_pc_spike"] = (pc.rolling(5).mean() > 0.80).astype(int)
        max_score = 5
    else:
        max_score = 4
    score["Total"] = score[[c for c in score.columns if c.startswith("S")]].sum(axis=1)

    n_panels = 6 if pc is not None else 5
    row_h = [0.24, 0.17, 0.15, 0.15, 0.15] + ([0.14] if pc is not None else [])
    titles = ["VIX (left) vs VVIX (right)", "VIX Premium (Implied - Realized Vol)", "Contango Ratio",
              "Bottom Signal Composite", "SPY  [green star = high score]"] + (["Put/Call Ratio (5d MA)"] if pc is not None else [])
    specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (n_panels - 1)

    fig = make_subplots(rows=n_panels, cols=1, shared_xaxes=True, row_heights=row_h, vertical_spacing=0.03,
                         subplot_titles=titles, specs=specs)

    fig.add_trace(go.Scatter(x=vix.index, y=vix["VIX"], mode="lines", line=dict(color=ORANGE, width=2), name="VIX",
                              hovertemplate="VIX: %{y:.2f}<extra></extra>"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=vvix.index, y=vvix, mode="lines", line=dict(color=PURPLE, width=1.5, dash="dot"), name="VVIX",
                              hovertemplate="VVIX: %{y:.2f}<extra></extra>"), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="VIX", row=1, col=1, secondary_y=False, gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(title_text="VVIX", row=1, col=1, secondary_y=True, gridcolor="rgba(0,0,0,0)",
                      tickfont=dict(color=PURPLE), title_font=dict(color=PURPLE))

    fig.add_hline(y=0, line_color=SUBTEXT, line_width=1, line_dash="dash", row=2, col=1)
    fig.add_trace(go.Scatter(x=vix_premium.index, y=vix_premium.clip(lower=0), fill="tozeroy",
        fillcolor="rgba(231,76,60,0.18)", line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=vix_premium.index, y=vix_premium, mode="lines", line=dict(color=RED, width=1.8),
        name="VIX Premium", hovertemplate="Premium: %{y:+.2f}pts<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=rvol.index, y=rvol, mode="lines", line=dict(color=BLUE, width=1.2, dash="dot"),
        name="Realized Vol (20d)", hovertemplate="RVol: %{y:.1f}%<extra></extra>"), row=2, col=1)

    fig.add_hline(y=1.0, line_color=SUBTEXT, line_width=1, line_dash="dash", row=3, col=1)
    fig.add_hline(y=1.10, line_color=RED, line_width=0.7, line_dash="dot", row=3, col=1)
    fig.add_trace(go.Scatter(x=cr.index, y=cr, mode="lines", line=dict(color=ORANGE, width=1.5),
        name="Contango Ratio", hovertemplate="Ratio: %{y:.3f}<extra></extra>"), row=3, col=1)

    score_colors = score["Total"].apply(
        lambda v: GREEN if v >= max_score * 0.75 else (YELLOW if v >= max_score * 0.5 else (ORANGE if v > 0 else LGRAY)))
    fig.add_trace(go.Bar(x=score.index, y=score["Total"], marker_color=score_colors.tolist(), name="Bottom Signals",
        hovertemplate="Score: %{y}/" + str(max_score) + "<extra></extra>"), row=4, col=1)
    fig.add_hline(y=max_score * 0.75, line_color=GREEN, line_width=0.8, line_dash="dot", row=4, col=1)

    fig.add_trace(go.Scatter(x=spy.index, y=spy, mode="lines", line=dict(color=ORANGE, width=1.5), name="SPY",
                              hovertemplate="SPY: $%{y:.0f}<extra></extra>"), row=5, col=1)
    score4_idx = score[score["Total"] >= 4].index
    spy_s4 = spy.reindex(score4_idx).dropna()
    if len(spy_s4) > 0:
        fig.add_trace(go.Scatter(x=spy_s4.index, y=spy_s4.values, mode="markers",
            marker=dict(color=GREEN, size=12, symbol="star", line=dict(color=DGRAY, width=0.5)),
            name="Score >= 4 (most signals firing)",
            hovertemplate="<b>%{x|%b %d %Y}</b><br>Score 4+ &mdash; signals firing<br>SPY: $%{y:.2f}<extra></extra>"), row=5, col=1)

    if pc is not None:
        pc_ma = pc.rolling(5).mean()
        fig.add_hline(y=0.80, line_color=GREEN, line_width=0.8, line_dash="dot", row=6, col=1)
        fig.add_hline(y=1.00, line_color=RED, line_width=0.8, line_dash="dot", row=6, col=1)
        fig.add_trace(go.Scatter(x=pc.index, y=pc, mode="lines", line=dict(color=SUBTEXT, width=0.7), opacity=0.4,
            name="P/C Daily", hovertemplate="P/C: %{y:.3f}<extra></extra>"), row=6, col=1)
        fig.add_trace(go.Scatter(x=pc_ma.index, y=pc_ma, mode="lines", line=dict(color=YELLOW, width=1.8),
            name="P/C 5d MA", hovertemplate="P/C 5d: %{y:.3f}<extra></extra>"), row=6, col=1)

    cur_score = int(score["Total"].iloc[-1])
    cur_color = GREEN if cur_score >= int(max_score * 0.75) else (YELLOW if cur_score >= int(max_score * 0.5) else ORANGE)
    fig.update_layout(
        title=dict(text="<b>VIX Bottom Signal Dashboard</b>  "
                        f'<span style="font-size:13px; color:{cur_color}">Score: {cur_score}/{max_score} signals firing</span>',
                    font=dict(size=18, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        hovermode="x unified", height=1050,
        legend=dict(bgcolor=LGRAY, bordercolor=LGRAY, font=dict(size=10), x=0.01, y=0.99),
        margin=dict(l=65, r=80, t=80, b=40),
    )
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY, showspikes=True, spikecolor=SUBTEXT, spikethickness=1)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY, zeroline=False)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=11, color=SUBTEXT)
    return fig


# ── 7. VIX-VIX3M Spread vs S&P 500 ───────────────────────────────────────────
def chart_vix3m_spread_bar(vix, spy):
    spread = vix["Spread"].dropna()
    spy_s = spy.reindex(spread.index).ffill()
    bar_colors = [RED if v > 0 else GREEN for v in spread]
    spread_ma90 = spread.rolling(90).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.60, 0.40], vertical_spacing=0.04,
        subplot_titles=["VIX - VIX3M Daily Spread  (Red = Backwardation | Green = Contango)", "S&P 500 - SPY"])
    fig.add_trace(go.Bar(x=spread.index, y=spread.values, marker_color=bar_colors, name="VIX-VIX3M Spread",
        hovertemplate="%{x|%b %d, %Y}<br>Spread: %{y:+.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=spread_ma90.index, y=spread_ma90.values, mode="lines",
        line=dict(color=YELLOW, width=2, dash="dot"), name="90d MA", hovertemplate="90d MA: %{y:+.2f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line_color=SUBTEXT, line_width=1.4, line_dash="dash", row=1, col=1)
    fig.add_trace(go.Scatter(x=spy_s.index, y=spy_s.values, mode="lines", line=dict(color=ORANGE, width=1.6), name="SPY",
                              hovertemplate="SPY: $%{y:.0f}<extra></extra>"), row=2, col=1)

    in_back = spread > 0
    transitions = in_back.astype(int).diff().fillna(0)
    starts = spread.index[transitions == 1].tolist()
    ends = spread.index[transitions == -1].tolist()
    if in_back.iloc[0]: starts.insert(0, spread.index[0])
    if in_back.iloc[-1]: ends.append(spread.index[-1])
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(231,76,60,0.08)", line_width=0, row=2, col=1)

    cur_spread = spread.iloc[-1]
    cur_color = RED if cur_spread > 0 else GREEN
    cur_label = "Backwardation" if cur_spread > 0 else "Contango"
    fig.update_layout(
        title=dict(text="<b>VIX - VIX3M Spread vs S&amp;P 500</b>  "
                        f'<span style="font-size:13px; color:{SUBTEXT}">Current: '
                        f'<b style="color:{cur_color}">{cur_spread:+.2f} ({cur_label})</b> | '
                        f"90d MA: {spread_ma90.iloc[-1]:+.2f}</span>",
                    font=dict(size=20, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        hovermode="x unified", height=740,
        legend=dict(bgcolor=LGRAY, bordercolor=LGRAY, font=dict(size=11)),
        margin=dict(l=70, r=70, t=90, b=40),
    )
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY, showspikes=True, spikecolor=SUBTEXT, spikethickness=1)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(title_text="Spread (pts)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    add_logo(fig)
    return fig


# ── 8. SPY Forward Returns by VIX-VIX3M Spread Level ─────────────────────────
def chart_spread_fwd_returns(vix, spy):
    combined_s = pd.DataFrame({"Spread": vix["Spread"], "SPY": spy}).dropna()
    for label, days in [("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)]:
        combined_s[f"fwd_{label}"] = combined_s["SPY"].pct_change(days).shift(-days) * 100

    bins_s = [-np.inf, -10, -5, -2, 0, 2, 5, 10, np.inf]
    labels_s = ["< -10", "-10 to -5", "-5 to -2", "-2 to 0", "0 to 2", "2 to 5", "5 to 10", "> 10"]
    combined_s["Bucket"] = pd.cut(combined_s["Spread"], bins=bins_s, labels=labels_s)

    fwd_stats_s = combined_s.groupby("Bucket", observed=True).agg(
        avg_1m=("fwd_1m", "mean"), avg_3m=("fwd_3m", "mean"), avg_6m=("fwd_6m", "mean"), avg_12m=("fwd_12m", "mean"),
        pos_1m=("fwd_1m", lambda x: (x > 0).mean() * 100), N=("Spread", "count"),
    ).round(2)

    cur_sp = vix["Spread"].iloc[-1]
    cur_bkt = labels_s[next(i for i, (lo, hi) in enumerate(zip(bins_s, bins_s[1:])) if lo < cur_sp <= hi)]
    periods = ["1m", "3m", "6m", "12m"]
    p_labels = ["1 Month", "3 Months", "6 Months", "12 Months"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"<b>{pl}</b> Forward Return" for pl in p_labels],
                         vertical_spacing=0.20, horizontal_spacing=0.10)
    for idx, (p, pl) in enumerate(zip(periods, p_labels)):
        row, col = idx // 2 + 1, idx % 2 + 1
        avgs = fwd_stats_s[f"avg_{p}"]
        ns = fwd_stats_s["N"]
        bar_c = [YELLOW if str(b) == cur_bkt else (GREEN if v >= 0 else RED) for b, v in zip(fwd_stats_s.index, avgs)]
        fig.add_trace(go.Bar(
            x=fwd_stats_s.index.astype(str), y=avgs.values, marker_color=bar_c,
            text=[f'{v:+.1f}%<br><span style="font-size:9px">n={n}</span>' for v, n in zip(avgs.values, ns.values)],
            textposition="outside", textfont=dict(color=TEXT, size=10), name=pl, showlegend=False,
            hovertemplate="%{x}<br>Avg: %{y:+.2f}%<extra></extra>",
        ), row=row, col=col)
        fig.add_hline(y=0, line_color=SUBTEXT, line_width=1, row=row, col=col)

    fig.update_layout(
        title=dict(text="<b>SPY Forward Returns by VIX-VIX3M Spread Level</b>  "
                        f'<span style="font-size:12px; color:{YELLOW}">Yellow = Current ({cur_sp:+.2f}, {cur_bkt})</span>',
                    font=dict(size=18, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=720, margin=dict(l=55, r=55, t=90, b=90),
    )
    fig.update_xaxes(tickangle=35, tickfont=dict(size=9), gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY, ticksuffix="%")
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    add_logo(fig)
    return fig


# ── 9. VIX Spike -> All-Clear Episodes ───────────────────────────────────────
def detect_spike_allclear_episodes(vix, spy):
    tsx_raw = yf.download("^GSPTSE", start=START_LONG, end=END_DATE, auto_adjust=True, progress=False)["Close"].squeeze()
    tsx_raw.index = pd.to_datetime(tsx_raw.index)
    tsx = tsx_raw.reindex(vix.index).ffill()

    def fwd_return(series, date, days):
        if date not in series.index:
            return np.nan
        loc = series.index.get_loc(date)
        future = loc + days
        if future >= len(series):
            return np.nan
        return (series.iloc[future] / series.iloc[loc] - 1) * 100

    vix_s = vix["VIX"]
    episodes = []
    open_ep = None
    state = "NORMAL"
    for i in range(len(vix_s)):
        v = vix_s.iloc[i]
        d = vix_s.index[i]
        if state == "NORMAL":
            if v >= SPIKE_THRESH:
                state = "SPIKED"
                open_ep = {"spike_date": d, "spike_vix": v, "peak_date": d, "peak_vix": v}
        elif state == "SPIKED":
            if v > open_ep["peak_vix"]:
                open_ep["peak_date"] = d
                open_ep["peak_vix"] = v
            if v < ALLCLEAR_THRESH:
                open_ep["allclear_date"] = d
                open_ep["allclear_vix"] = v
                episodes.append(open_ep)
                open_ep, state = None, "NORMAL"

    ep_df = pd.DataFrame(episodes)
    for hl, days in [("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)]:
        for sig, dcol in [("spike", "spike_date"), ("allclear", "allclear_date")]:
            for mkt, ser in [("spy", spy), ("tsx", tsx)]:
                ep_df[f"{mkt}_{sig}_{hl}"] = ep_df[dcol].apply(lambda d, s=ser, dy=days: fwd_return(s, d, dy))
    return ep_df, vix_s, tsx


def chart_spike_timeline(vix_s, spy, ep_df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.04,
                         subplot_titles=["VIX - Spike & All-Clear Episodes", "S&P 500 (SPY)"])
    fig.add_trace(go.Scatter(x=vix_s.index, y=vix_s.values, line=dict(color=ORANGE, width=1.2), name="VIX",
                              hovertemplate="%{x|%b %d, %Y}<br>VIX: %{y:.1f}<extra></extra>"), row=1, col=1)
    for lvl, col, lbl, pos in [(30, RED, "30 - Spike", "top right"), (20, GREEN, "20 - All-Clear", "bottom right")]:
        fig.add_hline(y=lvl, line_color=col, line_width=1.5, line_dash="dot", row=1, col=1,
                      annotation_text=lbl, annotation_font_color=col, annotation_position=pos, annotation_font_size=10)

    fig.add_trace(go.Scatter(x=ep_df["spike_date"], y=ep_df["spike_vix"] + 1.5, mode="markers",
        marker=dict(symbol="triangle-up", size=11, color=RED, line=dict(color="white", width=0.5)),
        name=f"Spike Signal (VIX >= {SPIKE_THRESH})",
        hovertemplate="<b>Spike</b>: %{x|%b %d, %Y}<br>VIX: %{customdata:.1f}<extra></extra>",
        customdata=ep_df["spike_vix"].values), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep_df["peak_date"], y=ep_df["peak_vix"] + 1.5, mode="markers",
        marker=dict(symbol="star", size=11, color="#FF6B6B", line=dict(color="white", width=0.5)),
        name="VIX Peak", hovertemplate="<b>Peak</b>: %{x|%b %d, %Y}<br>VIX: %{customdata:.1f}<extra></extra>",
        customdata=ep_df["peak_vix"].values), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep_df["allclear_date"], y=ep_df["allclear_vix"] - 1.5, mode="markers",
        marker=dict(symbol="circle", size=11, color=GREEN, line=dict(color="white", width=0.5)),
        name=f"All-Clear Signal (VIX < {ALLCLEAR_THRESH})",
        hovertemplate="<b>All-Clear</b>: %{x|%b %d, %Y}<br>VIX: %{customdata:.1f}<extra></extra>",
        customdata=ep_df["allclear_vix"].values), row=1, col=1)

    fig.add_trace(go.Scatter(x=spy.index, y=spy.values, line=dict(color=BLUE, width=1.2), name="SPY",
                              hovertemplate="%{x|%b %d, %Y}<br>SPY: $%{y:.2f}<extra></extra>"), row=2, col=1)
    for _, r in ep_df.iterrows():
        fig.add_vline(x=str(r["allclear_date"].date()), line_color=GREEN, line_width=0.8, line_dash="dot", row=2, col=1)
        fig.add_vline(x=str(r["spike_date"].date()), line_color=RED, line_width=0.8, line_dash="dot", row=2, col=1)

    fig.update_layout(
        title=dict(text=f"<b>VIX Spike &rarr; All-Clear Episodes  ({len(ep_df)} since {vix_s.index[0].year})</b>  "
                        f'<span style="font-size:12px; color:{SUBTEXT}">'
                        f'<span style="color:{RED}">Spike &ge; 30</span> | '
                        f'<span style="color:{GREEN}">All-Clear &lt; 20</span></span>',
                    font=dict(size=17, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=620, margin=dict(l=60, r=60, t=80, b=50),
        legend=dict(bgcolor=LGRAY, bordercolor=LGRAY, font=dict(size=10), orientation="h", x=0.5, xanchor="center", y=1.025),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=11, color=SUBTEXT)
    add_logo(fig)
    return fig


def chart_spike_fwd_returns(ep_df):
    hlabels = ["1m", "3m", "6m", "12m"]

    def signal_stats(prefix):
        out = {}
        for hl in hlabels:
            data = ep_df[f"{prefix}_{hl}"].dropna()
            out[hl] = {"avg": data.mean(), "win": (data > 0).mean() * 100}
        return out

    spy_spike_st, spy_allclear_st = signal_stats("spy_spike"), signal_stats("spy_allclear")
    tsx_spike_st, tsx_allclear_st = signal_stats("tsx_spike"), signal_stats("tsx_allclear")

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "S&P 500 - Average Forward Return", "S&P 500 - Win Rate",
        "TSX Composite - Average Forward Return", "TSX Composite - Win Rate",
    ], vertical_spacing=0.18, horizontal_spacing=0.10)

    def add_return_bars(spike_st, allclear_st, row, col, showlegend=False):
        fig.add_trace(go.Bar(x=hlabels, y=[spike_st[h]["avg"] for h in hlabels],
            name=f"At Spike (VIX >= {SPIKE_THRESH})", marker_color=RED,
            text=[f'{spike_st[h]["avg"]:+.1f}%' for h in hlabels], textposition="outside",
            textfont=dict(color=TEXT, size=11), showlegend=showlegend,
            hovertemplate="Spike<br>%{x}: %{y:+.2f}%<extra></extra>"), row=row, col=col)
        fig.add_trace(go.Bar(x=hlabels, y=[allclear_st[h]["avg"] for h in hlabels],
            name=f"At All-Clear (VIX < {ALLCLEAR_THRESH})", marker_color=GREEN,
            text=[f'{allclear_st[h]["avg"]:+.1f}%' for h in hlabels], textposition="outside",
            textfont=dict(color=TEXT, size=11), showlegend=showlegend,
            hovertemplate="All-Clear<br>%{x}: %{y:+.2f}%<extra></extra>"), row=row, col=col)
        fig.add_hline(y=0, line_color=SUBTEXT, line_width=1, row=row, col=col)

    def add_win_lines(spike_st, allclear_st, row, col):
        fig.add_trace(go.Scatter(x=hlabels, y=[spike_st[h]["win"] for h in hlabels], mode="lines+markers",
            line=dict(color=RED, width=2), marker=dict(size=8), name=f"At Spike (VIX >= {SPIKE_THRESH})",
            showlegend=False, hovertemplate="Spike<br>%{x}: %{y:.0f}% win<extra></extra>"), row=row, col=col)
        fig.add_trace(go.Scatter(x=hlabels, y=[allclear_st[h]["win"] for h in hlabels], mode="lines+markers",
            line=dict(color=GREEN, width=2), marker=dict(size=8), name=f"At All-Clear (VIX < {ALLCLEAR_THRESH})",
            showlegend=False, hovertemplate="All-Clear<br>%{x}: %{y:.0f}% win<extra></extra>"), row=row, col=col)
        fig.add_hline(y=50, line_color=SUBTEXT, line_width=1, line_dash="dash", row=row, col=col)

    add_return_bars(spy_spike_st, spy_allclear_st, 1, 1, showlegend=True)
    add_win_lines(spy_spike_st, spy_allclear_st, 1, 2)
    add_return_bars(tsx_spike_st, tsx_allclear_st, 2, 1)
    add_win_lines(tsx_spike_st, tsx_allclear_st, 2, 2)

    n = len(ep_df)
    fig.update_layout(
        title=dict(text="<b>Forward Returns: At the Spike vs At the All-Clear</b>  "
                        f'<span style="font-size:12px; color:{SUBTEXT}">{n} episodes | 2004-present</span>',
                    font=dict(size=17, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        barmode="group", height=680, margin=dict(l=60, r=60, t=90, b=60),
        legend=dict(bgcolor=LGRAY, bordercolor=LGRAY, font=dict(size=11), orientation="h", x=0.5, xanchor="center", y=1.03),
    )
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", title_text="Win Rate", row=1, col=2)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_yaxes(ticksuffix="%", title_text="Win Rate", row=2, col=2)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    add_logo(fig)
    return fig


def chart_spike_episode_heatmap(ep_df):
    rows_heat = []
    for _, r in ep_df.iterrows():
        ep_label = r.spike_date.strftime("%b '%y") + f" peak {r.peak_vix:.0f}"
        for sig, sig_label in [("spike", f'Spike ({r.spike_date.strftime("%b %d")} VIX {r.spike_vix:.0f})'),
                               ("allclear", f'All-Clear ({r.allclear_date.strftime("%b %d")} VIX {r.allclear_vix:.0f})')]:
            for hl in ["1m", "3m", "6m", "12m"]:
                rows_heat.append({"Episode": ep_label, "Signal": sig_label, "Horizon": hl,
                                   "SPY_Return": r.get(f"spy_{sig}_{hl}", np.nan)})
    heat_df = pd.DataFrame(rows_heat)
    heat_df["RowLabel"] = heat_df["Episode"] + "  |  " + heat_df["Signal"]
    pivot = heat_df.pivot_table(index="RowLabel", columns="Horizon", values="SPY_Return", aggfunc="first")[["1m", "3m", "6m", "12m"]]
    all_labels = heat_df.drop_duplicates("RowLabel")["RowLabel"].tolist()
    pivot = pivot.loc[[lbl for lbl in all_labels if lbl in pivot.index]]

    z_vals = pivot.values
    row_lbls = pivot.index.tolist()
    col_lbls = ["1-Month", "3-Month", "6-Month", "12-Month"]
    colorscale = [[0.0, "#8B0000"], [0.3, RED], [0.45, "#555555"], [0.5, LGRAY], [0.55, "#1a5c36"], [0.7, GREEN], [1.0, "#00FF7F"]]
    text_vals = [[f"{v:+.1f}%" if not np.isnan(v) else "—" for v in row] for row in z_vals]

    fig = go.Figure(go.Heatmap(
        z=z_vals, x=col_lbls, y=row_lbls, text=text_vals, texttemplate="%{text}",
        textfont=dict(size=10, color="white"), colorscale=colorscale, zmid=0,
        hovertemplate="%{y}<br>%{x}: %{z:+.1f}%<extra></extra>",
        colorbar=dict(title="SPY Return (%)", ticksuffix="%", title_font=dict(color=SUBTEXT),
                      tickfont=dict(color=SUBTEXT), bgcolor=MGRAY, bordercolor=LGRAY),
    ))
    n_eps = len(ep_df)
    fig.update_layout(
        title=dict(text="<b>S&amp;P 500 Forward Returns &mdash; Each Episode Detail</b>  "
                        f'<span style="font-size:12px; color:{SUBTEXT}">{n_eps} episodes | Spike row then All-Clear row per episode</span>',
                    font=dict(size=17, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=max(500, 38 * len(row_lbls) + 120), margin=dict(l=280, r=80, t=80, b=60),
        xaxis=dict(side="top", tickfont=dict(size=12), gridcolor=LGRAY, linecolor=LGRAY),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed", gridcolor=LGRAY, linecolor=LGRAY),
    )
    add_logo(fig, x=0.99, y=-0.02, sizex=0.08, sizey=0.08)
    return fig


# ── 10. Long-History Level Analysis ──────────────────────────────────────────
HORIZONS = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}
BINS = [0, 15, 20, 25, 30, 35, 40, 45, 999]
LEVEL_LABELS = ["<15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45+"]


def load_long_history():
    vix_h = yf.download("^VIX", start="1990-01-01", end=END_DATE, auto_adjust=True, progress=False)["Close"].squeeze()
    spx_h = yf.download("^GSPC", start="1990-01-01", end=END_DATE, auto_adjust=True, progress=False)["Close"].squeeze()
    tsx_h = yf.download("^GSPTSE", start="1990-01-01", end=END_DATE, auto_adjust=True, progress=False)["Close"].squeeze()
    vix3m_h = yf.download("^VIX3M", start="2001-01-01", end=END_DATE, auto_adjust=True, progress=False)["Close"].squeeze()
    for s in (vix_h, spx_h, tsx_h, vix3m_h):
        s.index = pd.to_datetime(s.index)
    return vix_h, spx_h, tsx_h, vix3m_h


def fwd_ret(series, days):
    return (series.shift(-days) / series - 1) * 100


def bucket_stats(df, total_days):
    rows = []
    for b in LEVEL_LABELS:
        sub = df[df["bucket"] == b]
        n = len(sub)
        row = {"bucket": b, "days": n, "pct_time": n / total_days * 100}
        for lbl in HORIZONS:
            for mkt in ["spx", "tsx"]:
                col = f"{mkt}_{lbl}"
                data = sub[col].dropna()
                row[f"{mkt}_{lbl}_mean"] = data.mean() if len(data) else np.nan
                row[f"{mkt}_{lbl}_n"] = len(data)
        rows.append(row)
    return pd.DataFrame(rows)


def return_heatmap(stats, mkt, title, subtitle):
    hor_list = list(HORIZONS.keys())
    z, text, customdata = [], [], []
    for _, r in stats.iterrows():
        row_z, row_t, row_c = [], [], []
        for h in hor_list:
            v, n = r[f"{mkt}_{h}_mean"], r[f"{mkt}_{h}_n"]
            row_z.append(v if not np.isnan(v) else None)
            row_t.append(f"{v:+.1f}%" if not np.isnan(v) else "—")
            row_c.append(n)
        z.append(row_z); text.append(row_t); customdata.append(row_c)

    fig = go.Figure(go.Heatmap(
        z=z, x=[h.upper() for h in hor_list], y=[str(b) for b in stats["bucket"]],
        text=text, texttemplate="%{text}", customdata=customdata,
        hovertemplate="<b>VIX %{y} | %{x}</b><br>Avg return: %{text}<br>n = %{customdata}<extra></extra>",
        colorscale=[[0.0, "#8B0000"], [0.3, RED], [0.45, ORANGE], [0.5, "#555555"], [0.55, "#A8E6CF"], [0.7, GREEN], [1.0, "#1a6b3c"]],
        zmid=0, showscale=True,
        colorbar=dict(title="Avg Return", ticksuffix="%", tickfont_color="white", title_font_color="white"),
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color="white", family="Arial"),
        title=dict(text=f"<b>{title}</b>  <span style=\"font-size:12px; color:{YELLOW}\">{subtitle}</span>", font_size=15, x=0.5, xanchor="center"),
        xaxis=dict(title="Forward Return Horizon", side="top", gridcolor=LGRAY, linecolor=LGRAY),
        yaxis=dict(title="VIX Level", autorange="reversed", gridcolor=LGRAY, linecolor=LGRAY),
        height=420, margin=dict(t=110, b=40, l=80, r=100),
    )
    add_logo(fig, x=0.99, y=-0.06, sizex=0.08, sizey=0.08)
    return fig


def chart_time_spent(stats_v, total_v, stats_v3, total_v3):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["VIX - Time Spent in Each Bucket (Since 1990)",
                                                          "VIX3M - Time Spent in Each Bucket (Since 2001)"])
    bar_colors = [GREEN if b in ["<15", "15-20"] else YELLOW if b in ["20-25"] else ORANGE if b in ["25-30"] else RED
                  for b in LEVEL_LABELS]
    for col, (stats, total) in enumerate([(stats_v, total_v), (stats_v3, total_v3)], 1):
        active = stats[stats["days"] > 0]
        fig.add_trace(go.Bar(x=[str(b) for b in active["bucket"]], y=active["pct_time"],
            marker_color=bar_colors[:len(active)], text=[f"{p:.1f}%" for p in active["pct_time"]],
            textposition="outside", textfont=dict(color="white", size=11), showlegend=False,
            hovertemplate="VIX %{x}: %{y:.1f}%% of days<extra></extra>"), row=1, col=col)

    fig.update_layout(
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color="white", family="Arial"),
        title=dict(text="<b>How Often Does VIX Spend Time at Each Level?</b>", font_size=15, x=0.5, xanchor="center"),
        height=420, margin=dict(t=100, b=60, l=60, r=30),
    )
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY, ticksuffix="%", title_text="% of Trading Days")
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY, title_text="VIX Level")
    add_logo(fig, x=0.99, y=-0.08, sizex=0.07, sizey=0.07)
    return fig


def combo_heatmap(df_combo, combo_active_v, combo_active_v3, mkt, mkt_label, horizon, hor_label):
    col = f"{mkt}_{horizon}"
    z, text = [], []
    for vb in combo_active_v:
        row_z, row_t = [], []
        for v3b in combo_active_v3:
            sub = df_combo[(df_combo["vix_b"] == vb) & (df_combo["vix3m_b"] == v3b)]
            data = sub[col].dropna()
            v, n = (data.mean() if len(data) >= 5 else np.nan), len(data)
            row_z.append(v if not np.isnan(v) else None)
            row_t.append(f"{v:+.0f}%<br>n={n}" if not np.isnan(v) else "")
        z.append(row_z); text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(b) for b in combo_active_v3], y=[str(b) for b in combo_active_v],
        text=text, texttemplate="%{text}", hovertemplate="VIX %{y} | VIX3M %{x}<br>%{text}<extra></extra>",
        colorscale=[[0.0, "#8B0000"], [0.3, RED], [0.45, ORANGE], [0.5, "#444444"], [0.55, "#A8E6CF"], [0.7, GREEN], [1.0, "#1a6b3c"]],
        zmid=0, showscale=True,
        colorbar=dict(title="Avg Return", ticksuffix="%", tickfont_color="white", title_font_color="white"),
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color="white", family="Arial"),
        title=dict(text=f"<b>VIX x VIX3M &rarr; {mkt_label} {hor_label} Forward Return</b>  "
                        f'<span style="font-size:12px; color:{YELLOW}">Rows = VIX | Cols = VIX3M | min n=5</span>',
                    font_size=14, x=0.5, xanchor="center"),
        xaxis=dict(title="VIX3M Level", side="top", gridcolor=LGRAY, linecolor=LGRAY),
        yaxis=dict(title="VIX Level", autorange="reversed", gridcolor=LGRAY, linecolor=LGRAY),
        height=460, margin=dict(t=110, b=40, l=80, r=100),
    )
    add_logo(fig, x=0.99, y=-0.06, sizex=0.08, sizey=0.08)
    return fig


def build_1pt_stats(vol_series, spx_series, tsx_series, days=252):
    df = pd.DataFrame({
        "vol": vol_series.round().astype(int),
        "spx": (spx_series.shift(-days) / spx_series - 1) * 100,
        "tsx": (tsx_series.shift(-days) / tsx_series - 1) * 100,
    }).dropna(subset=["vol"])
    return df.groupby("vol").agg(
        spx_mean=("spx", "mean"), tsx_mean=("tsx", "mean"), spx_n=("spx", "count"), tsx_n=("tsx", "count"),
    ).reset_index()


def bar_chart_1pt(grp, vol_series, vol_label, mkt_col, mkt_label, subtitle, min_obs=20):
    sub = grp[grp[f"{mkt_col}_n"] >= min_obs].copy()
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub[f"{mkt_col}_mean"]]
    customdata = sub[[f"{mkt_col}_n", f"{mkt_col}_mean"]].values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub["vol"], y=sub[f"{mkt_col}_mean"], marker_color=colors, marker_line_width=0, customdata=customdata,
        hovertemplate=(f"<b>{vol_label}: %{{x}}</b><br>Avg 1Y Return: %{{customdata[1]:+.1f}}%<br>"
                       f"Observations: %{{customdata[0]}}<extra></extra>"),
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)

    cur_val = vol_series.iloc[-1]
    cur_rounded = round(cur_val)
    if cur_rounded in sub["vol"].values:
        cur_ret = sub.loc[sub["vol"] == cur_rounded, f"{mkt_col}_mean"].values[0]
        fig.add_annotation(x=cur_rounded, y=cur_ret, text=f"Current<br>{vol_label}={cur_val:.1f}", showarrow=True,
            arrowhead=2, arrowcolor="white", font=dict(color="white", size=11), bgcolor="rgba(50,50,50,0.8)",
            bordercolor="white", borderwidth=1, ay=-40 if cur_ret >= 0 else 40)

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"<b>{vol_label} Level (1-pt Intervals) -&gt; {mkt_label} 1-Year Forward Return</b><br>"
                        f'<span style="font-size:13px;color:#aaa">{subtitle}</span>', x=0.5, xanchor="center", font=dict(size=18)),
        xaxis=dict(title=f"{vol_label} Level (rounded to nearest integer)", tickmode="linear", dtick=5, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Avg Forward 1-Year Return (%)", ticksuffix="%", gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        bargap=0.15, plot_bgcolor="#111", paper_bgcolor="#111", font=dict(color="white", size=13),
        margin=dict(t=90, b=60, l=70, r=40), height=520,
    )
    return fig


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def build_report() -> str:
    parts = []

    print("Loading term structure...")
    vix, spy, labels, maturity_labels, long_end = load_term_structure()

    parts.append(section_header("Term Structure - Curve Shifts Over Time"))
    parts.append(fig_to_div(chart_curve_shifts(vix, labels, maturity_labels)))

    parts.append(section_header("Term Structure - 5-Year History"))
    parts.append(fig_to_div(chart_history_dashboard(vix, spy, labels)))

    parts.append(section_header("Term Structure Heatmap - 90 Days"))
    parts.append(fig_to_div(chart_heatmap_90d(vix, labels, maturity_labels)))

    parts.append(section_header("SPY Forward Returns by VIX Regime"))
    parts.append(fig_to_div(chart_regime_returns(vix, spy)))

    parts.append(section_header("SPY Forward Returns - VIX9D Analysis"))
    parts.append(fig_to_div(chart_vix9d_analysis(vix, spy)))

    print("Building bottom signal dashboard (VVIX + P/C fetch)...")
    parts.append(section_header("VIX Bottom Signal Dashboard"))
    parts.append(fig_to_div(chart_bottom_signals(vix, spy)))

    parts.append(section_header("VIX - VIX3M Spread Analysis"))
    parts.append(fig_to_div(chart_vix3m_spread_bar(vix, spy)))
    parts.append(fig_to_div(chart_spread_fwd_returns(vix, spy)))

    print("Detecting spike -> all-clear episodes (SPY + TSX)...")
    ep_df, vix_s, tsx = detect_spike_allclear_episodes(vix, spy)
    parts.append(section_header("VIX Spike -> All-Clear Episodes", f"{len(ep_df)} episodes since {vix_s.index[0].year}"))
    parts.append(fig_to_div(chart_spike_timeline(vix_s, spy, ep_df)))
    parts.append(fig_to_div(chart_spike_fwd_returns(ep_df)))
    parts.append(fig_to_div(chart_spike_episode_heatmap(ep_df)))

    print("Loading long history (VIX/SPX/TSX/VIX3M since 1990/2001)...")
    vix_h, spx_h, tsx_h, vix3m_h = load_long_history()

    df_v = pd.DataFrame({"VIX": vix_h})
    for lbl, days in HORIZONS.items():
        df_v[f"spx_{lbl}"] = fwd_ret(spx_h, days).reindex(df_v.index)
        df_v[f"tsx_{lbl}"] = fwd_ret(tsx_h, days).reindex(df_v.index)
    df_v3 = pd.DataFrame({"VIX3M": vix3m_h})
    for lbl, days in HORIZONS.items():
        df_v3[f"spx_{lbl}"] = fwd_ret(spx_h, days).reindex(df_v3.index)
        df_v3[f"tsx_{lbl}"] = fwd_ret(tsx_h, days).reindex(df_v3.index)
    df_v["bucket"] = pd.cut(df_v["VIX"], bins=BINS, labels=LEVEL_LABELS, right=False)
    df_v3["bucket"] = pd.cut(df_v3["VIX3M"], bins=BINS, labels=LEVEL_LABELS, right=False)

    total_v, total_v3 = len(df_v), len(df_v3)
    stats_v = bucket_stats(df_v, total_v)
    stats_v3 = bucket_stats(df_v3, total_v3)

    parts.append(section_header("VIX Level -> Forward Return Heatmaps", "Since 1990 (VIX) / 2001 (VIX3M)"))
    parts.append(fig_to_div(return_heatmap(stats_v, "spx", "VIX Level -> S&P 500 Forward Returns (Since 1990)",
                                            "Avg annualized return from each VIX level bucket")))
    parts.append(fig_to_div(return_heatmap(stats_v, "tsx", "VIX Level -> TSX Composite Forward Returns (Since 1990)",
                                            "Avg annualized return from each VIX level bucket")))
    parts.append(fig_to_div(return_heatmap(stats_v3, "spx", "VIX3M Level -> S&P 500 Forward Returns (Since 2001)",
                                            "Avg annualized return from each VIX3M level bucket")))
    parts.append(fig_to_div(return_heatmap(stats_v3, "tsx", "VIX3M Level -> TSX Composite Forward Returns (Since 2001)",
                                            "Avg annualized return from each VIX3M level bucket")))

    parts.append(section_header("Time Spent at Each VIX Level"))
    parts.append(fig_to_div(chart_time_spent(stats_v, total_v, stats_v3, total_v3)))

    print("Building VIX x VIX3M combo heatmaps...")
    df_combo = pd.DataFrame({
        "VIX": vix_h, "VIX3M": vix3m_h,
        "spx_1m": fwd_ret(spx_h, 21), "spx_3m": fwd_ret(spx_h, 63), "spx_6m": fwd_ret(spx_h, 126), "spx_1y": fwd_ret(spx_h, 252),
        "tsx_1m": fwd_ret(tsx_h, 21), "tsx_3m": fwd_ret(tsx_h, 63), "tsx_6m": fwd_ret(tsx_h, 126), "tsx_1y": fwd_ret(tsx_h, 252),
    }).dropna(subset=["VIX", "VIX3M"])
    df_combo["vix_b"] = pd.cut(df_combo["VIX"], bins=BINS, labels=LEVEL_LABELS, right=False)
    df_combo["vix3m_b"] = pd.cut(df_combo["VIX3M"], bins=BINS, labels=LEVEL_LABELS, right=False)
    combo_active_v = [b for b in LEVEL_LABELS if (df_combo["vix_b"] == b).sum() > 0]
    combo_active_v3 = [b for b in LEVEL_LABELS if (df_combo["vix3m_b"] == b).sum() > 0]

    parts.append(section_header("VIX x VIX3M Combo Heatmaps", "Forward returns by joint VIX/VIX3M level bucket"))
    for hor, hor_label in [("1m", "1-Month"), ("3m", "3-Month"), ("6m", "6-Month"), ("1y", "1-Year")]:
        for mkt, mkt_label in [("spx", "S&P 500"), ("tsx", "TSX Composite")]:
            parts.append(fig_to_div(combo_heatmap(df_combo, combo_active_v, combo_active_v3, mkt, mkt_label, hor, hor_label)))

    print("Building 1-point interval bar charts...")
    grp_vix = build_1pt_stats(vix_h, spx_h, tsx_h)
    grp_v3m = build_1pt_stats(vix3m_h, spx_h, tsx_h)
    parts.append(section_header("VIX / VIX3M Level (1-Point Intervals)", "1-Year forward return, each bar = one integer level"))
    charts_1pt = [
        (grp_vix, vix_h, "VIX", "spx", "S&P 500", "Each bar = avg 1Y SPX return when VIX closed at that level (since 1990)"),
        (grp_vix, vix_h, "VIX", "tsx", "TSX Composite", "Each bar = avg 1Y TSX return when VIX closed at that level (since 1990)"),
        (grp_v3m, vix3m_h, "VIX3M", "spx", "S&P 500", "Each bar = avg 1Y SPX return when VIX3M closed at that level (since 2001)"),
        (grp_v3m, vix3m_h, "VIX3M", "tsx", "TSX Composite", "Each bar = avg 1Y TSX return when VIX3M closed at that level (since 2001)"),
    ]
    for grp, vol_ser, vol_lbl, mkt_col, mkt_lbl, subtitle in charts_1pt:
        parts.append(fig_to_div(bar_chart_1pt(grp, vol_ser, vol_lbl, mkt_col, mkt_lbl, subtitle)))

    return "\n".join(parts)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VIX Structure Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E5E5EA; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E5E5EA; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin-top: 6px; }}
</style>
</head>
<body>
<header>
  <h1>VIX Structure Report</h1>
  <div class="meta">Generated {date_str} &middot; Term structure, regime forward returns, bottom signals, spike/all-clear episodes, and long-history level analysis</div>
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

    out_path = os.path.join(OUTPUT_DIR, "VIX_Structure_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
