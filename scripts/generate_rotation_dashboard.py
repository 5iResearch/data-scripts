"""
Daily Rotation Dashboard.

Rotation is a *relative* phenomenon, so every indicator in this report is
computed on a RATIO (theme / benchmark, or theme / theme), never on raw
price. A weekly RSI of 75 on SMH tells you tech went up; a weekly RSI of 75
on SMH/SPY tells you tech's *leadership* is stretched -- which is the thing
that actually mean-reverts when the market rotates.

For each ratio the dashboard computes, on weekly bars:
  - Weekly RSI(14) of the ratio        -> leadership exhaustion / washout
  - Z-score of the log-ratio residual vs a 5y regression channel (same math
    as generate_ratio_channel_screener.py, applied to baskets instead of
    single names)                      -> how stretched vs its own RS trend
  - % distance from the 40-week MA of the ratio, normalized
  - 13-week ROC of the ratio           -> the confirmation / trigger leg

Those roll up into a Stretch score (mean of the three normalized stretch
legs; high = leadership extended and at risk of rotating OUT, low = washed
out and a candidate to rotate IN) and a five-state regime label:
  Leading / Extended / Weakening / Lagging / Washed Out / Improving

Sections:
  Rotation Scoreboard    (all ratios ranked by Stretch score)
  US Themes vs SPY
  Canadian vs XIC
  Style, Size & Breadth
  Head-to-Head Pairs     (XLE/XLK and friends -- the flip shows up here first)

Numerators may be a single ETF or an equal-weighted basket of tickers
(rebased daily equal-weight returns across whichever constituents have
history on a given day).
"""

import base64
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.stats import linregress
from ta.momentum import RSIIndicator

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "rotation-dashboard")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")

# -- Corporate colours (matches Industry RSI / Regime Monitors reports) -------
ORANGE = "#C67A29"
BLUE = "#1F79BE"
DGRAY = "#1C1C1E"
MGRAY = "#2C2C2E"
LGRAY = "#3A3A3C"
TEXT = "#E5E5EA"
SUBTEXT = "#8E8E93"
GREEN = "#2ECC71"
RED = "#E74C3C"
YELLOW = "#F4D03F"

START = "2010-01-01"
CHART_WEEKS = 260          # 5y of weekly bars shown / fitted
FIT_WEEKS = 260            # regression channel window
CHANNEL_SIGMA = 2.0
RSI_LEN = 14
MA_WEEKS = 40
ROC_WEEKS = 13
NORM_WEEKS = 156           # 3y window for normalizing the 40w stretch
RSI_OB, RSI_OS = 70, 30
EXTENDED_Z, WASHED_Z = 1.0, -1.0

STATE_CLR = {
    "Extended": RED,
    "Leading": GREEN,
    "Weakening": YELLOW,
    "Lagging": RED,
    "Washed Out": BLUE,
    "Improving": BLUE,
}


# -- Universe -----------------------------------------------------------------
# num: a single ticker, or a list = equal-weighted basket.
US_THEMES = [
    ("Semis / AI Hardware", "SMH", "SPY", "AI & Tech"),
    ("AI Hardware Basket", ["NVDA", "AVGO", "MU", "VRT", "CLS"], "SPY", "AI & Tech"),
    ("Software", "IGV", "SPY", "AI & Tech"),
    ("Magnificent 7", "MAGS", "SPY", "AI & Tech"),
    ("Nuclear / Uranium", "URA", "SPY", "Resources"),
    ("Metals & Mining", "XME", "SPY", "Resources"),
    ("Rare Earth / Crit. Minerals", "REMX", "SPY", "Resources"),
    ("Gold Miners", "GDX", "SPY", "Resources"),
    ("Energy", "XLE", "SPY", "Energy"),
    ("Oil & Gas E&P", "XOP", "SPY", "Energy"),
    ("Oil Services", "OIH", "SPY", "Energy"),
    ("Aerospace & Defense", "ITA", "SPY", "Industrial"),
    ("Infrastructure", "PAVE", "SPY", "Industrial"),
    ("Utilities", "XLU", "SPY", "Defensive"),
    ("Cons. Staples", "XLP", "SPY", "Defensive"),
    ("Health Care", "XLV", "SPY", "Defensive"),
]

CDN_THEMES = [
    ("TSX Technology", "XIT.TO", "XIC.TO", "AI & Tech"),
    ("Cdn Growth Basket", ["CLS.TO", "HPS-A.TO", "TIH.TO", "DSG.TO", "SHOP.TO"],
     "XIC.TO", "AI & Tech"),
    ("TSX Energy", "XEG.TO", "XIC.TO", "Energy"),
    ("Cdn Energy Producers", ["CNQ.TO", "SU.TO", "IMO.TO", "TOU.TO", "ARX.TO"],
     "XIC.TO", "Energy"),
    ("TSX Materials", "XMA.TO", "XIC.TO", "Resources"),
    ("TSX Financials", "XFN.TO", "XIC.TO", "Financials"),
    ("TSX Industrials", "ZIN.TO", "XIC.TO", "Industrial"),
    ("TSX Utilities", "XUT.TO", "XIC.TO", "Defensive"),
    ("TSX vs S&P 500", "XIC.TO", "SPY", "Country"),
]

STYLE_BREADTH = [
    ("Growth / Value", "IWF", "IWD", "Style"),
    ("Momentum / Market", "MTUM", "SPY", "Style"),
    ("High Beta / Low Vol", "SPHB", "SPLV", "Style"),
    ("Equal Wt / Cap Wt", "RSP", "SPY", "Breadth"),
    ("Small / Large", "IWM", "SPY", "Breadth"),
    ("Cyclicals / Defensives", ["XLI", "XLB", "XLE", "XLF"], ["XLU", "XLP", "XLV"], "Breadth"),
    ("Credit Risk Appetite", "HYG", "IEF", "Risk"),
    ("Copper / Gold", "CPER", "GLD", "Risk"),
]

PAIRS = [
    ("Energy vs Tech (US)", "XLE", "XLK", "The Flip"),
    ("Energy vs Tech (CDN)", "XEG.TO", "XIT.TO", "The Flip"),
    ("Energy vs Semis", "XLE", "SMH", "The Flip"),
    ("Cdn Energy vs Cdn Growth", ["CNQ.TO", "SU.TO", "IMO.TO", "TOU.TO", "ARX.TO"],
     ["CLS.TO", "HPS-A.TO", "TIH.TO", "DSG.TO", "SHOP.TO"], "The Flip"),
    ("Crit. Minerals vs Semis", "REMX", "SMH", "The Flip"),
    ("Momentum vs Equal Wt", "MTUM", "RSP", "Crowding"),
    ("Semis vs Software", "SMH", "IGV", "Crowding"),
]

SECTIONS = [
    ("US Themes vs SPY", "Relative strength of each US theme against the S&P 500", US_THEMES),
    ("Canadian Themes vs XIC", "Relative strength of each Canadian theme against the TSX", CDN_THEMES),
    ("Style, Size & Breadth", "The factor and participation ratios that frame every rotation", STYLE_BREADTH),
    ("Head-to-Head Pairs", "The growth-vs-energy flip shows up in a pair ratio before it shows up anywhere else", PAIRS),
]


# -- Data ---------------------------------------------------------------------
_px = {}


def all_tickers():
    out = set()
    for _title, _sub, spec in SECTIONS:
        for _name, num, den, _grp in spec:
            for side in (num, den):
                out.update(side if isinstance(side, list) else [side])
    return sorted(out)


def fetch(tickers):
    missing = [t for t in tickers if t not in _px]
    if not missing:
        return
    print(f"Downloading {len(missing)} tickers...")
    raw = yf.download(missing, start=START, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(missing[0])
    for t in missing:
        s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        _px[t] = s
        if len(s) < 260:
            print(f"  ! {t}: only {len(s)} daily bars")


def leg_series(spec):
    """Single ticker -> its close. List -> equal-weighted rebased basket."""
    if not isinstance(spec, list):
        s = _px.get(spec, pd.Series(dtype=float))
        return s.rename(spec if isinstance(spec, str) else "leg")
    frame = pd.DataFrame({t: _px[t] for t in spec if len(_px.get(t, [])) > 0})
    if frame.empty:
        return pd.Series(dtype=float)
    rets = frame.ffill().pct_change()
    # equal weight across whichever constituents have a return that day
    basket = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    return basket.rename("basket")


def label(spec):
    return "+".join(spec) if isinstance(spec, list) else spec


# -- Metrics ------------------------------------------------------------------
def compute_ratio(num_spec, den_spec):
    num, den = leg_series(num_spec), leg_series(den_spec)
    if num.empty or den.empty:
        return None
    aligned = pd.DataFrame({"n": num, "d": den}).dropna()
    if len(aligned) < 260:
        return None
    daily = aligned["n"] / aligned["d"]
    weekly = daily.resample("W-FRI").last().dropna()
    if len(weekly) < FIT_WEEKS // 2:
        return None
    return weekly


def fit_channel(logr):
    """Regression channel on the log ratio. Returns fitted line, sigma, z."""
    x = np.arange(len(logr))
    slope, intercept, r_value, _, _ = linregress(x, logr.values)
    fitted = pd.Series(intercept + slope * x, index=logr.index)
    resid = logr - fitted
    sd = resid.std()
    z = resid.iloc[-1] / sd if sd > 0 else 0.0
    return fitted, sd, float(z), float(r_value ** 2), float(slope)


def metrics(weekly):
    rsi = RSIIndicator(weekly, RSI_LEN).rsi()
    ma = weekly.rolling(MA_WEEKS).mean()
    roc = weekly.pct_change(ROC_WEEKS) * 100
    stretch = weekly / ma - 1
    stretch_z = ((stretch - stretch.rolling(NORM_WEEKS, min_periods=52).mean())
                 / stretch.rolling(NORM_WEEKS, min_periods=52).std())

    fit_win = weekly.iloc[-FIT_WEEKS:]
    fitted, sd, ch_z, r2, slope = fit_channel(np.log(fit_win))

    rsi_z = (rsi.iloc[-1] - 50) / 10.0
    sz = stretch_z.iloc[-1]
    legs = [v for v in (rsi_z, ch_z, sz) if pd.notna(v)]
    composite = float(np.mean(legs)) if legs else np.nan

    above_ma = bool(weekly.iloc[-1] > ma.iloc[-1]) if pd.notna(ma.iloc[-1]) else False
    roc_pos = bool(roc.iloc[-1] > 0) if pd.notna(roc.iloc[-1]) else False
    extended = (rsi.iloc[-1] > RSI_OB) or (ch_z > EXTENDED_Z and composite > EXTENDED_Z)
    washed = (rsi.iloc[-1] < RSI_OS) or (ch_z < WASHED_Z and composite < WASHED_Z)

    if above_ma and roc_pos:
        state = "Extended" if extended else "Leading"
    elif above_ma and not roc_pos:
        state = "Weakening"
    elif not above_ma and roc_pos:
        state = "Improving"
    else:
        state = "Washed Out" if washed else "Lagging"

    return dict(
        weekly=weekly, rsi=rsi, ma=ma, fitted=fitted, sd=sd,
        rsi_last=float(rsi.iloc[-1]), ch_z=ch_z, r2=r2,
        trend_ann=float((np.exp(slope * 52) - 1) * 100),
        roc13=float(roc.iloc[-1]) if pd.notna(roc.iloc[-1]) else np.nan,
        stretch_pct=float(stretch.iloc[-1] * 100) if pd.notna(stretch.iloc[-1]) else np.nan,
        stretch_z=float(sz) if pd.notna(sz) else np.nan,
        composite=composite, state=state,
    )


def build_rows():
    rows = []
    for sec_title, _sub, spec in SECTIONS:
        for name, num, den, grp in spec:
            weekly = compute_ratio(num, den)
            if weekly is None:
                print(f"  ! skipped {name} (insufficient history)")
                continue
            m = metrics(weekly)
            m.update(name=name, section=sec_title, group=grp,
                     num=label(num), den=label(den))
            rows.append(m)
    return rows


# -- Charts -------------------------------------------------------------------
with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = base64.b64encode(f.read()).decode()


def add_logo(fig):
    fig.add_layout_image(dict(
        source=f"data:image/png;base64,{LOGO_B64}", xref="paper", yref="paper",
        x=0.99, y=1.10, sizex=0.11, sizey=0.11, xanchor="right", yanchor="top",
        opacity=0.55, layer="above",
    ))


def ratio_chart(row):
    w = row["weekly"].iloc[-CHART_WEEKS:]
    ma = row["ma"].reindex(w.index)
    rsi = row["rsi"].reindex(w.index)
    fitted = np.exp(row["fitted"])
    upper = np.exp(row["fitted"] + CHANNEL_SIGMA * row["sd"])
    lower = np.exp(row["fitted"] - CHANNEL_SIGMA * row["sd"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)

    fig.add_trace(go.Scatter(x=upper.index, y=upper, mode="lines", name=f"+{CHANNEL_SIGMA:g}&sigma;",
                             line=dict(color=LGRAY, width=1, dash="dot"), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=lower.index, y=lower, mode="lines", name=f"-{CHANNEL_SIGMA:g}&sigma;",
                             line=dict(color=LGRAY, width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(58,58,60,0.28)", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted, mode="lines", name="RS trend",
                             line=dict(color=SUBTEXT, width=1.2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma.index, y=ma, mode="lines", name=f"{MA_WEEKS}w MA",
                             line=dict(color=ORANGE, width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w, mode="lines", name="Ratio",
                             line=dict(color=BLUE, width=2.2),
                             hovertemplate="%{x|%b %d, %Y}<br>Ratio: %{y:.4f}<extra></extra>"), row=1, col=1)

    fig.add_hrect(y0=RSI_OB, y1=100, fillcolor="rgba(231,76,60,0.12)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=RSI_OS, fillcolor="rgba(46,204,113,0.12)", line_width=0, row=2, col=1)
    fig.add_hline(y=50, line_color=LGRAY, line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode="lines", name="Weekly RSI(14)",
                             line=dict(color=TEXT, width=1.6),
                             hovertemplate="%{x|%b %d, %Y}<br>RSI: %{y:.1f}<extra></extra>"), row=2, col=1)

    ob = rsi[rsi > RSI_OB]
    os_ = rsi[rsi < RSI_OS]
    if len(ob):
        fig.add_trace(go.Scatter(x=ob.index, y=ob, mode="markers", name="Overbought",
                                 marker=dict(color=RED, size=5)), row=2, col=1)
    if len(os_):
        fig.add_trace(go.Scatter(x=os_.index, y=os_, mode="markers", name="Oversold",
                                 marker=dict(color=GREEN, size=5)), row=2, col=1)

    sclr = STATE_CLR[row["state"]]
    fig.update_layout(
        title=dict(
            text=f'<b>{row["name"]}</b>  <span style="font-size:11px;color:{SUBTEXT}">'
                 f'{row["num"]} / {row["den"]}  &middot;  weekly</span>'
                 f'   <span style="font-size:12px;color:{sclr}"><b>{row["state"].upper()}</b></span>',
            font=dict(size=15, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY,
        font=dict(color=TEXT, family="monospace", size=11),
        height=520, margin=dict(l=60, r=30, t=80, b=40),
        showlegend=True,
        legend=dict(bgcolor="rgba(28,28,30,0.8)", font=dict(size=9), orientation="h", x=0, y=-0.12),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Ratio", type="log", gridcolor=LGRAY, title_font=dict(color=SUBTEXT), row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], gridcolor=LGRAY,
                     title_font=dict(color=SUBTEXT), row=2, col=1)
    fig.update_xaxes(gridcolor=LGRAY, row=1, col=1)
    fig.update_xaxes(gridcolor=LGRAY, row=2, col=1)
    add_logo(fig)
    return fig


def scoreboard_chart(rows):
    """Stretch score ranked across every ratio -- the one-glance rotation read."""
    srt = sorted([r for r in rows if pd.notna(r["composite"])], key=lambda r: r["composite"])
    labels = [f'{r["name"]}' for r in srt]
    vals = [r["composite"] for r in srt]
    clrs = [STATE_CLR[r["state"]] for r in srt]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=clrs,
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
        textfont=dict(size=9, color=TEXT),
        customdata=[[r["state"], r["rsi_last"], r["ch_z"], r["roc13"]] for r in srt],
        hovertemplate="<b>%{y}</b><br>Stretch: %{x:+.2f}<br>State: %{customdata[0]}"
                      "<br>wRSI: %{customdata[1]:.0f}<br>Channel z: %{customdata[2]:+.2f}"
                      "<br>13w ROC: %{customdata[3]:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=SUBTEXT, line_width=1)
    fig.add_vline(x=1, line_color=RED, line_width=0.8, line_dash="dot")
    fig.add_vline(x=-1, line_color=BLUE, line_width=0.8, line_dash="dot")
    fig.update_layout(
        title=dict(text="<b>Rotation Scoreboard &mdash; Stretch Score</b>  "
                        f'<span style="font-size:11px;color:{SUBTEXT}">'
                        "mean of (weekly RSI z, channel z, 40w stretch z) &middot; "
                        "right = leadership extended &middot; left = washed out</span>",
                   font=dict(size=16, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY,
        font=dict(color=TEXT, family="monospace"),
        height=max(520, len(srt) * 26 + 140), bargap=0.3, showlegend=False,
        margin=dict(l=230, r=90, t=90, b=50),
        xaxis=dict(gridcolor=LGRAY, zeroline=False),
        yaxis=dict(gridcolor=LGRAY, tickfont=dict(size=10)),
    )
    add_logo(fig)
    return fig


# -- HTML ---------------------------------------------------------------------
def fig_to_div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def summary_table(rows):
    head = ("<tr><th>Ratio</th><th>Section</th><th>State</th><th>Stretch</th>"
            "<th>wRSI(14)</th><th>Channel z</th><th>R&sup2;</th><th>40w Stretch</th>"
            "<th>13w ROC</th><th>RS Trend/yr</th></tr>")
    body = []
    for r in sorted(rows, key=lambda r: -(r["composite"] if pd.notna(r["composite"]) else -99)):
        sclr = STATE_CLR[r["state"]]
        rsi_clr = RED if r["rsi_last"] > RSI_OB else (GREEN if r["rsi_last"] < RSI_OS else TEXT)
        roc_clr = GREEN if r["roc13"] > 0 else RED
        body.append(
            f'<tr><td class="nm">{r["name"]}</td><td class="sub">{r["section"]}</td>'
            f'<td style="color:{sclr}"><b>{r["state"]}</b></td>'
            f'<td><b>{r["composite"]:+.2f}</b></td>'
            f'<td style="color:{rsi_clr}">{r["rsi_last"]:.0f}</td>'
            f'<td>{r["ch_z"]:+.2f}</td><td class="sub">{r["r2"]:.2f}</td>'
            f'<td>{r["stretch_pct"]:+.1f}%</td>'
            f'<td style="color:{roc_clr}">{r["roc13"]:+.1f}%</td>'
            f'<td class="sub">{r["trend_ann"]:+.1f}%</td></tr>'
        )
    return f'<div class="tbl-wrap"><table>{head}{"".join(body)}</table></div>'


def how_to_read():
    return """
<div class="note">
  <b>How to read it.</b> Every number here is computed on a <i>ratio</i>, not on price.
  <ul>
    <li><b>Stretch</b> is the mean of three normalized legs: weekly RSI distance from 50,
        the z-score of the log-ratio residual vs its 5-year regression channel, and the
        normalized distance from the 40-week MA. Above +1 means leadership is extended and
        at risk of rotating <i>out</i>; below &minus;1 means it is washed out and is a
        candidate to rotate <i>in</i>.</li>
    <li><b>State</b> comes from position vs the 40-week MA and the sign of the 13-week ROC.
        Above MA + rising = Leading (Extended if stretched); above MA + falling = Weakening;
        below MA + rising = Improving; below MA + falling = Lagging (Washed Out if stretched).</li>
    <li><b>Stretch is the warning, ROC is the trigger.</b> An Extended reading says the trade
        is late, not that it is over. The flip is confirmed when the 13-week ROC crosses below
        zero or the ratio closes under its 40-week MA.</li>
    <li><b>R&sup2;</b> is the quality of the 5-year relative-strength trend fit. A low R&sup2;
        means the channel &mdash; and so the channel z &mdash; is describing noise; lean on the
        RSI and MA legs instead for those.</li>
  </ul>
</div>
"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rotation Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E5E5EA; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E5E5EA; border-bottom: 2px solid #1F79BE;
                 display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin-top: 6px; }}
  .note {{ margin: 12px 32px; padding: 14px 18px; background: #2C2C2E; border-left: 3px solid #C67A29;
           font-size: 13px; line-height: 1.55; }}
  .note ul {{ margin: 8px 0 0; padding-left: 18px; }}
  .note li {{ margin-bottom: 6px; }}
  .tbl-wrap {{ padding: 12px 32px 8px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-family: monospace; font-size: 12px; }}
  th {{ text-align: right; padding: 7px 10px; color: #8E8E93; border-bottom: 1px solid #3A3A3C;
        font-weight: normal; white-space: nowrap; }}
  th:first-child, th:nth-child(2), th:nth-child(3) {{ text-align: left; }}
  td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid #2C2C2E; white-space: nowrap; }}
  td.nm {{ text-align: left; color: #E5E5EA; }}
  td.sub {{ text-align: left; color: #8E8E93; }}
  td:nth-child(3) {{ text-align: left; }}
  tr:hover td {{ background: #2C2C2E; }}
</style>
</head>
<body>
<header>
  <h1>Rotation Dashboard</h1>
  <div class="meta">Generated {date_str} &middot; Weekly RSI, regression-channel z and 40-week stretch
  computed on relative-strength ratios &middot; {n} ratios tracked</div>
</header>
{body}
</body>
</html>
"""


def build_report(rows):
    parts = [section_header("Rotation Scoreboard",
                            "Every tracked ratio ranked by how stretched its leadership is")]
    parts.append(how_to_read())
    parts.append(fig_to_div(scoreboard_chart(rows)))
    parts.append(summary_table(rows))

    for sec_title, sub, _spec in SECTIONS:
        sec_rows = [r for r in rows if r["section"] == sec_title]
        if not sec_rows:
            continue
        parts.append(section_header(sec_title, sub))
        for r in sorted(sec_rows, key=lambda r: -r["composite"]):
            parts.append(fig_to_div(ratio_chart(r)))
    return "\n".join(parts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fetch(all_tickers())
    rows = build_rows()
    if not rows:
        raise SystemExit("No ratios could be computed.")
    body = build_report(rows)
    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"),
                                n=len(rows), body=body)
    out_path = os.path.join(OUTPUT_DIR, "Rotation_Dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")

    print("\nMost extended (rotation-out risk):")
    for r in sorted(rows, key=lambda r: -r["composite"])[:6]:
        print(f"  {r['name']:<30} {r['composite']:+.2f}  {r['state']:<12} "
              f"wRSI {r['rsi_last']:.0f}  13w ROC {r['roc13']:+.1f}%")
    print("\nMost washed out (rotation-in candidates):")
    for r in sorted(rows, key=lambda r: r["composite"])[:6]:
        print(f"  {r['name']:<30} {r['composite']:+.2f}  {r['state']:<12} "
              f"wRSI {r['rsi_last']:.0f}  13w ROC {r['roc13']:+.1f}%")


if __name__ == "__main__":
    main()
