"""
Website-ready Macro Market Model (MMM): a copy of generate_charts.py (S&P 500) and generate_charts_tsx.py
(TSX Composite) restyled to match the other website pages (5i logo, left-aligned titles, timeframe buttons
top-left with axes that refit, compact explainer).

Outputs, in outputs/mmm-web/:
  US_Macro_Market_Model.html       composite model + the five indicator charts, S&P 500
  Canada_Macro_Market_Model.html   the same for the TSX Composite
  charts/{sp500,tsx}-{market-model,rsi,vix,pmi,margin,breadth}.html
                                   each chart on its own, for embedding individually

The indicator scoring, weights and signal logic are copied unchanged from the two original scripts,
including where they differ:
  - TSX: PMI from "data/PMI - TSX.csv", breadth from data/TSX.csv, and before 2012 the composite averages
    only RSI, margin debt and breadth (1/3 each); US: all five at 20% throughout.
  - Signal colours: the US charts colour by the daily weighted average truncated to an integer, the TSX
    charts by the 21-day average rounded (Model_score), exactly as the originals do.
The original scripts and their outputs are untouched.

Data: Yahoo Finance (index, VIX, constituents for breadth), data/ISM.csv and data/PMI - TSX.csv (Koyfin,
monthly), data/margin_2.csv (FINRA, monthly).
"""

import os
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import ta
import yfinance as yf
from plotly.subplots import make_subplots

from generate_index_rsi_web import fig_to_div, timeframe_ranges
from generate_vix_web import AXIS_STYLE, FIT_AXES_JS, base_layout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "mmm-web")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

START_DATE = "1997-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

CORP = {"orange": "#C67A29", "blue": "#1F79BE", "dgrey": "#363636", "green": "#44A660", "red": "#A22A2A"}
GOLD = "#D4A820"
PATH = "#CCCCCC"

SIGNAL_COLORS = {0: "#A8A8A8", 1: CORP["red"], 2: "#8E6AC8", 3: GOLD, 4: CORP["blue"], 5: CORP["green"]}
SIGNAL_LABELS = {0: "N/A", 1: "Trim", 2: "Tactical Buy/Hold", 3: "Buy", 4: "Strong Buy", 5: "Very Strong Buy"}

MARKETS = {
    "us": dict(name="S&P 500", ticker="^GSPC", pmi_csv="ISM.csv", slug="sp500", page="US_Macro_Market_Model.html",
               country="US"),
    "cdn": dict(name="TSX Composite", ticker="^GSPTSE", pmi_csv="PMI - TSX.csv", slug="tsx",
                page="Canada_Macro_Market_Model.html", country="Canada"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Model data: copied from generate_charts.py / generate_charts_tsx.py
# ══════════════════════════════════════════════════════════════════════════════
def breadth_symbols(market: str) -> list:
    if market == "us":
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"})
        symbols = [t for t in pd.read_html(StringIO(resp.text))[0]["Symbol"].tolist()
                   if t not in ["SEDG", "OTIS", "NTAP"]]
        return [s.replace(".", "-") for s in symbols]
    symbols = [str(s).replace(".", "-") for s in pd.read_csv(os.path.join(DATA_DIR, "TSX.csv"))["Symbol"].dropna()]
    return [s if s.endswith(".TO") else s + ".TO" for s in symbols]


def download_close(tickers) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False, timeout=120)
            if raw is not None and not raw.empty:
                close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
                return close.dropna(axis=1, how="all")
        except Exception as e:
            print(f"  download attempt {attempt}/3 failed: {e}")
    raise ValueError("yfinance returned empty data after 3 attempts")


def build_model(market: str) -> pd.DataFrame:
    cfg = MARKETS[market]
    print(f"=== {cfg['name']} ===")
    df = yf.download(cfg["ticker"], start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    vix.columns = vix.columns.get_level_values(0)
    df["vix"] = vix["Close"]
    df = df.dropna(subset=["Close"])

    # RSI
    df["RSI_1yr"] = ta.momentum.RSIIndicator(df["Close"], window=252).rsi()
    df["RSI_change"] = (df["RSI_1yr"] / df["RSI_1yr"].shift(252)) - 1
    df["RSI_score"] = np.select([
        df["RSI_1yr"].between(43, 47),
        df["RSI_1yr"].between(47, 51) & (df["RSI_change"] < 0),
        df["RSI_1yr"].between(47, 51) & (df["RSI_change"] >= 0),
        df["RSI_1yr"].between(51, 55) & (df["RSI_change"] < 0),
        df["RSI_1yr"].between(51, 55) & (df["RSI_change"] >= 0),
        df["RSI_1yr"] > 55,
    ], [5, 4, 3, 2, 3, 1], default=3)

    # VIX
    df["VIX_score"] = np.select([
        df["vix"] < 18, df["vix"].between(18, 21), df["vix"].between(21, 27),
        df["vix"].between(27, 36), df["vix"] > 36,
    ], [2, 3, 3, 4, 5], default=3)

    # PMI (monthly CSV, forward-filled daily, 126-day average)
    pmi = pd.read_csv(os.path.join(DATA_DIR, cfg["pmi_csv"]))
    pmi["Date"] = pd.to_datetime(pmi[" Date"].str.strip(), format="%m-%d-%Y")
    pmi = pmi.set_index("Date")
    pmi.index = pmi.index + pd.offsets.MonthEnd(0)
    pmi = pmi[~pmi.index.duplicated(keep="last")]
    month_end = df.index.to_period("M").to_timestamp("M")
    df["pmi"] = pd.Series(month_end.map(pmi["NAPMPMI Close"]), index=df.index).ffill()
    df["rolling_PMI"] = df["pmi"].rolling(window=126).mean()
    df["PMI_score"] = np.select([
        df["rolling_PMI"] < 48, df["rolling_PMI"].between(48, 50), df["rolling_PMI"].between(50, 53),
        df["rolling_PMI"].between(53, 56), df["rolling_PMI"] > 56,
    ], [5, 4, 3, 2, 1], default=3)

    # Margin debt YoY (FINRA, monthly)
    margin = pd.read_csv(os.path.join(DATA_DIR, "margin_2.csv"))
    margin["Year-Month"] = pd.to_datetime(margin["Year-Month"])
    margin = margin.sort_values("Year-Month").set_index("Year-Month")
    margin.index = margin.index + pd.offsets.MonthEnd(0)
    margin = margin.rename(columns={"Debit Balances in Customers' Securities Margin Accounts": "debit"})
    margin["debit"] = margin["debit"].astype(str).str.replace(",", "").astype(float)
    margin["yoy"] = margin["debit"].pct_change(periods=12)
    margin = margin[~margin.index.duplicated(keep="last")]
    df["yoy"] = pd.Series(month_end.map(margin["yoy"]), index=df.index).ffill()
    df["rolling_yoy"] = df["yoy"].rolling(window=252).mean()
    df["Margin_score"] = np.select([
        df["rolling_yoy"] < -0.15, df["rolling_yoy"].between(-0.15, 0), df["rolling_yoy"].between(0, 0.15),
        df["rolling_yoy"].between(0.15, 0.30), df["rolling_yoy"] > 0.30,
    ], [5, 4, 3, 2, 1], default=3)

    # Breadth: share of constituents above their price a year ago
    print(f"  breadth download ({cfg['name']} constituents)...")
    try:
        stocks = download_close(breadth_symbols(market))
        adv = (stocks > stocks.shift(252)).sum(axis=1)
        dec = (stocks < stocks.shift(252)).sum(axis=1)
        df["net %"] = (adv / (adv + dec)).reindex(df.index).ffill()
        df["Breadth_score"] = np.select([
            df["net %"] < 0.40, df["net %"].between(0.40, 0.45), df["net %"].between(0.45, 0.55),
            df["net %"].between(0.55, 0.60), df["net %"] > 0.60,
        ], [5, 4, 3, 2, 1], default=3)
    except Exception as e:
        print(f"  breadth failed: {e}; breadth chart skipped, breadth scored neutral (3)")
        df["net %"], df["Breadth_score"] = np.nan, 3

    # Composite
    if market == "us":
        df["Weighted_Average"] = 0.2 * (df["PMI_score"] + df["VIX_score"] + df["RSI_score"]
                                        + df["Margin_score"] + df["Breadth_score"])
    else:
        df["Weighted_Average"] = np.where(
            df.index.year < 2012,
            (df["RSI_score"] + df["Margin_score"] + df["Breadth_score"]) / 3,
            0.25 * (df["PMI_score"] + df["VIX_score"] + df["RSI_score"] + df["Margin_score"] + df["Breadth_score"]),
        )
    df["Rolling_12MA"] = df["Weighted_Average"].rolling(window=21).mean()
    df["Model_score"] = df["Rolling_12MA"].round().clip(1, 5).fillna(3).astype(int)
    # Signal colouring, as each original script does it
    df["signal"] = (df["Weighted_Average"].fillna(0).astype(int) if market == "us" else df["Model_score"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Charts (5i website style)
# ══════════════════════════════════════════════════════════════════════════════
def _vals(s: pd.Series) -> list:
    # Plain lists with None for gaps: plotly 6 base64-encodes numpy arrays, which the page script can't read
    return [None if pd.isna(v) else float(v) for v in s]


def _dates(idx) -> list:
    return idx.strftime("%Y-%m-%d").tolist()


INDICATORS = {
    # key: (column, title, axis label, colourscale, tick/hover format, reference line, axis step)
    "rsi": ("RSI_1yr", "1-Year RSI", "1-Year RSI",
            [[0, CORP["green"]], [0.5, GOLD], [1, CORP["red"]]], ".1f", None, 1),
    "vix": ("vix", "VIX", "VIX", [[0, CORP["blue"]], [0.5, CORP["orange"]], [1, CORP["red"]]], ".1f", None, 1),
    "pmi": ("rolling_PMI", "ISM PMI (6-Mo. Avg.)", "PMI (6-mo avg)",
            [[0, CORP["red"]], [0.5, GOLD], [1, CORP["blue"]]], ".1f", 50, 1),
    "margin": ("yoy", "Margin Debt YoY %", "Margin debt YoY",
               [[0, CORP["red"]], [0.5, GOLD], [1, CORP["green"]]], ".0%", 0, 0.05),
    "breadth": ("net %", "Market Breadth (% Above 1-Year Ago)", "% advancing",
                [[0, CORP["red"]], [0.5, GOLD], [1, CORP["green"]]], ".0%", 0.5, 0.05),
}


# How the indicator charts are coloured:
#   "score"    each dot in the indicator's 1-5 model score, in the model's signal colours, with dashed lines at
#              the scoring cut-offs, so green always means "pushing the model towards buy" on every chart
#   "gradient" the original charts' continuous colour scale on the raw value (whose direction disagrees with
#              the model's scoring on VIX, PMI, margin debt and breadth)
COLOR_MODE = "score"

# key: (score column, cut-offs drawn in score mode, series plotted in score mode if different, its axis label)
SCORING = {
    "rsi": ("RSI_score", [43, 47, 51, 55], None, None),
    "vix": ("VIX_score", [18, 21, 27, 36], None, None),
    "pmi": ("PMI_score", [48, 50, 53, 56], None, None),
    # The model scores the 12-month average of margin debt growth, not the monthly reading the original plots
    "margin": ("Margin_score", [-0.15, 0, 0.15, 0.30], "rolling_yoy", "Margin debt YoY (12-mo avg)"),
    "breadth": ("Breadth_score", [0.40, 0.45, 0.55, 0.60], None, None),
}


def indicator_chart(df: pd.DataFrame, cfg: dict, key: str, mode: str = None) -> go.Figure:
    mode = mode or COLOR_MODE
    col, title, label, scale, fmt, ref, step = INDICATORS[key]
    score_col, cuts, score_series, score_label = SCORING[key]
    if mode == "score" and score_series:
        col, label = score_series, score_label
    d = df.dropna(subset=[col]) if key != "rsi" else df
    dates, price, val = _dates(d.index), _vals(d["Close"]), _vals(d[col])
    hover_fmt = ".1%" if fmt.endswith("%") else fmt
    if mode == "score":
        # One colour per score; rows with no reading yet (e.g. RSI's first year) in grey
        scores = [None if pd.isna(v) else int(sc) for v, sc in zip(d[col], d[score_col])]
        shade = [SIGNAL_COLORS[sc] if sc else "#D0D0D0" for sc in scores]
        scale = None
        score_text = [f"Score {sc}: {SIGNAL_LABELS[sc]}" if sc else "" for sc in scores]
    else:
        shade = d[col].to_numpy()   # marker colours: numeric array (Plotly rejects None there; gaps stay NaN)
        score_text = [""] * len(d)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    # Traces 0 / 2: grey paths (feed fitAxes); 1 / 3: dots coloured by the indicator (carry the hover)
    fig.add_trace(go.Scatter(x=dates, y=price, mode="lines", line=dict(color=PATH, width=1), hoverinfo="skip"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=price, mode="markers", name=cfg["name"],
                             marker=dict(size=4, color=shade, colorscale=scale, opacity=0.9, line=dict(width=0)),
                             hovertemplate=f"{cfg['name']}: %{{y:,.0f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=val, mode="lines", line=dict(color=PATH, width=1), hoverinfo="skip"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=val, mode="markers", name=label, customdata=score_text,
                             marker=dict(size=3, color=shade, colorscale=scale, opacity=0.9, line=dict(width=0)),
                             hovertemplate=f"{label}: %{{y:{hover_fmt}}}<br>%{{customdata}}<extra></extra>"),
                  row=2, col=1)
    if mode == "score":
        for c in cuts:
            fig.add_hline(y=c, line_dash="dash", line_color="#B0B0B0", line_width=1, row=2, col=1)
    elif ref is not None:
        fig.add_hline(y=ref, line_dash="dot", line_color="#9A9A9A", line_width=1.2, row=2, col=1)

    windows = timeframe_ranges(d.index)
    base_layout(fig, f"<b>{cfg['name']}</b>  ·  {title}", windows,
                fit=[dict(ax="yaxis", tr=[0], step=50),
                     dict(ax="yaxis2", tr=[2], step=step, inc=[] if ref is None else [ref])],
                xaxes=["xaxis", "xaxis2"], height=620)
    fig.update_yaxes(title_text=cfg["name"], tickformat=",.0f", row=1, col=1, **AXIS_STYLE)
    fig.update_yaxes(title_text=label, tickformat=fmt, row=2, col=1, **AXIS_STYLE)
    return fig


def model_chart(df: pd.DataFrame, cfg: dict) -> go.Figure:
    dates = _dates(df.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    present = [i for i in range(6) if (df["signal"] == i).any()]

    # Trace 0: grey price path (feeds fitAxes); then one dot trace per signal
    fig.add_trace(go.Scatter(x=dates, y=_vals(df["Close"]), mode="lines", line=dict(color=PATH, width=1),
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    for i in present:
        sub = df[df["signal"] == i]
        fig.add_trace(go.Scatter(
            x=_dates(sub.index), y=_vals(sub["Close"]), mode="markers", name=SIGNAL_LABELS[i], legendgroup=str(i),
            marker=dict(size=4, color=SIGNAL_COLORS[i], opacity=0.9, line=dict(width=0)),
            hovertemplate=f"{cfg['name']}: %{{y:,.0f}}<br>Signal: {SIGNAL_LABELS[i]}<extra></extra>"), row=1, col=1)
    score_path = len(fig.data)
    fig.add_trace(go.Scatter(x=dates, y=_vals(df["Rolling_12MA"]), mode="lines", line=dict(color=PATH, width=1),
                             hoverinfo="skip", showlegend=False), row=2, col=1)
    for i in present:
        sub = df[df["signal"] == i]
        fig.add_trace(go.Scatter(
            x=_dates(sub.index), y=_vals(sub["Rolling_12MA"]), mode="markers", legendgroup=str(i), showlegend=False,
            marker=dict(size=3, color=SIGNAL_COLORS[i], opacity=0.9, line=dict(width=0)),
            hovertemplate="Score: %{y:.2f}<extra></extra>"), row=2, col=1)
    for lvl in (2, 3, 4):
        fig.add_hline(y=lvl, line_dash="dot", line_color=SIGNAL_COLORS[lvl], opacity=0.5, row=2, col=1)

    windows = timeframe_ranges(df.index)
    base_layout(fig, f"<b>{cfg['name']}</b>  ·  Macro Market Model", windows,
                fit=[dict(ax="yaxis", tr=[0], step=50), dict(ax="yaxis2", tr=[score_path], step=0.25)],
                xaxes=["xaxis", "xaxis2"], height=660)
    fig.update_layout(showlegend=True, legend=dict(
        orientation="h", x=1.0, xanchor="right", y=1.02, yanchor="bottom", font=dict(size=12),
        itemsizing="constant", bgcolor="rgba(255,255,255,0.85)"))
    fig.update_yaxes(title_text=cfg["name"], tickformat=",.0f", row=1, col=1, **AXIS_STYLE)
    fig.update_yaxes(title_text="Model score (21-day avg)", tickformat=".2f", row=2, col=1, **AXIS_STYLE)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════
SECTION_NOTES = {
    "model": "Each of the five indicators below is scored from 1 (least favourable) to 5 (most favourable) and "
             "averaged; the line is the 21-day average of that score. Higher is better.",
    "rsi": "The index's momentum over the past year. A lower reading, especially one that is rising again, scores "
           "higher: it means the market is not stretched. Readings above 55 score lowest.",
    "vix": "Expected volatility. Fear is a contrarian positive: the VIX scores highest above 36 and lowest below 18.",
    "pmi": "US manufacturing activity (6-month average; 50 separates growth from contraction). Weak readings score "
           "highest, because markets tend to bottom while the economy still looks bad; readings above 56 score lowest.",
    "margin": "Growth in money borrowed to buy stocks, versus a year earlier (FINRA). Fast growth signals "
              "speculation and scores lowest; falling margin debt signals capitulation and scores highest.",
    "breadth": "The share of index members trading above their price of a year ago. Very broad strength (above 60%) "
               "scores lowest; below 40%, a washed-out market, scores highest.",
}

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{fit_js}
<style>
  body {{ background: #FFFFFF; color: #363636; font-family: Arial, sans-serif; font-size: 16px; line-height: 1.55; margin: 0; padding: 0 0 32px; }}
  header {{ padding: 18px 16px 12px; border-bottom: 1px solid #E6E6E6; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; color: #363636; }}
  header .meta {{ color: #555555; font-size: 15px; }}
  .intro {{ padding: 12px 16px 0; }}
  .intro p {{ margin: 0 0 8px; }}
  .intro .legend {{ font-size: 15px; color: #555555; margin-top: 16px; }}
  .status {{ font-size: 17px; }}
  .pill {{ display: inline-block; padding: 1px 10px; border-radius: 12px; font-size: 15px; font-weight: bold; color: #FFFFFF; }}
  .section {{ padding: 16px 16px 0; }}
  .section h2 {{ margin: 0; font-size: 23px; color: #363636; border-bottom: 3px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section p {{ margin: 8px 0 0; color: #555555; font-size: 15px; }}
  .source {{ color: #555555; font-size: 14px; padding: 20px 16px 0; border-top: 1px solid #E6E6E6; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">Updated {date_str} &middot; {index_name} &middot; daily since 1997</div>
</header>
<div class="intro">
  <p class="status">Current signal: <span class="pill" style="background:{signal_color}">{signal_label}</span></p>
  <p>The Macro Market Model combines five indicators of market conditions, covering momentum, fear, the economy,
  speculation and breadth, into a single reading of how favourable the backdrop is for {index_name} investors. Each
  indicator is scored from 1 to 5, and the scores are averaged into one of five signals, from <b>Trim</b> (conditions
  stretched) to <b>Very Strong Buy</b> (conditions washed out). The model is deliberately contrarian: it tends to be
  most positive when news and sentiment are worst.{weights_note}</p>
  <p class="legend">{legend}</p>
</div>
{charts}
<div class="source">Source: 5i Research, Yahoo Finance, Koyfin, FINRA. PMI and margin debt are monthly and update when new
data is released.</div>
</body>
</html>
"""

STANDALONE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{fit_js}
<style> body {{ background: #FFFFFF; margin: 0; font-family: Arial, sans-serif; }} </style>
</head>
<body>
{chart}
</body>
</html>
"""


def build_market(market: str):
    cfg = MARKETS[market]
    df = build_model(market)
    charts = {"model": model_chart(df, cfg)}
    for key in INDICATORS:
        if df[INDICATORS[key][0]].notna().any():
            charts[key] = indicator_chart(df, cfg, key)
        else:
            print(f"  {key}: no data, chart skipped")

    # Standalone files
    for key, fig in charts.items():
        name = f"{cfg['slug']}-{'market-model' if key == 'model' else key}.html"
        with open(os.path.join(CHART_DIR, name), "w", encoding="utf-8") as f:
            f.write(STANDALONE_TEMPLATE.format(title=f"{cfg['name']}: {name[:-5]}", fit_js=FIT_AXES_JS,
                                               chart=fig_to_div(fig)))
        print(f"  saved charts/{name}")

    # Full page
    titles = {"model": "Macro Market Model", **{k: v[1] for k, v in INDICATORS.items()}}
    body = []
    for key, fig in charts.items():
        body.append(f'<div class="section"><h2>{titles[key]}</h2><p>{SECTION_NOTES[key]}</p></div>')
        body.append(fig_to_div(fig))
    signal = int(df["signal"].iloc[-1])
    legend = "  ".join(
        f'<span class="pill" style="background:{SIGNAL_COLORS[i]}">{SIGNAL_LABELS[i]}</span>' for i in range(1, 6))
    if COLOR_MODE == "score":
        legend += ("<br>The indicator charts use the same colours: each reading is coloured by the score it gives "
                   "the model, and the dashed lines mark the scoring cut-offs.")
    weights_note = ("" if market == "us" else " For the TSX, PMI and VIX are only included from 2012 onward; before "
                    "that the model averages RSI, margin debt and breadth.")
    html = PAGE_TEMPLATE.format(
        title=f"Macro Market Model: {cfg['country']}", fit_js=FIT_AXES_JS,
        date_str=datetime.now().strftime("%B %d, %Y"), index_name=cfg["name"],
        signal_color=SIGNAL_COLORS[signal], signal_label=SIGNAL_LABELS[signal],
        weights_note=weights_note, legend=legend, charts="\n".join(body))
    with open(os.path.join(OUTPUT_DIR, cfg["page"]), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  saved {cfg['page']} (current signal: {SIGNAL_LABELS[signal]})")


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    for market in MARKETS:
        build_market(market)


if __name__ == "__main__":
    main()
