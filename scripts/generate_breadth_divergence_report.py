"""
Theme Breadth & Divergence report.

The rotation dashboard measures how *stretched* a theme's leadership is. This
report measures how *broad* it is -- and the gap between the two is the early
warning. Leadership tops narrow before they break: the cap-weighted RS line
makes a new high while fewer and fewer constituents are still participating.
That divergence typically leads the price flip by several weeks, which is
exactly the lead time a stretch score alone does not give you.

For each theme basket, on daily bars resampled to weekly:
  % above 50dma / 200dma       -- classic participation
  RS participation             -- % of constituents whose OWN ratio to the
                                  benchmark is above its own 200-day MA. This
                                  is the relative-strength version of breadth
                                  and is the most direct measure of whether a
                                  theme's leadership is broad or concentrated.
  Net new 52w highs            -- % making a 52w high in the last 20 sessions
                                  minus % making a 52w low
  Breadth composite            -- mean of (% >50dma, RS participation)

A divergence is flagged when the basket's 13-week RS change and its 13-week
breadth-composite change disagree in sign by more than DIV_THRESHOLD points:
  Negative  RS rising, breadth falling   -> leadership narrowing, top risk
  Positive  RS falling, breadth rising   -> washout ending, bottom forming

Constituents with stale data (delisted, merged) are dropped automatically --
a ticker whose history stopped in 2024 would otherwise sit permanently below
its 50dma and quietly drag every breadth reading down.

Outputs the HTML report plus data/breadth_series.csv (weekly breadth history
per theme) so the lead-lag study can consume it without refetching.
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_rotation_dashboard import (  # noqa: E402
    BLUE, DGRAY, GREEN, LGRAY, MGRAY, ORANGE, RED, SUBTEXT, TEXT, YELLOW,
    add_logo, fig_to_div, section_header,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "rotation-dashboard")
DATA_DIR = os.path.join(REPO_ROOT, "data")

START = "2012-01-01"
STALE_DAYS = 15          # drop constituents whose last bar is older than this
MIN_HISTORY = 300        # daily bars required to be included
MA_FAST, MA_SLOW = 50, 200
HIGH_WIN = 252           # 52w high/low window
HIGH_RECENT = 20         # "made a new high in the last N sessions"
CHG_WEEKS = 13           # divergence comparison window
DIV_THRESHOLD = 5.0      # breadth must move this many points against RS
CHART_WEEKS = 208        # 4y shown
MIN_CONSTITUENTS = 8


# -- Theme baskets ------------------------------------------------------------
# (theme, benchmark, [constituents])
BASKETS = [
    ("US AI & Semis", "SPY", [
        "NVDA", "AVGO", "AMD", "MU", "TSM", "ASML", "AMAT", "LRCX", "KLAC",
        "MRVL", "ADI", "TXN", "NXPI", "ON", "MCHP", "QCOM", "INTC", "SNPS",
        "CDNS", "ARM", "VRT", "SMCI", "DELL", "ANET", "CLS", "CIEN", "COHR",
        "MPWR", "TER", "ENTG",
    ]),
    ("US Energy Producers", "SPY", [
        "XOM", "CVX", "COP", "EOG", "OXY", "DVN", "FANG", "APA",
        "MTDR", "PR", "SM", "AR", "EQT", "RRC", "CHRD", "OVV", "MGY", "CNX",
        "MUR", "GPOR", "CRC", "NOG", "TPL", "CRGY",
    ]),
    ("Critical Minerals & Mining", "SPY", [
        "MP", "ALB", "SQM", "LAC", "UUUU", "FCX", "SCCO", "TECK",
        "RIO", "BHP", "VALE", "AA", "NUE", "CLF", "HBM", "ERO",
        "STLD", "CMC", "ATI", "MTRN", "TROX",
    ]),
    ("Nuclear & Uranium", "SPY", [
        "CCJ", "UEC", "DNN", "NXE", "UUUU", "URG", "LEU", "SMR", "OKLO",
        "BWXT", "VST", "CEG", "TLN", "NNE", "ASPI",
    ]),
    ("Aerospace & Defense", "SPY", [
        "LMT", "RTX", "NOC", "GD", "LHX", "BA", "HWM", "TDG", "HEI", "LDOS",
        "TXT", "AXON", "KTOS", "AVAV", "RKLB", "LUNR", "PLTR", "CW", "MOG-A",
        "HXL", "CACI", "BAH",
    ]),
    ("Cdn Growth", "XIC.TO", [
        "CLS.TO", "HPS-A.TO", "TIH.TO", "DSG.TO", "SHOP.TO", "CSU.TO",
        "OTEX.TO", "LSPD.TO", "KXS.TO", "DCBO.TO", "WELL.TO", "GSY.TO",
        "EFN.TO", "ATS.TO", "BYD.TO", "TFII.TO", "STN.TO", "WSP.TO",
        "TOI.V", "MDA.TO",
    ]),
    ("Cdn Energy Producers", "XIC.TO", [
        "CNQ.TO", "SU.TO", "IMO.TO", "CVE.TO", "TOU.TO", "ARX.TO", "OBE.TO",
        "BTE.TO", "WCP.TO", "CJ.TO", "SGY.TO", "PEY.TO", "PSK.TO", "TVE.TO",
        "BIR.TO", "IPCO.TO", "KEL.TO", "PXT.TO", "ATH.TO", "SDE.TO",
        "FRU.TO", "HWX.TO",
    ]),
    ("Cdn Materials & Miners", "XIC.TO", [
        "ABX.TO", "AEM.TO", "K.TO", "FM.TO", "TECK-B.TO", "IVN.TO", "LUN.TO",
        "CS.TO", "HBM.TO", "ERO.TO", "CCO.TO", "NXE.TO", "DML.TO", "WPM.TO",
        "FNV.TO", "AGI.TO", "ELD.TO", "OGC.TO", "IMG.TO", "TXG.TO",
        "LUG.TO", "CG.TO",
    ]),
]


# -- Data ---------------------------------------------------------------------
def fetch_closes(tickers):
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(sorted(set(tickers)), start=START, auto_adjust=True,
                      progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close


def clean_constituents(close, tickers, theme):
    """Drop short or stale series -- a delisted name would sit permanently
    below its 50dma and drag every breadth reading down."""
    last_date = close.index.max()
    keep, dropped = [], []
    for t in tickers:
        if t not in close.columns:
            dropped.append((t, "no data"))
            continue
        s = close[t].dropna()
        if len(s) < MIN_HISTORY:
            dropped.append((t, f"{len(s)} bars"))
        elif (last_date - s.index.max()).days > STALE_DAYS:
            dropped.append((t, f"stale {s.index.max():%Y-%m-%d}"))
        else:
            keep.append(t)
    if dropped:
        print(f"  {theme}: dropped {', '.join(f'{t} ({why})' for t, why in dropped)}")
    return keep


# -- Breadth math -------------------------------------------------------------
def pct_above_ma(px, window):
    ma = px.rolling(window, min_periods=window // 2).mean()
    above = (px > ma) & px.notna() & ma.notna()
    valid = px.notna() & ma.notna()
    return (above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)) * 100


def pct_rs_above_ma(px, bench, window=MA_SLOW):
    """% of constituents whose OWN ratio to the benchmark is above its own MA.
    The relative-strength version of breadth."""
    rs = px.div(bench, axis=0)
    return pct_above_ma(rs, window)


def net_new_highs(px):
    roll_max = px.rolling(HIGH_WIN, min_periods=HIGH_WIN // 2).max()
    roll_min = px.rolling(HIGH_WIN, min_periods=HIGH_WIN // 2).min()
    at_high = (px >= roll_max * 0.999)
    at_low = (px <= roll_min * 1.001)
    recent_high = at_high.rolling(HIGH_RECENT).max().fillna(0)
    recent_low = at_low.rolling(HIGH_RECENT).max().fillna(0)
    valid = px.notna() & roll_max.notna()
    n = valid.sum(axis=1).replace(0, np.nan)
    return ((recent_high[valid].sum(axis=1) - recent_low[valid].sum(axis=1)) / n) * 100


def basket_rs(px, bench):
    """Equal-weighted basket relative to benchmark."""
    rets = px.pct_change()
    basket = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    return basket / bench * bench.iloc[0]


def compute_theme(close, theme, bench_ticker, tickers):
    names = clean_constituents(close, tickers, theme)
    if len(names) < MIN_CONSTITUENTS:
        print(f"  ! {theme}: only {len(names)} usable constituents, skipping")
        return None
    px = close[names].ffill()
    bench = close[bench_ticker].ffill()
    aligned = px.join(bench.rename("__bench__"), how="inner").dropna(subset=["__bench__"])
    px, bench = aligned[names], aligned["__bench__"]

    daily = pd.DataFrame({
        "rs": basket_rs(px, bench),
        "pct50": pct_above_ma(px, MA_FAST),
        "pct200": pct_above_ma(px, MA_SLOW),
        "rs_part": pct_rs_above_ma(px, bench),
        "nnh": net_new_highs(px),
    })
    wk = daily.resample("W-FRI").last().dropna(subset=["rs"])
    wk["breadth"] = wk[["pct50", "rs_part"]].mean(axis=1)

    wk["rs_chg"] = wk["rs"].pct_change(CHG_WEEKS) * 100
    wk["breadth_chg"] = wk["breadth"].diff(CHG_WEEKS)
    wk["divergence"] = np.select(
        [(wk["rs_chg"] > 0) & (wk["breadth_chg"] < -DIV_THRESHOLD),
         (wk["rs_chg"] < 0) & (wk["breadth_chg"] > DIV_THRESHOLD)],
        ["Negative", "Positive"], default="Neutral")

    # Is RS at a 26w high while breadth is not? A second, stricter top tell.
    rs_hi = wk["rs"] >= wk["rs"].rolling(26).max() * 0.999
    br_hi = wk["breadth"] >= wk["breadth"].rolling(26).max() - 1
    wk["narrow_high"] = rs_hi & ~br_hi

    last = wk.iloc[-1]
    return dict(
        theme=theme, bench=bench_ticker, names=names, n=len(names), wk=wk,
        rs_chg=float(last["rs_chg"]), breadth_chg=float(last["breadth_chg"]),
        pct50=float(last["pct50"]), pct200=float(last["pct200"]),
        rs_part=float(last["rs_part"]), nnh=float(last["nnh"]),
        breadth=float(last["breadth"]), divergence=last["divergence"],
        narrow_high=bool(last["narrow_high"]),
        weeks_in_div=int((wk["divergence"].iloc[::-1] != last["divergence"]).argmax())
        if last["divergence"] != "Neutral" else 0,
    )


# -- Charts -------------------------------------------------------------------
DIV_CLR = {"Negative": "rgba(231,76,60,0.14)", "Positive": "rgba(46,204,113,0.14)"}


def div_runs(series, min_len=2):
    """Contiguous runs of a non-'None' divergence state."""
    runs, start, cur = [], None, "Neutral"
    for idx, val in series.items():
        if val != cur:
            if cur != "Neutral" and start is not None:
                runs.append((start, prev_idx, cur))
            start, cur = idx, val
        prev_idx = idx
    if cur != "Neutral" and start is not None:
        runs.append((start, prev_idx, cur))
    return [r for r in runs if (series.index.get_loc(r[1]) - series.index.get_loc(r[0])) >= min_len - 1]


def theme_chart(res):
    wk = res["wk"].iloc[-CHART_WEEKS:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.44, 0.34, 0.22], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(x=wk.index, y=wk["rs"], mode="lines", name="RS vs bench",
                             line=dict(color=BLUE, width=2.2),
                             hovertemplate="%{x|%b %d, %Y}<br>RS: %{y:.3f}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(x=wk.index, y=wk["pct50"], mode="lines", name=f"% > {MA_FAST}dma",
                             line=dict(color=ORANGE, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=wk.index, y=wk["pct200"], mode="lines", name=f"% > {MA_SLOW}dma",
                             line=dict(color=SUBTEXT, width=1.2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=wk.index, y=wk["rs_part"], mode="lines", name="RS participation",
                             line=dict(color=TEXT, width=2)), row=2, col=1)
    fig.add_hline(y=50, line_color=LGRAY, line_width=1, row=2, col=1)

    nnh = wk["nnh"]
    fig.add_trace(go.Bar(x=nnh.index, y=nnh, name="Net new 52w highs",
                         marker_color=[GREEN if v >= 0 else RED for v in nnh],
                         hovertemplate="%{x|%b %d, %Y}<br>Net new highs: %{y:.0f}%<extra></extra>"),
                  row=3, col=1)
    fig.add_hline(y=0, line_color=SUBTEXT, line_width=1, row=3, col=1)

    for start, end, kind in div_runs(wk["divergence"]):
        fig.add_vrect(x0=start, x1=end, fillcolor=DIV_CLR[kind], line_width=0, layer="below")

    dclr = {"Negative": RED, "Positive": GREEN, "Neutral": SUBTEXT}[res["divergence"]]
    dtxt = (f'{res["divergence"].upper()} DIVERGENCE ({res["weeks_in_div"]}w)'
            if res["divergence"] != "Neutral" else "NO DIVERGENCE")
    fig.update_layout(
        title=dict(
            text=f'<b>{res["theme"]}</b>  <span style="font-size:11px;color:{SUBTEXT}">'
                 f'{res["n"]} constituents vs {res["bench"]}  &middot;  weekly</span>'
                 f'   <span style="font-size:12px;color:{dclr}"><b>{dtxt}</b></span>',
            font=dict(size=15, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY,
        font=dict(color=TEXT, family="monospace", size=11),
        height=640, margin=dict(l=60, r=30, t=80, b=40), bargap=0.1,
        legend=dict(bgcolor="rgba(28,28,30,0.8)", font=dict(size=9), orientation="h", x=0, y=-0.1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="RS ratio", type="log", gridcolor=LGRAY,
                     title_font=dict(color=SUBTEXT), row=1, col=1)
    fig.update_yaxes(title_text="% of basket", range=[0, 100], gridcolor=LGRAY,
                     title_font=dict(color=SUBTEXT), row=2, col=1)
    fig.update_yaxes(title_text="Net 52w hi", gridcolor=LGRAY,
                     title_font=dict(color=SUBTEXT), row=3, col=1)
    for r in (1, 2, 3):
        fig.update_xaxes(gridcolor=LGRAY, row=r, col=1)
    add_logo(fig)
    return fig


def divergence_chart(results):
    """RS change vs breadth change -- the divergence quadrant."""
    fig = go.Figure()
    xs = [r["rs_chg"] for r in results]
    ys = [r["breadth_chg"] for r in results]
    xpad = max(8.0, (max(xs) - min(xs)) * 0.25)
    ypad = max(8.0, (max(ys) - min(ys)) * 0.25)
    x_rng = [min(xs) - xpad, max(xs) + xpad]
    y_rng = [min(ys) - ypad, max(ys) + ypad]

    fig.add_hrect(y0=0, y1=y_rng[1], fillcolor="rgba(46,204,113,0.05)", line_width=0)
    fig.add_hrect(y0=y_rng[0], y1=0, fillcolor="rgba(231,76,60,0.05)", line_width=0)
    fig.add_hline(y=0, line_color=SUBTEXT, line_width=1)
    fig.add_vline(x=0, line_color=SUBTEXT, line_width=1)

    for lbl, xp, yp, clr in [
        ("BROAD STRENGTH", x_rng[1] * 0.6, y_rng[1] * 0.75, GREEN),
        ("NARROWING &mdash; top risk", x_rng[1] * 0.6, y_rng[0] * 0.75, RED),
        ("BASE BUILDING", x_rng[0] * 0.6, y_rng[1] * 0.75, BLUE),
        ("BROAD WEAKNESS", x_rng[0] * 0.6, y_rng[0] * 0.75, YELLOW),
    ]:
        fig.add_annotation(x=xp, y=yp, text=f"<b>{lbl}</b>", font=dict(color=clr, size=12),
                           showarrow=False, opacity=0.55)

    for r in results:
        clr = {"Negative": RED, "Positive": GREEN, "Neutral": BLUE}[r["divergence"]]
        fig.add_trace(go.Scatter(
            x=[r["rs_chg"]], y=[r["breadth_chg"]], mode="markers+text",
            marker=dict(color=clr, size=13, line=dict(color=TEXT, width=0.8)),
            text=[f'  {r["theme"]}'], textposition="middle right",
            textfont=dict(size=10, color=TEXT), showlegend=False,
            hovertemplate=f'<b>{r["theme"]}</b><br>RS {CHG_WEEKS}w: {r["rs_chg"]:+.1f}%'
                          f'<br>Breadth {CHG_WEEKS}w: {r["breadth_chg"]:+.1f} pts'
                          f'<br>% >50dma: {r["pct50"]:.0f}%<br>RS participation: {r["rs_part"]:.0f}%'
                          "<extra></extra>"))

    fig.update_layout(
        title=dict(text="<b>Breadth Divergence Map</b>  "
                        f'<span style="font-size:11px;color:{SUBTEXT}">'
                        f"{CHG_WEEKS}-week change in relative strength vs {CHG_WEEKS}-week change "
                        "in breadth composite &middot; bottom-right = leadership narrowing</span>",
                   font=dict(size=16, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY,
        font=dict(color=TEXT, family="monospace"),
        height=680, margin=dict(l=70, r=70, t=90, b=60),
        xaxis=dict(title=f"RS {CHG_WEEKS}w change (%)", title_font=dict(color=SUBTEXT),
                   range=x_rng, gridcolor=LGRAY, zeroline=False),
        yaxis=dict(title=f"Breadth {CHG_WEEKS}w change (pts)", title_font=dict(color=SUBTEXT),
                   range=y_rng, gridcolor=LGRAY, zeroline=False),
        hovermode="closest",
    )
    add_logo(fig)
    return fig


# -- HTML ---------------------------------------------------------------------
def summary_table(results):
    head = (f"<tr><th>Theme</th><th>N</th><th>Divergence</th><th>Weeks</th>"
            f"<th>RS {CHG_WEEKS}w</th><th>Breadth {CHG_WEEKS}w</th>"
            f"<th>% &gt;50dma</th><th>% &gt;200dma</th><th>RS particip.</th>"
            f"<th>Net 52w hi</th><th>Narrow high</th></tr>")
    body = []
    order = {"Negative": 0, "Positive": 1, "Neutral": 2}
    for r in sorted(results, key=lambda r: (order[r["divergence"]], -r["rs_chg"])):
        dclr = {"Negative": RED, "Positive": GREEN, "Neutral": SUBTEXT}[r["divergence"]]
        body.append(
            f'<tr><td class="nm">{r["theme"]}</td><td class="sub">{r["n"]}</td>'
            f'<td style="color:{dclr}"><b>{r["divergence"]}</b></td>'
            f'<td class="sub">{r["weeks_in_div"] or "&mdash;"}</td>'
            f'<td style="color:{GREEN if r["rs_chg"] > 0 else RED}">{r["rs_chg"]:+.1f}%</td>'
            f'<td style="color:{GREEN if r["breadth_chg"] > 0 else RED}">{r["breadth_chg"]:+.1f}</td>'
            f'<td>{r["pct50"]:.0f}%</td><td>{r["pct200"]:.0f}%</td>'
            f'<td><b>{r["rs_part"]:.0f}%</b></td><td>{r["nnh"]:+.0f}%</td>'
            f'<td style="color:{RED if r["narrow_high"] else SUBTEXT}">'
            f'{"YES" if r["narrow_high"] else "&mdash;"}</td></tr>'
        )
    return f'<div class="tbl-wrap"><table>{head}{"".join(body)}</table></div>'


def how_to_read():
    return f"""
<div class="note">
  <b>How to read it.</b> The rotation dashboard tells you how stretched a theme's
  leadership is. This tells you how <i>broad</i> it is &mdash; and narrowing usually
  comes first.
  <ul>
    <li><b>RS participation</b> is the one to watch: the share of constituents whose own
        ratio to the benchmark is above its own {MA_SLOW}-day MA. When the basket's RS line
        is still rising but RS participation is falling, a handful of names are carrying
        the theme and the rotation out has already started underneath.</li>
    <li><b>Breadth composite</b> = mean of (% above {MA_FAST}dma, RS participation).
        A <b>Negative</b> divergence means RS rose over {CHG_WEEKS} weeks while breadth fell
        more than {DIV_THRESHOLD:g} points &mdash; leadership narrowing, top risk.
        <b>Positive</b> is the mirror: RS still falling but breadth improving, which is what
        the end of a washout looks like.</li>
    <li><b>Narrow high</b> is the stricter version: the RS line is at a 26-week high while
        the breadth composite is not. Rarer, and a more pointed warning.</li>
    <li><b>Shaded bands</b> on each chart mark historical divergence runs of two weeks or
        more. Scroll back through them to see how much lead time they actually gave &mdash;
        that is the question the lead-lag study will answer properly.</li>
    <li>Breadth is a <i>warning</i>, not a trigger. It says the leadership is fragile, not
        that it has broken. Pair it with the 13-week ROC cross on the rotation dashboard.</li>
  </ul>
</div>
"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Theme Breadth &amp; Divergence</title>
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
  th:first-child, th:nth-child(3) {{ text-align: left; }}
  td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid #2C2C2E; white-space: nowrap; }}
  td.nm {{ text-align: left; color: #E5E5EA; }}
  td.sub {{ text-align: left; color: #8E8E93; }}
  td:nth-child(3) {{ text-align: left; }}
  tr:hover td {{ background: #2C2C2E; }}
</style>
</head>
<body>
<header>
  <h1>Theme Breadth &amp; Divergence</h1>
  <div class="meta">Generated {date_str} &middot; Participation internals for {n} theme baskets
  &middot; Leadership narrows before it breaks</div>
</header>
{body}
</body>
</html>
"""


def save_series(results):
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []
    for r in results:
        df = r["wk"][["rs", "pct50", "pct200", "rs_part", "nnh", "breadth",
                      "rs_chg", "breadth_chg", "divergence", "narrow_high"]].copy()
        df.insert(0, "theme", r["theme"])
        frames.append(df)
    out = pd.concat(frames).rename_axis("date").reset_index()
    path = os.path.join(DATA_DIR, "breadth_series.csv")
    out.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(out):,} rows)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = [t for _, _, cons in BASKETS for t in cons]
    tickers += [b for _, b, _ in BASKETS]
    close = fetch_closes(tickers)

    results = []
    for theme, bench, cons in BASKETS:
        res = compute_theme(close, theme, bench, cons)
        if res:
            results.append(res)
    if not results:
        raise SystemExit("No themes could be computed.")

    parts = [section_header("Breadth Divergence Map",
                            "Where each theme sits on relative strength vs participation")]
    parts.append(how_to_read())
    parts.append(fig_to_div(divergence_chart(results)))
    parts.append(summary_table(results))
    parts.append(section_header("Theme Internals",
                                "RS line, participation, and net new 52-week highs per basket"))
    order = {"Negative": 0, "Positive": 1, "Neutral": 2}
    for r in sorted(results, key=lambda r: (order[r["divergence"]], -r["rs_chg"])):
        parts.append(fig_to_div(theme_chart(r)))

    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"),
                                n=len(results), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "Breadth_Divergence_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")
    save_series(results)

    print(f"\n{'Theme':<30}{'Divergence':<12}{'RS13w':>8}{'Brdth13w':>10}"
          f"{'>50dma':>8}{'RSpart':>8}{'Narrow':>8}")
    for r in sorted(results, key=lambda r: (order[r["divergence"]], -r["rs_chg"])):
        print(f"{r['theme']:<30}{r['divergence']:<12}{r['rs_chg']:>+8.1f}"
              f"{r['breadth_chg']:>+10.1f}{r['pct50']:>7.0f}%{r['rs_part']:>7.0f}%"
              f"{'YES' if r['narrow_high'] else '-':>8}")


if __name__ == "__main__":
    main()
