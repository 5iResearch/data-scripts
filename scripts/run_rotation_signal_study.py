"""
Rotation Signal Lead-Lag Study.

The dashboard and the breadth report both assume things that have never been
tested: that a weekly ratio RSI above 70 means leadership is about to flip,
that a channel z above +1.5 is "stretched", that a breadth divergence leads
the turn. This script tests those assumptions against labelled history.

Method
------
1. LABEL the turns. For each relative-strength ratio, run a zigzag over the
   weekly RS line to find local peaks (rotation OUT of the theme) and troughs
   (rotation IN). The swing threshold is adaptive -- a multiple of that
   ratio's own weekly vol, floored -- so a quiet ratio like XLP/SPY and a
   violent one like REMX/SPY both get sensible turn counts. Labelling is
   allowed to look ahead; it is the ground truth, not a signal.

2. COMPUTE the signals CAUSALLY. Every signal at week t uses only data up to
   and including week t, including a rolling 5-year regression channel refit
   each week. No full-sample fits anywhere.

3. DE-CLUSTER. A signal that stays true for ten weeks is one event, not ten.
   Firings within COOLDOWN weeks of a previous firing are dropped.

4. SCORE each signal three ways:
   - Hit rate / lead time: did a labelled turn of the right kind follow within
     HORIZON weeks, and by how many weeks did the signal lead it?
   - Recall: what share of labelled turns had the signal fire beforehand?
   - Forward RS return vs that ratio's own baseline. This is the metric that
     matters most because it needs no labels at all -- it just asks whether
     relative strength was actually worse (for top signals) after the signal
     fired than it normally is.

5. TEST it. A randomization test draws the same number of random dates from
   the same ratio's history and asks how often pure chance beats the signal's
   forward return. Reported as an empirical p-value.

Caveats, stated up front because they bound everything below: the number of
genuine leadership flips per ratio is small (typically 6-15 over the sample),
the randomization test does not fully account for overlapping forward windows,
and no parameter here has been optimized -- these are conventional defaults
being tested, not fitted values. Treat a p-value of 0.04 as "worth another
look", not as proof.
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_rotation_dashboard import (  # noqa: E402
    BLUE, DGRAY, GREEN, LGRAY, MGRAY, ORANGE, RED, SUBTEXT, TEXT, YELLOW,
    SECTIONS, add_logo, all_tickers, compute_ratio, fetch, fig_to_div,
    section_header,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "rotation-dashboard")
DATA_DIR = os.path.join(REPO_ROOT, "data")

FIT_WEEKS = 260        # rolling regression channel window
MA_WEEKS = 40
ROC_WEEKS = 13
NORM_WEEKS = 156
RSI_LEN = 14
MIN_WEEKS = 380        # need channel window + room to evaluate

SWING_VOL_MULT = 1.6   # zigzag threshold = this x 13-week vol of the ratio
SWING_FLOOR = 0.07     # ...but never less than 7%
SWING_CAP = 0.30

COOLDOWN = 8           # weeks between counted firings of the same signal
HORIZON = 13           # a "hit" = labelled turn within this many weeks
FWD_HORIZONS = (4, 8, 13, 26)
N_RANDOM = 2000        # randomization draws
RNG = np.random.default_rng(20260922)


# -- Labelling ----------------------------------------------------------------
def zigzag(values, theta):
    """Local peaks/troughs in a series, confirmed by a theta retracement.
    Look-ahead by construction -- this is ground truth, not a signal."""
    piv, dir_, ext_i, ext_v = [], 0, 0, values[0]
    for i in range(1, len(values)):
        v = values[i]
        if dir_ >= 0:
            if v > ext_v:
                ext_i, ext_v = i, v
            elif v <= ext_v * (1 - theta):
                piv.append((ext_i, "peak"))
                dir_, ext_i, ext_v = -1, i, v
        else:
            if v < ext_v:
                ext_i, ext_v = i, v
            elif v >= ext_v * (1 + theta):
                piv.append((ext_i, "trough"))
                dir_, ext_i, ext_v = 1, i, v
    return piv


def label_turns(rs):
    vol13 = rs.pct_change().rolling(13).std().mean() * np.sqrt(13)
    theta = float(np.clip(SWING_VOL_MULT * vol13, SWING_FLOOR, SWING_CAP))
    piv = zigzag(rs.values, theta)
    peaks = pd.DatetimeIndex([rs.index[i] for i, k in piv if k == "peak"])
    troughs = pd.DatetimeIndex([rs.index[i] for i, k in piv if k == "trough"])
    return peaks, troughs, theta


# -- Causal signal construction -----------------------------------------------
def rolling_channel_z(logrs, window=FIT_WEEKS):
    """Z-score of the last residual from a regression refit each week on the
    trailing `window` weeks. Causal by construction."""
    y = logrs.values
    n = len(y)
    out = np.full(n, np.nan)
    x = np.arange(window)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    for t in range(window - 1, n):
        yy = y[t - window + 1: t + 1]
        slope = (x_c * (yy - yy.mean())).sum() / denom
        intercept = yy.mean() - slope * x.mean()
        resid = yy - (intercept + slope * x)
        sd = resid.std()
        if sd > 0:
            out[t] = resid[-1] / sd
    return pd.Series(out, index=logrs.index)


def wilder_rsi(s, length=RSI_LEN):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(rs):
    f = pd.DataFrame(index=rs.index)
    f["rs"] = rs
    f["rsi"] = wilder_rsi(rs)
    f["ma"] = rs.rolling(MA_WEEKS).mean()
    f["roc"] = rs.pct_change(ROC_WEEKS) * 100
    stretch = rs / f["ma"] - 1
    f["stretch_z"] = ((stretch - stretch.rolling(NORM_WEEKS, min_periods=52).mean())
                      / stretch.rolling(NORM_WEEKS, min_periods=52).std())
    f["ch_z"] = rolling_channel_z(np.log(rs))
    f["rsi_z"] = (f["rsi"] - 50) / 10.0
    f["composite"] = f[["rsi_z", "ch_z", "stretch_z"]].mean(axis=1)
    f["mrsi"] = wilder_rsi(rs.resample("ME").last()).reindex(rs.index, method="ffill")
    return f


def cross_below(s, level):
    return (s < level) & (s.shift(1) >= level)


def cross_above(s, level):
    return (s > level) & (s.shift(1) <= level)


TOP_SIGNALS = {
    "wRSI > 70": lambda f: f["rsi"] > 70,
    "wRSI > 75": lambda f: f["rsi"] > 75,
    "Channel z > +1.5": lambda f: f["ch_z"] > 1.5,
    "Channel z > +2.0": lambda f: f["ch_z"] > 2.0,
    "40w stretch z > +1.5": lambda f: f["stretch_z"] > 1.5,
    "Composite > +1.0": lambda f: f["composite"] > 1.0,
    "Composite > +1.5": lambda f: f["composite"] > 1.5,
    "Monthly RSI > 70": lambda f: f["mrsi"] > 70,
    "13w ROC crosses < 0": lambda f: cross_below(f["roc"], 0),
    "Ratio crosses < 40w MA": lambda f: cross_below(f["rs"] / f["ma"], 1.0),
    "Extended + ROC cross": lambda f: cross_below(f["roc"], 0) & (f["composite"].shift(1) > 1.0),
    "Extended + MA cross": lambda f: cross_below(f["rs"] / f["ma"], 1.0) & (f["composite"].shift(1) > 1.0),
}

BOTTOM_SIGNALS = {
    "wRSI < 30": lambda f: f["rsi"] < 30,
    "wRSI < 25": lambda f: f["rsi"] < 25,
    "Channel z < -1.5": lambda f: f["ch_z"] < -1.5,
    "Channel z < -2.0": lambda f: f["ch_z"] < -2.0,
    "40w stretch z < -1.5": lambda f: f["stretch_z"] < -1.5,
    "Composite < -1.0": lambda f: f["composite"] < -1.0,
    "Composite < -1.5": lambda f: f["composite"] < -1.5,
    "13w ROC crosses > 0": lambda f: cross_above(f["roc"], 0),
    "Ratio crosses > 40w MA": lambda f: cross_above(f["rs"] / f["ma"], 1.0),
    "Washed + ROC cross": lambda f: cross_above(f["roc"], 0) & (f["composite"].shift(1) < -1.0),
    "Washed + MA cross": lambda f: cross_above(f["rs"] / f["ma"], 1.0) & (f["composite"].shift(1) < -1.0),
}

BREADTH_TOP = {
    "Breadth neg. divergence": lambda f: f["divergence"] == "Negative",
    "Narrow high": lambda f: f["narrow_high"],
    "RS particip. < 50 & RS up": lambda f: (f["rs_part"] < 50) & (f["roc"] > 0),
    "RS particip. falling 13w": lambda f: f["rs_part"].diff(13) < -15,
}
BREADTH_BOTTOM = {
    "Breadth pos. divergence": lambda f: f["divergence"] == "Positive",
    "RS particip. rising 13w": lambda f: f["rs_part"].diff(13) > 15,
}


# -- Scoring ------------------------------------------------------------------
def declustered_firings(mask, cooldown=COOLDOWN):
    idx = mask.index[mask.fillna(False).values]
    out, last = [], None
    for d in idx:
        if last is None or (d - last).days >= cooldown * 7:
            out.append(d)
            last = d
    return pd.DatetimeIndex(out)


def lead_times(firings, turns, horizon=HORIZON):
    """Weeks from each firing to the next labelled turn (None if none within
    horizon). Positive lead = the signal came first."""
    leads = []
    for f in firings:
        nxt = turns[turns >= f]
        if len(nxt) == 0:
            leads.append(np.nan)
            continue
        wks = (nxt[0] - f).days / 7.0
        leads.append(wks if wks <= horizon else np.nan)
    return np.array(leads, dtype=float)


def nearest_lead(firings, turns, window=HORIZON):
    """Signed distance to the NEAREST turn either side, for diagnosing whether
    a signal is early, on time, or chronically late."""
    out = []
    for f in firings:
        if len(turns) == 0:
            continue
        diffs = np.array([(t - f).days / 7.0 for t in turns])
        near = diffs[np.argmin(np.abs(diffs))]
        if abs(near) <= window:
            out.append(near)
    return np.array(out, dtype=float)


def forward_returns(rs, dates, horizon):
    vals = []
    for d in dates:
        loc = rs.index.get_loc(d)
        if loc + horizon < len(rs):
            vals.append(rs.iloc[loc + horizon] / rs.iloc[loc] - 1)
    return np.array(vals, dtype=float) * 100


def randomization_p(rs, n_draws, horizon, observed, tail):
    """How often do n random dates from the same series beat the observed mean
    forward return? Does not fully account for overlapping windows."""
    valid = np.arange(len(rs) - horizon)
    if len(valid) < n_draws or n_draws == 0:
        return np.nan
    fwd = (rs.values[horizon:] / rs.values[:-horizon] - 1) * 100
    draws = RNG.choice(fwd, size=(N_RANDOM, n_draws), replace=True).mean(axis=1)
    if tail == "low":
        return float((draws <= observed).mean())
    return float((draws >= observed).mean())


def score_signal(name, mask, feats, turns, kind):
    rs = feats["rs"]
    firings = declustered_firings(mask)
    firings = firings[firings >= rs.index[FIT_WEEKS]] if len(rs) > FIT_WEEKS else firings
    if len(firings) == 0:
        return None

    leads = lead_times(firings, turns)
    hits = int(np.isfinite(leads).sum())
    near = nearest_lead(firings, turns)

    row = dict(signal=name, kind=kind, n_firings=len(firings), n_turns=len(turns),
               hit_rate=hits / len(firings) * 100,
               median_lead=float(np.nanmedian(leads)) if hits else np.nan,
               median_nearest=float(np.median(near)) if len(near) else np.nan,
               pct_late=float((near < 0).mean() * 100) if len(near) else np.nan)

    # recall: share of turns preceded by a firing within HORIZON weeks
    if len(turns):
        covered = sum(any((t - f).days / 7.0 <= HORIZON and t >= f for f in firings) for t in turns)
        row["recall"] = covered / len(turns) * 100
    else:
        row["recall"] = np.nan

    tail = "low" if kind == "top" else "high"
    for h in FWD_HORIZONS:
        fwd = forward_returns(rs, firings, h)
        base = (rs.shift(-h) / rs - 1).dropna() * 100
        row[f"fwd{h}"] = float(fwd.mean()) if len(fwd) else np.nan
        row[f"base{h}"] = float(base.mean())
        row[f"edge{h}"] = row[f"fwd{h}"] - row[f"base{h}"]
        if h == HORIZON:
            row["p_value"] = randomization_p(rs, len(fwd), h, row[f"fwd{h}"], tail)
    return row


def run_universe(frames, signal_sets, tag):
    """frames: {ratio_name: feature DataFrame with 'rs'}."""
    rows = []
    for ratio, feats in frames.items():
        rs = feats["rs"].dropna()
        if len(rs) < MIN_WEEKS:
            continue
        peaks, troughs, theta = label_turns(rs)
        for kind, sigs in signal_sets:
            turns = peaks if kind == "top" else troughs
            for name, fn in sigs.items():
                try:
                    mask = fn(feats)
                except Exception:
                    continue
                r = score_signal(name, mask, feats, turns, kind)
                if r:
                    r.update(ratio=ratio, theta=theta * 100, universe=tag,
                             n_peaks=len(peaks), n_troughs=len(troughs))
                    rows.append(r)
    return pd.DataFrame(rows)


def aggregate(df):
    """Pool each signal across ratios. Edge is averaged per-ratio so a single
    volatile ratio cannot dominate."""
    g = df.groupby(["universe", "kind", "signal"])
    agg = g.agg(
        ratios=("ratio", "nunique"),
        firings=("n_firings", "sum"),
        hit_rate=("hit_rate", "mean"),
        recall=("recall", "mean"),
        median_lead=("median_lead", "median"),
        median_nearest=("median_nearest", "median"),
        pct_late=("pct_late", "mean"),
        edge4=("edge4", "median"),
        edge13=("edge13", "median"),
        edge26=("edge26", "median"),
        edge13_mean=("edge13", "mean"),
        fwd13=("fwd13", "median"),
        base13=("base13", "median"),
        p_value=("p_value", "median"),
    ).reset_index()
    # share of ratios where the edge pointed the right way for this signal kind
    sign_ok = df.assign(ok=np.where(df.kind == "top", df.edge13 < 0, df.edge13 > 0))
    agg2 = sign_ok.groupby(["universe", "kind", "signal"]).ok.mean().mul(100).rename("pct_correct")
    agg = agg.merge(agg2, on=["universe", "kind", "signal"], how="left")
    return agg.sort_values(["universe", "kind", "edge13"])


# -- Report -------------------------------------------------------------------
def edge_chart(agg, universe, kind, title):
    d = agg[(agg.universe == universe) & (agg.kind == kind)].sort_values("edge13")
    if d.empty:
        return None
    if kind == "bottom":
        d = d.sort_values("edge13", ascending=True)
    good = (d.edge13 < 0) if kind == "top" else (d.edge13 > 0)
    clrs = [GREEN if g else RED for g in good]
    fig = go.Figure(go.Bar(
        x=d.edge13, y=d.signal, orientation="h", marker_color=clrs,
        text=[f"{v:+.1f}%" for v in d.edge13], textposition="outside",
        textfont=dict(size=9, color=TEXT),
        customdata=np.stack([d.firings, d.hit_rate, d.median_lead, d.p_value,
                             d.pct_correct, d.edge13_mean], axis=-1),
        hovertemplate="<b>%{y}</b><br>13w edge vs baseline (median): %{x:+.2f}%"
                      "<br>mean across ratios: %{customdata[5]:+.2f}%"
                      "<br>ratios with right sign: %{customdata[4]:.0f}%"
                      "<br>firings: %{customdata[0]:.0f}<br>hit rate: %{customdata[1]:.0f}%"
                      "<br>median lead: %{customdata[2]:.0f}w<br>p: %{customdata[3]:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=SUBTEXT, line_width=1)
    sub = ("negative = relative strength was weaker than normal after the signal (what a top signal should do)"
           if kind == "top" else
           "positive = relative strength was stronger than normal after the signal (what a bottom signal should do)")
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>  "
                        f'<span style="font-size:11px;color:{SUBTEXT}">{sub}</span>',
                   font=dict(size=15, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY,
        font=dict(color=TEXT, family="monospace"),
        height=max(360, len(d) * 30 + 130), bargap=0.3, showlegend=False,
        margin=dict(l=210, r=90, t=80, b=50),
        xaxis=dict(title="Median across ratios of (13-week RS return after signal minus that ratio's baseline), %",
                   title_font=dict(color=SUBTEXT), gridcolor=LGRAY, zeroline=False),
        yaxis=dict(gridcolor=LGRAY, tickfont=dict(size=10)),
    )
    add_logo(fig)
    return fig


def agg_table(agg, universe, kind):
    d = agg[(agg.universe == universe) & (agg.kind == kind)].copy()
    if d.empty:
        return ""
    d = d.sort_values("edge13", ascending=(kind == "top"))
    head = ("<tr><th>Signal</th><th>Ratios</th><th>Firings</th><th>Hit rate</th>"
            "<th>Recall</th><th>Median lead</th><th>% late</th>"
            "<th>Fwd 13w</th><th>Baseline</th><th>Edge 13w (med)</th><th>Edge 13w (mean)</th>"
            "<th>% ratios right sign</th><th>Edge 26w</th><th>p</th></tr>")
    body = []
    for _, r in d.iterrows():
        good = (r.edge13 < 0) if kind == "top" else (r.edge13 > 0)
        eclr = GREEN if good else RED
        pclr = GREEN if (pd.notna(r.p_value) and r.p_value < 0.10) else SUBTEXT
        lead = f"{r.median_lead:.0f}w" if pd.notna(r.median_lead) else "&mdash;"
        body.append(
            f'<tr><td class="nm">{r.signal}</td><td class="sub">{r.ratios:.0f}</td>'
            f'<td>{r.firings:.0f}</td><td>{r.hit_rate:.0f}%</td><td>{r.recall:.0f}%</td>'
            f'<td>{lead}</td><td>{r.pct_late:.0f}%</td>'
            f'<td>{r.fwd13:+.1f}%</td><td class="sub">{r.base13:+.1f}%</td>'
            f'<td style="color:{eclr}"><b>{r.edge13:+.1f}%</b></td>'
            f'<td class="sub">{r.edge13_mean:+.1f}%</td>'
            f'<td style="color:{GREEN if r.pct_correct > 55 else SUBTEXT}">{r.pct_correct:.0f}%</td>'
            f'<td style="color:{eclr}">{r.edge26:+.1f}%</td>'
            f'<td style="color:{pclr}">{r.p_value:.3f}</td></tr>')
    return f'<div class="tbl-wrap"><table>{head}{"".join(body)}</table></div>'


def episode_table(frames, ratios, since="2026-01-01"):
    """What actually fired around the 2026 turns, ratio by ratio."""
    head = ("<tr><th>Ratio</th><th>Labelled turn</th><th>Type</th>"
            "<th>Signal</th><th>Fired</th><th>Lead (weeks)</th></tr>")
    body = []
    for ratio in ratios:
        feats = frames.get(ratio)
        if feats is None:
            continue
        rs = feats["rs"].dropna()
        peaks, troughs, _ = label_turns(rs)
        turns = [(d, "Peak") for d in peaks] + [(d, "Trough") for d in troughs]
        turns = sorted([t for t in turns if t[0] >= pd.Timestamp(since)])
        for tdate, ttype in turns:
            sigs = TOP_SIGNALS if ttype == "Peak" else BOTTOM_SIGNALS
            for name, fn in sigs.items():
                try:
                    fires = declustered_firings(fn(feats))
                except Exception:
                    continue
                near = fires[(fires >= tdate - pd.Timedelta(weeks=26))
                             & (fires <= tdate + pd.Timedelta(weeks=13))]
                if len(near) == 0:
                    continue
                f = near[np.argmin(np.abs([(x - tdate).days for x in near]))]
                lead = (tdate - f).days / 7.0
                lclr = GREEN if lead > 0 else RED
                body.append(
                    f'<tr><td class="nm">{ratio}</td><td>{tdate:%Y-%m-%d}</td>'
                    f'<td class="sub">{ttype}</td><td class="sub">{name}</td>'
                    f'<td>{f:%Y-%m-%d}</td>'
                    f'<td style="color:{lclr}"><b>{lead:+.0f}</b></td></tr>')
    if not body:
        return "<div class='note'>No labelled turns in this window.</div>"
    return f'<div class="tbl-wrap"><table>{head}{"".join(body)}</table></div>'


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rotation Signal Study</title>
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
  th:first-child {{ text-align: left; }}
  td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid #2C2C2E; white-space: nowrap; }}
  td.nm {{ text-align: left; color: #E5E5EA; }}
  td.sub {{ text-align: left; color: #8E8E93; }}
  tr:hover td {{ background: #2C2C2E; }}
</style>
</head>
<body>
<header>
  <h1>Rotation Signal Study</h1>
  <div class="meta">Generated {date_str} &middot; {n_ratios} ratios &middot; {n_turns} labelled turns
  &middot; Causal signals, de-clustered firings, forward RS returns vs per-ratio baselines</div>
</header>
{body}
</body>
</html>
"""


METHOD_NOTE = f"""
<div class="note">
  <b>How to read it.</b> Turns are labelled with a zigzag on each ratio's weekly RS line, using an
  adaptive swing threshold ({SWING_VOL_MULT:g}&times; the ratio's own 13-week vol, floored at
  {SWING_FLOOR*100:.0f}%). Every signal is computed causally &mdash; the regression channel is refit
  each week on the trailing {FIT_WEEKS} weeks &mdash; and repeat firings within {COOLDOWN} weeks
  collapse into one event.
  <ul>
    <li><b>Edge 13w</b> is the metric to trust. It is the mean 13-week forward RS return after the
        signal minus that same ratio's unconditional average over the same horizon. It needs no
        labels, so it does not inherit the zigzag's assumptions. For a <i>top</i> signal you want it
        <b>negative</b>; for a <i>bottom</i> signal, <b>positive</b>.</li>
    <li><b>Hit rate</b> is the share of firings followed by a labelled turn of the right kind within
        {HORIZON} weeks. <b>Recall</b> is the share of labelled turns the signal caught beforehand.
        A signal can have a high hit rate and useless recall &mdash; it fires rarely but is right
        when it does.</li>
    <li><b>% late</b> is the share of firings where the nearest turn had <i>already happened</i>.
        Above 50% means the signal is a confirmation, not a warning, whatever its edge.</li>
    <li><b>p</b> is a randomization test: the share of 2,000 draws of the same number of random dates
        from the same ratio that beat the signal's forward return. It does not correct for
        overlapping forward windows or for testing many signals at once, so a single p below 0.05
        here is weak evidence, not a discovery.</li>
    <li>Nothing here is optimized. These are conventional thresholds being tested, which is the
        point &mdash; a fitted threshold would look better and mean less.</li>
  </ul>
</div>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fetch(all_tickers())
    frames, total_turns = {}, 0
    for _title, _sub, spec in SECTIONS:
        for name, num, den, _grp in spec:
            rs = compute_ratio(num, den)
            if rs is None or len(rs) < MIN_WEEKS:
                continue
            frames[name] = build_features(rs)
    print(f"Built {len(frames)} ratio feature sets")

    px_df = run_universe(frames, [("top", TOP_SIGNALS), ("bottom", BOTTOM_SIGNALS)], "Price ratios")

    bpath = os.path.join(DATA_DIR, "breadth_series.csv")
    bframes = {}
    if os.path.exists(bpath):
        bd = pd.read_csv(bpath, parse_dates=["date"], keep_default_na=False, na_values=[""])
        for theme, g in bd.groupby("theme"):
            g = g.set_index("date").sort_index()
            f = build_features(g["rs"].astype(float))
            for col in ("rs_part", "pct50", "breadth", "divergence", "narrow_high"):
                if col in g.columns:
                    f[col] = g[col]
            if "narrow_high" in f.columns:
                f["narrow_high"] = f["narrow_high"].astype(str).str.lower().eq("true")
            bframes[theme] = f
        print(f"Built {len(bframes)} breadth feature sets")
    else:
        print(f"! {bpath} not found -- run generate_breadth_divergence_report.py first")

    br_sets = [("top", {**TOP_SIGNALS, **BREADTH_TOP}),
               ("bottom", {**BOTTOM_SIGNALS, **BREADTH_BOTTOM})]
    br_df = run_universe(bframes, br_sets, "Theme baskets") if bframes else pd.DataFrame()

    full = pd.concat([px_df, br_df], ignore_index=True) if len(br_df) else px_df
    full.to_csv(os.path.join(DATA_DIR, "signal_study_results.csv"), index=False)
    agg = aggregate(full)
    agg.to_csv(os.path.join(DATA_DIR, "signal_study_summary.csv"), index=False)

    for _r, f in frames.items():
        p, t, _ = label_turns(f["rs"].dropna())
        total_turns += len(p) + len(t)

    parts = [section_header("Method &amp; Caveats", "What is being tested, and how far it can be trusted")]
    parts.append(METHOD_NOTE)

    for uni in full.universe.unique():
        for kind, lab in (("top", "Top signals"), ("bottom", "Bottom signals")):
            parts.append(section_header(f"{uni} &mdash; {lab}",
                                        "Ranked by 13-week forward RS edge against each ratio's own baseline"))
            fig = edge_chart(agg, uni, kind, f"{uni}: {lab}")
            if fig:
                parts.append(fig_to_div(fig))
            parts.append(agg_table(agg, uni, kind))

    key = ["Semis / AI Hardware", "Cdn Growth Basket", "Energy", "TSX Energy",
           "Energy vs Tech (US)", "Cdn Energy vs Cdn Growth", "Rare Earth / Crit. Minerals"]
    parts.append(section_header("The 2026 Episode",
                                "Every labelled turn since January 2026 and which signals fired near it"))
    parts.append(episode_table(frames, [k for k in key if k in frames]))

    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"),
                                n_ratios=len(frames) + len(bframes), n_turns=total_turns,
                                body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "Rotation_Signal_Study.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Saved: {out_path}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for uni in agg.universe.unique():
        for kind in ("top", "bottom"):
            d = agg[(agg.universe == uni) & (agg.kind == kind)].sort_values(
                "edge13", ascending=(kind == "top"))
            if d.empty:
                continue
            print(f"\n=== {uni} / {kind} signals (best first) ===")
            print(d[["signal", "firings", "hit_rate", "recall", "median_lead",
                     "pct_late", "edge13", "edge13_mean", "pct_correct",
                     "edge26", "p_value"]].to_string(
                index=False, float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
