"""
Complement Finder engine: the method from the "Complement Finder" notebook (Sept 2026 Market Musings), as
reusable functions. For a holding it finds the high-quality stocks that would most have smoothed the ride:

  1. Quality: beats its home benchmark's total return (XIC for Canadian names, SPY for US) and makes money in
     each of three equal sub-periods.
  2. Low correlation that holds up: weekly returns, worst of the three sub-periods <= 0.50.
  3. Holds up in sell-offs: beats the holding in >= 60% of stress tests (COVID, 2022 rate reset, Apr-2025
     tariffs, Mar-2026 shock, and the holding's own worst drawdown).
  4. Smooths the ride: a 50/50 blend rebalanced quarterly has a shallower max drawdown than the holding alone and
     no more than 5% more volatility.
  5. Ranked on correlation, stress-test edge and blend return per unit of risk; one per sector.

Total returns (dividends reinvested), converted to CAD. Backtested and hypothetical, not advice.

Differences from the notebook (for the website's Stock Lookup page): up to 5 complements instead of 3, drawn from
both markets, and the candidate universe (market cap, sector, name) comes from the revision CSVs
(data/us_1w_rev_est_screener.csv, data/cdn_1w_rev_est_screener.csv) rather than the Koyfin files, with the same
floors: TSX $3B+, S&P 500 members $10B+.
"""

import math
import os
import re

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

START = "2015-06-01"
PRICE_START = "2014-12-01"
MIN_YEARS = 5
MIN_MCAP_CDN_M = 3_000
MIN_MCAP_US_M = 10_000
N_SUBPERIODS = 3
MAX_SUBPERIOD_CORR = 0.50
MIN_HELD_UP_SHARE = 0.6
VOL_TOLERANCE = 0.05
TOP_N = 5
HOME_BENCH = {"CDN": "XIC.TO", "US": "SPY"}
FX = "CAD=X"
EPISODES = {
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate reset": ("2021-12-31", "2022-10-14"),
    "Apr-2025 tariffs": ("2025-02-19", "2025-04-08"),
    "Mar-2026 geopolitical": ("2026-03-02", "2026-03-20"),
}

# Revision-CSV Canadian tickers have their punctuation stripped; map back to Yahoo (as in the notebook)
CDN_SPECIAL = {"TECKB": "TECK-B", "BBDB": "BBD-B", "QSPUN": "QSR", "RCIB": "RCI-B", "GIBA": "GIB-A", "CCLB": "CCL-B",
               "QBRA": "QBR-B", "EMPA": "EMP-A", "CTCA": "CTC-A", "ACOX": "ACO-X"}


def cdn_to_yahoo(t: str) -> str:
    if t in CDN_SPECIAL:
        return CDN_SPECIAL[t] + ".TO"
    if len(t) >= 4 and t.endswith("UN"):
        return t[:-2] + "-UN.TO"
    return t + ".TO"


def is_cad_listed(t: str) -> bool:
    return t.endswith((".TO", ".V", ".NE", ".CN"))


def disp(t: str) -> str:
    return re.sub(r"\.(TO|V|NE|CN)$", "", t)


SHORT_OVERRIDE = {"Canadian Natural Resources Limited": "Canadian Natural"}


def short_name(name: str) -> str:
    if name in SHORT_OVERRIDE:
        return SHORT_OVERRIDE[name]
    out = re.sub(r"^The\s+|\s*\(The\)$", "", name)
    out = re.sub(r"\s+(and|&)\s+(Company|Co\.?)$|\s+Company$", "", out)
    out = re.sub(r"\b(Companies|Corporation|Limited|Incorporated|Inc\.?|Corp\.?|Ltd\.?|plc|N\.V\.|S\.A\.|Co\.)(?=\s|,|$)", "", out)
    return re.sub(r"\s+", " ", out).strip(" ,") or name


def load_universe(sp500_symbols: set) -> pd.DataFrame:
    """Candidate complements, indexed by Yahoo symbol: Name, Sector, Home. TSX $3B+ and S&P 500 members $10B+."""
    frames = []
    for fname, home, floor, mapper in (
        ("cdn_1w_rev_est_screener.csv", "CDN", MIN_MCAP_CDN_M, cdn_to_yahoo),
        ("us_1w_rev_est_screener.csv", "US", MIN_MCAP_US_M, lambda t: t.replace(".", "-")),
    ):
        d = pd.read_csv(os.path.join(DATA_DIR, fname), converters={"Ticker": lambda v: str(v).strip()})
        d["Market Cap"] = pd.to_numeric(d["Market Cap"], errors="coerce")
        d = d[d["Market Cap"] >= floor].copy()
        d["yf"] = d["Ticker"].map(mapper)
        if home == "US":
            d = d[d["yf"].isin(sp500_symbols)]
        d["Home"] = home
        frames.append(d[["yf", "Name", "Sector", "Home"]])
    return pd.concat(frames).drop_duplicates("yf").set_index("yf")


def to_cad(raw: pd.DataFrame) -> pd.DataFrame:
    """Weekday closes with every non-Canadian listing converted to CAD at the daily USD/CAD rate."""
    px = raw[raw.index.dayofweek < 5].copy()
    usd = [c for c in px.columns if c != FX and not is_cad_listed(c)]
    px[usd] = px[usd].mul(px[FX].ffill(), axis=0)
    return px.drop(columns=FX)


# ── Maths (unchanged from the notebook) ────────────────────────────────────────
def years(ix):
    return (ix[-1] - ix[0]).days / 365.25


def cagr(nav):
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / years(nav.index)) - 1


def ann_vol(nav):
    return nav.resample("W-FRI").last().pct_change(fill_method=None).iloc[1:].std() * np.sqrt(52)


def max_dd(nav):
    return (nav / nav.cummax() - 1).min()


def episode_returns(nav, episodes):
    return pd.DataFrame({k: nav.loc[:b].iloc[-1] / nav.loc[:a].iloc[-1] - 1 for k, (a, b) in episodes.items()})


def blend_navs(anchor_px, cand_px, w=0.5):
    """w * anchor + (1 - w) * each candidate, rebalanced every calendar quarter."""
    a = anchor_px.pct_change(fill_method=None).fillna(0).to_numpy()
    R = cand_px.pct_change(fill_method=None).fillna(0).to_numpy()
    periods = np.asarray(anchor_px.index.to_period("Q"))
    out, level = np.empty_like(R), np.ones(R.shape[1])
    for p in pd.unique(periods):
        m = periods == p
        out[m] = level * (w * np.cumprod(1 + a[m])[:, None] + (1 - w) * np.cumprod(1 + R[m], axis=0))
        level = out[m][-1]
    return pd.DataFrame(out, index=cand_px.index, columns=cand_px.columns)


# ── The screen ─────────────────────────────────────────────────────────────────
def analyze(anchor: str, px: pd.DataFrame, universe: pd.DataFrame, top_n: int = TOP_N, exclude=()):
    """Complements for one holding. Returns (result dict or None, reason string when there's no result)."""
    if anchor not in px.columns:
        return None, "no price data"
    s = px[anchor].dropna()
    if s.empty:
        return None, "no price data"
    start = max(pd.Timestamp(START), s.index[0])
    if years(pd.DatetimeIndex([start, px.index[-1]])) < MIN_YEARS:
        return None, f"only {years(s.index):.1f} years of history (at least {MIN_YEARS} needed)"

    excluded = set(exclude) | {anchor}
    W_all = px.loc[start:]
    W_all = W_all[W_all[anchor].notna()]          # the holding's own trading calendar (TSX and US holidays differ)
    cands = [c for c in universe.index if c in W_all.columns and c not in excluded]
    first = W_all[cands].apply(pd.Series.first_valid_index)
    cands = [c for c in cands if pd.notna(first[c]) and first[c] <= start + pd.Timedelta(days=10)
             and W_all[c].notna().mean() >= 0.95]
    benches = [b for b in HOME_BENCH.values() if b in W_all.columns and W_all[b].notna().mean() >= 0.95]
    W = W_all[[anchor] + cands + benches].ffill().bfill(limit=5)

    cut = np.linspace(0, len(W) - 1, N_SUBPERIODS + 1).round().astype(int)
    subs = [(W.index[cut[i]], W.index[cut[i + 1]]) for i in range(N_SUBPERIODS)]
    full_cagr = cagr(W)
    sub_cagr = pd.concat([cagr(W.loc[a:b]) for a, b in subs], axis=1)
    hurdle = universe.loc[cands, "Home"].map({h: full_cagr[b] for h, b in HOME_BENCH.items() if b in benches})
    quality = [c for c in cands if pd.notna(hurdle[c]) and full_cagr[c] >= hurdle[c] and (sub_cagr.loc[c] > 0).all()]
    if not quality:
        return None, "no candidates passed the quality screen"

    wk = W.resample("W-FRI").last().pct_change(fill_method=None).iloc[1:]
    worst_corr = pd.concat([wk.loc[a:b, quality].corrwith(wk.loc[a:b, anchor]) for a, b in subs], axis=1).max(axis=1)

    eps = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in EPISODES.items() if pd.Timestamp(a) >= start}
    nav_a = W[anchor]
    trough = (nav_a / nav_a.cummax() - 1).idxmin()
    peak = nav_a.loc[:trough].idxmax()
    if trough > peak:
        eps["Own worst drawdown"] = (peak, trough)
    ep = episode_returns(W[[anchor] + quality], eps)
    edge = ep.loc[quality].sub(ep.loc[anchor], axis=1)

    blends = blend_navs(nav_a, W[quality])
    a_stats = {"CAGR": full_cagr[anchor], "Vol": ann_vol(nav_a), "Max DD": max_dd(nav_a)}
    df = pd.DataFrame({
        "Name": universe.loc[quality, "Name"], "Sector": universe.loc[quality, "Sector"],
        "Home": universe.loc[quality, "Home"],
        "Worst sub-period corr": worst_corr, "Complement CAGR": full_cagr[quality],
        "Blend CAGR": cagr(blends), "Blend vol": ann_vol(blends), "Blend max DD": max_dd(blends),
        "Held up": (edge > 0).sum(axis=1), "Avg stress edge": edge.mean(axis=1),
    }, index=quality)
    df["Blend return/vol"] = df["Blend CAGR"] / df["Blend vol"]
    min_held = math.ceil(MIN_HELD_UP_SHARE * len(eps))
    df["Eligible"] = ((df["Worst sub-period corr"] <= MAX_SUBPERIOD_CORR) & (df["Held up"] >= min_held)
                      & (df["Blend max DD"] > a_stats["Max DD"]) & (df["Blend vol"] <= a_stats["Vol"] * (1 + VOL_TOLERANCE)))
    el = df[df["Eligible"]]
    df.loc[el.index, "Score"] = (el["Worst sub-period corr"].rank(pct=True, ascending=False)
                                 + el["Avg stress edge"].rank(pct=True) + el["Blend return/vol"].rank(pct=True)) / 3
    df = df.sort_values("Score", ascending=False)

    chosen, sectors = [], set()
    for t, r in df[df["Eligible"]].iterrows():
        if r["Sector"] not in sectors:
            chosen.append(t)
            sectors.add(r["Sector"])
        if len(chosen) == top_n:
            break
    if not chosen:
        return None, "no stock passed all the tests"
    return dict(anchor=anchor, start=W.index[0], end=W.index[-1], a_stats=a_stats, top=df.loc[chosen],
                n_eps=len(eps), n_cands=len(cands), n_quality=len(quality)), ""
