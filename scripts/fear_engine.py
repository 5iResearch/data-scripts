"""
Leader Selloff engine (v2, simplified), shared by fear_study.py (backtest)
and generate_peak_fear_report.py (daily report).

Buy signal = all four conditions, each point-in-time:

  1. Historic leader - beat SPY over the 5 years ending 3 months ago (so the
                       current selloff can't disqualify it) and ranks in the
                       top third of its universe on that measure
  2. Weekly RSI(14)  - at or below its own q_w percentile of every completed
                       weekly RSI in the name's history
  3. Monthly RSI(14) - at or below its own q_m percentile of every completed
                       monthly RSI in the name's history
  4. Volatility spike- 20-day realized (Garman-Klass) volatility reached the
                       stock's own "VIX 30" within the last 20 trading days:
                       the same percentile of ITS history that VIX 30 is of
                       VIX's history since 1990

RSIs are "live" - the current week/month bar uses today's close, like a
charting platform - while thresholds only use completed bars, so a value on
date t never uses anything after the close of t.

Also here: the market-driven / sector / stock-specific split of each 21-day
move (a diagnostic), and the market fear gauge (VIX term structure).
Price/volume/volatility only - no fundamentals.
"""

import io
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_screening import load_nasdaq100_table, load_sp500_sectors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "data", "cache")
CONFIG_PATH = os.path.join(REPO_ROOT, "data", "fear_model_config.json")

BENCH = "SPY"
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
CHUNK_SIZE = 200
HISTORY_START = "2000-01-01"  # thresholds are vs each name's full history, so the daily report needs it too

RSI_LEN = 14
W_MIN_BARS = 156      # 3 years of weekly bars before a weekly threshold exists
M_MIN_BARS = 60       # 5 years of monthly bars before a monthly threshold exists
VOL_MIN_YEARS = 3     # years of daily volatility before a spike threshold exists
VOL_MIN_DAYS = VOL_MIN_YEARS * 252
MKT_PCT_WINDOW = 1260  # market gauge: 5-year percentile window
MKT_PCT_MIN = 504
VIX_PCT_FALLBACK = {25: 0.85, 30: 0.93, 35: 0.96}  # approximate; only used if CBOE is unreachable

# XLRE (2015) and XLC (2018) are too young for a 20-year study; IYR and VOX
# cover the same sectors back to 2000/2004.
SECTOR_ETF_BY_GICS = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "IYR", "Communication Services": "VOX",
}
# Nasdaq-100-only names come with ICB industries; map them onto the GICS names above
ICB_TO_GICS = {
    "Technology": "Information Technology", "Telecommunications": "Communication Services",
    "Basic Materials": "Materials", "Health Care": "Health Care", "Financials": "Financials",
    "Consumer Discretionary": "Consumer Discretionary", "Consumer Staples": "Consumer Staples",
    "Industrials": "Industrials", "Energy": "Energy", "Utilities": "Utilities", "Real Estate": "Real Estate",
}

CRYPTO = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"}
CRYPTO_PPY = 365  # crypto trades every calendar day

DEFAULT_CONFIG = {
    "w_rsi_pct": 0.10,       # weekly RSI at or below its own 10th percentile
    "m_rsi_pct": 0.20,       # monthly RSI at or below its own 20th percentile
    "vix_equiv_level": 30,   # vol spike = the stock's own equivalent of VIX at this level
    "vol_lookback": 20,      # the spike must be within this many trading days
    "lead_years": 5,         # leadership = relative return vs SPY over this many years...
    "lead_lag": 63,          # ...ending this many trading days ago
    "lead_top_pct": 0.33,    # ...positive and in this top fraction of the universe
    "cooldown": 63,          # min trading days between events for the same ticker
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


# ── Universes ─────────────────────────────────────────────────────────────────
def sp500_universe():
    df = load_sp500_sectors()
    df["Ticker"] = df["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    names = dict(zip(df["Ticker"], df["Security"]))
    sectors = dict(zip(df["Ticker"], df["GICS Sector"]))
    return df["Ticker"].tolist(), names, sectors


def etf_universe():
    from generate_regime_monitors_report import SECTOR_UNIVERSE, THEME_UNIVERSE

    seen = {}
    for name, ticker, group in SECTOR_UNIVERSE + THEME_UNIVERSE:
        seen.setdefault(ticker, (name, group))
    tickers = list(seen)
    return tickers, {t: seen[t][0] for t in tickers}, {t: seen[t][1] for t in tickers}


def us_stock_universe():
    """S&P 500 + Nasdaq-100 (union), ranked together as one universe."""
    tickers, names, sectors = sp500_universe()
    try:
        ndx = load_nasdaq100_table()
    except Exception as exc:
        print(f"  Nasdaq-100 list unavailable ({exc}); using S&P 500 only")
        return tickers, names, sectors
    for _, r in ndx.iterrows():
        t = str(r["Ticker"]).strip().upper().replace(".", "-")
        if t and t not in names:
            tickers.append(t)
            names[t] = r["Company"]
            industry = str(r["Sector"]).strip()
            sectors[t] = ICB_TO_GICS.get(industry, industry)
    return tickers, names, sectors


def crypto_universe():
    return list(CRYPTO), dict(CRYPTO), {t: "Crypto" for t in CRYPTO}


def crypto_config(cfg):
    """Crypto trades every calendar day and the universe is just two coins:
    convert trading-day settings to calendar days, and make the leader rule
    simply 'beat SPY over the lookback' (ranking two coins means nothing)."""
    k = CRYPTO_PPY / 252
    return dict(cfg, periods_per_year=CRYPTO_PPY, lead_top_pct=1.0,
                lead_lag=round(cfg["lead_lag"] * k), vol_lookback=round(cfg["vol_lookback"] * k),
                cooldown=round(cfg["cooldown"] * k))


# ── Data ──────────────────────────────────────────────────────────────────────
def download_ohlcv(tickers, start=None, period=None, chunk_size=CHUNK_SIZE, retries=3):
    """Batched yfinance download -> {field: DataFrame(dates x tickers)}.
    Yahoo intermittently rate-limits large pulls and returns empty frames for
    whole batches, so failed tickers are retried in smaller batches after a pause."""
    tickers = list(dict.fromkeys(tickers))
    frames = {f: {} for f in FIELDS}

    def fetch(batch, size):
        for i in range(0, len(batch), size):
            chunk = batch[i : i + size]
            print(f"  downloading {i + 1}-{i + len(chunk)} of {len(batch)}...")
            try:
                raw = yf.download(
                    chunk, start=start, period=period, group_by="ticker",
                    auto_adjust=True, threads=True, progress=False,
                )
            except Exception as exc:
                print(f"  batch error: {exc}")
                continue
            for t in chunk:
                try:
                    sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                except KeyError:
                    continue
                if sub.empty or sub["Close"].dropna().empty:
                    continue
                for f in FIELDS:
                    frames[f][t] = sub[f]

    fetch(tickers, chunk_size)
    for attempt in range(1, retries + 1):
        missing = [t for t in tickers if t not in frames["Close"]]
        if len(missing) <= max(2, 0.02 * len(tickers)):
            break  # a couple of misses are delisted/renamed tickers, not rate-limiting - don't wait on them
        wait = 20 * attempt
        print(f"  {len(missing)} tickers failed; retrying in smaller batches after {wait}s (attempt {attempt}/{retries})...")
        time.sleep(wait)
        fetch(missing, max(10, chunk_size // (4 * attempt)))
    missing = [t for t in tickers if t not in frames["Close"]]
    if missing:
        print(f"  gave up on {len(missing)} tickers: {missing[:25]}{' ...' if len(missing) > 25 else ''}")
    panel = {f: pd.DataFrame(v).sort_index() for f, v in frames.items()}
    return clean_panel(panel)


def clean_panel(panel):
    close = panel["Close"].dropna(how="all")
    out = {}
    for f in FIELDS:
        df = panel[f].reindex(index=close.index, columns=close.columns).astype(float)
        df = df.where(df > 0)
        # bridge the odd missing print, but never carry a price across a real gap
        out[f] = df if f == "Volume" else df.ffill(limit=3)
    return out


def load_panel(tickers, start, cache_name=None, refresh=False):
    """download_ohlcv with an optional local pickle cache (data/cache/)."""
    path = os.path.join(CACHE_DIR, f"{cache_name}.pkl") if cache_name else None
    if path and os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            panel = pickle.load(f)
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        print(f"  loaded cached panel {cache_name} ({panel['Close'].shape[1]} tickers, {age.days}d old)")
        missing = [t for t in dict.fromkeys(tickers) if t not in panel["Close"].columns]
        if missing:
            print(f"  cache lacks {len(missing)} tickers; downloading just those...")
            new = download_ohlcv(missing, start=start)
            if new["Close"].shape[1]:
                idx = panel["Close"].index  # keep the cache's dates so the panel stays aligned
                panel = {f: pd.concat([panel[f], new[f].reindex(idx)], axis=1) for f in FIELDS}
                with open(path, "wb") as f:
                    pickle.dump(panel, f)
        return panel
    panel = download_ohlcv(tickers, start=start)
    if path:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(panel, f)
    return panel


def subset_panel(panel, tickers):
    cols = [t for t in dict.fromkeys(tickers) if t in panel["Close"].columns]
    return {f: panel[f][cols] for f in FIELDS}


def sector_close_frame(columns, sector_of, etf_close):
    """Close of each stock's sector ETF, laid out in the same columns as the stocks."""
    data = {}
    for t in columns:
        etf = SECTOR_ETF_BY_GICS.get(sector_of.get(t))
        if etf in etf_close.columns:
            data[t] = etf_close[etf]
    return pd.DataFrame(data, index=etf_close.index).reindex(columns=columns)


# ── Indicators (vectorized across a dates x tickers frame) ────────────────────
def live_rsi(close, freq, n=RSI_LEN):
    """Wilder RSI on `freq` bars ("W-FRI" / "M"), evaluated every day with the
    current bar closing at today's price. Returns (daily live RSI, RSI of each
    bar, the bar key of every day)."""
    key = close.index.to_period(freq)
    bars = close.groupby(key).last()
    d = bars.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    bar_rsi = 100 - 100 / (1 + ag / al)

    def prev(x):  # state as of the last COMPLETED bar, laid out on the daily index
        return x.shift(1).loc[key].set_axis(close.index)

    chg = close - prev(bars)
    ag_live = (prev(ag) * (n - 1) + chg.clip(lower=0)) / n
    al_live = (prev(al) * (n - 1) + (-chg).clip(lower=0)) / n
    return 100 - 100 / (1 + ag_live / al_live), bar_rsi, key


def own_quantile(bar_rsi, key, index, q, min_bars):
    """q-quantile of every COMPLETED bar's RSI up to (not including) the
    current bar, laid out on the daily index."""
    thr = bar_rsi.expanding(min_periods=min_bars).quantile(q).shift(1)
    return thr.loc[key].set_axis(index)


def own_percentile(bar_rsi, key, ticker, value):
    """Where today's live RSI sits in the name's completed-bar history (0-1)."""
    hist = bar_rsi.loc[bar_rsi.index < key[-1], ticker].dropna()
    return float((hist <= value).mean()) if len(hist) and pd.notna(value) else np.nan


def gk_vol(o, h, l, c, n=20, ppy=252):
    """Annualized Garman-Klass volatility over `n` bars (`ppy` bars per year)."""
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(gk.rolling(n, min_periods=int(n * 0.75)).mean().clip(lower=0) * ppy)


def vol_percentile(level):
    """Share of trading days since 1990 that VIX closed below `level` - a
    stock's own "VIX 30" is its volatility at that same percentile."""
    from generate_vix_structure_report import fetch_cboe_index
    try:
        vix = fetch_cboe_index("VIX").dropna()
        return float((vix < level).mean())
    except Exception as exc:
        print(f"  CBOE VIX fetch failed ({exc}); using approximate percentile")
        return VIX_PCT_FALLBACK.get(level, 0.93)


# ── Signal ────────────────────────────────────────────────────────────────────
def compute_signal(panel, bench_close, sector_close=None, cfg=None, driver_tail=None):
    """Inputs for the four conditions (dates x tickers frames). Threshold-
    dependent pieces live in conditions() so the study can grid-search them.
    driver_tail: compute the market/sector/specific split on only the last N
    rows (the daily report needs today's value, not 26 years of it)."""
    cfg = cfg or load_config()
    o, h, l, c = (panel[f] for f in ("Open", "High", "Low", "Close"))
    bench = bench_close.reindex(c.index).ffill()

    w_rsi, w_bars, w_key = live_rsi(c, "W-FRI")
    m_rsi, m_bars, m_key = live_rsi(c, "M")

    ppy = cfg.get("periods_per_year", 252)
    lag, span = cfg["lead_lag"], int(cfg["lead_years"] * ppy)
    rel5 = (c.shift(lag) / c.shift(lag + span) - 1).sub(bench.shift(lag) / bench.shift(lag + span) - 1, axis=0)
    leader = (rel5 > 0) & (rel5.rank(axis=1, pct=True) >= 1 - cfg["lead_top_pct"])

    return {
        "close": c, "rel5": rel5, "leader": leader,
        "w_rsi": w_rsi, "w_bars": w_bars, "w_key": w_key,
        "m_rsi": m_rsi, "m_bars": m_bars, "m_key": m_key,
        "rv": gk_vol(o, h, l, c, ppy=ppy),
        "sma50": c.rolling(50, min_periods=40).mean(),
        "sma200": c.rolling(200, min_periods=160).mean(),
        "drawdown": 1 - c / c.rolling(252, min_periods=126).max(),
        **driver_frames(c.iloc[-driver_tail:] if driver_tail else c, bench, sector_close),
        "has_sector": sector_close is not None,
    }


def conditions(s, cfg, vol_pct):
    """The four boolean conditions + their thresholds, and the buy signal."""
    idx = s["close"].index
    w_thr = own_quantile(s["w_bars"], s["w_key"], idx, cfg["w_rsi_pct"], W_MIN_BARS)
    m_thr = own_quantile(s["m_bars"], s["m_key"], idx, cfg["m_rsi_pct"], M_MIN_BARS)
    v_thr = s["rv"].expanding(min_periods=int(VOL_MIN_YEARS * cfg.get("periods_per_year", 252))).quantile(vol_pct)
    spike_day = s["rv"] >= v_thr
    cond = {
        "leader": s["leader"],
        "weekly": s["w_rsi"] <= w_thr,
        "monthly": s["m_rsi"] <= m_thr,
        "vol_spike": spike_day.astype(float).rolling(cfg["vol_lookback"], min_periods=1).max() > 0,
        "spike_day": spike_day, "w_thr": w_thr, "m_thr": m_thr, "v_thr": v_thr,
    }
    cond["buy"] = cond["leader"] & cond["weekly"] & cond["monthly"] & cond["vol_spike"]
    return cond


def extract_events(signal, cooldown):
    """One row per (date, ticker) hit, keeping at most one event per ticker per
    `cooldown` trading days so a long selloff isn't counted a dozen times."""
    arr = signal.fillna(False).to_numpy(bool)
    rows = []
    for j, ticker in enumerate(signal.columns):
        last = -(10 ** 9)
        for i in np.flatnonzero(arr[:, j]):
            if i - last >= cooldown:
                rows.append((signal.index[i], ticker, i))
                last = i
    return pd.DataFrame(rows, columns=["date", "ticker", "i"])


# ── What drove the move (diagnostic) ──────────────────────────────────────────
def driver_frames(close, bench, sector_close=None, window=21, est=252):
    """Split every asset's last `window`-day log return into market / sector /
    specific pieces (dates x tickers frames). Betas come from the `est` days
    BEFORE the window, so the move being explained never estimates its own
    betas. The sector ETF is orthogonalized to SPY first: 'market' is the plain
    market beta x SPY's move, 'sector' only the sector's move beyond what SPY
    explains. ETFs (no sector) split into market / specific."""
    lr = np.log(close).diff()
    m = np.log(bench.reindex(close.index)).diff()

    def roll(x):
        return x.rolling(est, min_periods=int(est * 0.8))

    var_m = roll(m).var()
    cov_ym = roll(lr).cov(m)
    beta_m = cov_ym.div(var_m, axis=0).shift(window)
    total = lr.rolling(window, min_periods=window).sum()
    m_move = m.rolling(window, min_periods=window).sum()
    market = beta_m.mul(m_move, axis=0)

    if sector_close is None:
        sector = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    else:
        s = np.log(sector_close.reindex(index=close.index, columns=close.columns)).diff()
        cov_sm = roll(s).cov(m)
        g = cov_sm.div(var_m, axis=0)
        a_s = roll(s).mean() - g.mul(roll(m).mean(), axis=0)
        var_es = roll(s).var() - cov_sm.pow(2).div(var_m, axis=0)
        b_s = (roll(lr).cov(s) - g * cov_ym) / var_es
        g, a_s, b_s = g.shift(window), a_s.shift(window), b_s.shift(window)
        es_move = s.rolling(window, min_periods=window).sum() - window * a_s - g.mul(m_move, axis=0)
        sector = b_s * es_move
    specific = total - market - sector
    market_driven = (total < 0) & (market <= sector) & (market <= specific)
    return {"move21": total, "mkt_part": market, "sec_part": sector, "spec_part": specific,
            "market_driven": market_driven}


def driver_labels(mkt, sec, spec, total, has_sector=True):
    """Label each move by its dominant piece: the most negative piece for a
    drop, the most positive for a rise. Inputs are equal-length 1-D arrays."""
    names = np.array(["Market", "Sector", "Specific"] if has_sector else ["Market", "Specific"])
    parts = np.column_stack([mkt, sec, spec] if has_sector else [mkt, spec]).astype(float)
    total = np.asarray(total, dtype=float)
    ok = np.isfinite(parts).all(axis=1) & np.isfinite(total)
    filled = np.where(ok[:, None], parts, 0.0)
    idx = np.where(total < 0, filled.argmin(axis=1), filled.argmax(axis=1))
    return np.where(ok, names[idx], "n/a")


# ── Market fear gauge ─────────────────────────────────────────────────────────
def fetch_credit_stress(start):
    """High-yield OAS from FRED if reachable with long history; otherwise a
    HYG/IEF drawdown (credit underperforming Treasuries) from yfinance."""
    try:
        r = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0])
        s = s.dropna()
        if len(s) > 2500 and s.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=365 * 3):
            return s, "HY OAS (FRED)"
        print("  FRED HY OAS history too short, using HYG/IEF")
    except Exception as exc:
        print(f"  FRED unavailable ({type(exc).__name__}), using HYG/IEF")
    raw = yf.download(["HYG", "IEF"], start=start, auto_adjust=True, progress=False)["Close"]
    ratio = (raw["HYG"] / raw["IEF"]).dropna()
    return 1 - ratio / ratio.rolling(252, min_periods=63).max(), "HYG/IEF drawdown"


def market_gauge(member_close, start):
    """0-100 market fear gauge on the trading dates of `member_close` (the
    S&P 500 panel) from VIX level, VIX/VIX3M and credit stress, each a
    percentile of its own 5-year history. Regime: Panic = VIX term structure
    inverted (VIX > VIX3M), Elevated = gauge >= 70, else Calm. Breadth is
    returned separately in .attrs["breadth"] as trend context."""
    from generate_vix_structure_report import fetch_cboe_index

    idx = member_close.index
    comps = {}
    try:
        vix = fetch_cboe_index("VIX").reindex(idx).ffill()
        comps["VIX level"] = vix
        vix3m = fetch_cboe_index("VIX3M").reindex(idx).ffill(limit=5)
        comps["VIX / VIX3M"] = vix / vix3m
    except Exception as exc:
        print(f"  CBOE fetch failed: {exc}")

    credit, credit_label = fetch_credit_stress(start)
    comps[f"Credit ({credit_label})"] = credit.reindex(idx).ffill(limit=5)

    live = member_close.notna()
    n = live.sum(axis=1).where(lambda x: x >= 50)
    sma50 = member_close.rolling(50, min_periods=40).mean()
    sma200 = member_close.rolling(200, min_periods=160).mean()
    at_low = member_close <= member_close.rolling(252, min_periods=200).min()
    breadth = pd.DataFrame({
        "% below 50-day": (member_close < sma50).sum(axis=1) / n,
        "% below 200-day": (member_close < sma200).sum(axis=1) / n,
        "% at 52-wk lows": (at_low.sum(axis=1) / n).rolling(5, min_periods=1).mean(),
    })

    raw = pd.DataFrame(comps)
    pct = raw.rolling(MKT_PCT_WINDOW, min_periods=MKT_PCT_MIN).rank(pct=True)
    out = pct.mul(100)
    out["gauge"] = pct.mean(axis=1) * 100
    ratio = raw["VIX / VIX3M"] if "VIX / VIX3M" in raw else pd.Series(np.nan, index=idx)
    out["vix_ratio"] = ratio
    out["inverted"] = ratio > 1
    out["regime"] = np.where(out["inverted"], "Panic", np.where(out["gauge"] >= 70, "Elevated", "Calm"))
    out.loc[out["gauge"].isna() & ~out["inverted"], "regime"] = "n/a"
    credit_col = [c for c in pct.columns if c.startswith("Credit")][0]
    out["credit_stress"] = pct[credit_col] >= 0.90
    out.attrs["raw"] = raw
    out.attrs["breadth"] = breadth
    return out
