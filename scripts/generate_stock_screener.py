"""
Daily Stock Screener — Sharpe leaderboard, sector snapshot, and a risk/return
scatter with an efficient frontier, adapted from the "Stock Screener" research
notebook (S&P 500 / NASDAQ-100 / TSX, Sharpe ratio, min-variance frontier).

Differences from the source notebook, deliberate for CI reliability/runtime:
  - Universes are S&P 500 + NASDAQ-100 + TSX (this repo's existing
    data/tsx_universe.csv, already proven in generate_benchmark_beaters.py /
    generate_relative_performance_charts.py) instead of also including
    Russell 2000 — ~2,000 extra tickers isn't a reliable size for a scheduled
    job on ubuntu-latest.
  - One combined download (deduped across the three lists) instead of three
    separate ones, since S&P 500 and NASDAQ-100 overlap heavily.
  - 10 years of daily history instead of 25 — still long enough to see a
    full-cycle Sharpe/vol/frontier, well under the download+compute cost.
  - Per-ticker detail chart from the notebook is dropped (no natural "which
    ticker" choice for an unattended report); everything else is kept.
  - The risk/return + efficient-frontier chart is built four times instead
    of once per index: S&P 500 alone, NASDAQ-100 alone, TSX alone, and a
    consolidated S&P 500 + NASDAQ-100 chart (deduped where the two overlap,
    e.g. AAPL/MSFT/NVDA sit in both).

── Manually adding your own tickers ──────────────────────────────────────────
Two ways in, same effect either way — the ticker gets fetched even if it
isn't in any index, is forced into every chart's efficient-frontier blend
alongside the top-Sharpe basket, and is always rendered with its own labeled
gold-diamond marker on all four scatter charts:

  1. Edit the WATCHLIST constant a few lines below (e.g. WATCHLIST =
     ["RY.TO", "SHOP.TO"]) and re-run the script locally.
  2. On GitHub: Actions tab -> "Stock Screener (Sharpe / Efficient Frontier)"
     -> "Run workflow" -> type a comma-separated list into the
     "extra_tickers" field. That's the workflow_dispatch input wired to the
     EXTRA_TICKERS environment variable this script reads at runtime — no
     code edit needed, and it doesn't affect the regular scheduled run.
"""

import base64
import json
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.optimize import minimize

from common_screening import load_nasdaq100_symbols, load_sp500_symbols, load_tsx_symbols

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "stock-screener")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")

RISK_FREE = 0.045   # annual risk-free rate, used in Sharpe
DATA_YEARS = 10       # years of daily history to pull
TOP_N = 25            # top-Sharpe stocks labeled on the scatter / used for the frontier basket
LEADERBOARD_N = 50   # rows in each Sharpe leaderboard table
MIN_FRONTIER_HISTORY = 252  # trading days required to be eligible for "top Sharpe" / frontier ranking

# ── Manually enter tickers to add to every efficient-frontier chart here ─────
WATCHLIST: list = []  # e.g. ["RY.TO", "SHOP.TO", "ASML"] — local-run convenience;
                       # for a one-off run without editing this file, use the
                       # EXTRA_TICKERS env var / the workflow's "extra_tickers" input instead.

# ── Corporate colours (matches Industry RSI / other reports in this repo) ────
ORANGE = "#C67A29"
BLUE = "#1F79BE"
DGREY = "#363636"
LGREY = "#4A4A4A"
GREEN = "#44A660"
RED = "#A22A2A"
GOLD = "#E8B84B"
TEXTCLR = "#E8E8E8"
BG = "#1C1C1E"
GRID = "#3A3A3C"

with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLY": "Cons. Disc.", "XLP": "Cons. Staples", "XLE": "Energy",
    "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate",
    "XLU": "Utilities", "XLC": "Comm. Services",
}


def get_extra_tickers() -> list:
    env_val = os.environ.get("EXTRA_TICKERS", "")
    env_tickers = [t.strip().upper() for t in env_val.replace(";", ",").split(",") if t.strip()]
    combined = list(dict.fromkeys(WATCHLIST + env_tickers))  # de-duped, order preserved
    return combined


# ── Universe loading ──────────────────────────────────────────────────────────
def load_universes(extra_tickers: list) -> dict:
    universes = {}
    try:
        universes["S&P 500"] = [t.replace(".", "-") for t in load_sp500_symbols()]
    except Exception as e:
        print(f"S&P 500 fetch failed: {e}")
        universes["S&P 500"] = []
    try:
        universes["NASDAQ-100"] = [t.replace(".", "-") for t in load_nasdaq100_symbols()]
    except Exception as e:
        print(f"NASDAQ-100 fetch failed: {e}")
        universes["NASDAQ-100"] = []
    try:
        universes["TSX"] = load_tsx_symbols()
    except Exception as e:
        print(f"TSX fetch failed: {e}")
        universes["TSX"] = []

    for name, tickers in universes.items():
        print(f"{name}: {len(tickers)} tickers")
    if extra_tickers:
        print(f"Custom/manual watchlist: {len(extra_tickers)} tickers — {', '.join(extra_tickers)}")
    return universes


# ── Chunked download ──────────────────────────────────────────────────────────
def download_prices(tickers: list, years: int, chunk_size: int = 150) -> pd.DataFrame:
    end = datetime.now().date()
    start = end - timedelta(days=int(years * 365.25))
    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(chunk, start=start, end=end, auto_adjust=True, threads=True, progress=False)
            close = raw["Close"] if "Close" in raw.columns else raw
            if isinstance(close, pd.Series):
                close = close.to_frame(chunk[0])
            frames.append(close)
            print(f"  chunk {i // chunk_size + 1}/{(len(tickers) - 1) // chunk_size + 1} done")
        except Exception as e:
            print(f"  chunk {i // chunk_size + 1} error: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    return df.loc[:, ~df.columns.duplicated()]


# ── Signal computation ────────────────────────────────────────────────────────
def compute_signals(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    prices = prices.dropna(axis=1, how="all").copy()
    today = prices.index[-1]
    ytd_idx = prices.index[prices.index >= f"{today.year}-01-01"]
    ytd_start_price = prices.loc[ytd_idx[0]] if len(ytd_idx) else None

    def _rsi(p, n):
        d = p.diff(1)
        g = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
        return (100 - 100 / (1 + g / l.replace(0, float("nan")))).iloc[-1]

    rows = []
    for tkr in prices.columns:
        p = prices[tkr].dropna()
        if len(p) < 60:
            continue
        try:
            ret = p.pct_change()
            last = p.iloc[-1]

            def _ret(n):
                return (last / p.iloc[-n] - 1) * 100 if len(p) >= n else float("nan")

            ytd = (last / ytd_start_price[tkr] - 1) * 100 if ytd_start_price is not None and tkr in ytd_start_price else float("nan")
            ma50 = p.rolling(50).mean().iloc[-1]
            ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else float("nan")
            hi52 = p.rolling(252).max().iloc[-1]
            lo52 = p.rolling(252).min().iloc[-1]
            ann_ret = ret.mean() * 252 * 100
            ann_vol = ret.std() * (252 ** 0.5) * 100
            sharpe = (ann_ret / 100 - RISK_FREE) / (ann_vol / 100) if ann_vol else float("nan")

            rows.append({
                "Ticker": tkr,
                "1W%": round(_ret(5), 1), "1M%": round(_ret(21), 1), "3M%": round(_ret(63), 1),
                "6M%": round(_ret(126), 1), "1Y%": round(_ret(252), 1), "YTD%": round(ytd, 1),
                "RSI14": round(_rsi(p, 14), 1),
                "vMA50%": round((last / ma50 - 1) * 100, 1),
                "vMA200%": round((last / ma200 - 1) * 100, 1) if ma200 == ma200 else float("nan"),
                "52wHi%": round((last / hi52 - 1) * 100, 1), "52wLo%": round((last / lo52 - 1) * 100, 1),
                "AnnRet%": round(ann_ret, 1), "Vol%": round(ann_vol, 1), "Sharpe": round(sharpe, 2),
                "Price": round(last, 2), "NObs": len(p),
            })
        except Exception:
            pass

    return pd.DataFrame(rows).set_index("Ticker")


# ── Efficient frontier ────────────────────────────────────────────────────────
def _largest_overlap_basket(returns: pd.DataFrame, tickers: list, min_rows: int) -> list:
    """Real small/thin-history tickers can each individually have plenty of history but no single
    date where ALL of them traded (a halt or listing gap in just one name blanks out the whole row).
    Greedily drop whichever ticker's removal grows the row-wise overlap the most, until enough
    common rows remain or too few tickers are left."""
    tickers = list(dict.fromkeys(tickers))
    dropped = []
    while len(tickers) >= 2:
        if len(returns[tickers].dropna()) >= min_rows:
            break
        best_ticker, best_len = None, -1
        for t in tickers:
            trial = [x for x in tickers if x != t]
            trial_len = len(returns[trial].dropna())
            if trial_len > best_len:
                best_len, best_ticker = trial_len, t
        tickers.remove(best_ticker)
        dropped.append(best_ticker)
    if dropped:
        print(f"    (dropped from frontier basket to restore overlapping history: {', '.join(dropped)})")
    return tickers


def efficient_frontier(returns: pd.DataFrame, tickers: list, n_points: int = 40, min_rows: int = 60):
    tickers = _largest_overlap_basket(returns, tickers, min_rows)
    if len(tickers) < 2:
        return None, None
    ret = returns[tickers].dropna()
    if len(ret) < min_rows:
        return None, None
    mu = ret.mean().values * 252
    cov = ret.cov().values * 252
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n

    ef_vol, ef_ret = [], []
    for target in np.linspace(mu.min(), mu.max(), n_points):
        res = minimize(
            lambda w: float(w @ cov @ w), w0, method="SLSQP", bounds=bounds,
            constraints=[
                {"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
            ],
            options={"ftol": 1e-9, "maxiter": 600},
        )
        if res.success:
            ef_vol.append(np.sqrt(max(res.fun, 0)) * 100)
            ef_ret.append(target * 100)
    return ef_vol, ef_ret


# ── Plot builders ──────────────────────────────────────────────────────────────
_RET_COLS = {"1W%", "1M%", "3M%", "6M%", "1Y%", "YTD%", "vMA50%", "vMA200%", "52wHi%", "52wLo%"}


def _bg(val, col):
    if col in _RET_COLS:
        if val != val:
            return "#2A2A2C"
        return "#1a4731" if val > 0 else "#4a1a1a"
    if col == "RSI14":
        if val != val:
            return "#2A2A2C"
        if val > 70:
            return "#4a1a1a"
        if val < 30:
            return "#1a4731"
    return "#2A2A2C"


def _fg(val, col):
    if col in _RET_COLS:
        if val != val:
            return "#8E8E93"
        return GREEN if val > 0 else RED
    return TEXTCLR


def make_leaderboard_table(df: pd.DataFrame, title: str, top_n: int) -> go.Figure:
    cols = ["1M%", "3M%", "1Y%", "Sharpe", "Vol%", "RSI14", "vMA200%", "52wHi%"]
    d = df.sort_values("Sharpe", ascending=False).head(top_n)[cols].reset_index()
    disp_cols = d.columns.tolist()
    fig = go.Figure(go.Table(
        columnwidth=[80] + [65] * (len(disp_cols) - 1),
        header=dict(values=[f"<b>{c}</b>" for c in disp_cols], fill_color="#12171e",
                    font=dict(color="white", size=11), align="center", height=28),
        cells=dict(
            values=[d[c] for c in disp_cols],
            fill_color=[[_bg(v, c) for v in d[c]] for c in disp_cols],
            font=dict(color=[[_fg(v, c) for v in d[c]] for c in disp_cols], size=10),
            align=["left"] + ["right"] * (len(disp_cols) - 1), height=22,
        ),
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(color=TEXTCLR, size=14)),
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=40, b=0), height=min(1000, 70 + 22 * len(d)),
    )
    return fig


def make_scatter_frontier(
    signals: pd.DataFrame, returns: pd.DataFrame, universe_tickers: list, extra_tickers: list, title: str,
) -> go.Figure:
    """Risk/return scatter + efficient frontier for one universe (a subset of `signals`' index).
    `extra_tickers` (the manual/custom watchlist) are overlaid regardless of universe membership —
    that's the mechanism for "add this ticker to the frontier" even if it's not an index member."""
    members = [t for t in universe_tickers if t in signals.index]
    d = signals.loc[members].dropna(subset=["Vol%", "AnnRet%", "Sharpe"])

    # The frontier basket needs every member to share a long overlapping return history, or the
    # row-wise dropna() in efficient_frontier() collapses to ~0 rows and the whole frontier vanishes.
    # A recently-listed ticker can post an inflated Sharpe on a short hot run and would otherwise
    # dominate the "top by Sharpe" selection and poison the basket for everyone else in it — so
    # ranking/selection for the frontier is restricted to sufficiently long-history names, while the
    # scatter's grey background still shows every screened stock in the universe, thin-history ones included.
    eligible = d[d["NObs"] >= MIN_FRONTIER_HISTORY]
    top_tickers = eligible.nlargest(TOP_N, "Sharpe").index.tolist()
    custom_in_data = [t for t in extra_tickers if t in signals.index]
    custom_for_frontier = [t for t in custom_in_data if signals.loc[t, "NObs"] >= MIN_FRONTIER_HISTORY]
    skipped_custom = [t for t in custom_in_data if t not in custom_for_frontier]
    if skipped_custom:
        print(f"  [{title}] Custom tickers shown but excluded from the frontier blend (too little history, <{MIN_FRONTIER_HISTORY}d): {', '.join(skipped_custom)}")
    top_tickers_plot = [t for t in top_tickers if t not in custom_in_data]

    d_all = d.reset_index()
    d_top = d.loc[top_tickers_plot].reset_index()
    d_custom = signals.loc[custom_in_data].reset_index() if custom_in_data else pd.DataFrame()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=d_all["Vol%"], y=d_all["AnnRet%"], mode="markers",
        marker=dict(color="#5a6272", size=5, opacity=0.55),
        hovertemplate="<b>%{customdata[0]}</b><br>Vol: %{x:.1f}%<br>Ann Ret: %{y:.1f}%<br>Sharpe: %{customdata[1]:.2f}<extra></extra>",
        customdata=d_all[["Ticker", "Sharpe"]].values, name=f"{title} stocks", showlegend=True,
    ))

    if not d_top.empty:
        fig.add_trace(go.Scatter(
            x=d_top["Vol%"], y=d_top["AnnRet%"], mode="markers+text",
            marker=dict(color=d_top["Sharpe"], colorscale="RdYlGn",
                        cmin=d_all["Sharpe"].quantile(0.1), cmax=d_all["Sharpe"].quantile(0.9),
                        size=11, line=dict(color="white", width=0.8),
                        colorbar=dict(title=dict(text="Sharpe", font=dict(color=TEXTCLR, size=10)),
                                      thickness=12, tickfont=dict(color=TEXTCLR, size=9))),
            text=d_top["Ticker"], textposition="top center", textfont=dict(size=9, color="#dddddd"),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Ann Ret: %{y:.1f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
            name=f"Top {TOP_N} by Sharpe",
        ))

    if not d_custom.empty:
        fig.add_trace(go.Scatter(
            x=d_custom["Vol%"], y=d_custom["AnnRet%"], mode="markers+text",
            marker=dict(color=GOLD, size=15, symbol="diamond", line=dict(color="white", width=1.2)),
            text=d_custom["Ticker"], textposition="bottom center", textfont=dict(size=10, color=GOLD),
            hovertemplate="<b>%{text}</b> (custom)<br>Vol: %{x:.1f}%<br>Ann Ret: %{y:.1f}%<br>Sharpe: %{customdata:.2f}<extra></extra>",
            customdata=d_custom["Sharpe"], name="Custom / manual watchlist",
        ))

    frontier_basket = list(dict.fromkeys(top_tickers + custom_for_frontier))
    frontier_basket = [t for t in frontier_basket if t in returns.columns]
    if len(frontier_basket) >= 2:
        ef_vol, ef_ret = efficient_frontier(returns, frontier_basket)
        if not ef_vol:
            print(f"  Efficient frontier: optimization returned no points for {len(frontier_basket)} tickers (check for overlapping-history gaps).")
        if ef_vol:
            label = f"Efficient frontier (top {TOP_N}" + (" + custom)" if custom_for_frontier else ")")
            fig.add_trace(go.Scatter(
                x=ef_vol, y=ef_ret, mode="lines", line=dict(color=ORANGE, width=2.5, dash="dot"),
                name=label, hovertemplate="Frontier<br>Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
            ))

    fig.add_hline(y=0, line_dash="dash", line_color="#666", opacity=0.5)
    fig.add_hline(y=RISK_FREE * 100, line_dash="dot", line_color=BLUE, opacity=0.7,
                  annotation_text=f"Risk-free rate ({RISK_FREE * 100:.1f}%)",
                  annotation_font_color=BLUE, annotation_font_size=10, annotation_position="right")

    fig.update_layout(
        title=dict(text=f"<b>{title} — Risk vs. Return + Efficient Frontier</b>",
                    font=dict(color=TEXTCLR, size=15)),
        xaxis_title="Annualized Volatility (%)", yaxis_title=f"Annualized Return % ({DATA_YEARS}yr daily avg)",
        paper_bgcolor=BG, plot_bgcolor=LGREY, font=dict(color=TEXTCLR), height=680,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), x=0.01, y=0.99),
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def make_movers_chart(signals: pd.DataFrame, n: int = TOP_N) -> go.Figure:
    d = signals.dropna(subset=["1M%"])
    top, bot = d["1M%"].nlargest(n), d["1M%"].nsmallest(n)
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"<b>Top {n} Gainers</b> (1M%)", f"<b>Bottom {n} Laggards</b> (1M%)"])
    fig.add_trace(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=GREEN,
                          text=[f"{v:+.1f}%" for v in top.values], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=bot.values, y=bot.index, orientation="h", marker_color=RED,
                          text=[f"{v:+.1f}%" for v in bot.values], textposition="outside"), row=1, col=2)
    fig.update_layout(title="<b>1-Month Return Leaders & Laggards — All Screened Stocks</b>",
                       paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXTCLR), showlegend=False,
                       height=max(420, 22 * n + 90))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, tickfont=dict(size=9))
    return fig


def make_sector_snapshot() -> go.Figure:
    end, start = datetime.now().date(), datetime.now().date() - timedelta(days=420)
    sec_px = yf.download(list(SECTOR_ETFS), start=start, end=end, auto_adjust=True, progress=False)["Close"]

    def _rsi14(p):
        d = p.diff(1)
        g = d.clip(lower=0).ewm(com=13, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        return (100 - 100 / (1 + g / l.replace(0, float("nan")))).iloc[-1]

    rows = []
    for tkr, name in SECTOR_ETFS.items():
        if tkr not in sec_px.columns:
            continue
        p = sec_px[tkr].dropna()
        ytd_idx = p.index[p.index >= f"{end.year}-01-01"]
        ytd_start = p.loc[ytd_idx[0]] if len(ytd_idx) else p.iloc[0]
        rows.append({
            "Sector": name, "ETF": tkr,
            "1W%": round((p.iloc[-1] / p.iloc[-5] - 1) * 100, 1),
            "1M%": round((p.iloc[-1] / p.iloc[-21] - 1) * 100, 1),
            "3M%": round((p.iloc[-1] / p.iloc[-63] - 1) * 100, 1),
            "YTD%": round((p.iloc[-1] / ytd_start - 1) * 100, 1),
            "RSI14": round(_rsi14(p), 1),
            "vMA200%": round((p.iloc[-1] / p.rolling(200).mean().iloc[-1] - 1) * 100, 1),
            "52wHi%": round((p.iloc[-1] / p.rolling(252).max().iloc[-1] - 1) * 100, 1),
        })
    sec_df = pd.DataFrame(rows).sort_values("3M%", ascending=False).reset_index(drop=True)
    metric_cols = ["1W%", "1M%", "3M%", "YTD%", "RSI14", "vMA200%", "52wHi%"]
    all_cols = ["Sector", "ETF"] + metric_cols

    fig = go.Figure(go.Table(
        columnwidth=[130, 45] + [62] * len(metric_cols),
        header=dict(values=[f"<b>{c}</b>" for c in all_cols], fill_color="#12171e",
                    font=dict(color="white", size=11), align="center", height=30),
        cells=dict(
            values=[sec_df[c] for c in all_cols],
            fill_color=[["#2A2A2C"] * len(sec_df) if c in ("Sector", "ETF") else [_bg(v, "1W%") for v in sec_df[c]] for c in all_cols],
            font=dict(color=[["white"] * len(sec_df) if c in ("Sector", "ETF") else [_fg(v, "1W%") for v in sec_df[c]] for c in all_cols], size=11),
            align=["left", "center"] + ["right"] * len(metric_cols), height=26,
        ),
    ))
    fig.update_layout(title="<b>S&P 500 Sector Snapshot</b>  <span style='font-size:11px;color:#888'>Sorted by 3M return</span>",
                       paper_bgcolor=BG, margin=dict(l=0, r=0, t=50, b=0), height=420)
    return fig


SCREENS = {
    "Oversold Bounce (RSI14<30, 1M>0)": lambda d: d[(d["RSI14"] < 30) & (d["1M%"] > 0)],
    "Overbought (RSI14>70)": lambda d: d[d["RSI14"] > 70],
    "High Momentum (3M top quartile, Sharpe>1)": lambda d: d[(d["3M%"] >= d["3M%"].quantile(0.75)) & (d["Sharpe"] > 1)],
    "Near 52w High (within 5%)": lambda d: d[d["52wHi%"] >= -5],
    "Near 52w Low (within 10%)": lambda d: d[d["52wLo%"] <= 15],
    "Healthy Uptrend (above MA200, RSI14 40-60)": lambda d: d[(d["vMA200%"] > 0) & d["RSI14"].between(40, 60)],
    "Low Vol High Sharpe (Vol<20%, Sharpe>1.5)": lambda d: d[(d["Vol%"] < 20) & (d["Sharpe"] > 1.5)],
}


def screens_html(signals: pd.DataFrame) -> str:
    rows = []
    for name, fn in SCREENS.items():
        try:
            res = fn(signals)
            tickers = ", ".join(res.index[:25].tolist()) + ("..." if len(res) > 25 else "") if not res.empty else "no matches"
            rows.append(f"<tr><td>{name}</td><td>{len(res)}</td><td>{tickers}</td></tr>")
        except Exception as e:
            rows.append(f"<tr><td>{name}</td><td>error</td><td>{e}</td></tr>")
    return (
        '<table class="screens">'
        "<thead><tr><th>Screen</th><th>Matches</th><th>Tickers</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def fig_to_div(fig: go.Figure, div_id: str = None) -> str:
    kwargs = {"div_id": div_id} if div_id else {}
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True}, **kwargs)


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


FRONTIER_DIV_IDS = ["frontier-sp500", "frontier-nasdaq100", "frontier-tsx", "frontier-consolidated"]


def search_box_html(signals: pd.DataFrame) -> str:
    """Client-side, no-rerun ticker highlighter — same pattern as Channel_Lookup.html: every
    already-screened ticker's Vol%/AnnRet%/Sharpe is embedded as JSON at build time, so typing one
    in just looks it up and adds a marker in the browser. A ticker outside this run's universe
    (S&P 500 / NASDAQ-100 / TSX) isn't in this JSON and can't be added this way — that still needs
    the workflow's `extra_tickers` input (see the footer), which actually fetches new data."""
    payload = {
        tkr: [row["Vol%"], row["AnnRet%"], round(row["Sharpe"], 2)]
        for tkr, row in signals[["Vol%", "AnnRet%", "Sharpe"]].dropna().iterrows()
    }
    data_json = json.dumps(payload, separators=(",", ":"))

    return f"""
<div class="search-box">
  <label for="tickerSearch">Highlight a ticker on all 4 charts below (instant, no rerun — searches the {len(payload):,} already-screened tickers above):</label>
  <div class="search-row">
    <input type="text" id="tickerSearch" placeholder="e.g. AAPL, RY.TO, SHOP.TO" autocomplete="off">
    <button onclick="addTickers()">Highlight</button>
    <button onclick="clearTickers()" class="secondary">Clear highlights</button>
  </div>
  <div id="searchStatus" class="search-status"></div>
</div>
<script>
const ALL_SIGNALS = {data_json};
const FRONTIER_DIVS = {json.dumps(FRONTIER_DIV_IDS)};
let baseTraceCounts = null;
let addedCount = {{}};

function ensureBaseCounts() {{
  if (baseTraceCounts) return;
  baseTraceCounts = {{}};
  FRONTIER_DIVS.forEach(function (id) {{
    const el = document.getElementById(id);
    baseTraceCounts[id] = el ? el.data.length : 0;
    addedCount[id] = 0;
  }});
}}

function addTickers() {{
  ensureBaseCounts();
  const raw = document.getElementById('tickerSearch').value;
  const tickers = raw.split(/[,\\s]+/).map(function (t) {{ return t.trim().toUpperCase(); }}).filter(Boolean);
  const found = [], missing = [];
  tickers.forEach(function (t) {{
    if (!(t in ALL_SIGNALS)) {{ missing.push(t); return; }}
    found.push(t);
    const vol = ALL_SIGNALS[t][0], ret = ALL_SIGNALS[t][1], sharpe = ALL_SIGNALS[t][2];
    const trace = {{
      x: [vol], y: [ret], mode: 'markers+text', type: 'scatter',
      marker: {{ color: '{GOLD}', size: 15, symbol: 'diamond', line: {{ color: 'white', width: 1.2 }} }},
      text: [t], textposition: 'bottom center', textfont: {{ size: 10, color: '{GOLD}' }},
      hovertemplate: '<b>' + t + '</b> (searched)<br>Vol: ' + vol.toFixed(1) + '%<br>Ann Ret: ' + ret.toFixed(1) + '%<br>Sharpe: ' + sharpe.toFixed(2) + '<extra></extra>',
      name: 'Searched: ' + t, showlegend: true,
    }};
    FRONTIER_DIVS.forEach(function (id) {{
      const el = document.getElementById(id);
      if (el) {{ Plotly.addTraces(id, trace); addedCount[id]++; }}
    }});
  }});
  const parts = [];
  if (found.length) parts.push('Added: ' + found.join(', '));
  if (missing.length) parts.push('Not in this run\\'s universe (use the workflow\\'s extra_tickers input instead): ' + missing.join(', '));
  document.getElementById('searchStatus').textContent = parts.join('  \\u2014  ');
  document.getElementById('tickerSearch').value = '';
}}

function clearTickers() {{
  ensureBaseCounts();
  FRONTIER_DIVS.forEach(function (id) {{
    const el = document.getElementById(id);
    if (!el || addedCount[id] === 0) return;
    const base = baseTraceCounts[id];
    const indices = [];
    for (let i = base; i < base + addedCount[id]; i++) indices.push(i);
    Plotly.deleteTraces(id, indices);
    addedCount[id] = 0;
  }});
  document.getElementById('searchStatus').textContent = 'Cleared.';
}}

document.addEventListener('DOMContentLoaded', ensureBaseCounts);
document.getElementById && document.addEventListener('keydown', function (e) {{
  if (e.key === 'Enter' && document.activeElement && document.activeElement.id === 'tickerSearch') addTickers();
}});
</script>
"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stock Screener — Sharpe &amp; Efficient Frontier</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; display: flex; justify-content: space-between; align-items: flex-start; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  header img {{ height: 48px; opacity: 0.9; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E8E8E8; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin-top: 6px; }}
  .leaderboards {{ display: flex; flex-wrap: wrap; gap: 16px; padding: 8px 24px; }}
  .leaderboards > div {{ flex: 1 1 380px; min-width: 340px; }}
  table.screens {{ width: calc(100% - 48px); margin: 8px 24px; border-collapse: collapse; font-size: 13px; }}
  table.screens th {{ background: #12171e; color: white; text-align: left; padding: 8px 10px; }}
  table.screens td {{ padding: 7px 10px; border-bottom: 1px solid #2A2A2C; vertical-align: top; }}
  table.screens tr:nth-child(even) {{ background: #232326; }}
  .search-box {{ margin: 8px 24px 20px; padding: 14px 16px; background: #232326; border: 1px solid #3A3A3C; border-radius: 6px; }}
  .search-box label {{ display: block; font-size: 13px; color: #C9C9CC; margin-bottom: 8px; }}
  .search-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .search-row input[type=text] {{ flex: 1 1 260px; background: #1C1C1E; color: #E8E8E8; border: 1px solid #3A3A3C;
    border-radius: 4px; padding: 8px 10px; font-size: 13px; }}
  .search-row button {{ background: #E8B84B; color: #1C1C1E; border: none; border-radius: 4px; padding: 8px 16px;
    font-size: 13px; font-weight: bold; cursor: pointer; }}
  .search-row button.secondary {{ background: #3A3A3C; color: #E8E8E8; font-weight: normal; }}
  .search-status {{ margin-top: 8px; font-size: 12px; color: #8E8E93; min-height: 16px; }}
  .frontier-chart {{ margin-bottom: 4px; }}
  footer {{ padding: 24px 32px; color: #8E8E93; font-size: 12px; border-top: 1px solid #3A3A3C; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Stock Screener — Sharpe Leaderboard &amp; Efficient Frontier</h1>
    <div class="meta">Generated {date_str} &middot; Universe: S&amp;P 500 + NASDAQ-100 + TSX ({n_universe} tickers) &middot; {years}yr daily history &middot; Risk-free rate {rf:.1f}%{custom_note}</div>
  </div>
  <img src="{logo_b64}" alt="logo">
</header>
{body}
<footer>
  Research tool, not investment advice. Sharpe uses a {rf:.1f}% annual risk-free rate; the efficient
  frontier is a long-only minimum-variance blend of the top {top_n} Sharpe stocks (plus any custom
  watchlist tickers), not a recommended portfolio. To add tickers to the frontier, re-run this
  workflow manually with a comma-separated ticker list in the "extra_tickers" input.
</footer>
</body>
</html>
"""


def build_report(extra_tickers: list):
    universes = load_universes(extra_tickers)
    all_tickers = list(dict.fromkeys(
        universes["S&P 500"] + universes["NASDAQ-100"] + universes["TSX"] + extra_tickers
    ))
    print(f"\nDownloading {len(all_tickers)} tickers, {DATA_YEARS}yr history...")
    prices = download_prices(all_tickers, DATA_YEARS)
    print(f"Downloaded {prices.shape[1]} tickers with data.\n")

    print("Computing signals...")
    signals = compute_signals(prices)
    returns = prices.pct_change(fill_method=None)
    print(f"{len(signals)} stocks with usable signals.\n")

    parts = []

    parts.append(section_header("Sharpe Leaderboards (Top 50 per index)"))
    leaderboard_html = ['<div class="leaderboards">']
    for uname in ("S&P 500", "NASDAQ-100", "TSX"):
        members = [t for t in universes[uname] if t in signals.index]
        if not members:
            continue
        sub = signals.loc[members]
        fig = make_leaderboard_table(sub, uname, LEADERBOARD_N)
        leaderboard_html.append(f"<div>{fig_to_div(fig)}</div>")
    leaderboard_html.append("</div>")
    parts.append("".join(leaderboard_html))

    print("Building sector snapshot...")
    try:
        parts.append(section_header("Sector Snapshot"))
        parts.append(fig_to_div(make_sector_snapshot()))
    except Exception as e:
        print(f"Sector snapshot error: {e}")

    print("Building risk/return scatter + efficient frontier charts...")
    parts.append(section_header(
        "Risk vs. Return + Efficient Frontier",
        f"One chart per universe, plus a consolidated S&amp;P 500 + NASDAQ-100 view &middot; "
        f"Grey = all screened stocks in that universe &middot; Colored+labeled = top {TOP_N} by Sharpe &middot; "
        "Gold diamonds = custom/manual watchlist (always included, any universe) &middot; Orange dashed = efficient frontier",
    ))
    parts.append(search_box_html(signals))
    sp500_nasdaq100 = list(dict.fromkeys(universes["S&P 500"] + universes["NASDAQ-100"]))
    frontier_charts = [
        ("S&P 500", universes["S&P 500"], "frontier-sp500"),
        ("NASDAQ-100", universes["NASDAQ-100"], "frontier-nasdaq100"),
        ("TSX", universes["TSX"], "frontier-tsx"),
        ("S&P 500 + NASDAQ-100 (consolidated)", sp500_nasdaq100, "frontier-consolidated"),
    ]
    for chart_title, chart_universe, div_id in frontier_charts:
        if not chart_universe:
            continue
        print(f"  {chart_title}...")
        fig = make_scatter_frontier(signals, returns, chart_universe, extra_tickers, chart_title)
        parts.append(f'<div class="frontier-chart">{fig_to_div(fig, div_id=div_id)}</div>')

    print("Building movers chart...")
    parts.append(section_header("1-Month Movers"))
    parts.append(fig_to_div(make_movers_chart(signals)))

    print("Building signal screens...")
    parts.append(section_header("Signal Screens", "Pre-defined screens run across the full combined universe"))
    parts.append(screens_html(signals))

    custom_note = f" &middot; Custom watchlist: {', '.join(extra_tickers)}" if extra_tickers else ""
    return "\n".join(parts), len(all_tickers), custom_note


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extra_tickers = get_extra_tickers()
    body, n_universe, custom_note = build_report(extra_tickers)

    html = PAGE_TEMPLATE.format(
        date_str=datetime.now().strftime("%B %d, %Y"), body=body, n_universe=n_universe,
        years=DATA_YEARS, rf=RISK_FREE * 100, top_n=TOP_N, custom_note=custom_note, logo_b64=LOGO_B64,
    )
    out_path = os.path.join(OUTPUT_DIR, "Stock_Screener.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
