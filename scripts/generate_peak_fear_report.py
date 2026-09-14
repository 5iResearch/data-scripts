"""
Daily Leader Selloff report (the signal in fear_engine.py).

Buy Signal = a historic market leader whose weekly AND monthly RSI are at
statistically low points vs its own history, within 20 trading days of a
stock-level volatility spike (its own equivalent of VIX jumping to 30+).

  1. Market regime   - VIX level, VIX/VIX3M, credit -> 0-100; Panic = VIX > VIX3M,
                       plus each component's own history and where it sits today
  2. Buy Signals     - all four conditions met in the last 5 trading days
  3. Radar           - fired in the last 30 trading days, with the conditions still on today
  4. Setting Up      - leaders meeting 2 of the 3 other conditions today
  5. Charts          - price, weekly/monthly RSI vs their own thresholds, vol vs its spike level
  6. Signal history  - pick any past date: the gauge that day and what fired

Universes: US stocks (the Uptrend Channel Screener list - S&P 500 +
Nasdaq-100 + data/koyfin_us.csv + data/us_1w_rev_est_screener.csv, ~2,000
names), BTC/ETH, and the sector/theme ETFs. Stocks are downloaded and scored
250 at a time so ~2,000 names x 26 years never sit in memory at once.

Thresholds and historical base rates come from data/fear_model_config.json,
written by `python scripts/fear_study.py --write-config`; built-in defaults
are used until that file exists.
"""

import base64
import gc
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fear_engine as fe

OUTPUT_DIR = os.path.join(fe.REPO_ROOT, "outputs", "peak-fear")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
LOGO_PATH = os.path.join(fe.REPO_ROOT, "assets", "Logo_Transparent_1200px.png")

SIGNAL_KEEP_DAYS = 5      # "fresh" buy signals
RADAR_DAYS = 30           # anything that fired in the last ~6 weeks stays on the radar
DOWNLOAD_CHUNK = 50       # tickers per yfinance request (26 years each)
STOCK_PIECE = 250         # stocks downloaded + scored this many at a time
KEEP_ROWS = 520           # daily rows kept per name after scoring (charts show 2 years)
DRIVER_TAIL = 400         # rows needed for today's market/sector/specific split (252d betas + 2 x 21d)
GAUGE_MA_DAYS = 100       # moving average drawn on the fear gauge chart
DEFAULT_ZOOM_DAYS = 1260  # gauge charts open on the last 5 years, zoom buttons for more
PICKER_DATES = 600        # how many signal days the time-machine dropdown lists
MAX_SETUP_ROWS = 40
MAX_NAME_CHARTS = 30
EARNINGS_LOOKBACK = 30
EARNINGS_LOOKAHEAD = 14
STOCK_LABEL = "US Stocks (S&P 500, Nasdaq-100 + screener list)"
CONDITIONS = [("Leader", "leader"), ("Weekly RSI", "weekly"), ("Monthly RSI", "monthly"), ("Vol spike", "vol_spike")]

DGRAY, MGRAY, LGRAY = "#1C1C1E", "#2C2C2E", "#3A3A3C"
TEXT, SUBTEXT = "#E5E5EA", "#8E8E93"
GREEN, RED, ORANGE, BLUE, YELLOW, PURPLE = "#2ECC71", "#E74C3C", "#C67A29", "#1F79BE", "#F4D03F", "#9B59B6"

with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()


def add_logo(fig, x=0.99, y=0.99, sizex=0.10, sizey=0.10, opacity=0.45):
    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=x, y=y, sizex=sizex, sizey=sizey,
        xanchor="right", yanchor="top", opacity=opacity, layer="above",
    ))


# ── Data ──────────────────────────────────────────────────────────────────────
def load_universes(cfg):
    st_tickers, st_names, st_sectors = fe.us_stock_universe()
    sp_members = set(fe.sp500_universe()[0])
    etf_tickers, etf_names, etf_groups = fe.etf_universe()
    crypto_tickers, crypto_names, crypto_groups = fe.crypto_universe()
    sector_etfs = sorted(set(fe.SECTOR_ETF_BY_GICS.values()))
    vol_pct = cfg.get("vol_percentile") or fe.vol_percentile(cfg["vix_equiv_level"])

    print(f"Downloading SPY, sector and theme ETFs since {fe.HISTORY_START}...")
    base = fe.download_ohlcv(etf_tickers + sector_etfs + [fe.BENCH], start=fe.HISTORY_START, chunk_size=DOWNLOAD_CHUNK)
    if fe.BENCH not in base["Close"].columns:
        raise RuntimeError(f"{fe.BENCH} download failed - Yahoo is likely rate-limiting; rerun later.")
    spy = base["Close"][fe.BENCH].copy()
    sec_etf_close = base["Close"][[e for e in sector_etfs if e in base["Close"].columns]].copy()
    etfs = fe.subset_panel(base, etf_tickers)
    del base

    def stock_pieces():
        for i in range(0, len(st_tickers), STOCK_PIECE):
            batch = st_tickers[i : i + STOCK_PIECE]
            print(f"Stocks {i + 1}-{i + len(batch)} of {len(st_tickers)}:")
            p = fe.download_ohlcv(batch, start=fe.HISTORY_START, chunk_size=DOWNLOAD_CHUNK)
            p = {f: df.reindex(spy.index) for f, df in p.items()}  # one calendar for every piece
            yield p, fe.sector_close_frame(p["Close"].columns, st_sectors, sec_etf_close)
            del p
            gc.collect()

    s, cond, fired = fe.scan_pieces(stock_pieces(), spy, cfg, vol_pct, keep_rows=KEEP_ROWS, driver_tail=DRIVER_TAIL)
    n_ok = 0 if s is None else s["close"].shape[1]
    if n_ok < 0.8 * len(st_tickers):
        raise RuntimeError(f"Only {n_ok} of {len(st_tickers)} stocks downloaded - Yahoo is likely rate-limiting; "
                           "rerun later rather than publish a partial report.")
    print(f"Scored {n_ok} stocks")
    universes = [dict(key="stocks", label=STOCK_LABEL, s=s, cond=cond, fired=fired, names=st_names, groups=st_sectors)]

    print("Downloading crypto...")
    crypto = fe.download_ohlcv(crypto_tickers, start=fe.HISTORY_START)  # own 7-day calendar
    for key, label, p, names, groups, ucfg in (
        ("crypto", "Crypto", crypto, crypto_names, crypto_groups, fe.crypto_config(cfg)),
        ("etfs", "Sector/Theme ETFs", etfs, etf_names, etf_groups, cfg),
    ):
        if p["Close"].empty:
            print(f"  no data for {label}, skipping")
            continue
        s, cond, fired = fe.scan_pieces([(p, None)], spy, ucfg, vol_pct, keep_rows=KEEP_ROWS, driver_tail=DRIVER_TAIL)
        universes.append(dict(key=key, label=label, s=s, cond=cond, fired=fired, names=names, groups=groups))

    print("Building market fear gauge...")
    st_close = universes[0]["s"]["close"]
    gauge = fe.market_gauge(st_close[[t for t in st_close.columns if t in sp_members]], fe.HISTORY_START, index=spy.index)
    return universes, gauge, spy, vol_pct


# ── Lists ─────────────────────────────────────────────────────────────────────
def last(frame, t):
    return frame[t].iloc[-1]


def earnings_flag(ticker, today):
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if ed is None or ed.empty:
            return ""
        dates = pd.to_datetime(ed.index).tz_localize(None).normalize()
        past = [(today - d).days for d in dates if 0 <= (today - d).days <= EARNINGS_LOOKBACK]
        ahead = [(d - today).days for d in dates if 0 < (d - today).days <= EARNINGS_LOOKAHEAD]
        if past:
            return f"Reported {min(past)}d ago"
        if ahead:
            return f"Due in {min(ahead)}d"
    except Exception:
        pass
    return ""


def base_rate(cfg, key, driver=None):
    rates = cfg.get("base_rates", {}).get(key, {})
    r = rates.get("by_driver", {}).get(driver) if driver else None
    r = r or rates.get("signal")
    if not r or r.get("6m_mean") is None:
        return ""
    return f"{r['6m_mean'] * 100:+.1f}% 6m xs, {r['6m_hit'] * 100:.0f}% hit (n={r['n']:,})"


def build_rows(u, tickers, list_name, cfg, fired_map=None):
    s, cond = u["s"], u["cond"]
    today = s["close"].index[-1]
    rows = []
    for t in tickers:
        w_live, m_live = last(s["w_rsi"], t), last(s["m_rsi"], t)
        spikes = cond["spike_day"][t].iloc[-120:].to_numpy()
        hit_days = np.flatnonzero(spikes)
        fired = (fired_map or {}).get(t)
        driver = fe.driver_labels(*(np.array([last(s[k], t)]) for k in ("mkt_part", "sec_part", "spec_part", "move21")),
                                  has_sector=s["has_sector"])[0]
        rows.append({
            "List": list_name, "Universe": u["label"], "Ticker": t,
            "Name": u["names"].get(t, t), "Group": u["groups"].get(t, ""),
            **{label: bool(last(cond[k], t)) for label, k in CONDITIONS},
            "5Y rel": last(s["rel5"], t),
            "Weekly RSI": w_live, "Weekly pct": fe.own_percentile(s["w_bars"], s["w_key"], t, w_live),
            "Weekly thr": last(cond["w_thr"], t),
            "Monthly RSI": m_live, "Monthly pct": fe.own_percentile(s["m_bars"], s["m_key"], t, m_live),
            "Monthly thr": last(cond["m_thr"], t),
            "Vol 20d": last(s["rv"], t), "Vol spike level": last(cond["v_thr"], t),
            "Days since spike": int(len(spikes) - 1 - hit_days[-1]) if len(hit_days) else None,
            "Drawdown": last(s["drawdown"], t),
            "21d move": float(np.exp(last(s["move21"], t)) - 1),
            "Driver": driver,
            "Signal date": fired[0] if fired else "",
            "Days ago": fired[1] if fired else None,
            "Earnings": earnings_flag(t, today) if u["key"] == "stocks" else "",
            "Base rate": base_rate(cfg, u["key"], driver if u["key"] == "stocks" else None) if fired else "",
        })
    return rows


def build_lists(universes, cfg):
    lists = {"Buy Signal": [], "Radar": [], "Setting Up": []}
    for u in universes:
        s, cond, fired = u["s"], u["cond"], u["fired"]
        idx = s["close"].index
        w_now = s["w_rsi"].iloc[-1]

        fired_map = {}
        if fired is not None and len(fired):
            window_start = idx[-min(RADAR_DAYS, len(idx))]
            recent = fired[(fired["date"] >= window_start) & (fired["ticker"].isin(s["close"].columns))]
            for t, d in recent.groupby("ticker")["date"].max().items():
                fired_map[t] = (d.strftime("%Y-%m-%d"), int((idx > d).sum()))
        fresh = [t for t, (_, ago) in fired_map.items() if ago < SIGNAL_KEEP_DAYS]
        radar = [t for t in fired_map if t not in fresh]

        n_met = sum(cond[k].iloc[-1].astype(int) for k in ("weekly", "monthly", "vol_spike"))
        setup = n_met[(n_met >= 2) & cond["leader"].iloc[-1]].index
        setup = [t for t in setup if t not in fired_map]

        by_rsi = lambda ts: sorted(ts, key=lambda t: np.nan_to_num(w_now.get(t, np.nan), nan=100))
        print(f"  {u['label']}: {len(fresh)} buy signals, {len(radar)} on radar, {len(setup)} setting up")
        lists["Buy Signal"] += build_rows(u, by_rsi(fresh), "Buy Signal", cfg, fired_map)
        lists["Radar"] += build_rows(u, sorted(radar, key=lambda t: fired_map[t][1]), "Radar", cfg, fired_map)
        lists["Setting Up"] += build_rows(u, by_rsi(setup)[:MAX_SETUP_ROWS], "Setting Up", cfg)
    return {k: pd.DataFrame(v) for k, v in lists.items()}


# ── HTML formatting ───────────────────────────────────────────────────────────
def pct(v, signed=True):
    if v is None or pd.isna(v):
        return "&ndash;"
    s = f"{v * 100:+.1f}%" if signed else f"{v * 100:.0f}%"
    cls = "pos" if v > 0 else "neg"
    return f'<span class="{cls}">{s}</span>' if signed else s


def ordinal(p):
    if p is None or pd.isna(p):
        return "&ndash;"
    n = int(round(p * 100))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def rsi_cell(value, own_pct, met):
    if value is None or pd.isna(value):
        return "&ndash;"
    cls = "met" if met else ""
    return f'<span class="{cls}">{value:.1f}</span> <span class="sub">({ordinal(own_pct)} pct)</span>'


def list_table(df, kind):
    if df.empty:
        return '<div class="empty">No names today.</div>'
    out = []
    for _, r in df.iterrows():
        o = {"Ticker": f"<b>{r['Ticker']}</b>", "Name": r["Name"], "Group": r["Group"]}
        if kind in ("buy", "radar"):
            o["Signal"] = r["Signal date"]
        if kind == "radar":
            o["Days ago"] = r["Days ago"]
        if kind in ("radar", "setup"):
            o["Conditions still on" if kind == "radar" else "Conditions met"] = " ".join(
                f'<span class="chip {"on" if r[label] else "off"}">{label}</span>' for label, _ in CONDITIONS[1:])
        o["5Y rel vs SPY"] = pct(r["5Y rel"])
        o["Weekly RSI"] = rsi_cell(r["Weekly RSI"], r["Weekly pct"], r["Weekly RSI"] <= r["Weekly thr"] if pd.notna(r["Weekly thr"]) else False)
        o["Monthly RSI"] = rsi_cell(r["Monthly RSI"], r["Monthly pct"], r["Monthly RSI"] <= r["Monthly thr"] if pd.notna(r["Monthly thr"]) else False)
        o["Vol (20d)"] = "&ndash;" if pd.isna(r["Vol 20d"]) else f"{r['Vol 20d'] * 100:.0f}%"
        o["Spike level"] = "&ndash;" if pd.isna(r["Vol spike level"]) else f"{r['Vol spike level'] * 100:.0f}%"
        o["Days since spike"] = "&ndash;" if r["Days since spike"] is None or pd.isna(r["Days since spike"]) else int(r["Days since spike"])
        o["From 52w high"] = pct(-r["Drawdown"])
        o["21d move"] = pct(r["21d move"])
        o["Driver"] = r["Driver"]
        o["Earnings"] = r["Earnings"]
        if kind in ("buy", "radar"):
            o["Hist. base rate"] = r["Base rate"] or "&ndash;"
        out.append(o)
    return '<div class="wrap">' + pd.DataFrame(out).to_html(escape=False, index=False, classes="tbl", border=0) + "</div>"


def fig_to_div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def base_layout(fig, title, height):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=TEXT), x=0.01),
        paper_bgcolor=DGRAY, plot_bgcolor=MGRAY, font=dict(color=TEXT, family="monospace"),
        height=height, margin=dict(l=60, r=30, t=60, b=40), hovermode="x unified",
        legend=dict(bgcolor=LGRAY, bordercolor=LGRAY, font=dict(size=10), orientation="h", y=1.02, x=0.3),
    )
    fig.update_xaxes(gridcolor=LGRAY, linecolor=LGRAY)
    fig.update_yaxes(gridcolor=LGRAY, linecolor=LGRAY)
    return fig


def add_time_controls(fig, index, row=None):
    """Zoom buttons + scrollbar, opening on the last DEFAULT_ZOOM_DAYS."""
    kw = dict(row=row, col=1) if row else {}
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[dict(count=1, label="1y", step="year", stepmode="backward"),
                     dict(count=3, label="3y", step="year", stepmode="backward"),
                     dict(count=5, label="5y", step="year", stepmode="backward"),
                     dict(step="all", label="all")],
            bgcolor=LGRAY, activecolor=ORANGE, font=dict(color=TEXT, size=10), x=0.01, y=1.12,
        ), row=1, col=1)
    if len(index) > DEFAULT_ZOOM_DAYS:
        fig.update_xaxes(range=[index[-DEFAULT_ZOOM_DAYS], index[-1]], **kw)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04, bgcolor=MGRAY), **kw)
    return fig


# ── Charts ────────────────────────────────────────────────────────────────────
def chart_gauge_history(gauge, spy):
    g = gauge.dropna(subset=["gauge"])
    s = spy.reindex(g.index)
    inv = g[g["inverted"] == True]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.06,
                        subplot_titles=["SPY (red = VIX above VIX3M)", "Market Fear Gauge (0 = calm, 100 = panic)"])
    fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=ORANGE, width=1.5), name="SPY",
                             hovertemplate="SPY %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=inv.index, y=s.reindex(inv.index), mode="markers", name="VIX > VIX3M (Panic)",
                             marker=dict(color=RED, size=5), hovertemplate="Inverted: SPY %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=ORANGE, opacity=0.10, line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=inv.index, y=inv["gauge"], mode="markers", name="VIX > VIX3M (Panic)", showlegend=False,
                             marker=dict(color=RED, size=5), hovertemplate="Inverted, gauge %{y:.0f}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["gauge"], line=dict(color=TEXT, width=1.6), name="Gauge",
                             hovertemplate="Gauge %{y:.0f}<extra></extra>"), row=2, col=1)
    # context, not a trigger: the gauge's slower "normal fear level" over the last ~5 months. Panics that
    # arrive after months of elevated fear (this line >= 60) were historically the better entries.
    ma = gauge["gauge"].rolling(GAUGE_MA_DAYS, min_periods=GAUGE_MA_DAYS).mean().reindex(g.index)
    fig.add_trace(go.Scatter(x=g.index, y=ma, line=dict(color=YELLOW, width=1.2, dash="dot"),
                             name=f"{GAUGE_MA_DAYS}-day avg",
                             hovertemplate=f"{GAUGE_MA_DAYS}-day avg %{{y:.0f}}<extra></extra>"), row=2, col=1)
    base_layout(fig, "Market Fear Gauge &mdash; full history", 600)
    fig.update_layout(showlegend=False, margin=dict(l=60, r=30, t=80, b=40))
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    add_time_controls(fig, g.index, row=2)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    add_logo(fig)
    return fig


def chart_components_history(gauge):
    """Each gauge input on its own panel, with the 'extreme' zone it has to
    reach (its own 90th percentile) and where it stands today."""
    raw, breadth = gauge.attrs["raw"], gauge.attrs["breadth"]
    panels = [(c, raw[c]) for c in raw.columns] + [("% of S&P 500 below their 200-day", breadth["% below 200-day"])]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        subplot_titles=[p[0] for p in panels])
    for i, (label, series) in enumerate(panels, 1):
        ser = series.dropna()
        if ser.empty:
            continue
        thr = ser.rolling(fe.MKT_PCT_WINDOW, min_periods=fe.MKT_PCT_MIN).quantile(0.90)
        fig.add_trace(go.Scatter(x=ser.index, y=ser, line=dict(color=TEXT, width=1.2), showlegend=False,
                                 name=label, hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"), row=i, col=1)
        fig.add_trace(go.Scatter(x=thr.index, y=thr, line=dict(color=RED, width=1, dash="dot"), showlegend=False,
                                 name="90th pct", hovertemplate="extreme zone (90th pct): %{y:.2f}<extra></extra>"),
                      row=i, col=1)
        if "VIX / VIX3M" in label:
            fig.add_hline(y=1.0, line_color=YELLOW, line_dash="dash", line_width=1, row=i, col=1)
        now, level = ser.iloc[-1], thr.iloc[-1]
        hot = pd.notna(level) and now >= level
        fig.add_trace(go.Scatter(x=[ser.index[-1]], y=[now], mode="markers", showlegend=False,
                                 marker=dict(color=RED if hot else GREEN, size=9),
                                 hovertemplate=f"now {now:,.2f}<extra></extra>"), row=i, col=1)
    base_layout(fig, "Gauge components &mdash; where each one stands vs its own history", 230 * len(panels))
    fig.update_layout(hovermode="x unified", margin=dict(l=60, r=30, t=90, b=40))
    add_time_controls(fig, raw.index, row=len(panels))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    return fig


def chart_signals_per_month(fired_all):
    g = fired_all.groupby([fired_all["date"].dt.to_period("M").dt.to_timestamp(), "universe"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for col, color in zip(g.columns, [BLUE, PURPLE, ORANGE]):
        fig.add_trace(go.Bar(x=g.index, y=g[col], name=col, marker_color=color,
                             hovertemplate="%{x|%b %Y}: %{y} signals<extra>" + col + "</extra>"))
    base_layout(fig, "Buy signals per month &mdash; full history", 380)
    fig.update_layout(barmode="stack", hovermode="x unified", showlegend=True,
                      legend=dict(orientation="h", y=1.02, x=0.3, font=dict(size=10)))
    add_time_controls(fig, g.index)
    return fig


def chart_name(u, t, cfg):
    s, cond = u["s"], u["cond"]
    n = 504
    c, s50, s200 = s["close"][t].iloc[-n:], s["sma50"][t].iloc[-n:], s["sma200"][t].iloc[-n:]
    buys = cond["buy"][t].iloc[-n:]
    buys = buys[buys].index
    spikes = cond["spike_day"][t].iloc[-n:]
    spikes = spikes[spikes].index
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.28, 0.22], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=c.index, y=c, name="Price", line=dict(color=TEXT, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s50.index, y=s50, name="50-day", line=dict(color=BLUE, width=1.1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s200.index, y=s200, name="200-day", line=dict(color=ORANGE, width=1.1)), row=1, col=1)
    if len(buys):
        fig.add_trace(go.Scatter(x=buys, y=c.reindex(buys), mode="markers", name="Buy signal",
                                 marker=dict(color=GREEN, size=11, symbol="triangle-up")), row=1, col=1)
    for key, thr, color, label in (("w_rsi", "w_thr", BLUE, "Weekly"), ("m_rsi", "m_thr", ORANGE, "Monthly")):
        fig.add_trace(go.Scatter(x=c.index, y=s[key][t].iloc[-n:], name=f"{label} RSI", line=dict(color=color, width=1.4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=c.index, y=cond[thr][t].iloc[-n:], name=f"{label} own low", showlegend=False,
                                 line=dict(color=color, width=1, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=s["rv"][t].iloc[-n:] * 100, name="Vol 20d", line=dict(color=TEXT, width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=cond["v_thr"][t].iloc[-n:] * 100, name=f"Own VIX-{cfg['vix_equiv_level']} level",
                             line=dict(color=RED, width=1, dash="dot")), row=3, col=1)
    if len(spikes):
        fig.add_trace(go.Scatter(x=spikes, y=s["rv"][t].reindex(spikes) * 100, mode="markers", showlegend=False,
                                 marker=dict(color=RED, size=4)), row=3, col=1)
    base_layout(fig, f"{u['names'].get(t, t)} [{t}]", 520)
    fig.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Vol %", row=3, col=1)
    add_logo(fig, sizex=0.08, sizey=0.10)
    return fig


# ── Time machine ──────────────────────────────────────────────────────────────
def time_machine(universes, gauge):
    """Pick any past date: what the gauge said, and which names fired."""
    signals = {}
    for u in universes:
        f, names = u["fired"], u["names"]
        if f is None or not len(f):
            continue
        short = u["label"].split(" (")[0]
        for d, t in zip(f["date"], f["ticker"]):
            signals.setdefault(d.strftime("%Y-%m-%d"), []).append([t, str(names.get(t, t))[:44], short])
    g = gauge.dropna(subset=["gauge"])
    payload = {
        "signals": signals,
        "gauge": {d.strftime("%Y-%m-%d"): [round(float(v), 1), str(r)] for d, v, r in zip(g.index, g["gauge"], g["regime"])},
    }
    dates = sorted(signals, reverse=True)[:PICKER_DATES]
    options = "".join(f'<option value="{d}">{d} ({len(signals[d])})</option>' for d in dates)
    first, last_d = g.index[0].strftime("%Y-%m-%d"), g.index[-1].strftime("%Y-%m-%d")
    return f"""
<div class="tm">
  <label>Date <input type="date" id="tmDate" min="{first}" max="{last_d}" value="{last_d}"></label>
  <label>Days with signals <select id="tmPick">{options}</select></label>
  <button class="tmbtn" id="tmPrev">&#9664; previous signal day</button>
  <button class="tmbtn" id="tmNext">next signal day &#9654;</button>
</div>
<div id="tmOut" class="tm-out"></div>
<script>
const TM = {json.dumps(payload)};
const TM_SIG_DATES = Object.keys(TM.signals).sort();
const TM_G_DATES = Object.keys(TM.gauge).sort();
function tmNearest(d) {{
  let lo = 0, hi = TM_G_DATES.length - 1, best = null;
  while (lo <= hi) {{ const mid = (lo + hi) >> 1;
    if (TM_G_DATES[mid] <= d) {{ best = TM_G_DATES[mid]; lo = mid + 1; }} else {{ hi = mid - 1; }} }}
  return best;
}}
function tmShow(d) {{
  const gd = tmNearest(d), g = gd ? TM.gauge[gd] : null, sigs = TM.signals[d] || [];
  const colour = g ? ({{Calm: "#2ECC71", Elevated: "#C67A29", Panic: "#E74C3C"}}[g[1]] || "#8E8E93") : "#8E8E93";
  let html = '<div class="tm-head">' + d + (g ? ' &middot; Market Fear Gauge <b style="color:' + colour + '">' + g[0] +
      '</b> <span class="pill" style="background:' + colour + ';color:#111">' + g[1] + '</span>' +
      (gd !== d ? ' <span class="sub">(last close ' + gd + ')</span>' : '') : ' &middot; no gauge data') + '</div>';
  if (!sigs.length) {{ html += '<div class="empty">No buy signals fired on this date.</div>'; }}
  else {{
    html += '<div class="wrap"><table class="tbl"><thead><tr><th>Ticker</th><th>Name</th><th>Universe</th></tr></thead><tbody>' +
      sigs.map(r => '<tr><td><b>' + r[0] + '</b></td><td>' + r[1] + '</td><td>' + r[2] + '</td></tr>').join('') +
      '</tbody></table></div>';
  }}
  document.getElementById('tmOut').innerHTML = html;
  document.getElementById('tmDate').value = d;
}}
function tmStep(dir) {{
  const cur = document.getElementById('tmDate').value;
  let i = TM_SIG_DATES.findIndex(x => x >= cur);
  if (i < 0) i = TM_SIG_DATES.length - 1;
  if (TM_SIG_DATES[i] === cur) i += dir; else if (dir < 0) i -= 1;
  i = Math.max(0, Math.min(TM_SIG_DATES.length - 1, i));
  const d = TM_SIG_DATES[i];
  tmShow(d);
  const pick = document.getElementById('tmPick');
  if ([...pick.options].some(o => o.value === d)) pick.value = d;
}}
document.getElementById('tmDate').addEventListener('change', e => tmShow(e.target.value));
document.getElementById('tmPick').addEventListener('change', e => tmShow(e.target.value));
document.getElementById('tmPrev').addEventListener('click', () => tmStep(-1));
document.getElementById('tmNext').addEventListener('click', () => tmStep(1));
tmShow(document.getElementById('tmDate').value);
</script>
"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Leader Selloff Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E5E5EA; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E5E5EA; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin-top: 6px; max-width: 1050px; line-height: 1.5; }}
  h3 {{ margin: 18px 32px 4px; font-size: 15px; color: #C67A29; }}
  .note {{ color: #AEAEB2; font-size: 13px; margin: 8px 32px; max-width: 1050px; line-height: 1.55; }}
  .wrap {{ overflow-x: auto; margin: 8px 32px; }}
  .empty {{ color: #8E8E93; margin: 8px 32px; font-style: italic; }}
  table.tbl {{ border-collapse: collapse; font-size: 12.5px; white-space: nowrap; }}
  .tbl th {{ background: #2C2C2E; color: #E5E5EA; padding: 6px 9px; text-align: right; border-bottom: 1px solid #3A3A3C; }}
  .tbl td {{ padding: 5px 9px; text-align: right; border-bottom: 1px solid #2C2C2E; }}
  .tbl td:nth-child(-n+3), .tbl th:nth-child(-n+3) {{ text-align: left; }}
  .met {{ color: #2ECC71; font-weight: bold; }} .sub {{ color: #8E8E93; font-size: 11px; }}
  .chip {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; margin-right: 2px; }}
  .chip.on {{ background: rgba(46,204,113,0.25); color: #2ECC71; }} .chip.off {{ background: #2C2C2E; color: #8E8E93; }}
  .pos {{ color: #2ECC71; }} .neg {{ color: #E74C3C; }}
  .gauge-box {{ display: flex; flex-wrap: wrap; gap: 18px; align-items: center; margin: 16px 32px; }}
  .gauge-num {{ font-size: 46px; font-weight: bold; }}
  .pill {{ padding: 4px 12px; border-radius: 14px; font-weight: bold; font-size: 14px; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 6px; padding: 0 24px; }}
  .tm {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin: 14px 32px; font-size: 13px; color: #AEAEB2; }}
  .tm input, .tm select {{ background: #2C2C2E; color: #E5E5EA; border: 1px solid #3A3A3C; border-radius: 4px;
     padding: 5px 8px; font-size: 13px; font-family: inherit; }}
  .tmbtn {{ background: #2C2C2E; color: #E5E5EA; border: 1px solid #3A3A3C; border-radius: 4px; padding: 5px 10px;
     font-size: 13px; cursor: pointer; }}
  .tmbtn:hover {{ border-color: #C67A29; }}
  .tm-out {{ margin-bottom: 10px; }}
  .tm-head {{ margin: 8px 32px; font-size: 15px; }}
  @media (max-width: 600px) {{ .section, header {{ padding-left: 16px; padding-right: 16px; }}
    .note, .wrap, h3, .gauge-box, .empty, .tm, .tm-head {{ margin-left: 16px; margin-right: 16px; }}
    .charts {{ grid-template-columns: 1fr; padding: 0 8px; }} }}
</style>
</head>
<body>
<header>
  <h1>Leader Selloff Report</h1>
  <div class="meta">Generated {date_str} &middot; prices through {asof} &middot; US stocks (S&amp;P 500, Nasdaq-100 + screener list), BTC/ETH, sector/theme ETFs &middot; {config_note}</div>
</header>
{body}
</body>
</html>
"""


def main():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    cfg = fe.load_config()
    universes, gauge, spy, vol_pct = load_universes(cfg)
    asof = universes[0]["s"]["close"].index[-1]
    print("Building lists...")
    lists = build_lists(universes, cfg)

    rule = (f"weekly RSI &le; its own {cfg['w_rsi_pct']:.0%} percentile, monthly RSI &le; its own "
            f"{cfg['m_rsi_pct']:.0%} percentile, and 20-day volatility hit its own VIX-{cfg['vix_equiv_level']} "
            f"equivalent ({vol_pct * 100:.0f}th percentile of its history) in the last {cfg['vol_lookback']} trading days")
    parts = []

    g_now = gauge.iloc[-1]
    ma_now = gauge["gauge"].rolling(GAUGE_MA_DAYS, min_periods=GAUGE_MA_DAYS).mean().iloc[-1]
    regime_color = {"Calm": GREEN, "Elevated": ORANGE, "Panic": RED}.get(g_now["regime"], SUBTEXT)
    parts.append(section_header("1. Market regime",
                                "Context, not part of the signal: VIX level, VIX/VIX3M term structure and credit stress, "
                                "each as a percentile of its own 5-year history"))
    parts.append(
        f'<div class="gauge-box"><div class="gauge-num" style="color:{regime_color}">{g_now["gauge"]:.0f}</div>'
        f'<span class="pill" style="background:{regime_color};color:#111">{g_now["regime"]}</span>'
        + ('<span class="pill" style="background:#8B0000;color:#fff">Credit stress</span>' if g_now["credit_stress"] else "")
        + f'<span class="note" style="margin:0">Panic = VIX term structure inverted (VIX above VIX3M). '
          f"Elevated = gauge &ge; 70. Calm otherwise. {GAUGE_MA_DAYS}-day average of the gauge: {ma_now:.0f} &mdash; "
          "historically, panics that arrived after months of elevated fear (average &ge; 60) were the better entries; "
          "sudden panics from calm worked too but with deeper drawdowns first.</span></div>"
    )
    br = gauge.attrs["breadth"].iloc[-1]
    parts.append(f'<div class="note"><b>Breadth:</b> {br["% below 50-day"] * 100:.0f}% of S&amp;P 500 members below their '
                 f'50-day, {br["% below 200-day"] * 100:.0f}% below their 200-day, {br["% at 52-wk lows"] * 100:.1f}% at '
                 f'52-week lows. VIX / VIX3M = {g_now["vix_ratio"]:.2f}.</div>')
    parts.append(fig_to_div(chart_gauge_components(gauge)))
    parts.append(fig_to_div(chart_components_history(gauge)))
    parts.append(fig_to_div(chart_gauge_history(gauge, spy)))

    def per_universe(list_name, kind):
        for u in universes:
            parts.append(f"<h3>{u['label']}</h3>")
            df = lists[list_name]
            parts.append(list_table(df[df["Universe"] == u["label"]] if len(df) else df, kind))

    parts.append(section_header("2. Buy Signals &mdash; fired in the last 5 trading days",
                                f"Historic leaders (beat SPY over the {cfg['lead_years']} years ending ~3 months ago, top "
                                f"{cfg['lead_top_pct']:.0%} of the universe) where {rule}. Crypto (BTC, ETH) uses the same "
                                "rules on its 7-day calendar, with leader = beat SPY over 5 years."))
    per_universe("Buy Signal", "buy")
    parts.append(section_header(f"3. Radar &mdash; fired in the last {RADAR_DAYS} trading days",
                                "Everything that has fired in roughly the last six weeks, newest first, so nothing is missed "
                                "between visits. \"Conditions still on\" shows which rules hold TODAY: all green means the "
                                "setup is still live, greyed-out means the name has already moved on."))
    per_universe("Radar", "radar")
    parts.append(section_header("4. Setting Up &mdash; one condition away",
                                "Historic leaders meeting 2 of the 3 other conditions today. Watch these, they are not signals."))
    per_universe("Setting Up", "setup")
    parts.append('<div class="note"><b>Columns.</b> RSI values are live (the current week/month closes at today\'s price); '
                 "the percentile is where today's value sits in every completed bar of the name's history, and green means "
                 "it is at or below the signal threshold. Spike level = the name's own VIX-equivalent volatility. Driver = "
                 "whether the 21-day move is mostly explained by the market, the sector ETF, or the stock alone. Hist. base "
                 "rate = average 6-month excess return vs SPY and hit rate for this signal in the 2005+ backtest.</div>")

    chart_names = [(r["Universe"], r["Ticker"]) for name in ("Buy Signal", "Radar", "Setting Up")
                   for _, r in lists[name].iterrows()]
    chart_names = chart_names[:MAX_NAME_CHARTS]
    if chart_names:
        parts.append(section_header("5. Charts", "Price with 50/200-day; weekly and monthly RSI against their own "
                                    "signal thresholds (dotted); volatility against its spike level (2 years)"))
        by_label = {u["label"]: u for u in universes}
        parts.append('<div class="charts">' + "".join(
            f"<div>{fig_to_div(chart_name(by_label[lab], t, cfg))}</div>" for lab, t in chart_names) + "</div>")

    fired_all = pd.concat([u["fired"].assign(universe=u["label"].split(" (")[0]) for u in universes
                           if u["fired"] is not None and len(u["fired"])], ignore_index=True)
    parts.append(section_header("6. Signal history",
                                "Every day the signal has fired since 2005. Pick a date to see the market gauge that day "
                                "and which names fired."))
    if len(fired_all):
        parts.append(fig_to_div(chart_signals_per_month(fired_all)))
    parts.append(time_machine(universes, gauge))

    parts.append('<div class="note" style="margin-top:28px">Price, volume and volatility only &mdash; no fundamentals, news or '
                 "estimates. Backtest base rates use today's listed names (survivorship bias makes levels optimistic). "
                 "A model output, not a recommendation.</div>")

    if "study_date" in cfg:
        config_note = f"thresholds from study {cfg['study_date']} ({'passed' if cfg.get('study_passed_oos') else 'NOT proven'} out-of-sample)"
    else:
        config_note = "default thresholds (run fear_study.py --write-config)"
    today = datetime.now()
    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), asof=asof.strftime("%b %d, %Y"),
                                config_note=config_note, body="\n".join(parts))
    out_html = os.path.join(OUTPUT_DIR, "Peak_Fear_Report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_html}")

    frames = [df for df in lists.values() if len(df)]
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(all_rows):
        all_rows.insert(0, "Date", asof.strftime("%Y-%m-%d"))
        all_rows.insert(1, "Market gauge", round(float(g_now["gauge"]), 1))
    all_rows.to_csv(os.path.join(OUTPUT_DIR, "Peak_Fear_latest.csv"), index=False, float_format="%.4f")
    all_rows.to_csv(os.path.join(HISTORY_DIR, f"Peak_Fear_{asof.strftime('%Y-%m-%d')}.csv"), index=False, float_format="%.4f")
    if len(fired_all):
        fired_all.sort_values("date").to_csv(os.path.join(OUTPUT_DIR, "signal_history.csv"), index=False)
        print(f"Saved signal_history.csv ({len(fired_all):,} signals since {fired_all['date'].min().date()})")
    print(f"Saved CSVs ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
