"""
Daily Leader Selloff report (the simplified v2 signal in fear_engine.py).

Buy Signal = a historic market leader whose weekly AND monthly RSI are at
statistically low points vs its own history, within 20 trading days of a
stock-level volatility spike (its own equivalent of VIX jumping to 30+).

  1. Market regime - VIX level, VIX/VIX3M, credit -> 0-100; Panic = VIX > VIX3M
  2. Buy Signals   - all four conditions met in the last 5 trading days
  3. Setting Up    - leaders meeting 2 of the 3 other conditions today
  4. Charts        - price, weekly/monthly RSI vs their own thresholds, vol vs its spike level

Thresholds and historical base rates come from data/fear_model_config.json,
written by `python scripts/fear_study.py --write-config`; built-in defaults
are used until that file exists.
"""

import base64
import gc
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

SIGNAL_KEEP_DAYS = 5      # a signal stays on the list for a week
DOWNLOAD_CHUNK = 50
DRIVER_TAIL = 400        # rows needed for today's market/sector/specific split (252d betas + 2 x 21d)
MAX_SETUP_ROWS = 40
MAX_NAME_CHARTS = 30
EARNINGS_LOOKBACK = 30
EARNINGS_LOOKAHEAD = 14
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
    sp_tickers, sp_names, sp_sectors = fe.us_stock_universe()  # S&P 500 + Nasdaq-100
    etf_tickers, etf_names, etf_groups = fe.etf_universe()
    sector_etfs = sorted(set(fe.SECTOR_ETF_BY_GICS.values()))
    print(f"Downloading {len(sp_tickers) + len(etf_tickers)} tickers since {fe.HISTORY_START}...")
    # 26 years x 200 tickers per request spikes memory; smaller batches cost little time
    panel = fe.download_ohlcv(sp_tickers + etf_tickers + sector_etfs + [fe.BENCH], start=fe.HISTORY_START,
                              chunk_size=DOWNLOAD_CHUNK)
    n_ok = panel["Close"].shape[1]
    if fe.BENCH not in panel["Close"].columns or n_ok < 0.8 * len(sp_tickers):
        raise RuntimeError(f"Price download failed ({n_ok} tickers, {fe.BENCH} "
                           f"{'present' if fe.BENCH in panel['Close'].columns else 'missing'}) - Yahoo is likely "
                           "rate-limiting; rerun later rather than publish a partial report.")
    spy = panel["Close"][fe.BENCH].copy()
    vol_pct = cfg.get("vol_percentile") or fe.vol_percentile(cfg["vix_equiv_level"])

    stocks = fe.subset_panel(panel, sp_tickers)
    etfs = fe.subset_panel(panel, etf_tickers)
    sec_close = fe.sector_close_frame(stocks["Close"].columns, sp_sectors,
                                      panel["Close"][[e for e in sector_etfs if e in panel["Close"].columns]])
    del panel  # only the subsets are needed from here; keeps peak memory down
    gc.collect()
    # separate download: crypto's 7-day calendar must not add weekend rows to the stocks
    crypto_tickers, crypto_names, crypto_groups = fe.crypto_universe()
    crypto = fe.download_ohlcv(crypto_tickers, start=fe.HISTORY_START)
    universes = []
    for key, label, p, sec, names, groups, ucfg in (
        ("stocks", "US Stocks (S&P 500 + Nasdaq-100)", stocks, sec_close, sp_names, sp_sectors, cfg),
        ("crypto", "Crypto", crypto, None, crypto_names, crypto_groups, fe.crypto_config(cfg)),
        ("etfs", "Sector/Theme ETFs", etfs, None, etf_names, etf_groups, cfg),
    ):
        if p["Close"].empty:
            print(f"  no data for {label}, skipping")
            continue
        print(f"Computing {label} ({p['Close'].shape[1]} names)...")
        s = fe.compute_signal(p, spy, sec, ucfg, driver_tail=DRIVER_TAIL)
        universes.append(dict(key=key, label=label, s=s, cond=fe.conditions(s, ucfg, vol_pct), names=names, groups=groups))
    print("Building market fear gauge...")
    gauge = fe.market_gauge(stocks["Close"], fe.HISTORY_START)
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


def build_rows(u, tickers, list_name, cfg):
    s, cond = u["s"], u["cond"]
    today = s["close"].index[-1]
    rows = []
    for t in tickers:
        w_live, m_live = last(s["w_rsi"], t), last(s["m_rsi"], t)
        spikes = cond["spike_day"][t].iloc[-120:].to_numpy()
        hit_days = np.flatnonzero(spikes)
        sig = cond["buy"][t].iloc[-SIGNAL_KEEP_DAYS:]
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
            "Signal date": sig[sig].index[-1].strftime("%Y-%m-%d") if sig.any() else "",
            "Earnings": earnings_flag(t, today) if u["key"] == "stocks" else "",
            "Base rate": base_rate(cfg, u["key"], driver if u["key"] == "stocks" else None) if list_name == "Buy Signal" else "",
        })
    return rows


def build_lists(universes, cfg):
    lists = {"Buy Signal": [], "Setting Up": []}
    for u in universes:
        s, cond = u["s"], u["cond"]
        w_now = s["w_rsi"].iloc[-1]
        recent = cond["buy"].iloc[-SIGNAL_KEEP_DAYS:].any()
        buys = sorted(recent[recent].index, key=lambda t: np.nan_to_num(w_now[t], nan=100))
        n_met = sum(cond[k].iloc[-1].astype(int) for k in ("weekly", "monthly", "vol_spike"))
        setup = n_met[(n_met >= 2) & cond["leader"].iloc[-1]].index
        setup = sorted([t for t in setup if t not in buys], key=lambda t: np.nan_to_num(w_now[t], nan=100))[:MAX_SETUP_ROWS]
        print(f"  {u['label']}: {len(buys)} buy signals, {len(setup)} setting up")
        lists["Buy Signal"] += build_rows(u, buys, "Buy Signal", cfg)
        lists["Setting Up"] += build_rows(u, setup, "Setting Up", cfg)
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
        if kind == "setup":
            o["Conditions met"] = " ".join(
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
        if kind == "buy":
            o["Signal"] = r["Signal date"]
        o["Earnings"] = r["Earnings"]
        if kind == "buy":
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


# ── Charts ────────────────────────────────────────────────────────────────────
def chart_gauge_history(gauge, spy):
    g = gauge.iloc[-1260:]
    s = spy.reindex(g.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.06,
                        subplot_titles=["SPY", "Market Fear Gauge (0 = calm, 100 = panic)"])
    fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=ORANGE, width=1.5), name="SPY",
                             hovertemplate="SPY %{y:.2f}<extra></extra>"), row=1, col=1)
    inv = g[g["inverted"] == True]
    fig.add_trace(go.Scatter(x=inv.index, y=s.reindex(inv.index), mode="markers", name="VIX > VIX3M (Panic)",
                             marker=dict(color=RED, size=5), hovertemplate="Inverted: SPY %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=ORANGE, opacity=0.10, line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=inv.index, y=inv["gauge"], mode="markers", name="VIX > VIX3M (Panic)",
                             marker=dict(color=RED, size=5), hovertemplate="Inverted, gauge %{y:.0f}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=g.index, y=g["gauge"], line=dict(color=TEXT, width=1.6), name="Gauge",
                             hovertemplate="Gauge %{y:.0f}<extra></extra>"), row=2, col=1)
    # context, not a trigger: panic days with the gauge already below this line (fear fading) were the
    # best historical entries; waiting for a cross back below it after a panic came too late
    ma20 = gauge["gauge"].rolling(20, min_periods=20).mean().reindex(g.index)
    fig.add_trace(go.Scatter(x=g.index, y=ma20, line=dict(color=YELLOW, width=1.2, dash="dot"), name="20-day avg",
                             hovertemplate="20-day avg %{y:.0f}<extra></extra>"), row=2, col=1)
    base_layout(fig, "Market Fear Gauge &mdash; 5 years", 560)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color=SUBTEXT)
    add_logo(fig)
    return fig


def chart_gauge_components(gauge):
    comps = list(gauge.attrs["raw"].columns)
    vals = gauge[comps].iloc[-1]
    raw = gauge.attrs["raw"].iloc[-1]
    labels = [f"{c}  ({raw[c]:.2f})" if abs(raw[c]) < 5 else f"{c}  ({raw[c]:.1f})" for c in comps]
    colors = [RED if v >= 85 else ORANGE if v >= 60 else GREEN for v in vals.fillna(0)]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:.0f}" if pd.notna(v) else "n/a" for v in vals], textposition="outside",
                           hovertemplate="%{y}: %{x:.0f}th pct<extra></extra>"))
    base_layout(fig, "Today's components (percentile vs own 5-year history)", 280)
    fig.update_layout(hovermode="closest", margin=dict(l=260, r=40, t=60, b=40))
    fig.update_xaxes(range=[0, 108])
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
  @media (max-width: 600px) {{ .section, header {{ padding-left: 16px; padding-right: 16px; }}
    .note, .wrap, h3, .gauge-box, .empty {{ margin-left: 16px; margin-right: 16px; }}
    .charts {{ grid-template-columns: 1fr; padding: 0 8px; }} }}
</style>
</head>
<body>
<header>
  <h1>Leader Selloff Report</h1>
  <div class="meta">Generated {date_str} &middot; prices through {asof} &middot; S&amp;P 500 + Nasdaq-100, BTC/ETH, sector/theme ETFs &middot; {config_note}</div>
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
    regime_color = {"Calm": GREEN, "Elevated": ORANGE, "Panic": RED}.get(g_now["regime"], SUBTEXT)
    parts.append(section_header("1. Market regime",
                                "Context, not part of the signal: VIX level, VIX/VIX3M term structure and credit stress, "
                                "each as a percentile of its own 5-year history"))
    parts.append(
        f'<div class="gauge-box"><div class="gauge-num" style="color:{regime_color}">{g_now["gauge"]:.0f}</div>'
        f'<span class="pill" style="background:{regime_color};color:#111">{g_now["regime"]}</span>'
        + ('<span class="pill" style="background:#8B0000;color:#fff">Credit stress</span>' if g_now["credit_stress"] else "")
        + '<span class="note" style="margin:0">Panic = VIX term structure inverted (VIX above VIX3M). '
          "Elevated = gauge &ge; 70. Calm otherwise.</span></div>"
    )
    br = gauge.attrs["breadth"].iloc[-1]
    parts.append(f'<div class="note"><b>Breadth:</b> {br["% below 50-day"] * 100:.0f}% of S&amp;P 500 members below their '
                 f'50-day, {br["% below 200-day"] * 100:.0f}% below their 200-day, {br["% at 52-wk lows"] * 100:.1f}% at '
                 f'52-week lows. VIX / VIX3M = {g_now["vix_ratio"]:.2f}.</div>')
    parts.append(fig_to_div(chart_gauge_components(gauge)))
    parts.append(fig_to_div(chart_gauge_history(gauge, spy)))

    def per_universe(list_name, kind):
        for u in universes:
            parts.append(f"<h3>{u['label']}</h3>")
            df = lists[list_name]
            parts.append(list_table(df[df["Universe"] == u["label"]] if len(df) else df, kind))

    parts.append(section_header("2. Buy Signals",
                                f"Historic leaders (beat SPY over the {cfg['lead_years']} years ending ~3 months ago, top "
                                f"{cfg['lead_top_pct']:.0%} of the universe) where {rule}. Listed for {SIGNAL_KEEP_DAYS} "
                                "trading days after the signal. Crypto (BTC, ETH) uses the same rules on its 7-day "
                                "calendar, with leader = beat SPY over 5 years."))
    per_universe("Buy Signal", "buy")
    parts.append(section_header("3. Setting Up &mdash; one condition away",
                                "Historic leaders meeting 2 of the 3 other conditions today. Watch these, they are not signals."))
    per_universe("Setting Up", "setup")
    parts.append('<div class="note"><b>Columns.</b> RSI values are live (the current week/month closes at today\'s price); '
                 "the percentile is where today's value sits in every completed bar of the name's history, and green means "
                 "it is at or below the signal threshold. Spike level = the name's own VIX-equivalent volatility. Driver = "
                 "whether the 21-day move is mostly explained by the market, the sector ETF, or the stock alone. Hist. base "
                 "rate = average 6-month excess return vs SPY and hit rate for this signal in the 2005+ backtest.</div>")

    chart_names = [(r["Universe"], r["Ticker"]) for name in ("Buy Signal", "Setting Up") for _, r in lists[name].iterrows()]
    chart_names = chart_names[:MAX_NAME_CHARTS]
    if chart_names:
        parts.append(section_header("4. Charts", "Price with 50/200-day; weekly and monthly RSI against their own "
                                    "signal thresholds (dotted); volatility against its spike level (2 years)"))
        by_label = {u["label"]: u for u in universes}
        parts.append('<div class="charts">' + "".join(
            f"<div>{fig_to_div(chart_name(by_label[lab], t, cfg))}</div>" for lab, t in chart_names) + "</div>")

    parts.append('<div class="note" style="margin-top:28px">Price, volume and volatility only &mdash; no fundamentals, news or '
                 "estimates. Backtest base rates use today's S&amp;P 500 members (survivorship bias makes levels optimistic). "
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
    print(f"Saved CSVs ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
