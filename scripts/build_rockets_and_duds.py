"""
Weekly "Rockets and Duds" blog post: colleague's email -> charts + blog HTML.

Reads input.txt (the email pasted verbatim) from an issue folder, then:
  * splits it into the intro and one entry per stock (company name, ticker,
    rocket/dud emoji, write-up)
  * looks up each ticker's exchange so headings link to the right
    5iresearch.ca/company/<exchange>/<TICKER> page (cached in exchanges.json)
  * draws the 1-year price/volume chart for each ticker, the same design as the
    Rockets and Duds notebook, saved as <TICKER>_institutional.png
  * writes post.html in the site's markup and kit.html to copy/preview from

The PNGs are uploaded to the site's file manager by hand, keeping their names:
the <img> tags already point at /files/www/<TICKER>_institutional.png.

Usage:  python scripts/build_rockets_and_duds.py ["<issue folder>"] [--date YYYY-MM-DD]
        (or double-click "Build Rockets and Duds.bat", which uses this week's folder)
"""

import argparse
import html
import json
import os
import re
import sys
import webbrowser
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_TEMPLATE = os.path.join(REPO_ROOT, "templates", "rockets_kit.html")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")
ISSUES_ROOT = os.path.join(os.path.expanduser("~"), "Documents", "Rockets and Duds")
SITE_IMAGE_DIR = "/files/www"
SITE_BASE = "https://www.5iresearch.ca"

# Week 84 was posted Monday September 14, 2026; later weeks count up from there.
WEEK_ANCHOR = (datetime(2026, 9, 14), 84)

ROCKET, DUD = "\U0001F680", "\U0001F6AB"

# notebook palette
orange, blue = "#C67A29", "#1F79BE"
green, red = "#2ECC71", "#E74C3C"
dgray, mgray, lgray = "#1C1C1E", "#2C2C2E", "#3A3A3C"
text_col, subtext = "#E5E5EA", "#8E8E93"

EXCHANGE_SLUGS = {
    "NYQ": "nyse", "NYS": "nyse", "PCX": "nyse", "ASE": "nyse", "AMEX": "nyse",
    "NMS": "nasdaq", "NGM": "nasdaq", "NCM": "nasdaq", "NAS": "nasdaq",
    "TOR": "tsx", "TSX": "tsx", "VAN": "tsxv", "CNQ": "cse", "NEO": "neo",
}


# --------------------------------------------------------------------------- email

def parse_email(raw):
    """Intro paragraphs + one entry per stock. A stock paragraph looks like
    'Pershing Square Inc. PS ROCKETx3 Everyone loves Billionaire Bill...'."""
    entry_re = re.compile(
        rf"^(?P<name>.+?)\s+(?P<ticker>[A-Z][A-Z.\-]{{0,7}})\s*"
        rf"(?P<emoji>(?:[{ROCKET}{DUD}]\s*)+)\s*(?P<body>.*)$",
        re.S,
    )
    intro, entries = [], []
    for para in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").strip()):
        para = " ".join(line.strip() for line in para.splitlines()).strip()
        if not para:
            continue
        m = entry_re.match(para)
        if m:
            emoji = re.sub(r"\s+", "", m.group("emoji"))
            entries.append({
                "name": m.group("name").strip(),
                "ticker": m.group("ticker").strip(),
                "emoji": emoji,
                "kind": "rocket" if ROCKET in emoji else "dud",
                "body": m.group("body").strip(),
            })
        elif entries:
            entries[-1]["body"] += " " + para  # continuation of the last write-up
        else:
            intro.append(para)
    return intro, entries


def issue_week(when):
    return WEEK_ANCHOR[1] + round((when - WEEK_ANCHOR[0]).days / 7)


def issue_title(when):
    return f"Rockets and Duds: Week {issue_week(when)} - {when:%B} {when.day}, {when.year}"


def issue_slug(when):
    """Post URL slug, e.g. rockets-and-duds-week-83-september-8-2026."""
    return (f"rockets-and-duds-week-{issue_week(when)}-"
            f"{when.strftime('%B').lower()}-{when.day}-{when.year}")


# --------------------------------------------------------------------------- exchanges

def load_cache(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def exchange_slug(ticker, cache, warnings):
    if ticker in cache:
        return cache[ticker]
    slug = ""
    try:
        info = yf.Ticker(ticker).info
        code = (info.get("exchange") or "").upper()
        slug = EXCHANGE_SLUGS.get(code, "")
        if not slug:
            warnings.append(f"{ticker}: unknown exchange '{code}' - check the heading link.")
    except Exception as e:  # network/ticker problems shouldn't stop the build
        warnings.append(f"{ticker}: couldn't look up the exchange ({e.__class__.__name__}).")
    cache[ticker] = slug
    return slug


# --------------------------------------------------------------------------- chart

def load_logo():
    raw = mpimg.imread(LOGO_PATH)
    if raw.dtype == np.uint8:
        raw = raw.astype(np.float32) / 255.0
    if raw.ndim == 2:
        rgba = np.stack([raw] * 3 + [np.ones_like(raw)], axis=-1)
    elif raw.shape[2] == 3:
        rgba = np.concatenate([raw, np.ones((*raw.shape[:2], 1), dtype=np.float32)], axis=-1)
    else:
        rgba = raw.copy()
    white = (rgba[:, :, 0] > 0.88) & (rgba[:, :, 1] > 0.88) & (rgba[:, :, 2] > 0.88)
    rgba[white, 3] = 0.0
    return rgba


def render_chart(stock, out_path, logo_img, as_of):
    """1-year price + volume chart, ported from the Rockets and Duds notebook."""
    start_date = as_of - timedelta(weeks=52)
    x_end = as_of + timedelta(days=55)
    as_of_str = as_of.strftime("%B %d, %Y")

    data = yf.download(stock, start=start_date.strftime("%Y-%m-%d"),
                       end=as_of.strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        return None, f"{stock}: no price data from Yahoo Finance - chart skipped."

    info = yf.Ticker(stock).info
    company_name = info.get("longName") or info.get("shortName") or stock

    close, volume, dates = data["Close"].squeeze(), data["Volume"].squeeze(), data.index
    start_price, end_price = float(close.iloc[0]), float(close.iloc[-1])
    performance = (end_price - start_price) / start_price * 100
    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    perf_sign = "+" if performance >= 0 else ""
    line_color = green if performance >= 0 else red
    w52_high, w52_low = float(close.max()), float(close.min())
    vol_colors = [green if float(close.iloc[i]) >= float(data["Open"].iloc[i]) else red
                  for i in range(len(data))]

    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor(dgray)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1], hspace=0.04,
                           left=0.06, right=0.97, top=0.88, bottom=0.12)
    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    for ax in (ax_price, ax_vol):
        ax.set_facecolor(mgray)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=subtext, length=0, labelsize=9)

    fig.add_artist(mlines.Line2D([0.06, 0.97], [0.935, 0.935], transform=fig.transFigure,
                                 color=orange, linewidth=2.5, solid_capstyle="round"))
    fig.text(0.06, 0.956, f"{stock}  ·  {company_name}", fontsize=17, fontweight="bold",
             color=text_col, va="bottom", ha="left")
    fig.text(0.06, 0.928, f"1-Year Price Performance  ·  As of {as_of_str}",
             fontsize=9.5, color=subtext, va="top", ha="left")

    imagebox = OffsetImage(logo_img, zoom=0.115)
    imagebox.image.axes = ax_price
    fig.add_artist(AnnotationBbox(imagebox, xy=(0.955, 0.975), xycoords="figure fraction",
                                  box_alignment=(1.0, 1.0), frameon=False, pad=0.0))

    ax_price.fill_between(dates, close, float(close.min()) * 0.97,
                          color=line_color, alpha=0.07, zorder=1)
    if ma200.notna().sum() > 5:
        ax_price.plot(dates, ma200, color=subtext, lw=1.2, linestyle=":",
                      label="200-day MA", zorder=2, alpha=0.7)
    ax_price.plot(dates, ma50, color=orange, lw=1.5, linestyle="--",
                  label="50-day MA", zorder=3, alpha=0.9)
    ax_price.plot(dates, close, color=line_color, lw=2.2, label="Close", zorder=4,
                  solid_capstyle="round")

    for idx, lbl, va in [(close.idxmax(), f"52W High\n${w52_high:,.2f}", "bottom"),
                         (close.idxmin(), f"52W Low\n${w52_low:,.2f}", "top")]:
        ax_price.scatter(idx, float(close[idx]), color=orange, s=55, zorder=6, edgecolors="none")
        ax_price.annotate(lbl, xy=(idx, float(close[idx])),
                          xytext=(0, 16 if va == "bottom" else -16), textcoords="offset points",
                          ha="center", va=va, fontsize=8, color=text_col,
                          bbox=dict(boxstyle="round,pad=0.35", fc=lgray, ec=orange,
                                    linewidth=0.8, alpha=0.92))
    ax_price.scatter([dates[0], dates[-1]], [start_price, end_price], color=orange, s=55,
                     zorder=6, edgecolors="none")
    ax_price.text(0.985, 0.965, f"{perf_sign}{performance:.1f}%", transform=ax_price.transAxes,
                  fontsize=18, fontweight="bold", color=line_color, ha="right", va="top",
                  bbox=dict(boxstyle="round,pad=0.45", fc="#1a3d2b" if performance >= 0 else "#3d1a1a",
                            ec=line_color, linewidth=1.2, alpha=0.95))

    ax_price.set_xlim(dates[0], x_end)
    ax_price.yaxis.grid(True, color=lgray, linewidth=0.55, linestyle="-")
    ax_price.xaxis.grid(False)
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_price.get_yticklabels(), color=subtext, fontsize=9)
    ax_price.legend(fontsize=8.5, loc="upper left", facecolor=lgray, edgecolor=lgray,
                    labelcolor=subtext, framealpha=0.9, borderpad=0.6, handlelength=1.8)

    bar_width = (dates[-1] - dates[0]).days / len(dates) * 0.75
    ax_vol.bar(dates, volume, color=vol_colors, alpha=0.55, width=bar_width, zorder=2)
    ax_vol.plot(dates, volume.rolling(20).mean(), color=orange, lw=1.2, alpha=0.8, zorder=3)
    ax_vol.yaxis.grid(True, color=lgray, linewidth=0.45, linestyle="-")
    ax_vol.xaxis.grid(False)
    ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"))
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_vol.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_vol.get_xticklabels(), color=subtext, fontsize=9)
    plt.setp(ax_vol.get_yticklabels(), color=subtext, fontsize=8)
    ax_vol.set_ylabel("Volume", color=subtext, fontsize=8.5, labelpad=6)

    stats_y = 0.038
    fig.add_artist(mlines.Line2D([0.06, 0.97], [stats_y + 0.025] * 2, transform=fig.transFigure,
                                 color=lgray, linewidth=0.8))
    x_pos = 0.06
    for label, val in (("Current", f"${end_price:,.2f}"), ("52W High", f"${w52_high:,.2f}"),
                       ("52W Low", f"${w52_low:,.2f}"),
                       ("Period Rtn", f"{perf_sign}{performance:.1f}%")):
        fig.text(x_pos, stats_y + 0.015, label, fontsize=7.5, color=subtext, ha="left",
                 va="center", transform=fig.transFigure)
        color_val = (line_color if label == "Period Rtn" else
                     green if label == "52W High" else red if label == "52W Low" else text_col)
        fig.text(x_pos, stats_y + 0.003, val, fontsize=9.5, fontweight="bold", color=color_val,
                 ha="left", va="center", transform=fig.transFigure)
        x_pos += 0.12
    fig.text(0.97, stats_y + 0.008, "Source: Yahoo Finance", ha="right", va="center",
             fontsize=8, color=subtext, transform=fig.transFigure)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=dgray)
    plt.close(fig)
    return f"{perf_sign}{performance:.1f}%", None


# --------------------------------------------------------------------------- html

FONT = "Arial,Helvetica,sans-serif"
BLUE = blue
INK, MUTED, RULE = "#363636", "#6F6F6F", "#E3E6EA"
ROCKET_GREEN, DUD_RED = "#2E7D4F", "#A22A2A"


def build_post(intro, entries):
    """Blog body in 5i colours: an intro, then one card per stock with a
    ROCKET/DUD badge, the ticker linked to its company page, the write-up and
    the chart. Divs only - the blog editor outlines every <table>."""
    para_style = f"font-family:{FONT};font-size:16px;line-height:1.6;color:{INK};margin:0 0 14px;"
    parts = [f'<div style="{para_style}">{html.escape(p)}</div>' for p in intro]

    for e in entries:
        accent = ROCKET_GREEN if e["kind"] == "rocket" else DUD_RED
        badge = (f'<span style="display:inline-block;background:{accent};color:#ffffff;'
                 f'font-family:{FONT};font-size:11px;font-weight:bold;letter-spacing:1px;'
                 f'padding:3px 9px;border-radius:3px;margin:0 10px 0 0;vertical-align:3px;">'
                 f'{"ROCKET" if e["kind"] == "rocket" else "DUD"}</span>')
        name = (f'<span style="font-family:{FONT};font-size:22px;font-weight:bold;color:{INK};">'
                f'{html.escape(e["name"])} </span>'
                f'<span style="font-family:{FONT};font-size:22px;font-weight:bold;color:{BLUE};">'
                f'{html.escape(e["ticker"])}</span>')
        if e["href"]:
            name = f'<a href="{e["href"]}" target="_blank" style="text-decoration:none;">{name}</a>'
        parts.append(
            f'<div style="margin:30px 0 12px;padding:0 0 8px;border-bottom:2px solid {accent};">'
            f'{badge}{name} <span style="font-size:18px;">{e["emoji"]}</span></div>'
        )
        parts.append(f'<div style="{para_style}">{html.escape(e["body"])}</div>')
        if e.get("image"):
            parts.append(
                f'<div style="margin:0 0 10px;"><img src="{SITE_IMAGE_DIR}/{e["image"][0]}" '
                f'alt="{html.escape(e["ticker"])} one-year price performance" '
                f'style="width:100%;height:auto;border-radius:6px;display:block;" /></div>'
            )
            move = f' &middot; 1-year move {html.escape(e["move"])}' if e.get("move") else ""
            parts.append(f'<div style="font-family:{FONT};font-size:13px;color:{MUTED};'
                         f'margin:0 0 6px;"><em>{html.escape(e["ticker"])} one-year price '
                         f'performance{move}. Source: Yahoo Finance.</em></div>')
    return "\n".join(parts) + "\n"


SIGNATURE_IMG = ("https://mcusercontent.com/70a01e3dae2f947876a36d9c2/images/"
                 "e92c40eb-ca4e-4f64-9ceb-5574ffa1c633.png")
TRIAL_BANNER_IMG = "/files/www/inline/241ec2a2e05b8d378992aa60cda08f6658aa3fa0.png"
TRIAL_BANNER_HREF = "/index.php?p=qt_redeem_code.NewAccount&amp;code=14day"
PROMO_BULLETS = [
    ("Investor Q&amp;A:", "Have burning questions? Get answers from our team of experts and "
                          "fellow investors in our dedicated Q&amp;A section."),
    ("Research Reports:", "With over 60 meticulously researched Canadian stocks, our reports "
                          "offer in-depth analysis, giving you the confidence to invest wisely."),
    ("Model Portfolios, Alerts, Forums, Portfolio Tracking, and Much More..", ""),
]
DISCLOSURE = (
    "Analysts of 5i Research responsible for this report do not have a financial or other interest "
    "in securities mentioned. The i2i Fund does not have a financial or other interest in securities "
    "mentioned. Clients of i2i Capital Managements Private Investment Counsel service (i2i PIC) may "
    "hold a financial interest in any companies discussed and the views of i2i PIC may differ from "
    "the views of 5i Research."
)


def previous_post_url(when, warnings):
    """Last week's post, assuming the usual weekly cadence. Checked over the web
    so a skipped or differently dated week shows up as a warning, not a dead link."""
    url = f"{SITE_BASE}/blog/{issue_slug(when - timedelta(days=7))}"
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status != 200:
                raise OSError(r.status)
    except Exception:
        warnings.append(f"The 'Missed last week's' link didn't load: {url} - "
                        "check it, or fix that one link in the blog editor.")
    return url


def build_footer(when, warnings):
    """Back-link to last week's post, sign-off, the 5i promo block and the
    disclosure (which the user edits by hand when it changes)."""
    para = f"font-family:{FONT};font-size:16px;line-height:1.6;color:{INK};margin:0 0 14px;"
    prev_url = previous_post_url(when, warnings)
    bullets = "".join(
        f'<li style="margin:0 0 8px;"><strong>{label}</strong>{" " + text if text else ""}</li>'
        for label, text in PROMO_BULLETS)
    return f"""
<div style="border-top:2px solid {orange};margin:34px 0 18px;"></div>
<div style="{para}">Missed last week's <a href="{prev_url}" target="_blank" style="color:{BLUE};font-weight:bold;">Rockets and Duds?</a></div>
<div style="{para}margin:0 0 6px;">Take Care,</div>
<div style="margin:0 0 20px;"><img title="Peter's Signature" src="{SIGNATURE_IMG}" alt="Peter's Signature" width="180" height="108" /></div>
<div style="background:#F6F8FA;border:1px solid {RULE};border-radius:6px;padding:20px 22px;margin:0 0 18px;">
<div style="font-family:{FONT};font-size:20px;font-weight:bold;color:{orange};text-align:center;margin:0 0 12px;">Unlock the Power of Informed Investing with 5i Research!</div>
<div style="{para}">DIY investing doesn't have to mean going it alone. At 5i Research, we're your trusted partner in navigating the stock market. Our platform offers comprehensive stock and market research, empowering you to make smart investment decisions.</div>
<ul style="font-family:{FONT};font-size:16px;line-height:1.6;color:{INK};margin:0 0 16px;padding-left:22px;">{bullets}</ul>
<div style="text-align:center;"><a href="{TRIAL_BANNER_HREF}" target="_blank"><img src="{TRIAL_BANNER_IMG}" alt="Start your free 14-day trial of 5i Research" width="420" height="128" style="display:block;margin:0 auto;max-width:100%;height:auto;" /></a></div>
</div>
<div style="font-family:{FONT};font-size:12px;line-height:1.5;color:{MUTED};font-style:italic;">{DISCLOSURE}</div>
"""


def build_kit(title, post_html, entries, warnings, as_of):
    with open(KIT_TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    data = {"title": title, "post": post_html, "warnings": warnings,
            "images": [e["image"][0] for e in entries if e.get("image")],
            "stocks": [{"ticker": e["ticker"], "name": e["name"], "kind": e["kind"],
                        "move": e.get("move", ""), "href": e["href"]} for e in entries]}
    return page.replace("%%DATA_JSON%%", json.dumps(data).replace("</", "<\\/")) \
               .replace("%%TITLE%%", html.escape(title)) \
               .replace("%%DATE%%", f"{as_of:%B} {as_of.day}, {as_of.year}")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="Issue folder (default: this week's)")
    ap.add_argument("--date", help="Post date YYYY-MM-DD (default: this week's Monday)")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    when = (datetime.strptime(args.date, "%Y-%m-%d") if args.date
            else datetime.now() - timedelta(days=datetime.now().weekday()))
    folder = os.path.abspath(args.folder.strip('"')) if args.folder \
        else os.path.join(ISSUES_ROOT, when.strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)

    input_path = os.path.join(folder, "input.txt")
    if not os.path.exists(input_path):
        with open(input_path, "w", encoding="utf-8") as f:
            f.write("Paste the Rockets and Duds email here, then run this again.\n")
        print(f"Created {input_path}\nPaste the email in, save, then run this again.")
        if hasattr(os, "startfile"):
            os.startfile(input_path)
        return

    with open(input_path, encoding="utf-8") as f:
        raw = f.read()
    intro, entries = parse_email(raw)
    warnings = []
    if not entries:
        warnings.append("No stocks found in input.txt - each stock needs a ticker and rocket/dud emoji.")
    print(f"{len(entries)} stock(s): {', '.join(e['ticker'] for e in entries)}")

    cache_path = os.path.join(ISSUES_ROOT, "exchanges.json")
    cache = load_cache(cache_path)
    logo_img = load_logo()
    for e in entries:
        slug = exchange_slug(e["ticker"], cache, warnings)
        e["href"] = f"{SITE_BASE}/company/{slug}/{e['ticker']}" if slug else ""
        name = f"{e['ticker']}_institutional.png"
        out_path = os.path.join(folder, name)
        move, err = render_chart(e["ticker"], out_path, logo_img, when)
        if err:
            warnings.append(err)
            e["image"] = None
            continue
        e["move"] = move
        with open(out_path, "rb") as fh:
            head = fh.read(33)
        w, h = (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")) \
            if head[:8] == b"\x89PNG\r\n\x1a\n" else (0, 0)
        e["image"] = (name, w, h)
        print(f"  {e['ticker']}: {move}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    title = issue_title(when)
    post_html = build_post(intro, entries) + build_footer(when, warnings)
    with open(os.path.join(folder, "post.html"), "w", encoding="utf-8") as f:
        f.write(post_html)
    kit_path = os.path.join(folder, "kit.html")
    with open(kit_path, "w", encoding="utf-8") as f:
        f.write(build_kit(title, post_html, entries, warnings, when))

    print(f"\n{title}")
    for w in warnings:
        print(f"  ! {w}")
    print(f"Kit: {kit_path}")
    if not args.no_open:
        webbrowser.open("file:///" + kit_path.replace(os.sep, "/"))


if __name__ == "__main__":
    main()
