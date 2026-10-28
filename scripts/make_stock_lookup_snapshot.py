"""
Frozen Stock Lookup for a Market Update blog post: the live page (outputs/stock-lookup/Stock_Lookup.html), cut down
to a fixed set of stocks with their data baked in, so it never updates and can't look up anything else.

The live page fetches data/<ticker>.json on demand, so anyone who copies its address gets every stock, every day.
This copy has no search box (buttons for the chosen stocks instead), no fetch (each stock's JSON is inlined), and
only the chosen stocks in its ticker list, so complement links to other stocks show as plain text. Charts stay
interactive (hover, zoom, time frames) but the data stops at the live page's "as of" date.

Run after generate_stock_lookup_web.py so the snapshot uses that day's data:
    python scripts/make_stock_lookup_snapshot.py
Output: outputs/market-update-snapshots/stock-lookup-<as of date>/Stock_Lookup_Snapshot.html (one self-contained file)
"""

import json
import os
import re
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR = os.path.join(REPO_ROOT, "outputs", "stock-lookup")
OUT_ROOT = os.path.join(REPO_ROOT, "outputs", "market-update-snapshots")
JOIN_URL = "https://www.5iresearch.ca/join-investor-suite"

# (ticker, held in a 5i model portfolio). Model names are from the Sept 30, 2026 factsheets.
CDN = [("CLS.TO", True), ("CSU.TO", True), ("WPM.TO", True), ("ENB.TO", True),
       ("RY.TO", False), ("TD.TO", False), ("CNQ.TO", False), ("BCE.TO", False), ("CNR.TO", False), ("MFC.TO", False)]
US = [(t, False) for t in ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "LLY")]
MODEL_ASOF = "September 30, 2026"


def swap(html: str, old: str, new: str) -> str:
    """Replace exactly one occurrence, so a change to the live page fails loudly instead of leaking a feature."""
    if html.count(old) != 1:
        raise SystemExit(f"Expected one match in Stock_Lookup.html, found {html.count(old)}: {old[:70]!r}")
    return html.replace(old, new)


def chips(label: str, names: list, by_sym: dict) -> str:
    links = "".join(
        f'<a data-t="{t}" title="{by_sym[t][1]}">{t.replace(".TO", "")}{"<span class=dot></span>" if held else ""}</a>'
        for t, held in names)
    return f'<div class="chips"><span class="lbl">{label}</span>{links}</div>'


def main():
    with open(os.path.join(LIVE_DIR, "Stock_Lookup.html"), encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"const META = (\{.*?\});\n", html)
    meta = json.loads(m.group(1))
    by_sym = {r[0]: r for r in meta["index"]}

    picks = CDN + US
    missing = [t for t, _ in picks if t not in by_sym]
    if missing:
        raise SystemExit(f"Not in Stock Lookup coverage: {missing}")
    snap = {}
    for t, _ in picks:
        with open(os.path.join(LIVE_DIR, "data", t + ".json"), encoding="utf-8") as f:
            snap[t] = json.load(f)
        thin = [k for k in ("rev", "fund", "risk", "comp") if not snap[t].get(k)]
        if thin:
            print(f"  note: {t} has no {', '.join(thin)} data; that section will show as unavailable")

    meta["index"] = [by_sym[t] for t, _ in picks]
    asof = datetime.strptime(meta["asof"], "%Y-%m-%d")
    asof_txt = f"{asof:%B} {asof.day}, {asof.year}"

    html = html.replace(m.group(1), json.dumps(meta, separators=(",", ":")), 1)
    # Inline data instead of fetching data/<ticker>.json
    html = swap(html, "const BY_SYM = {};",
                "const SNAP = " + json.dumps(snap, separators=(",", ":")) + ";\nconst BY_SYM = {};")
    html = swap(html, "fetch(dataFile(t))",
                "Promise.resolve({ ok: !!SNAP[t], status: 404, json: function () { return SNAP[t]; } })")
    # Search box -> stock buttons; the input stays (hidden) because lookup() writes the chosen ticker into it
    html = swap(html, '<input id="q" list="tickers"', '<input id="q" type="hidden" list="tickers"')
    html = swap(html, '<button onclick="lookup()">Look up</button>', "")
    html = re.sub(r'<div class="chips">Try:.*?</div>',
                  lambda _: chips("Canada", CDN, by_sym) + chips("U.S.", US, by_sym) +
                  f'<div class="key"><span class="dot"></span> Held in a 5i model portfolio as of {MODEL_ASOF}</div>',
                  html, count=1, flags=re.S)
    html = swap(html, "<h1>Stock Lookup</h1>", "<h1>Stock Lookup: 20 stocks</h1>")
    html = re.sub(r'<div class="meta">Prices as of .*?</div>',
                  f'<div class="meta">A snapshot from the Portfolio Risk Lab &middot; prices as of {asof_txt}. '
                  f'This page does not update.</div>', html, count=1)
    html = re.sub(r"<p>Enter a ticker or company name to see.*?</p>",
                  "<p>Pick a stock to see its long-term trend, what analysts are doing with their revenue forecasts, "
                  "how its risk and return compare with the rest of its market, and whether it has beaten simply "
                  "owning the index.</p>", html, count=1, flags=re.S)
    cta = ('<div class="cta"><p>In the Portfolio Risk Lab, Stock Lookup covers every TSX, S&amp;P 500 and Nasdaq 100 '
           'stock and updates every weekday.</p><a href="' + JOIN_URL + '" target="_top">Join Investor Suite</a></div>')
    html = re.sub(r'(<div class="source">)(.*?)(</div>)',
                  lambda g: cta + g.group(1) + "Source: 5i Research, Koyfin, Yahoo Finance. Prices as of " + asof_txt +
                  ". For information only; not a recommendation to buy or sell any security." + g.group(3),
                  html, count=1, flags=re.S)
    html = swap(html, "</style>", """  .chips { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 8px 0 0; }
  .chips .lbl { width: 64px; font-size: 13px; font-weight: bold; color: #555555; text-transform: uppercase; letter-spacing: .03em; }
  .chips a { margin: 0; border: 1px solid #CFCFCF; border-radius: 4px; padding: 4px 10px; font-weight: bold; color: #1F79BE; }
  .chips a:hover { text-decoration: none; border-color: #1F79BE; }
  .chips a.on { background: #1F79BE; border-color: #1F79BE; color: #FFFFFF; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #C67A29; margin-left: 5px; vertical-align: 2px; }
  .key { font-size: 13px; color: #555555; margin: 8px 0 0; }
  .key .dot { margin: 0 3px 0 0; }
  .cta { margin: 24px 16px 8px; padding: 16px 18px; background: #13243A; color: #FFFFFF; border-radius: 6px;
         display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px; }
  .cta p { margin: 0; flex: 1 1 320px; font-size: 15px; }
  .cta a { background: #C67A29; color: #FFFFFF; font-weight: bold; text-decoration: none; padding: 9px 18px; border-radius: 4px; white-space: nowrap; }
  @media (max-width: 600px) { .chips .lbl { width: 100%; } }
</style>""")
    # Highlight the chosen stock's button; open on the first Canadian name; tell an embedding page our height
    html = swap(html, "    render(d);\n",
                "    render(d);\n    document.querySelectorAll('.chips a').forEach(function (a) "
                "{ a.classList.toggle('on', a.dataset.t === t); });\n    setTimeout(postHeight, 300);\n")
    html = swap(html, "  if (t) lookup(t);\n})();",
                "  lookup(t && SNAP[t] ? t : " + json.dumps(CDN[0][0]) + ");\n})();\n"
                "function postHeight() { try { parent.postMessage({ slsnapHeight: document.documentElement.scrollHeight }, '*'); } catch (e) {} }\n"
                "window.addEventListener('resize', postHeight);")

    out_dir = os.path.join(OUT_ROOT, f"stock-lookup-{meta['asof']}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "Stock_Lookup_Snapshot.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, prices as of {asof_txt})")


if __name__ == "__main__":
    main()
