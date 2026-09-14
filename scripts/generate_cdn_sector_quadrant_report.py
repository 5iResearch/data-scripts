"""
Daily Canadian & US Sector Quadrant report.

Reuses the momentum z-score / 20-day displacement quadrant from the Regime
Monitors report, applied to two universes:
  Canadian sector ETFs  (vs XIC; XLY stands in for Cons. Discret., no Health Care)
  US SPDR sector ETFs   (vs SPY)
Each section adds a table of each ETF's current quadrant, with names in the
IMPROVING quadrant highlighted.
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_regime_monitors_report import (  # noqa: E402
    COMPARE_DAYS, SECTOR_CLR, build_quadrant, compute_universe, fig_to_div, section_header,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "regime-monitors")

CDN_SECTOR_UNIVERSE = [
    ("Technology", "XIT.TO", "Technology"),
    ("Financials", "XFN.TO", "Financials"),
    ("Energy", "XEG.TO", "Energy"),
    ("Materials", "XMA.TO", "Materials"),
    ("Industrials", "ZIN.TO", "Industrials"),
    ("Utilities", "XUT.TO", "Utilities"),
    ("Real Estate", "XRE.TO", "Real Estate"),
    ("Cons. Staples", "XST.TO", "Cons. Staples"),
    ("Cons. Discret.", "XLY", "Cons. Discret."),
]
CDN_BENCHMARK = "XIC.TO"

US_SECTOR_UNIVERSE = [
    ("Technology", "XLK", "Technology"),
    ("Financials", "XLF", "Financials"),
    ("Energy", "XLE", "Energy"),
    ("Materials", "XLB", "Materials"),
    ("Industrials", "XLI", "Industrials"),
    ("Utilities", "XLU", "Utilities"),
    ("Real Estate", "XLRE", "Real Estate"),
    ("Cons. Staples", "XLP", "Cons. Staples"),
    ("Cons. Discret.", "XLY", "Cons. Discret."),
    ("Health Care", "XLV", "Health Care"),
    ("Comm. Services", "XLC", "Comm. Services"),
]
US_BENCHMARK = "SPY"

QUADRANT_CLR = {"LEADING": "#2ECC71", "WEAKENING": "#F4D03F", "IMPROVING": "#1F79BE", "LAGGING": "#E74C3C"}


def quadrant(z, d):
    if z >= 0:
        return "LEADING" if d >= 0 else "WEAKENING"
    return "IMPROVING" if d >= 0 else "LAGGING"


def build_quadrant_table(ctx, region):
    z_comp, z_d20, today, labels = ctx["z_comp"], ctx["z_d20"], ctx["today"], ctx["labels"]
    past_idx = z_comp.index[z_comp.index.get_loc(today) - COMPARE_DAYS]

    rows = []
    for t in ctx["available"]:
        z, d = z_comp.loc[today, t], z_d20.loc[today, t]
        zp, dp = z_comp.loc[past_idx, t], z_d20.loc[past_idx, t]
        if pd.isna(z) or pd.isna(d):
            continue
        prev = quadrant(zp, dp) if pd.notna(zp) and pd.notna(dp) else "n/a"
        rows.append((labels[t], z, d, quadrant(z, d), prev))

    order = {"IMPROVING": 0, "LEADING": 1, "WEAKENING": 2, "LAGGING": 3}
    rows.sort(key=lambda r: (order[r[3]], -r[2]))

    improving = [r[0] for r in rows if r[3] == "IMPROVING"]
    summary = (f"Currently improving: <b>{', '.join(improving)}</b>" if improving
               else f"No {region} sector ETFs are currently in the IMPROVING quadrant.")

    body = []
    for label, z, d, q, prev in rows:
        cls = ' class="improving"' if q == "IMPROVING" else ""
        prev_clr = QUADRANT_CLR.get(prev, "#8E8E93")
        body.append(
            f"<tr{cls}><td>{label}</td><td>{z:+.2f}</td><td>{d:+.2f}</td>"
            f'<td style="color:{QUADRANT_CLR[q]};font-weight:bold">{q}</td>'
            f'<td style="color:{prev_clr}">{prev}</td></tr>'
        )

    return f"""
<div class="table-wrap">
  <div class="summary">{summary}</div>
  <table>
    <thead><tr><th>ETF</th><th>Composite Rel. Z</th><th>20-Day &Delta; Z</th>
      <th>Quadrant ({today.strftime("%b %d")})</th><th>Quadrant ({past_idx.strftime("%b %d")})</th></tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
  <div class="note">IMPROVING = below-benchmark momentum (Z &lt; 0) that is gaining over the last 20 days (&Delta; Z &gt; 0).</div>
</div>"""


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sector Quadrant Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E5E5EA; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E5E5EA; border-bottom: 2px solid #1F79BE; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin-top: 6px; }}
  .table-wrap {{ padding: 16px 32px; overflow-x: auto; }}
  .summary {{ font-size: 15px; margin-bottom: 12px; }}
  table {{ border-collapse: collapse; font-family: monospace; font-size: 13px; min-width: 640px; }}
  th, td {{ padding: 8px 14px; border-bottom: 1px solid #3A3A3C; text-align: left; }}
  th {{ color: #8E8E93; font-weight: normal; }}
  tr.improving td {{ background: rgba(31,121,190,0.22); }}
  tr.improving td:first-child {{ border-left: 3px solid #1F79BE; font-weight: bold; }}
  .note {{ color: #8E8E93; font-size: 12px; margin-top: 10px; }}
</style>
</head>
<body>
<header>
  <h1>Canadian &amp; US Sector Quadrant Report</h1>
  <div class="meta">Generated {date_str} &middot; Canadian sector ETFs (vs XIC) and US sector ETFs (vs SPY) &middot; Momentum z-score &middot; 20-day displacement quadrant</div>
</header>
{body}
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parts = []
    for region, universe, bench, sub_note in [
        ("Canadian", CDN_SECTOR_UNIVERSE, CDN_BENCHMARK, " (XLY used for Consumer Discretionary)"),
        ("US", US_SECTOR_UNIVERSE, US_BENCHMARK, ""),
    ]:
        print(f"=== {region} Sectors ===")
        ctx = compute_universe(universe, start="2015-01-01", benchmark=bench)
        parts.append(section_header(f"{region} Sector Quadrant", f"Momentum z-score relative to {bench}{sub_note}"))
        fig = build_quadrant(ctx, ctx["available"], SECTOR_CLR, f"{region} Sectors", benchmark=bench)
        if fig:
            parts.append(fig_to_div(fig))
        parts.append(section_header(f"{region} Quadrant Table", "Names currently in the IMPROVING quadrant are highlighted"))
        parts.append(build_quadrant_table(ctx, region))

    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "CDN_Sector_Quadrant_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
