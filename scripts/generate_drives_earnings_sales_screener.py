"""
"Drives Earnings & Sales" screener — $2B–$35B market cap universe.

Automated port of drives_earnings_sales_screener_2_35B.ipynb, which itself
replicates the "Final Ranking (3)" tab of `Drives Earnings and Sales - 35+.xlsm`.
Ranks every name on who is actually driving sales/profit dollars (and price
performance) across the universe, its sector, and its industry:

  1. Overall ranking   (xlsm cols S–W)   — Q = avg(elite Sales$ rank, elite
     Price3Y rank) / # of top-35 criteria hit; lower = better. Top 50 shown.
  2. Sector ranking    (xlsm cols AB–AK) — dropdown per sector
  3. Industry ranking  (xlsm cols AT–BC) — dropdown per industry

No price downloads — everything comes from one manually-refreshed screener
export (same 20 columns as the notebook's final_table_input_2_35.csv):
  data/drives_earnings_sales_2_35.csv
plus the ETF holdings list used for the GRNJ flag:
  data/etf_holdings.csv   (two columns: ETF, Ticker)
"""

import os
import subprocess
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "drives-earnings-sales-screener")
INPUT_CSV_PATH = os.path.join(REPO_ROOT, "data", "drives_earnings_sales_2_35.csv")
ETF_HOLDINGS_PATH = os.path.join(REPO_ROOT, "data", "etf_holdings.csv")

PRICE_THRESHOLD = 0.25
N_ELITE = 35
OVERALL_TOP_N = 50

BG_CELL, BG_HEADER, FG_MAIN, FG_HEADER = "#1a1d27", "#252840", "#c8ccd8", "#e2e5f0"
BORDER = "1px solid #2d3148"
C_GREEN, C_LGREEN, C_RED, C_NEUTRAL = "#2ecc71", "#27ae60", "#e74c3c", "#8b90a4"

RENAME = {
    "Last Price": "Price", "Market Cap": "MarketCap", "Trading Region": "Region",
    "Net Income - (IS) (LTM)": "NI_LTM", "Net Income - (IS) (FY)": "NI_FY",
    "Net Income - (IS) (-1FY)": "NI_1FY", "Net Income - (IS) (-2FY)": "NI_2FY",
    "Total Revenues (LTM)": "Rev_LTM", "Total Revenues (FY)": "Rev_FY",
    "Total Revenues (-1FY)": "Rev_1FY", "Total Revenues (1YTGLTM)": "Rev_Growth1Y",
    "Price Chg. % (1Y)": "PriceChg1Y", "Price Chg. %/CAGR (3Y)": "PriceChg3Y",
    "Price Chg. %/CAGR (5Y)": "PriceChg5Y", "Price Chg. %/CAGR (10Y)": "PriceChg10Y",
}
NUM_COLS = ["Price", "MarketCap", "NI_LTM", "NI_FY", "NI_1FY", "NI_2FY",
            "Rev_LTM", "Rev_FY", "Rev_1FY", "Rev_Growth1Y",
            "PriceChg1Y", "PriceChg3Y", "PriceChg5Y", "PriceChg10Y"]

# GRNY fallback kept from the notebook in case etf_holdings.csv is missing
_FALLBACKS = {"GRNY": {
    "GE", "ANET", "GOOGL", "AVGO", "COST", "EMR", "BK", "AXP", "LYV", "AXON",
    "CAT", "PWR", "GRMN", "CDNS", "VST", "MNST", "NVDA", "WTW", "MSTR", "MSFT",
    "GEV", "JPM", "PANW", "SPGI", "PLTR", "HOOD", "NFLX", "AMD", "GS", "ETN",
    "EXPE", "TSLA", "META", "AMZN", "KLAC", "LRCX", "AAPL", "CRWD", "ORCL",
}}

EXPORT_COLS = [
    "Ticker", "Name", "Sector", "Industry", "Price", "MarketCap",
    "PriceChg1Y", "PriceChg3Y", "PriceChg5Y", "PriceChg10Y",
    "Sales_D", "Sales_Pct", "Profits_D", "Profits_Pct",
    "Q", "P", "O",
    "Score_Sector", "Score_Sector_Adj", "Score_Ind", "Score_Ind_Adj",
    "Rank_Sales_D", "Rank_Price3Y", "Rank_Sales_TotalShare", "Rank_Profits_D",
    "Rank_Sales_D_Sec", "Rank_Price3Y_Sec", "Rank_Sales_D_Ind", "Rank_Price3Y_Ind",
    "Sales_D_SecShare", "Sales_D_IndShare", "OnGRNJ",
]


def load_etf_holdings(etf):
    """(set_of_tickers, source_description) for `etf` from etf_holdings.csv,
    falling back to _FALLBACKS[etf] if the file is missing or has no rows."""
    etf = etf.upper()
    if os.path.exists(ETF_HOLDINGS_PATH):
        try:
            h = pd.read_csv(ETF_HOLDINGS_PATH, converters={"Ticker": lambda v: str(v).strip()})
            h.columns = h.columns.str.strip()
            tickers = {t for t in h[h["ETF"].str.upper() == etf]["Ticker"] if t}
            if tickers:
                return tickers, f"etf_holdings.csv ({len(tickers)} tickers)"
        except Exception as e:
            print(f"  warning: could not read {ETF_HOLDINGS_PATH}: {e}")
    fallback = _FALLBACKS.get(etf, set())
    if fallback:
        return set(fallback), f"hardcoded fallback ({len(fallback)} tickers)"
    print(f"  warning: no holdings found for {etf} and no fallback configured")
    return set(), "unavailable"


def data_as_of(path):
    """Date the input CSV was last committed (the CI checkout resets mtimes, so
    git is the only honest 'data as of' signal); falls back to file mtime locally."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", path], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")


def load_input(path):
    # converters on Ticker: pandas otherwise parses the ticker "NA" as NaN
    df = pd.read_csv(path, converters={"Ticker": lambda v: str(v).strip()})
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["Ticker"].str.len() > 0].reset_index(drop=True)


def rank_desc(series):
    return series.rank(ascending=False, method="min", na_option="bottom")


def rank_desc_group(df, col, group):
    return df.groupby(group)[col].rank(ascending=False, method="min", na_option="bottom")


def compute_metrics(df):
    """Straight port of the notebook's metric cell — formulas mirror the xlsm."""
    df["Sales_D"] = df["Rev_LTM"] - df["Rev_LTM"] / (1 + df["Rev_Growth1Y"])

    ltm_eq_fy_rev = df["Rev_LTM"] == df["Rev_FY"]
    df["Sales_Pct"] = pd.to_numeric(np.where(
        ltm_eq_fy_rev, df["Rev_FY"] / df["Rev_1FY"] - 1, df["Rev_LTM"] / df["Rev_FY"] - 1), errors="coerce")

    ltm_eq_fy_ni = df["NI_LTM"] == df["NI_FY"]
    df["Profits_D"] = np.where(ltm_eq_fy_ni, df["NI_FY"] - df["NI_1FY"], df["NI_LTM"] - df["NI_FY"])
    df["Profits_Pct"] = pd.to_numeric(np.where(
        ltm_eq_fy_ni, df["NI_FY"] / df["NI_1FY"] - 1, df["NI_LTM"] / df["NI_FY"] - 1), errors="coerce")

    df["Sales_TotalShare"] = df["Rev_LTM"] / df["Rev_LTM"].sum()
    df["Profits_TotalShare"] = df["NI_LTM"] / df["NI_LTM"].sum()

    positive_sum = lambda x: x[x > 0].sum() if (x > 0).any() else np.nan
    df["Sales_D_SecShare"] = df["Sales_D"] / df.groupby("Sector")["Sales_D"].transform(positive_sum)
    df["Sales_D_IndShare"] = df["Sales_D"] / df.groupby("Industry")["Sales_D"].transform(positive_sum)

    for col in ["Sales_D", "Sales_Pct", "Profits_D", "Profits_Pct", "Sales_TotalShare", "Profits_TotalShare"]:
        df[f"Rank_{col}"] = rank_desc(df[col])
    for yrs in ["1Y", "3Y", "5Y", "10Y"]:
        df[f"Rank_Price{yrs}"] = rank_desc(df[f"PriceChg{yrs}"])

    for group, suffix in [("Sector", "Sec"), ("Industry", "Ind")]:
        df[f"Rank_Sales_D_{suffix}"] = rank_desc_group(df, "Sales_D", group)
        df[f"Rank_Sales_Pct_{suffix}"] = rank_desc_group(df, "Sales_Pct", group)
        df[f"Rank_Profits_D_{suffix}"] = rank_desc_group(df, "Profits_D", group)
        df[f"Rank_Price3Y_{suffix}"] = rank_desc_group(df, "PriceChg3Y", group)
        df[f"Rank_Price1Y_{suffix}"] = rank_desc_group(df, "PriceChg1Y", group)

    # Overall Q:  P = avg(elite Sales$ rank, elite Price3Y rank),  O = # top-N criteria hit,  Q = P / O
    elite = pd.DataFrame({c: df[c].where(df[c] <= N_ELITE) for c in [
        "Rank_Sales_D", "Rank_Sales_Pct", "Rank_Profits_D", "Rank_Profits_Pct",
        "Rank_Sales_TotalShare", "Rank_Profits_TotalShare",
        "Rank_Price1Y", "Rank_Price3Y", "Rank_Price5Y", "Rank_Price10Y"]})
    df["O"] = elite.notna().sum(axis=1)
    df["P"] = elite[["Rank_Sales_D", "Rank_Price3Y"]].mean(axis=1)
    df["Q"] = (df["P"] / df["O"]).where(df["O"] > 0)

    df["Score_Sector"] = df[["Rank_Sales_D_Sec", "Rank_Price3Y_Sec", "Rank_Sales_D"]].mean(axis=1)
    df["Score_Ind"] = df[["Rank_Sales_D_Ind", "Rank_Price3Y_Ind", "Rank_Sales_D_Sec"]].mean(axis=1)
    has_momentum = (df["PriceChg1Y"] > PRICE_THRESHOLD) | (df["PriceChg3Y"] > PRICE_THRESHOLD)
    df["Score_Sector_Adj"] = np.where(has_momentum, df["Score_Sector"], 1000)
    df["Score_Ind_Adj"] = np.where(has_momentum, df["Score_Ind"], 1000)
    return df, int(has_momentum.sum())


def fmt_pct(v):
    return f"{v:+.1%}" if pd.notna(v) else ""


def fmt_num(v):
    return f"{v:,.0f}" if pd.notna(v) else ""


def fmt_share(v):
    return f"{v:.1%}" if pd.notna(v) else ""


def _color_pct(val):
    try:
        v = float(str(val).replace("%", "").replace("+", "").replace(",", ""))
    except Exception:
        return ""
    if v >= 25:
        return f"color: {C_GREEN}; font-weight: bold"
    if v >= 5:
        return f"color: {C_LGREEN}"
    if v >= 0:
        return f"color: {C_NEUTRAL}"
    return f"color: {C_RED}"


def _color_rank(val, n_total):
    try:
        r = int(val)
    except Exception:
        return ""
    pct = r / max(n_total, 1)
    if pct <= 0.10:
        return f"color: {C_GREEN}; font-weight: bold"
    if pct <= 0.25:
        return f"color: {C_LGREEN}"
    if pct >= 0.75:
        return f"color: {C_RED}"
    return ""


def _check(v):
    return f"color: {C_GREEN}; font-weight: bold" if v == "✓" else ""


def _base_style(styler):
    return (
        styler
        .set_properties(**{"background-color": BG_CELL, "color": FG_MAIN, "border": BORDER,
                           "font-size": "12px", "text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", BG_HEADER), ("color", FG_HEADER),
                                         ("font-weight", "bold"), ("border", BORDER), ("text-align", "center")]},
            {"selector": "caption", "props": [("color", C_NEUTRAL), ("font-size", "11px"),
                                              ("text-align", "left"), ("padding-bottom", "6px")]},
        ])
        .hide(axis="index")
    )


def build_overall_html(df, grnj_source):
    n = len(df)
    overall = df.sort_values("Q", na_position="last").reset_index(drop=True)
    ranked_n = int(overall["Q"].notna().sum())
    top = overall.head(OVERALL_TOP_N)
    tbl = pd.DataFrame({
        "Rank": range(1, len(top) + 1),
        "Ticker": top["Ticker"], "Name": top["Name"], "Sector": top["Sector"],
        "Q Score": top["Q"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-"),
        "P": top["P"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-"),
        "O": top["O"].astype(int),
        "Sales$ Rank": top["Rank_Sales_D"].astype(int),
        "Price3Y Rank": top["Rank_Price3Y"].astype(int),
        "Price 1Y": top["PriceChg1Y"].map(fmt_pct), "Price 3Y": top["PriceChg3Y"].map(fmt_pct),
        "Sales $": top["Sales_D"].map(fmt_num), "Sales %": top["Sales_Pct"].map(fmt_pct),
        "Profits $": top["Profits_D"].map(fmt_num),
        "GRNJ": top["OnGRNJ"].map({True: "✓", False: ""}),
    })
    styled = (
        _base_style(tbl.style)
        .map(_color_pct, subset=["Price 1Y", "Price 3Y", "Sales %"])
        .map(lambda v: _color_rank(v, n), subset=["Sales$ Rank", "Price3Y Rank"])
        .map(_check, subset=["GRNJ"])
        .set_caption(
            f"Top {OVERALL_TOP_N} of {ranked_n} scored ({n - ranked_n} unranked) | "
            f"Q = AVERAGE(elite Sales$ rank, elite Price3Y rank) ÷ # top-{N_ELITE} criteria hit | "
            f"lower Q = better | GRNJ: {grnj_source}")
    )
    return styled.to_html()


def build_group_html(df, group, n_total):
    """One styled table for a single sector or industry (the notebook's dropdown views)."""
    is_sector = group == "Sector"
    sfx, other = ("Sec", "Industry") if is_sector else ("Ind", "Sector")
    score_adj, score = ("Score_Sector_Adj", "Score_Sector") if is_sector else ("Score_Ind_Adj", "Score_Ind")
    # third rank column: sectors show the global Sales$ rank, industries the sector Sales$ rank
    third_label, third_col = ("Global Sales$ Rank", "Rank_Sales_D") if is_sector else ("Sec Sales$ Rank", "Rank_Sales_D_Sec")

    g = df.sort_values(score_adj).reset_index(drop=True)
    n_g = len(g)
    tbl = pd.DataFrame({
        f"{sfx} Rank": range(1, n_g + 1),
        "Ticker": g["Ticker"], "Name": g["Name"], other: g[other],
        "Score (adj)": g[score_adj].map(lambda x: f"{x:.1f}" if x < 1000 else "n/a"),
        "Unadj Score": g[score].map(lambda x: f"{x:.1f}"),
        f"{sfx} Sales$ Rank": g[f"Rank_Sales_D_{sfx}"].astype(int),
        f"{sfx} Price3Y Rank": g[f"Rank_Price3Y_{sfx}"].astype(int),
        third_label: g[third_col].astype(int),
        "Price 1Y": g["PriceChg1Y"].map(fmt_pct), "Price 3Y": g["PriceChg3Y"].map(fmt_pct),
        "Sales $": g["Sales_D"].map(fmt_num), "Sales %": g["Sales_Pct"].map(fmt_pct),
        "Profits $": g["Profits_D"].map(fmt_num),
        f"Sales$% of {sfx}": g[f"Sales_D_{sfx}Share"].map(fmt_share),
        "MarketCap": g["MarketCap"].map(fmt_num),
        "GRNJ": g["OnGRNJ"].map({True: "✓", False: ""}),
    })
    # matches the notebook: sector tables colour the global rank against the whole universe,
    # industry tables colour the sector rank against the industry size
    third_n = n_total if is_sector else n_g
    styled = (
        _base_style(tbl.style)
        .map(_color_pct, subset=["Price 1Y", "Price 3Y", "Sales %"])
        .map(lambda v: _color_rank(v, n_g), subset=[f"{sfx} Sales$ Rank", f"{sfx} Price3Y Rank"])
        .map(lambda v: _color_rank(v, third_n), subset=[third_label])
        .map(_check, subset=["GRNJ"])
        .map(lambda v: f"color: {C_NEUTRAL}" if v == "n/a" else "", subset=["Score (adj)"])
        .set_caption(
            f"{g[group].iloc[0]} | {n_g} tickers | score penalises stocks with Price 1Y & 3Y "
            f"both &lt; {PRICE_THRESHOLD * 100:.0f}% (shown as n/a)")
    )
    return styled.to_html()


def build_dropdown_section(df, group, title, subtitle):
    """All per-group tables rendered up front, toggled by a <select> — the static
    stand-in for the notebook's ipywidgets dropdown."""
    key = group.lower()
    names = sorted(df[group].dropna().unique())
    options, panels = [], []
    for i, name in enumerate(names):
        options.append(f'<option value="{i}">{escape(name)}</option>')
        hidden = "" if i == 0 else " hidden"
        panels.append(f'<div class="table-wrap" data-{key}="{i}"{hidden}>'
                      f'{build_group_html(df[df[group] == name], group, len(df))}</div>')
    return (
        f'<div class="section"><h2>{title}</h2><div class="section-sub">{subtitle}</div>'
        f'<div class="picker"><label for="{key}Select">{group}:</label>'
        f'<select id="{key}Select" onchange="showGroup(\'{key}\', this.value)">{"".join(options)}</select>'
        f'</div></div>' + "\n".join(panels)
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_input(INPUT_CSV_PATH)
    as_of = data_as_of(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} tickers from {INPUT_CSV_PATH} (data as of {as_of})")
    print(f"  sectors: {df['Sector'].nunique()} | industries: {df['Industry'].nunique()}")

    df, n_momentum = compute_metrics(df)
    grnj, grnj_source = load_etf_holdings("GRNJ")
    df["OnGRNJ"] = df["Ticker"].isin(grnj)
    print(f"  {(df['O'] > 0).sum()} tickers in at least one top-{N_ELITE} list")
    print(f"  {n_momentum} tickers pass the >{PRICE_THRESHOLD * 100:.0f}% price momentum filter")
    print(f"  GRNJ: {grnj_source}, {df['OnGRNJ'].sum()} matched")

    parts = [
        f'<div class="section"><h2>Overall Ranking</h2><div class="section-sub">'
        f'Replicates cols S–W of Final Ranking (3) &middot; sorted by Q score</div></div>'
        f'<div class="table-wrap">{build_overall_html(df, grnj_source)}</div>',
        build_dropdown_section(df, "Sector", "Sector Ranking",
                               "Replicates cols AB–AK &middot; avg(sector Sales$ rank, sector Price3Y rank, global Sales$ rank)"),
        build_dropdown_section(df, "Industry", "Industry Ranking",
                               "Replicates cols AT–BC &middot; avg(industry Sales$ rank, industry Price3Y rank, sector Sales$ rank)"),
    ]

    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"), as_of=as_of,
                                n=len(df), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "Drives_Earnings_Sales_Screener.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")

    csv_path = os.path.join(OUTPUT_DIR, "drives_earnings_sales_2_35B_ranked.csv")
    df[[c for c in EXPORT_COLS if c in df.columns]].sort_values("Q", na_position="last").to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Drives Earnings &amp; Sales Screener</title>
<style>
  body {{ background: #0f1117; color: #c8ccd8; font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #2d3148; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; color: #e2e5f0; }}
  header .meta {{ color: #8b90a4; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #e2e5f0; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8b90a4; font-size: 13px; margin-top: 6px; }}
  .table-wrap {{ padding: 12px 32px 8px; overflow-x: auto; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  td:nth-child(3), th:nth-child(3) {{ text-align: left !important; }}
  caption {{ text-align: left; }}
  .picker {{ margin-top: 12px; font-size: 13px; }}
  .picker label {{ color: #8b90a4; margin-right: 6px; }}
  .picker select {{ background: #1a1d27; color: #e2e5f0; border: 1px solid #2d3148;
    border-radius: 4px; padding: 4px 10px; font-size: 13px; }}
  [hidden] {{ display: none !important; }}
</style>
<script>
function showGroup(key, idx) {{
  document.querySelectorAll('[data-' + key + ']').forEach(function (el) {{
    el.hidden = el.getAttribute('data-' + key) !== idx;
  }});
}}
</script>
</head>
<body>
<header>
  <h1>Drives Earnings &amp; Sales Screener &mdash; $2B&ndash;$35B</h1>
  <div class="meta">Generated {date_str} &middot; Input data as of {as_of} &middot; {n} tickers &middot; Replicates Final Ranking (3) of Drives Earnings and Sales - 35+.xlsm</div>
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
