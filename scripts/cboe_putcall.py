"""Cboe put/call ratios (total, equity, index) from Cboe's free sources.

History: cdn.cboe.com/resources/options/volume_and_call_put_ratios/{totalpc,equitypc,indexpc}.csv (Nov 2006 - Oct 4 2019).
Since Oct 7 2019: Cboe daily market statistics JSON, one file per trading day. Days already fetched are cached in
outputs/vix-structure/cboe_putcall_cache.csv (committed by the workflow) so each run only fetches new days.
The older cdn.cboe.com/api/.../PC_History.csv link now returns 403 and is no longer used.
"""
import os
import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
OLD_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{}.csv"
DAY_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{}_daily_options"
KINDS = {"total": ("totalpc", "TOTAL PUT/CALL RATIO"),
         "equity": ("equitypc", "EQUITY PUT/CALL RATIO"),
         "index": ("indexpc", "INDEX PUT/CALL RATIO")}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO_ROOT, "outputs", "vix-structure", "cboe_putcall_cache.csv")


def _old(kind):
    r = requests.get(OLD_URL.format(KINDS[kind][0]), headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text.replace("\r", "")), skiprows=2)
    df.columns = [c.strip() for c in df.columns]
    s = pd.Series(pd.to_numeric(df["P/C Ratio"], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"].astype(str).str.strip(), errors="coerce"), name=kind)
    return s[s.index.notna()].dropna().sort_index()


def _day(d):
    try:
        r = requests.get(DAY_URL.format(d.strftime("%Y-%m-%d")), headers=HEADERS, timeout=20)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return d, None
        vals = {}
        for row in json.loads(r.text).get("ratios", []):
            for k, (_, label) in KINDS.items():
                if row.get("name") == label:
                    v = float(row["value"])
                    vals[k] = v if v > 0 else None
        return d, vals or None
    except Exception:
        return d, None


def load_putcall(start="2006-11-01", workers=8):
    """DataFrame with columns total, equity, index (daily Cboe put/call ratios)."""
    old = pd.concat([_old(k) for k in KINDS], axis=1)
    cache = pd.read_csv(CACHE, index_col=0, parse_dates=True) if os.path.exists(CACHE) else pd.DataFrame(columns=list(KINDS))
    today = pd.Timestamp.today().normalize()
    days = pd.bdate_range(old.index[-1] + pd.Timedelta(days=1), today)
    recent = today - pd.Timedelta(days=7)
    todo = [d for d in days if d not in cache.index or d >= recent]
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            got = {d: v for d, v in ex.map(_day, todo) if v}
        if got:
            new = pd.DataFrame.from_dict(got, orient="index")
            cache = pd.concat([cache[~cache.index.isin(new.index)], new]).sort_index()
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            cache.to_csv(CACHE, index_label="date")
        print(f"  Cboe put/call: fetched {len(got)} of {len(todo)} days; data through {cache.index.max().date()}")
    out = pd.concat([old, cache.astype(float)]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.loc[start:]


if __name__ == "__main__":
    df = load_putcall()
    print(df.tail())
