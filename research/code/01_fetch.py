"""
01_fetch.py -- Download the raw daily series for the connectedness study.

Sources
-------
* Yahoo Finance chart API (adjclose: splits + dividends) for indices and ETFs.
* FRED fredgraph.csv (no API key) for the WTI spot price and the VIX.

All downloads are cached in research/data/raw/ so the analysis is fully
reproducible offline once the files exist.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/121.0 Safari/537.36")
UA_LINUX = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120 Safari/537.36")

YAHOO = {
    "^GSPC":  "US equities (S&P 500)",
    "^STOXX": "Europe equities (STOXX Europe 600)",
    "^N225":  "Japan equities (Nikkei 225)",
    "^NSEI":  "India equities (NIFTY 50)",
    "IEF":    "US Treasuries 7-10y (IEF, total return)",
    "GLD":    "Gold (GLD, total return)",
    "EEM":    "Emerging markets ex. listed (EEM, total return)",
}
FRED = {"DCOILWTICO": "Crude oil (WTI spot, Cushing)", "VIXCLS": "VIX"}


def _read(url: str, tries: int = 6) -> bytes:
    last = None
    for i in range(tries):
        req = urllib.request.Request(url, headers={
            "User-Agent": UA if i % 2 == 0 else UA_LINUX,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
        })
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(4.0 * (i + 1))
    raise RuntimeError(f"failed {url}: {last}")


def fetch_yahoo(ticker: str) -> pd.Series:
    path = os.path.join(RAW, f"yahoo_{ticker.replace('^', 'IDX_')}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    q = urllib.parse.quote(ticker, safe="")
    js = json.loads(_read(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{q}"
        f"?period1={int(pd.Timestamp('2003-01-01').timestamp())}"
        f"&period2={int(time.time())}&interval=1d&events=div%2Csplit"
    ))
    res = js["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(
        res["meta"].get("exchangeTimezoneName", "UTC")).normalize().tz_localize(None)
    ind = res["indicators"]
    vals = ind.get("adjclose", [{}])[0].get("adjclose") or ind["quote"][0]["close"]
    s = pd.Series(vals, index=idx, dtype="float64", name=ticker)
    s = s[~s.index.duplicated(keep="last")].dropna()
    s = s[s > 0]
    s.to_csv(path, header=[ticker])
    return s


def fetch_fred(series: str) -> pd.Series:
    path = os.path.join(RAW, f"fred_{series}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    df = pd.read_csv(io.BytesIO(_read(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"].astype(str).str.strip(), errors="coerce")
    s = df.dropna().set_index("date")["v"].sort_index()
    s.name = series
    s.to_csv(path, header=[series])
    return s


def main() -> None:
    out = {}
    for t in YAHOO:
        s = fetch_yahoo(t)
        out[t] = s
        print(f"{t:7s} {s.index.min().date()} -> {s.index.max().date()} n={len(s)}", flush=True)
        time.sleep(4.0)
    for f in FRED:
        s = fetch_fred(f)
        out[f] = s
        print(f"{f:11s} {s.index.min().date()} -> {s.index.max().date()} n={len(s)}", flush=True)
        time.sleep(1.0)
    common = None
    for s in out.values():
        common = s.index if common is None else common.intersection(s.index)
    print("\nintersection:", len(common), common.min().date(), "->", common.max().date())


if __name__ == "__main__":
    sys.exit(main())
