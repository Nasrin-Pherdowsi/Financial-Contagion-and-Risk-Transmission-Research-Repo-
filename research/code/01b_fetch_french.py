"""
01b_fetch_french.py -- Download regional equity market returns from the
Kenneth R. French Data Library (daily) and macro/market series from FRED.

Design note
-----------
All equity legs are value-weighted market returns expressed in US dollars,
which puts every cross-border equity series on a common currency footing and
removes local-currency inflation of returns. Bond, oil and volatility legs come
from FRED (Board of Governors / CBOE / EIA series).
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import urllib.request
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

FRENCH = {
    # local name -> (zip file on the French server, column to keep)
    "US":  ("F-F_Research_Data_Factors_daily_CSV.zip", "Mkt-RF"),
    "EU":  ("Europe_3_Factors_Daily_CSV.zip", "Mkt-RF"),
    "JP":  ("Japan_3_Factors_Daily_CSV.zip", "Mkt-RF"),
    "EM":  ("Emerging_5_Factors_Daily_CSV.zip", "Mkt-RF"),
}
FRED = ["DGS10", "DCOILWTICO", "VIXCLS"]


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_french_daily(blob: bytes, col: str) -> pd.Series:
    """The daily French files hold several tables; the first is the factor table."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("latin-1")
    lines = text.splitlines()
    # header line holding the column names
    hi = next(i for i, ln in enumerate(lines) if ln.strip().startswith(",") and col in ln)
    header = [h.strip() for h in lines[hi].split(",")]
    ci = header.index(col)
    rows = []
    for ln in lines[hi + 1:]:
        s = ln.strip()
        if not s or not re.match(r"^\d{8},", s):
            break
        parts = s.split(",")
        rows.append((parts[0], parts[ci]))
    s = pd.Series(
        [float(v) / 100.0 for _, v in rows],
        index=pd.to_datetime([d for d, _ in rows], format="%Y%m%d"),
        name=col,
    )
    return s.sort_index()


def fetch_french(key: str, fname: str, col: str) -> pd.Series:
    path = os.path.join(RAW, f"french_{key}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{fname}"
    blob = _get(url)
    s = parse_french_daily(blob, col)
    s.to_csv(path, header=[key])
    return s


def fetch_fred(series: str) -> pd.Series:
    path = os.path.join(RAW, f"fred_{series}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    df = pd.read_csv(io.BytesIO(_get(url)))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"].astype(str).str.strip(), errors="coerce")
    s = df.dropna().set_index("date")["v"].sort_index()
    s.name = series
    s.to_csv(path, header=[series])
    return s


def main() -> None:
    out = {}
    for k, (fn, col) in FRENCH.items():
        s = fetch_french(k, fn, col)
        out[k] = s
        print(f"{k:3s} {s.index.min().date()} -> {s.index.max().date()} n={len(s)}", flush=True)
        time.sleep(1.0)
    for f in FRED:
        s = fetch_fred(f)
        out[f] = s
        print(f"{f:11s} {s.index.min().date()} -> {s.index.max().date()} n={len(s)}", flush=True)
    common = None
    for s in out.values():
        common = s.index if common is None else common.intersection(s.index)
    print("\nintersection:", len(common), common.min().date(), "->", common.max().date())


if __name__ == "__main__":
    sys.exit(main())
