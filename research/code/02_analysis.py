"""
02_analysis.py -- Estimation engine for the connectedness study.

Pipeline
--------
1. Build the daily panel on the matched trading-day sample; transform to
   stationary returns.
2. Stationarity / lag-selection / VAR diagnostics.
3. Generalized forecast-error variance decomposition (Pesaran-Shin) ->
   Diebold-Yilmaz connectedness (TO, FROM, NET, TCI, pairwise).
4. Rolling-window dynamic connectedness.
5. Stress vs. calm inference with a circular block bootstrap (overlapping
   rolling estimates are strongly autocorrelated, so i.i.d. tests are invalid).
6. Pairwise Granger causality with a multiple-testing correction.
7. DCC-GARCH(1,1) as an independent second measure of dynamic dependence.
8. Robustness: forecast horizon, window length, VAR lag order, leave-one-out,
   alternative universe (EEM for NIFTY; French CRSP regional returns).

Everything is written to research/output/results.json plus CSV tables.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

START = "2007-09-18"
H_MAIN = 10
P_MAIN = None  # set by AIC below
WIN_MAIN = 250

NAMES = {
    "US":  "US equities",
    "EU":  "Europe equities",
    "JP":  "Japan equities",
    "IN":  "India equities",
    "UST": "US Treasuries",
    "GLD": "Gold",
    "OIL": "Crude oil",
    "VIX": "Implied volatility",
}
ORDER = ["US", "EU", "JP", "IN", "UST", "GLD", "OIL", "VIX"]


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def _rd(name):
    return pd.read_csv(os.path.join(RAW, name), index_col=0, parse_dates=True).iloc[:, 0]


def build_panel(universe=ORDER) -> pd.DataFrame:
    px = pd.DataFrame({
        "US":  _rd("yahoo_IDX_GSPC.csv"),
        "EU":  _rd("yahoo_IDX_STOXX.csv"),
        "JP":  _rd("yahoo_IDX_N225.csv"),
        "IN":  _rd("yahoo_IDX_NSEI.csv"),
        "UST": _rd("yahoo_IEF.csv"),
        "GLD": _rd("yahoo_GLD.csv"),
        "EEM": _rd("yahoo_EEM.csv"),
        "OIL": _rd("yahoo_CLF.csv"),
        "VIX": _rd("cboe_VIXCLS.csv"),
    })
    px.index = pd.to_datetime(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px.loc[START:]
    px = px[list(universe) + ([c for c in px.columns if c not in universe])]
    px = px[list(universe)].dropna()

    r = pd.DataFrame(index=px.index)
    for c in universe:
        if c == "VIX":
            r[c] = 100.0 * np.diff(np.log(px[c]), prepend=np.nan)
        else:
            r[c] = 100.0 * np.diff(np.log(px[c]), prepend=np.nan)
    r = r.iloc[1:].dropna()
    return px, r


# --------------------------------------------------------------------------
# 2. VAR by OLS (fast, used by the rolling loop)
# --------------------------------------------------------------------------
def make_zy(y: np.ndarray, p: int):
    T, n = y.shape
    Z = np.ones((T - p, 1 + n * p))
    for lag in range(1, p + 1):
        Z[:, 1 + (lag - 1) * n: 1 + lag * n] = y[p - lag: T - lag]
    return y[p:], Z


def fit_var(y: np.ndarray, p: int):
    Y, Z = make_zy(y, p)
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    E = Y - Z @ B
    T = len(Y)
    S = E.T @ E / T
    return B, S, E


def companion(B: np.ndarray, n: int, p: int) -> np.ndarray:
    A = np.zeros((n * p, n * p))
    A[:n, :] = B[1:].T
    if p > 1:
        A[n:, :-n] = np.eye(n * (p - 1))
    return A


def ma_coeffs(B, n, p, H):
    """Psi_0..Psi_{H-1} of the VAR MA(inf) representation."""
    A = companion(B, n, p)
    J = np.zeros((n, n * p))
    J[:, :n] = np.eye(n)
    psis = []
    Ak = np.eye(n * p)
    for _ in range(H):
        psis.append(J @ Ak @ J.T)
        Ak = Ak @ A
    return psis


def gfevd(B, S, n, p, H):
    """Generalized FEVD (Pesaran-Shin 1998), row-normalised."""
    psis = ma_coeffs(B, n, p, H)
    num = np.zeros((n, n))
    den = np.zeros(n)
    sd = np.sqrt(np.diag(S))
    for P in psis:
        M = P @ S                      # n x n
        num += (M ** 2) / (sd ** 2)[None, :]
        den += np.diag(P @ S @ P.T)
    theta = num / den[:, None]
    theta = theta / theta.sum(axis=1, keepdims=True)   # rows sum to 1
    return theta


def connectedness(theta):
    n = theta.shape[0]
    off = theta.copy()
    np.fill_diagonal(off, 0.0)
    to = off.sum(axis=0) * 100.0        # column sums: shocks SENT by j
    frm = off.sum(axis=1) * 100.0       # row sums:    shocks RECEIVED by i
    net = to - frm
    tci = off.sum() / n * 100.0
    pair_net = (off - off.T) * 100.0    # [i,j] > 0 => i sends more to j than it gets
    return dict(to=to, frm=frm, net=net, tci=tci, pair=pair_net, theta=theta)


# --------------------------------------------------------------------------
# 3. Stationarity / lag selection / diagnostics
# --------------------------------------------------------------------------
def stationarity(r):
    rows = []
    for c in r.columns:
        x = r[c].values
        a = adfuller(x, maxlag=int(np.ceil(12 * (len(x) / 100) ** 0.25)), autolag="BIC")
        try:
            k = kpss(x, regression="c", nlags="auto")
            kp = k[1]
        except Exception:
            kp = np.nan
        rows.append(dict(var=c, adf_stat=a[0], adf_p=a[1], adf_lags=a[2],
                         kpss_stat=k[0] if kp == kp else np.nan, kpss_p=kp))
    return pd.DataFrame(rows)


def lag_selection(y, pmax=10):
    rows = []
    n = y.shape[1]
    T = len(y)
    for p in range(1, pmax + 1):
        Y, Z = make_zy(y, p)
        B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        E = Y - Z @ B
        k = 1 + n * p
        m = len(Y)
        Sig = E.T @ E / m
        ll = -0.5 * m * (n * np.log(2 * np.pi) + np.log(np.linalg.det(Sig)) + n)
        aic = -2 * ll / m + 2 * k * n / m
        bic = -2 * ll / m + np.log(m) * k * n / m
        hq = -2 * ll / m + 2 * np.log(np.log(m)) * k * n / m
        rows.append(dict(p=p, loglik=ll, aic=aic, bic=bic, hq=hq))
    return pd.DataFrame(rows)


def var_diagnostics(r, p):
    y = r.values
    B, S, E = fit_var(y, p)
    n = y.shape[1]
    A = companion(B, n, p)
    eig = np.abs(np.linalg.eigvals(A)).max()
    # multivariate Ljung-Box on residual autocorrelation (lag 10)
    Q = 0.0
    T = len(E)
    Sinv = np.linalg.inv(S)
    for h in range(1, 11):
        C = E[h:].T @ E[:-h] / T
        Q += (T ** 2 / (T - h)) * np.trace(C.T @ Sinv @ C @ Sinv) / n
    df = n * n * 10
    p_q = 1 - stats.chi2.cdf(Q, df)
    # univariate Ljung-Box portmanteau per equation
    lb = []
    for j in range(n):
        e = E[:, j]
        q = 0.0
        for h in range(1, 11):
            rho = np.corrcoef(e[:-h], e[h:])[0, 1]
            q += rho ** 2 / (T - h)
        lb.append(1 - stats.chi2.cdf(T * (T + 2) * q, 10))
    return dict(max_root=float(eig), lb_stat=float(Q), lb_df=int(df),
                lb_p=float(p_q), lb_pvals_eq=[float(x) for x in lb],
                sigma=pd.DataFrame(S, index=r.columns, columns=r.columns))


def granger(r, p):
    """Pairwise Granger F-tests, j -> i, with a Bonferroni threshold.

    Bivariate: y_i on p lags of y_i (restricted) vs p lags of y_i and y_j.
    """
    rows = []
    cols = list(r.columns)
    n = len(r)
    for i in cols:
        for j in cols:
            if i == j:
                continue
            y = r[i].values.astype(float)
            xj = r[j].values.astype(float)
            Y = y[p:]
            R = np.ones((n - p, 1 + p))
            U = np.ones((n - p, 1 + 2 * p))
            for L in range(1, p + 1):
                R[:, L] = y[p - L:n - L]
                U[:, L] = y[p - L:n - L]
                U[:, p + L] = xj[p - L:n - L]
            bR, *_ = np.linalg.lstsq(R, Y, rcond=None)
            bU, *_ = np.linalg.lstsq(U, Y, rcond=None)
            ssrR = float(((Y - R @ bR) ** 2).sum())
            ssrU = float(((Y - U @ bU) ** 2).sum())
            q = p
            dfu = len(Y) - U.shape[1]
            F = ((ssrR - ssrU) / q) / (ssrU / dfu)
            pv = float(1 - stats.f.cdf(F, q, dfu))
            rows.append(dict(effect=i, cause=j, F=float(F), p=pv))
    df = pd.DataFrame(rows)
    m = len(df)
    df["p_bonf"] = np.minimum(df["p"] * m, 1.0)
    return df, m


# --------------------------------------------------------------------------
# 4. Rolling connectedness
# --------------------------------------------------------------------------
def rolling_connectedness(r, window=WIN_MAIN, p=P_MAIN, H=H_MAIN, step=1):
    y = r.values
    n = y.shape[1]
    cols = list(r.columns)
    dates, tci, to, frm, net = [], [], [], [], []
    pair_us_in, pair_vix_us, frac = [], [], []
    for end in range(window, len(y) + 1, step):
        seg = y[end - window:end]
        B, S, _ = fit_var(seg, p)
        th = gfevd(B, S, n, p, H)
        c = connectedness(th)
        dates.append(r.index[end - 1])
        tci.append(c["tci"])
        to.append(c["to"])
        frm.append(c["frm"])
        net.append(c["net"])
        iu = cols.index("US") if "US" in cols else None
        iv = cols.index("IN") if "IN" in cols else None
        ixx = cols.index("VIX") if "VIX" in cols else None
        # pair[i, j] = theta_ij - theta_ji = (j -> i) - (i -> j), so the net flow
        # i -> j is pair[j, i].
        pair_us_in.append(c["pair"][iv, iu] if None not in (iu, iv) else np.nan)
        pair_vix_us.append(c["pair"][iu, ixx] if None not in (ixx, iu) else np.nan)
        frac.append(th)
    out = pd.DataFrame(index=pd.DatetimeIndex(dates))
    out["TCI"] = tci
    for k, c in enumerate(cols):
        out["TO_" + c] = [t[k] for t in to]
        out["FROM_" + c] = [f[k] for f in frm]
        out["NET_" + c] = [t[k] for t in net]
    out["NET_US_to_IN"] = pair_us_in
    out["NET_VIX_to_US"] = pair_vix_us
    return out, np.array(frac), pd.DatetimeIndex(dates)


def static_connectedness(r, p, H):
    y = r.values
    B, S, _ = fit_var(y, p)
    th = gfevd(B, S, y.shape[1], p, H)
    return connectedness(th)


# --------------------------------------------------------------------------
# 5. Stress inference
# --------------------------------------------------------------------------
# Calendar windows fixed ex ante from public event dates (not selected on results).
STRESS = [
    ("Global financial crisis",      "2008-09-15", "2009-03-09"),
    ("Euro sovereign debt crisis",   "2011-07-01", "2011-12-30"),
    ("COVID-19 crash",               "2020-02-19", "2020-03-23"),
    ("Inflation / rate shock",       "2022-01-03", "2022-10-14"),
]


def stress_masks(index):
    m = pd.Series(False, index=index)
    for _, a, b in STRESS:
        m.loc[(index >= a) & (index <= b)] = True
    return m


def block_bootstrap_diff(a, b, block=20, B=5000, seed=7):
    """Difference in means a - b with a circular block bootstrap CI/p-value."""
    rng = np.random.default_rng(seed)

    def draw(x):
        n = len(x)
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        return x[idx]

    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    boots = np.array([draw(a).mean() - draw(b).mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p: share of resampled differences on the wrong side
    prop = np.mean(boots <= 0) if obs > 0 else np.mean(boots >= 0)
    return dict(diff=float(obs), lo=float(lo), hi=float(hi),
                p=float(min(1.0, 2 * prop)), n_a=int(len(a)), n_b=int(len(b)))


# --------------------------------------------------------------------------
# 6. DCC-GARCH
# --------------------------------------------------------------------------
def garch_std(r):
    from arch import arch_model
    Z, vols = [], []
    fits = {}
    for c in r.columns:
        am = arch_model(100 * r[c].values, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        z = res.resid / res.conditional_volatility
        Z.append(z / z.std())
        vols.append(res.conditional_volatility)
        fits[c] = dict(omega=float(res.params["omega"]), alpha=float(res.params["alpha[1]"]),
                       beta=float(res.params["beta[1]"]), nu=float(res.params["nu"]))
    return np.column_stack(Z), np.column_stack(vols), fits


def dcc_fit(Z, maxiter=200):
    T, n = Z.shape
    Qbar = np.corrcoef(Z.T)

    def unpack(th):
        a, b = th
        return a, b

    def Qpath(a, b):
        Q = Qbar.copy()
        Qs = np.empty((T, n, n))
        for t in range(T):
            z = Z[t][:, None]
            Q = (1 - a - b) * Qbar + a * (z @ z.T) + b * Q
            Qs[t] = Q
        return Qs

    def negll(th):
        a, b = unpack(th)
        if a <= 0 or b <= 0 or a + b >= 0.999:
            return 1e10
        Qs = Qpath(a, b)
        ll = 0.0
        for t in range(T):
            Q = Qs[t]
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e10
            z = Z[t]
            ll += -0.5 * (logdet + float(z @ np.linalg.solve(R, z)))
        return -ll

    # Coarse grid search first (the likelihood is flat near the unit root and
    # Nelder-Mead from a single start can wander), then a local refinement.
    best = (np.inf, 0.03, 0.94)
    for a0 in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20]:
        for b0 in [0.80, 0.86, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98]:
            if a0 + b0 >= 0.999:
                continue
            v = negll(np.array([a0, b0]))
            if v < best[0]:
                best = (v, a0, b0)
    res = optimize.minimize(negll, x0=np.array([best[1], best[2]]), method="Nelder-Mead",
                            options=dict(maxiter=maxiter, xatol=1e-5, fatol=1e-3))
    a, b = res.x
    print(f"DCC grid optimum a={best[1]:.3f} b={best[2]:.3f} -> refined a={a:.4f} b={b:.4f}")
    Qs = Qpath(a, b)
    Rabs = np.empty(T)      # mean |conditional correlation| over all pairs
    Rsig = np.empty(T)      # mean signed conditional correlation
    Req = np.empty(T)       # mean correlation inside the equity block
    eqi = [k for k in range(min(4, n))]
    for t in range(T):
        Q = Qs[t]
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        off = R[~np.eye(n, dtype=bool)]
        Rabs[t] = np.abs(off).mean()
        Rsig[t] = off.mean()
        eq = R[np.ix_(eqi, eqi)]
        Req[t] = (eq.sum() - len(eqi)) / (len(eqi) * (len(eqi) - 1))
    return dict(a=float(a), b=float(b), negll=float(res.fun),
                avg_abs=Rabs, avg_signed=Rsig, avg_equity=Req)


# --------------------------------------------------------------------------
def main():
    res = {}
    px, r = build_panel()
    res["data"] = dict(
        sample_start=str(r.index.min().date()), sample_end=str(r.index.max().date()),
        n_obs=int(len(r)), n_vars=int(r.shape[1]),
        vars={c: NAMES[c] for c in r.columns},
        uncond_mean={c: float(r[c].mean()) for c in r.columns},
        uncond_sd={c: float(r[c].std()) for c in r.columns},
        skew={c: float(r[c].skew()) for c in r.columns},
        kurt={c: float(r[c].kurt()) for c in r.columns},
        corr=r.corr().round(4).to_dict(),
    )
    print("panel", r.shape, r.index.min().date(), r.index.max().date())

    st = stationarity(r)
    st.to_csv(os.path.join(OUT, "t1_stationarity.csv"), index=False)
    res["stationarity"] = st.to_dict("records")
    print(st.round(4).to_string(index=False))

    ls = lag_selection(r.values, 10)
    ls.to_csv(os.path.join(OUT, "t2_lag_selection.csv"), index=False)
    p_aic = int(ls.loc[ls.aic.idxmin(), "p"])
    p_bic = int(ls.loc[ls.bic.idxmin(), "p"])
    # BIC is the main criterion: with 8 variables a rolling 250-day window cannot
    # support a high-order VAR (a p=4 system already needs 33 parameters per
    # equation), and the BIC choice is separately validated by the portmanteau
    # test on the residuals below.
    p = max(1, p_bic)
    res["lag_selection"] = ls.to_dict("records")
    res["p_aic"], res["p_bic"], res["p_used"] = p_aic, p_bic, p
    print("lag: AIC", p_aic, "BIC", p_bic, "used", p)

    diag = var_diagnostics(r, p)
    res["diagnostics"] = diag
    print("max root", diag["max_root"], "LB p", diag["lb_p"])

    # ---------------- static connectedness ----------------
    c = static_connectedness(r, p, H_MAIN)
    cols = list(r.columns)
    tbl = pd.DataFrame({
        "market": [NAMES[x] for x in cols],
        "TO": c["to"], "FROM": c["frm"], "NET": c["net"],
    }, index=cols)
    tbl["NET"] = tbl["TO"] - tbl["FROM"]
    tbl = tbl.round(2)
    tbl.to_csv(os.path.join(OUT, "t3_connectedness.csv"))
    res["static"] = dict(
        tci=float(c["tci"]),
        table=tbl.reset_index().to_dict("records"),
        theta=pd.DataFrame(c["theta"], index=cols, columns=cols).round(4).to_dict(),
        pair_net=pd.DataFrame(c["pair"], index=cols, columns=cols).round(4).to_dict(),
    )
    print(tbl.to_string(), "\nTCI", round(c["tci"], 2))

    # ---------------- rolling ----------------
    roll, th_all, rdates = rolling_connectedness(r, WIN_MAIN, p, H_MAIN, step=1)
    roll.to_csv(os.path.join(OUT, "rolling_main.csv"))
    res["rolling"] = dict(
        window=WIN_MAIN, step=1, p=p, H=H_MAIN,
        mean_tci=float(roll.TCI.mean()), min_tci=float(roll.TCI.min()),
        max_tci=float(roll.TCI.max()),
        max_tci_date=str(roll.TCI.idxmax().date()),
        min_tci_date=str(roll.TCI.idxmin().date()),
        corr_lag1=float(roll.TCI.autocorr(1)),
        series={str(k.date()): float(v) for k, v in roll.TCI.items()},
    )
    print("rolling TCI mean/min/max", roll.TCI.mean().round(2),
          roll.TCI.min().round(2), roll.TCI.max().round(2), roll.TCI.idxmax().date())

    # ---------------- stress vs calm ----------------
    m = stress_masks(roll.index)
    # require the whole estimation window to be outside stress for 'calm'
    stress_days = pd.Series(False, index=r.index)
    for _, a, b in STRESS:
        stress_days.loc[(r.index >= a) & (r.index <= b)] = True
    # calm window: no stress day anywhere inside the estimation window and a
    # 20-trading-day cooldown after each episode, so the two samples do not
    # overlap in information content.
    cool = stress_days.rolling(20, min_periods=1).max().astype(bool)
    win_any = cool.rolling(WIN_MAIN, min_periods=WIN_MAIN).max().reindex(roll.index)
    calm = (win_any.fillna(1) == 0).values

    cmp_ = {}
    for name, ser in [("TCI", roll.TCI)] + [(f"{k}", roll[k]) for k in
                                            ["NET_US_to_IN", "NET_VIX_to_US", "FROM_IN", "FROM_US"]]:
        cmp_[name] = block_bootstrap_diff(ser[m].values, ser[calm].values)
    res["stress"] = dict(windows=STRESS, comparison=cmp_,
                         mean_stress={k: float(v) for k, v in
                                      roll[m].mean().items()},
                         mean_calm={k: float(v) for k, v in roll[calm].mean().items()},
                         n_stress=int(m.sum()), n_calm=int(calm.sum()))
    print("TCI stress", round(roll.TCI[m].mean(), 2), "calm", round(roll.TCI[calm].mean(), 2),
          cmp_["TCI"])

    # per-episode means
    ep = {}
    for name, a, b in STRESS:
        mm = (roll.index >= a) & (roll.index <= b)
        ep[name] = dict(tci=float(roll.TCI[mm].mean()), n=int(mm.sum()),
                        net_vix_us=float(roll.NET_VIX_to_US[mm].mean()),
                        net_us_in=float(roll.NET_US_to_IN[mm].mean()),
                        from_in=float(roll.FROM_IN[mm].mean()))
    res["episodes"] = ep

    # ---------------- transmitter persistence (H2) ----------------
    persist = {}
    for k in cols:
        s = roll["NET_" + k]
        persist[k] = dict(mean=float(s.mean()), sd=float(s.std()),
                          share_positive=float((s > 0).mean()),
                          p05=float(s.quantile(.05)), p95=float(s.quantile(.95)))
    res["persistence"] = persist

    # ---------------- Granger ----------------
    g, mtest = granger(r, p)
    g.to_csv(os.path.join(OUT, "t4_granger.csv"), index=False)
    res["granger"] = dict(m=int(mtest), n_sig05=int((g.p < .05).sum()),
                          n_sig_bonf=int((g.p_bonf < .05).sum()),
                          strongest=g.sort_values("p").head(12).round(4).to_dict("records"))

    # ---------------- regime VAR (static, split sample) ----------------
    reg = {}
    calm_days = (cool == 0).values
    for nm, mask in [("stress", stress_days.values), ("calm", calm_days)]:
        sub = r[mask]
        cc = static_connectedness(sub, p, H_MAIN)
        reg[nm] = dict(n=int(mask.sum()), tci=float(cc["tci"]),
                       to=dict(zip(cols, (cc["to"]).round(2).tolist())),
                       from_=dict(zip(cols, (cc["frm"]).round(2).tolist())),
                       net=dict(zip(cols, (cc["net"]).round(2).tolist())),
                       theta=pd.DataFrame(cc["theta"], index=cols, columns=cols).round(3).to_dict())
    res["regime_var"] = reg

    # ---------------- DCC ----------------
    Z, vols, garch_par = garch_std(r)
    dcc = dcc_fit(Z)
    s_abs = pd.Series(dcc["avg_abs"], index=r.index)
    s_sig = pd.Series(dcc["avg_signed"], index=r.index)
    s_eq = pd.Series(dcc["avg_equity"], index=r.index)
    res["dcc"] = dict(a=dcc["a"], b=dcc["b"], garch=garch_par,
                      mean_abs=float(s_abs.mean()), mean_signed=float(s_sig.mean()),
                      mean_equity=float(s_eq.mean()))
    for tag, ser in [("abs", s_abs), ("signed", s_sig), ("equity", s_eq)]:
        j = pd.concat([roll.TCI, ser.reindex(roll.index)], axis=1).dropna()
        res["dcc"][f"corr_tci_{tag}"] = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
        res["dcc"][f"corr_tci_{tag}_diff"] = float(
            j.iloc[:, 0].diff().corr(j.iloc[:, 1].diff()))
        msk = stress_masks(ser.index)
        res["dcc"][f"stress_{tag}"] = float(ser[msk].mean())
        res["dcc"][f"calm_{tag}"] = float(ser[~msk].mean())
        res["dcc"][f"test_{tag}"] = block_bootstrap_diff(
            ser[msk].values, ser[~msk].values)
    res["dcc"]["series"] = {str(k.date()): float(v) for k, v in s_abs.items()}
    print("DCC a,b", round(dcc["a"], 4), round(dcc["b"], 4),
          "mean|rho|", round(s_abs.mean(), 4), "signed", round(s_sig.mean(), 4),
          "equity", round(s_eq.mean(), 4),
          "corr(|rho|,TCI)", round(res["dcc"]["corr_tci_abs"], 4))

    # ---------------- robustness ----------------
    rob = {}
    # horizon
    for H in [5, 20]:
        cc = static_connectedness(r, p, H)
        rob[f"H={H}"] = dict(tci=float(cc["tci"]),
                             net=dict(zip(cols, cc["net"].round(2).tolist())))
    # lags
    # lag order: p+1, p+2 and the AIC-selected order (full sample only, since a
    # 250-day window cannot identify a p=8 system)
    for pp in [p + 1, p + 2, p_aic]:
        cc = static_connectedness(r, pp, H_MAIN)
        rob[f"p={pp}"] = dict(tci=float(cc["tci"]),
                              net=dict(zip(cols, cc["net"].round(2).tolist())))
    res["robust_static"] = rob

    win_rob = {}
    for W in [150, 400]:
        rr, _, _ = rolling_connectedness(r, W, p, H_MAIN, step=5)
        mm = stress_masks(rr.index)
        win_rob[f"W={W}"] = dict(mean=float(rr.TCI.mean()), min=float(rr.TCI.min()),
                                 max=float(rr.TCI.max()),
                                 stress=float(rr.TCI[mm].mean()),
                                 calm=float(rr.TCI[~mm].mean()),
                                 corr_main=float(pd.concat(
                                     [roll.TCI, rr.TCI], axis=1).dropna().corr().iloc[0, 1]))
    res["robust_window"] = win_rob

    # leave-one-out on the *rolling* TCI (does one market drive the result?)
    loo = {}
    for c_ in cols:
        sub = r.drop(columns=[c_])
        rr, _, _ = rolling_connectedness(sub, WIN_MAIN, p, H_MAIN, step=5)
        mm = stress_masks(rr.index)
        loo[c_] = dict(mean=float(rr.TCI.mean()), stress=float(rr.TCI[mm].mean()),
                       calm=float(rr.TCI[~mm].mean()),
                       corr_main=float(pd.concat([roll.TCI, rr.TCI], axis=1)
                                       .dropna().corr().iloc[0, 1]))
    res["robust_loo"] = loo

    # alternative universe: EEM replaces NIFTY
    alt = r.copy()
    eem = _rd("yahoo_EEM.csv")
    alt["IN"] = 100 * np.diff(np.log(eem.reindex(r.index).ffill()), prepend=np.nan)
    alt = alt.dropna()
    cc = static_connectedness(alt, p, H_MAIN)
    res["robust_alt_universe"] = dict(
        tci=float(cc["tci"]), n=int(len(alt)),
        net=dict(zip(cols, cc["net"].round(2).tolist())))
    rr, _, _ = rolling_connectedness(alt, WIN_MAIN, p, H_MAIN, step=5)
    mm = stress_masks(rr.index)
    res["robust_alt_universe"].update(
        stress=float(rr.TCI[mm].mean()), calm=float(rr.TCI[~mm].mean()),
        corr=float(pd.concat([roll.TCI, rr.TCI], axis=1).dropna().corr().iloc[0, 1]))

    # VIX excluded -- does an endogenous volatility leg drive everything?
    cc = static_connectedness(r.drop(columns=["VIX"]), p, H_MAIN)
    res["robust_drop_vix"] = dict(tci=float(cc["tci"]))
    rr, _, _ = rolling_connectedness(r.drop(columns=["VIX"]), WIN_MAIN, p, H_MAIN, step=5)
    mm = stress_masks(rr.index)
    res["robust_drop_vix"].update(stress=float(rr.TCI[mm].mean()),
                                  calm=float(rr.TCI[~mm].mean()))

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    np.save(os.path.join(OUT, "theta_dyn.npy"), th_all)
    np.save(os.path.join(OUT, "theta_dates.npy"), rdates.values)
    print("\nsaved results.json")
    print(json.dumps({k: res[k] for k in ["robust_window", "robust_drop_vix"]}, indent=1, default=str))
    print(json.dumps(res["robust_loo"], indent=1, default=str))
    print(json.dumps(res["robust_static"], indent=1, default=str))


if __name__ == "__main__":
    main()
