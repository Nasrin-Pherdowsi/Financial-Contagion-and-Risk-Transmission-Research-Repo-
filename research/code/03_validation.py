"""
03_validation.py -- Statistical audit of the connectedness engine.

Tests actually executed
-----------------------
A. Monte-Carlo validation of the generalized FEVD. For a synthetic VAR whose
   innovations are uncorrelated the generalized decomposition must collapse to
   the (unambiguous) orthogonal variance share, so we can compare the analytic
   formula against simulation.
B. Order invariance: TCI computed from Cholesky-orthogonalized FEVD under 2,000
   random variable orderings vs. the generalized (order-free) TCI.
C. Fast OLS VAR vs. statsmodels VAR (coefficients, residual covariance) on a
   real rolling window, plus a manual recomputation of TO/FROM/NET identities.
D. Data audit: coverage, gaps, duplicates, return moments, missing values.
E. Rolling-window reproducibility: recompute TCI for randomly chosen windows in
   a fresh process path and compare with the stored series.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
A = importlib.import_module("02_analysis")

OUT = A.OUT
rep = {}


# ---------------------------------------------------------------- A
def test_mc_fevd(n=3, p=2, H=10, T=200000, seed=11):
    rng = np.random.default_rng(seed)
    # stable random VAR with diagonal innovation covariance
    A1 = np.array([[0.35, 0.10, 0.05], [0.15, 0.40, 0.05], [0.05, 0.10, 0.30]])
    A2 = np.array([[0.10, 0.00, 0.00], [0.00, 0.10, 0.00], [0.05, 0.00, 0.10]])
    sig = np.array([1.0, 1.6, 0.7])          # diagonal, different scales
    S = np.diag(sig ** 2)
    # parameter matrix in the (intercept + coefficients) x equation layout
    Bmat = np.vstack([np.zeros((1, n)), np.vstack([A1, A2])])
    psis = A.ma_coeffs(Bmat, n, p, H)
    theta = A.gfevd(Bmat, S, n, p, H)

    # simulate orthogonal shocks and accumulate forecast-error variance by source
    eps = rng.standard_normal((T + H + 50, n)) * sig
    x = np.zeros((T + H + 50, n))
    for t in range(p, len(x)):
        x[t] = A1 @ x[t - 1] + A2 @ x[t - 2] + eps[t]
    start = 50  # leaves H extra observations at the end for the forecast terms
    # analytic attribution: variance contributed by shock j at step h
    contrib = np.zeros((n, n))
    tot = np.zeros(n)
    for h in range(H):
        P = psis[h]
        contrib += (P ** 2) * (sig ** 2)[None, :]
        tot += np.diag(P @ S @ P.T)
    analytic = contrib / tot[:, None]
    # Simulation-based attribution: the realised h-step forecast error of series
    # i is sum_j c_ij(t,h), with c_ij(t,h) = sum_{k<=h} Psi_{h-k}[i,j] eps_{t+k,j}.
    # With a diagonal covariance the components are uncorrelated, so the share of
    # forecast-error variance attributable to shock j is Var(c_j)/sum_j Var(c_j).
    # Compare at the terminal horizon, where both sides are cumulative: the
    # analytic denominator sums per-step increments Psi_h S Psi_h', and the
    # H-step forecast error already cumulates Psi_0..Psi_{H-1}.
    sim = np.zeros((n, n))
    for h in range(H):
        comp = np.zeros((T, n, n))                     # t, equation, shock
        for k in range(h + 1):
            Pm = psis[h - k]
            E = eps[start + k: start + k + T]          # T x n shocks
            comp += E[:, None, :] * Pm[None, :, :]     # T x n_eq x n_shock
        sim = comp.reshape(T, -1).var(axis=0).reshape(n, n)
    sim_share = sim / sim.sum(axis=1, keepdims=True)
    return dict(analytic=analytic.round(6).tolist(), simulated=sim_share.round(6).tolist(),
                max_abs_diff=float(np.abs(analytic - sim_share).max()),
                max_abs_diff_vs_formula=float(np.abs(analytic - theta).max()))


# ---------------------------------------------------------------- B
def test_ordering(r, p, H):
    y = r.values
    B, S, _ = A.fit_var(y, p)
    n = y.shape[1]
    psis = A.ma_coeffs(B, n, p, H)
    den = np.zeros(n)
    for P in psis:
        den += np.diag(P @ S @ P.T)
    rng = np.random.default_rng(3)
    tcis = []
    for _ in range(2000):
        order = rng.permutation(n)
        L = np.linalg.cholesky(S[np.ix_(order, order)])
        num = np.zeros((n, n))
        for Pm in psis:
            Pm_o = Pm[np.ix_(order, order)]
            M = Pm_o @ L
            num += M ** 2
        th = np.zeros((n, n))
        th[np.ix_(order, order)] = num / den[order][:, None]
        th = th / th.sum(axis=1, keepdims=True)
        off = th.copy(); np.fill_diagonal(off, 0)
        tcis.append(off.sum() / n * 100)
    tcis = np.array(tcis)
    gen = A.connectedness(A.gfevd(B, S, n, p, H))["tci"]
    return dict(generalized=float(gen), chol_min=float(tcis.min()),
                chol_mean=float(tcis.mean()), chol_max=float(tcis.max()),
                chol_sd=float(tcis.std()))


# ---------------------------------------------------------------- C
def test_against_statsmodels(r, p, H):
    from statsmodels.tsa.api import VAR
    w = 250
    seg = r.iloc[-w:]
    m = VAR(seg)
    res = m.fit(p, trend="c")
    y = seg.values
    B, S, E = A.fit_var(y, p)
    Bs = res.params.values            # statsmodels also puts 'const' first
    n = y.shape[1]
    k = 1 + n * p
    dof = (len(y) - p) / (len(y) - p - k)
    S_adj = S * dof                   # ML vs. degrees-of-freedom covariance
    tci_a = A.connectedness(A.gfevd(B, S, n, p, H))["tci"]
    tci_b = A.connectedness(A.gfevd(Bs, res.sigma_u.values, n, p, H))["tci"]
    return dict(max_coef_diff=float(np.abs(B - Bs).max()),
                max_sigma_diff_after_dof=float(np.abs(S_adj - res.sigma_u.values).max()),
                dof_factor=float(dof), tci_ml=float(tci_a), tci_statsmodels=float(tci_b),
                tci_abs_diff=float(abs(tci_a - tci_b)), window=w, p=p)


# ---------------------------------------------------------------- D
def data_audit(r, px):
    idx = r.index
    gaps = pd.Series(idx).diff().dt.days.dropna()
    zero = {c: float((r[c].abs() < 1e-10).mean()) for c in r.columns}
    return dict(
        n_obs=int(len(r)), start=str(idx.min().date()), end=str(idx.max().date()),
        n_duplicate_dates=int(idx.duplicated().sum()),
        n_missing=int(r.isna().sum().sum()),
        median_gap_days=float(gaps.median()), p95_gap_days=float(gaps.quantile(.95)),
        max_gap_days=float(gaps.max()),
        max_gap_date=str(idx[gaps.idxmax()].date()) if gaps.max() > 10 else "",
        n_gaps_over_7d=int((gaps > 7).sum()),
        sd=r.std().round(3).to_dict(), skew=r.skew().round(3).to_dict(),
        exkurt=r.kurt().round(2).to_dict(), zero_share={k: round(v, 4) for k, v in zero.items()},
        jb_p={c: float(stats.jarque_bera(r[c]).pvalue) for c in r.columns},
        corr=r.corr().round(3).to_dict(),
        mean_pairwise_corr=float((r.corr().sum().sum() - r.shape[1]) /
                                (r.shape[1] * (r.shape[1] - 1))),
        eq_block_corr=float((r[["US", "EU", "JP", "IN"]].corr().sum().sum() - 4) / 12),
        corr_us_in=float(r["US"].corr(r["IN"])),
        corr_us_vix=float(r["US"].corr(r["VIX"])),
    )


# ---------------------------------------------------------------- E
def test_reproducibility(r, roll, p, H, k=8, seed=5):
    rng = np.random.default_rng(seed)
    y = r.values
    n = y.shape[1]
    diffs = []
    for _ in range(k):
        end = int(rng.integers(250, len(y) + 1))
        seg = y[end - 250:end]
        B, S, _ = A.fit_var(seg, p)
        th = A.gfevd(B, S, n, p, H)
        tci = A.connectedness(th)["tci"]
        stored = roll.TCI.iloc[end - 250]
        diffs.append(abs(tci - stored))
    return dict(max_abs_diff=float(max(diffs)), k=k)


def main():
    px, r = A.build_panel()
    p = 1
    H = 10
    rep["A_mc_fevd"] = test_mc_fevd()
    rep["B_ordering"] = test_ordering(r, p, H)
    rep["C_statsmodels"] = test_against_statsmodels(r, p, H)
    rep["D_data"] = data_audit(r, px)
    roll = pd.read_csv(os.path.join(OUT, "rolling_main.csv"), index_col=0, parse_dates=True)
    rep["E_reproducibility"] = test_reproducibility(r, roll, p, H)

    # identities
    res = json.load(open(os.path.join(OUT, "results.json")))
    tbl = pd.DataFrame(res["static"]["table"]).set_index("index")
    rep["F_identities"] = dict(
        sum_to=float(tbl.TO.sum()), sum_from=float(tbl.FROM.sum()),
        n_times_tci=float(r.shape[1] * res["static"]["tci"]),
        sum_net=float(tbl.NET.sum()),
    )
    theta = pd.DataFrame(res["static"]["theta"])
    rep["F_identities"]["row_sum_max_dev"] = float(np.abs(theta.sum(axis=1) - 1).max())

    json.dump(rep, open(os.path.join(OUT, "validation.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "D_data"} | {
        "D_data_subset": {kk: rep["D_data"][kk] for kk in
                          ["n_obs", "start", "end", "median_gap_days", "p95_gap_days",
                           "max_gap_days", "n_gaps_over_7d", "n_duplicate_dates",
                           "n_missing", "mean_pairwise_corr", "eq_block_corr",
                           "corr_us_in", "corr_us_vix"]}}, indent=1))


if __name__ == "__main__":
    main()
