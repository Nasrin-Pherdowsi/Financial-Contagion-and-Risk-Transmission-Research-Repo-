# Cross-Market Financial Contagion

**Directional risk transmission across global equities, rates, commodities, and volatility**

This repository implements a reproducible connectedness framework for identifying which financial markets transmit shocks, which absorb them, and how the transmission structure changes across market regimes.

The study combines a multivariate VAR with generalized forecast-error variance decomposition, 250-day rolling estimation, stress-period inference, DCC-GARCH cross-validation, robustness testing, and publication-ready reporting.

---

## Executive Summary

The model produces a persistent shock hierarchy:

| Market | TO | FROM | NET | Role |
|---|---:|---:|---:|---|
| US equities | 87.11 | 58.79 | +28.32 | Net transmitter |
| European equities | 74.39 | 57.72 | +16.67 | Net transmitter |
| VIX | 58.25 | 50.92 | +7.33 | Net transmitter |
| US Treasuries | 22.01 | 27.58 | -5.56 | Net receiver |
| Gold | 6.57 | 8.64 | -2.07 | Net receiver |
| WTI oil | 15.15 | 21.69 | -6.54 | Net receiver |
| Indian equities | 28.72 | 41.07 | -12.36 | Net receiver |
| Japanese equities | 27.68 | 53.46 | -25.79 | Net receiver |

**Key findings**

- Full-sample Total Connectedness Index: **39.98%**
- Rolling TCI mean: **45.52%**
- Rolling TCI range: **31.53%–62.30%**
- Maximum connectedness: **62.30% on 16 March 2020**
- S&P 500 is a net transmitter in **100%** of rolling windows
- Nikkei 225 is a net receiver in **100%** of rolling windows
- Stress versus calm TCI difference: **+3.63 percentage points**
- Bootstrap confidence interval: **[1.19, 6.01] percentage points**
- Bootstrap p-value: **0.0032**

> The results indicate that crises increase the intensity of cross-market transmission while largely preserving the underlying shock hierarchy.

---

## Research Reports

The repository contains a concise research brief and an extended methodology note.

- [One-page research brief](paper_1page.pdf)
- [Two-page research brief](paper_2page.pdf)
- [Reproducibility report](reproducibility_report.md)
- [One-page PDF in research output](research/output/paper_1page.pdf)
- [Two-page PDF in research output](research/output/paper_2page.pdf)

The PDF reports provide the publication view of the results. The README documents the implementation and reproducible workflow.

---

## Investment Universe

| Code | Instrument | Economic Role |
|---|---|---|
| US | S&P 500 | US growth and developed-equity risk |
| EU | STOXX Europe 600 | European equity exposure |
| JP | Nikkei 225 | Japanese equity exposure |
| IN | NIFTY 50 | Indian equity exposure |
| UST | IEF | US 7–10-year Treasury duration |
| GLD | Gold ETF | Defensive precious-metal exposure |
| OIL | WTI front-month proxy | Energy and inflation expectations |
| VIX | CBOE Volatility Index | Equity risk premium and market stress |

The primary panel contains **4,195 matched observations** from **19 September 2007 to 19 November 2026**.

---

## Methodology

### 1. Data Acquisition

[01_fetch.py](research/code/01_fetch.py) downloads and caches:

- Yahoo Finance adjusted prices
- FRED WTI and VIX observations

[01b_fetch_french.py](research/code/01b_fetch_french.py) provides an optional alternative source using Kenneth French regional market factors and FRED series.

Raw files are retained in [research/data/raw](research/data/raw). Existing observations are reused, making the pipeline offline-capable after the initial data download.

### 2. Stationarity Transformation

The model uses percentage log returns:

\[
r_t = 100 \times \log\left(\frac{P_t}{P_{t-1}}\right)
\]

Where $P_t$ represents the end-of-day price or index level.

ADF and KPSS tests are run in [02_analysis.py](research/code/02_analysis.py). Stationarity tests and descriptive statistics are exported to `t1_stationarity.csv` and `results.json`.

### 3. Vector Autoregression

The baseline specification is:

\[
Y_t = c + A_1Y_{t-1} + \varepsilon_t
\]

A lag grid from 1 to 10 is evaluated using AIC, BIC, and Hannan–Quinn criteria.

The main model uses:

- VAR order: **1**
- Forecast horizon: **10**
- Rolling window: **250 trading days**
- Estimator: OLS

BIC selects VAR(1), which provides a stable and estimable specification for repeated 250-day windows.

The companion-matrix maximum eigenvalue is below one, indicating covariance-stationary dynamics.

### 4. Generalized Forecast-Error Variance Decomposition

The VAR is converted to an infinite moving-average representation:

\[
Y_t = \Psi_0\varepsilon_t +
      \Psi_1\varepsilon_{t-1} +
      \Psi_2\varepsilon_{t-2} + \cdots
\]

The generalized variance share is:

\[
\theta_{ij}(H)
=
\frac{
\sigma_{jj}^{-1}
\displaystyle\sum_{h=0}^{H-1}
\left(e_i'\Psi_h\Sigma e_j\right)^2
}{
\displaystyle\sum_{h=0}^{H-1}
e_i'\Psi_h\Sigma\Psi_h'e_i
}
\]

Rows are normalized to sum to one. This decomposition is insensitive to variable ordering and avoids the economically arbitrary ordering assumptions imposed by Cholesky orthogonalization.

### 5. Connectedness Metrics

For an $N$-market variance-share matrix:

\[
\mathrm{TO}_j
=
\sum_{i \ne j}\theta_{ij}
\times 100
\]

\[
\mathrm{FROM}_i
=
\sum_{j \ne i}\theta_{ij}
\times 100
\]

\[
\mathrm{NET}_i
=
\mathrm{TO}_i-\mathrm{FROM}_i
\]

\[
\mathrm{TCI}
=
\frac{100}{N}
\sum_{i \ne j}\theta_{ij}
\]

Definitions:

- **TO:** shock transmission to other markets
- **FROM:** shock reception from other markets
- **NET:** net transmitter or net receiver status
- **TCI:** aggregate cross-market connectedness

---

## Rolling Estimation

The rolling engine performs one VAR and generalized FEVD estimation for every 250-trading-day window.

```text
For each trading day t:
    data_t = Y[t-250:t]
    fit VAR(1)
    compute moving-average coefficients
    decompose 10-step forecast variance
    calculate TCI, TO, FROM, NET and pairwise flows
```

Outputs:

- [rolling_main.csv](research/output/rolling_main.csv)
- [theta_dyn.npy](research/output/theta_dyn.npy)
- [theta_dates.npy](research/output/theta_dates.npy)

Rolling TCI:

- Mean: **45.52%**
- Minimum: **31.53%**
- Maximum: **62.30%**
- Maximum date: **16 March 2020**
- Lag-1 autocorrelation: high due to overlapping estimation windows

---

## Stress-Regime Inference

The project uses four ex ante stress episodes:

| Episode | Period |
|---|---|
| Global financial crisis | 15 Sep 2008–09 Mar 2009 |
| Euro sovereign-debt crisis | 01 Jul 2011–30 Dec 2011 |
| COVID-19 shock | 19 Feb 2020–23 Mar 2020 |
| Inflation and rate shock | 03 Jan 2022–14 Oct 2022 |

Calm observations include a 20-trading-day cooldown. A calm rolling estimate is retained only when its entire estimation window is outside stress and cooldown regions.

Because rolling estimates overlap substantially, inference uses a:

- 20-observation circular block bootstrap
- 5,000 bootstrap repetitions
- Two-sided percentile confidence interval

The stress-versus-calm result is:

| Regime | Mean TCI |
|---|---:|
| Stress | 46.79% |
| Calm | 43.16% |
| Difference | +3.63 pp |

---

## Cross-Model Validation

### DCC-GARCH

The DCC-GARCH model provides an independent measure of dynamic dependence:

\[
Q_t
=
(1-a-b)\bar Q
+az_tz_t'
+bQ_{t-1}
\]

\[
R_t
=
\operatorname{diag}(Q_t)^{-1/2}
Q_t
\operatorname{diag}(Q_t)^{-1/2}
\]

The implementation first filters each return series through a GARCH(1,1)-t model, standardizes residuals, and then estimates DCC parameters.

Estimated parameters:

- $a = 0.176$
- $b = 0.811$
- Mean absolute dynamic correlation: **0.356**
- Equity-block dynamic correlation: **0.363**

### Independent Validation

[03_validation.py](research/code/03_validation.py) performs:

- Monte-Carlo validation of generalized FEVD
- 2,000 random-variable-ordering tests
- Agreement testing against statsmodels VAR
- Missing-data and duplicate-date checks
- Rolling-window reproducibility checks
- Connectedness accounting identity checks

The manual VAR implementation and statsmodels implementation produce identical coefficients and TCI estimates in the validation window.

---

## Robustness Analysis

The primary specifications are tested against:

| Sensitivity | Result |
|---|---|
| Forecast horizon H = 5 | TCI 39.98% |
| Forecast horizon H = 20 | TCI 39.98% |
| VAR lag p = 2 | TCI 40.21% |
| VAR lag p = 3 | TCI 40.39% |
| VAR lag p = 8 | TCI 41.14% |
| Rolling window W = 150 | Mean TCI 47.36% |
| Rolling window W = 400 | Mean TCI 44.28% |
| VIX excluded | TCI 33.87% |

Additional checks include:

- Leave-one-market-out estimation
- EEM replacing NIFTY as the emerging-market proxy
- Alternative 150- and 400-day windows
- Regime-specific VAR estimation
- Pairwise Granger-causality testing
- Bonferroni correction for multiple testing

The level of connectedness is sensitive to window and variable selection. The qualitative hierarchy remains stable: US and Europe transmit, while Japan and India receive.

---

## Reproducing the Analysis

### Environment

Verified environment:

- Python 3.12
- statsmodels 0.15.0
- arch 9.0.0

Create a virtual environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install numpy pandas scipy statsmodels arch matplotlib pillow reportlab pypdfium2
```

### Execution Pipeline

```powershell
# 1. Populate the raw data cache
python research/code/01_fetch.py

# 2. Optional: populate alternative French/FRED data
python research/code/01b_fetch_french.py

# 3. Estimate VARs, FEVD, rolling connectedness and robustness
python research/code/02_analysis.py

# 4. Run independent validation checks
python research/code/03_validation.py

# 5. Render PNG and PDF figures
python research/code/04_figures.py

# 6. Typeset the research reports
python research/code/06_build_pdfs.py
```

The pipeline is deterministic for the cached data and fixed random seeds.

---

## Output Reference

| File | Purpose |
|---|---|
| `t1_stationarity.csv` | ADF, KPSS and stationarity diagnostics |
| `t2_lag_selection.csv` | AIC, BIC and Hannan–Quinn lag comparison |
| `t3_connectedness.csv` | Full-sample TO, FROM and NET estimates |
| `t4_granger.csv` | Pairwise Granger-causality tests |
| `rolling_main.csv` | Rolling TCI and directional connectedness |
| `theta_dyn.npy` | Rolling variance-share matrices |
| `theta_dates.npy` | Dates corresponding to rolling matrices |
| `results.json` | Complete machine-readable analysis output |
| `validation.json` | Independent validation and audit output |
| `figures/` | Publication-quality figures |

---

## Figures

The reporting system produces five figures:

1. **Net spillover network** — static transmitter/receiver map
2. **Dynamic TCI** — rolling connectedness and stress episodes
3. **Persistent net roles** — transmitter and receiver ranking
4. **Stress-versus-calm variance shares** — regime comparison matrix
5. **Rolling net connectedness** — role stability over time

Figures are generated by [04_figures.py](research/code/04_figures.py) and exported as PNG and PDF.

---

## PDF Generation

[06_build_pdfs.py](research/code/06_build_pdfs.py) is the maintained report-generation system.

It:

- Reads `results.json`
- Generates equation images
- Uses ReportLab for A4 typesetting
- Builds measured two-column layouts
- Embeds the generated figures
- Verifies exact one-page and two-page output

At the root of the project:

- `paper_1page.pdf`
- `paper_2page.pdf`
