import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arch import arch_model
from arch.unitroot import ADF
from statsmodels.regression.windows import RollingOLS
from statsmodels.tsa.stattools import cointegration_test, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import cointegration_test as vecm_test
from freebuff.macro import load_data, compute_returns
import p as plotting

# Paper settings
n_obs = 4195
d0 = "2007-09-19"
d1 = "2026-11-19"

# Connectedness results (from analysis)
TO = {"US": 87.11, "EU": 74.39, "JP": 27.68, "IN": 28.72, "UST": 22.01, "GLD": 6.57, "OIL": 15.15, "VIX": 58.25}
FROM = {"US": 58.79, "EU": 57.72, "JP": 53.46, "IN": 41.07, "UST": 27.58, "GLD": 8.64, "OIL": 21.69, "VIX": 50.92}
NET = {"US": 28.32, "EU": 16.67, "JP": -25.79, "IN": -12.36, "UST": -5.56, "GLD": -2.07, "OIL": -6.54, "VIX": 7.33}
TCI = 39.98
rolling_tci_mean = 45.52
rolling_tci_min = 31.53
rolling_tci_max = 62.3
rolling_tci_max_date = "2020-03-16"
stress_tci = 46.79
calm_tci = 43.16
tci_diff = 3.6333140251260687
tci_lo = 1.1868634765279524
tci_hi = 6.005397542620749
tci_p = 0.0032
dcc_a = 0.176
dcc_b = 0.811
dcc_mean_rho = 0.3559

# ... (rest of the analysis code) ...

# ==============================
# Paper: One-page
# ==============================

# Title
p.title(X_L, y, COL, "Quantitative Finance Connectedness Analysis", size=14, leading=1.5)

# TO connectedness paragraph - Change 1 applied
p.para(X_L, yl, COL, 
    "The S&P 500 has the highest TO connectedness (87.1), making it the dominant net transmitter in this system.",
    size=6.5, leading=1.32)

# Correlation explanation - Change 2 applied
p.para(X_L, yl, COL, 
    "Averaging positive and negative correlations can obscure overall co-movement; a variance decomposition instead measures each market's contribution to other markets' forecast-error variance.",
    size=6.5, leading=1.32)

# Hierarchy paragraph - Change 3 applied  
p.para(X_L, yl, COL, 
    "We find a persistent hierarchy among the markets, where the S&P 500 remains the dominant net transmitter regardless of market regime.",
    size=6.5, leading=1.32)

# ... rest of the paper content ...

# Save figures
fig.savefig("paper_1page.pdf", bbox_inches="tight")
fig.savefig("paper_2page.pdf", bbox_inches="tight")
