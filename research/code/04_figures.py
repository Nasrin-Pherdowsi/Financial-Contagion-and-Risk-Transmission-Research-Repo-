"""
04_figures.py -- Figure generation from the estimated connectedness objects.

fig1  hero network of net directional spillovers (full sample)
fig2  dynamic total connectedness + DCC-GARCH cross-check
fig3  transmitter / receiver ranking with persistence
fig4  calm vs. stress variance-share matrices and their difference
fig5  rolling net connectedness: stability of the architecture
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

INK = "#16222E"
NAVY = "#1F3A5F"
TRANSMIT = "#C0503C"
RECEIVE = "#2A7F79"
NEUTRAL = "#8B98A5"
GRID = "#DDE3E8"
PAPER = "#FFFFFF"

SHORT = {"US": "S&P 500", "EU": "STOXX 600", "JP": "Nikkei 225", "IN": "NIFTY 50",
         "UST": "US 7–10y", "GLD": "Gold", "OIL": "WTI", "VIX": "VIX"}
GROUP = {"US": "equity", "EU": "equity", "JP": "equity", "IN": "equity",
         "UST": "safe haven", "GLD": "safe haven", "OIL": "real asset", "VIX": "risk gauge"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "font.size": 8,
    "savefig.dpi": 320,
})

R = json.load(open(os.path.join(OUT, "results.json")))
roll = pd.read_csv(os.path.join(OUT, "rolling_main.csv"), index_col=0, parse_dates=True)
tbl = pd.DataFrame(R["static"]["table"]).set_index("index")
theta = pd.DataFrame(R["static"]["theta"])
cols = list(theta.columns)
N = len(cols)
TCI = R["static"]["tci"]

STRESS = [(n, pd.Timestamp(a), pd.Timestamp(b)) for n, a, b in R["stress"]["windows"]]


# --------------------------------------------------------------------- fig 1
def fig1():
    """Wide hero: horizontal position encodes NET connectedness."""
    net = tbl.NET
    lo, hi = net.min(), net.max()
    xa, xb = 0.105, 0.895
    xs = {c: xb - (xb - xa) * (net[c] - lo) / (hi - lo) for c in cols}
    pos = {
        "US":  (xs["US"],  0.50), "EU":  (xs["EU"],  0.775),
        "VIX": (xs["VIX"], 0.205), "GLD": (xs["GLD"], 0.790),
        "UST": (xs["UST"], 0.430), "OIL": (xs["OIL"], 0.185),
        "IN":  (xs["IN"],  0.520), "JP":  (xs["JP"],  0.520),
    }
    conn = (tbl.TO + tbl.FROM)
    scl = (conn - conn.min()) / (conn.max() - conn.min())
    size = {c: float(150 + 520 * scl[c]) for c in cols}

    fig = plt.figure(figsize=(7.35, 1.95))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.14); ax.axis("off")

    edges = []
    for i in cols:
        for j in cols:
            if i >= j:
                continue
            d = theta.loc[i, j] - theta.loc[j, i]
            edges.append((j if d > 0 else i, i if d > 0 else j, abs(d)))
    edges.sort(key=lambda e: -e[2])
    emax = edges[0][2]

    for src, dst, m in edges:
        if m < 0.011:
            continue
        x1, y1 = pos[src]; x2, y2 = pos[dst]
        w = 0.45 + 3.0 * (m / emax)
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.10" if y2 > y1 else "arc3,rad=-0.10",
            arrowstyle="-|>", mutation_scale=6 + 6 * (m / emax),
            lw=w, color="#2E4A66",
            alpha=0.26 + 0.52 * (m / emax),
            shrinkA=np.sqrt(size[src] / np.pi) * 0.80,
            shrinkB=np.sqrt(size[dst] / np.pi) * 0.80,
            zorder=1, capstyle="round"))

    for c in cols:
        x, y = pos[c]
        v = net[c]
        col = TRANSMIT if v > 1.5 else (RECEIVE if v < -1.5 else NEUTRAL)
        ax.scatter([x], [y], s=size[c], color=col, edgecolor=PAPER, linewidth=1.5, zorder=3)
        ax.scatter([x], [y], s=size[c], facecolor="none", edgecolor=INK, linewidth=0.45,
                   alpha=0.30, zorder=3)
        ax.text(x, y - 0.008, f"{v:+.0f}", ha="center", va="center", fontsize=7.4,
                color=PAPER, zorder=4, fontweight="bold")
        lab = {"US": (0, 0.150, "center", "bottom"), "EU": (0, 0.150, "center", "bottom"),
               "GLD": (0, 0.150, "center", "bottom"), "JP": (0, -0.150, "center", "top"),
               "IN": (0, -0.150, "center", "top"), "VIX": (0, -0.150, "center", "top"),
               "UST": (0, -0.150, "center", "top"), "OIL": (0.085, -0.02, "left", "center")}[c]
        ax.text(x + lab[0], y + lab[1], SHORT[c], ha=lab[2], va=lab[3], fontsize=7.6,
                color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.13", fc=PAPER, ec="none", alpha=0.88))

    # axis annotation
    ax.annotate("", xy=(xa - 0.03, 0.028), xytext=(xb + 0.03, 0.028),
                arrowprops=dict(arrowstyle="-", lw=0.5, color=NEUTRAL))
    ax.text(xa - 0.02, 0.045, "NET TRANSMITTERS", fontsize=5.6, color=TRANSMIT,
            fontweight="bold", ha="left")
    ax.text(xb + 0.02, 0.045, "NET RECEIVERS", fontsize=5.6, color=RECEIVE,
            fontweight="bold", ha="right")

    ax.text(0.995, 1.075, f"TCI = {TCI:.1f}%", fontsize=9.2, ha="right", color=INK,
            fontweight="bold")
    ax.text(0.995, 1.010, "full sample, H = 10", fontsize=5.7, ha="right", color=NEUTRAL)

    ax.scatter([0.012], [1.055], s=62, color=TRANSMIT, edgecolor=PAPER, lw=1.0)
    ax.text(0.026, 1.055, "net transmitter", fontsize=5.7, color=INK, va="center")
    ax.scatter([0.100], [1.055], s=62, color=RECEIVE, edgecolor=PAPER, lw=1.0)
    ax.text(0.114, 1.055, "net receiver", fontsize=5.7, color=INK, va="center")
    ax.text(0.192, 1.055, "node area $\propto$ TO + FROM   ·   arrow width $\propto$ net pairwise "
                          "spillover (pp), edges $\geq$ 1.1 pp   ·   number in node = NET",
            fontsize=5.4, color=NEUTRAL, va="center")
    fig.savefig(os.path.join(FIG, "fig1_network.png"), dpi=360)
    fig.savefig(os.path.join(FIG, "fig1_network.pdf"))
    plt.close(fig)
    return edges


# --------------------------------------------------------------------- fig 2
def fig2():
    """Page-1 dynamic figure: one generous TCI panel."""
    fig = plt.figure(figsize=(7.35, 1.58))
    ax = fig.add_axes([0.056, 0.20, 0.930, 0.66])
    for n, a, b in STRESS:
        ax.axvspan(a, b, color=TRANSMIT, alpha=0.085, lw=0, zorder=0)
    ax.plot(roll.index, roll.TCI, color=NAVY, lw=1.0, zorder=3)
    ax.axhline(roll.TCI.mean(), color=NEUTRAL, lw=0.6, ls=(0, (4, 3)), zorder=2)
    ax.set_ylabel("TCI (\%)", fontsize=7.0)
    ax.set_ylim(28, 66)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=6.4, length=2.4)

    pk = roll.TCI.idxmax(); pv = roll.TCI.max()
    ax.annotate(f"{pv:.1f}%  Mar 2020", xy=(pk, pv), xytext=(pd.Timestamp("2020-12-01"), 61.0),
                fontsize=6.6, color=TRANSMIT, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=TRANSMIT,
                                connectionstyle="arc3,rad=-0.2"))
    tr = roll.TCI.idxmin(); tv = roll.TCI.min()
    ax.plot([tr], [tv], marker="o", ms=2.4, color=NEUTRAL, zorder=4)
    ax.annotate(f"{tv:.1f}%", xy=(tr, tv), xytext=(pd.Timestamp("2022-06-01"), 31.0),
                fontsize=6.6, color=NEUTRAL, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=NEUTRAL,
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate(f"{roll.TCI.loc['2011-07':'2012-02'].max():.1f}%",
                xy=(pd.Timestamp("2011-11-15"), roll.TCI.loc["2011-07":"2012-02"].max()),
                xytext=(pd.Timestamp("2012-09-01"), 62.0), fontsize=6.6, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, connectionstyle="arc3,rad=-0.25"))
    ax.text(pd.Timestamp("2024-08-01"), 46.2, "mean 45.5%", fontsize=5.9, color=NEUTRAL,
            bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec="none"))
    for n, a, b in STRESS:
        mid = a + (b - a) / 2
        lab = "GFC" if "Global" in n else ("EURO" if "Euro" in n else
                                          ("COVID" if "COVID" in n else "RATES"))
        ax.text(mid, 64.4, lab, fontsize=5.2, color=TRANSMIT, ha="center")
    fig.savefig(os.path.join(FIG, "fig2_tci.png"), dpi=360)
    fig.savefig(os.path.join(FIG, "fig2_tci.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------- fig 3
def fig3():
    order = tbl.sort_values("NET").index.tolist()
    fig = plt.figure(figsize=(4.35, 2.75))
    ax = fig.add_axes([0.255, 0.14, 0.52, 0.74])
    y = np.arange(len(order))
    net = tbl.loc[order, "NET"].values
    colors = [TRANSMIT if v > 0 else RECEIVE for v in net]
    ax.barh(y, net, height=0.56, color=colors, alpha=0.92, zorder=3)
    ax.axvline(0, color=INK, lw=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[c] for c in order], fontsize=7.2)
    ax.set_xlim(-38, 38)
    ax.set_xlabel("NET connectedness (pp)", fontsize=7.0)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6.5, length=2.2)
    for yi, v in zip(y, net):
        ax.text(v + (1.3 if v > 0 else -1.3), yi, f"{v:+.1f}", va="center",
                ha="left" if v > 0 else "right", fontsize=6.4, color=INK)

    # persistence strip
    ax2 = fig.add_axes([0.795, 0.14, 0.135, 0.74])
    share = [R["persistence"][c]["share_positive"] * 100 for c in order]
    ax2.barh(np.arange(len(order)), [100] * len(order), color=GRID, height=0.56, zorder=1)
    ax2.barh(np.arange(len(order)), share, color=NAVY, height=0.56, alpha=0.85, zorder=2)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel("% of\nwindows", fontsize=5.9, labelpad=1)
    ax2.tick_params(labelsize=5.8, length=0)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.set_title("NET > 0", fontsize=5.9, pad=2)
    fig.text(0.255, 0.925, "Persistent roles: who sends, who absorbs",
             fontsize=7.2, color=INK)
    fig.savefig(os.path.join(FIG, "fig3_net.png"), dpi=340)
    fig.savefig(os.path.join(FIG, "fig3_net.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------- fig 4
def fig4():
    cal = pd.DataFrame(R["regime_var"]["calm"]["theta"]).loc[cols, cols]
    st = pd.DataFrame(R["regime_var"]["stress"]["theta"]).loc[cols, cols]
    diff = st - cal
    fig = plt.figure(figsize=(7.1, 2.22))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.065, right=0.905, top=0.845, bottom=0.225)
    vmax = max(cal.values.max(), st.values.max())
    for k, (M, title, kw) in enumerate([
        (cal, f"Calm   TCI {R['regime_var']['calm']['tci']:.1f}%", dict(vmin=0, vmax=vmax, cmap="Blues")),
        (st, f"Stress   TCI {R['regime_var']['stress']['tci']:.1f}%", dict(vmin=0, vmax=vmax, cmap="Oranges")),
        (diff, "$\Delta$ stress $-$ calm", dict(vmin=-0.22, vmax=0.22, cmap="RdBu_r")),
    ]):
        ax = fig.add_subplot(gs[k])
        im = ax.imshow(M.values, **kw)
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=5.9)
        ax.set_yticklabels(cols, fontsize=6.1)
        ax.set_title(title, fontsize=6.9, pad=5)
        for i in range(N):
            for j in range(N):
                v = M.values[i, j]
                if i == j or abs(v) < 0.045:
                    continue
                dark = (k == 2 and abs(v) > 0.15) or (k < 2 and v > 0.55 * vmax)
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=5.2,
                        color="white" if dark else INK)
        ax.set_xticks(np.arange(-.5, N, 1), minor=True)
        ax.set_yticks(np.arange(-.5, N, 1), minor=True)
        ax.grid(which="minor", color=PAPER, lw=1.2)
        ax.tick_params(which="both", length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        if k == 0:
            ax.set_ylabel("receives (row)", fontsize=6.3)
        else:
            ax.set_yticklabels([])
    cax = fig.add_axes([0.912, 0.28, 0.010, 0.50])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$\Delta$ variance share, pp", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.2, length=1.6)
    cb.outline.set_linewidth(0.4)
    fig.text(0.065, 0.975, "Shock origin migrates to the core: variance shares, calm vs. stress",
             fontsize=7.4, color=INK)
    fig.text(0.42, 0.055, "transmits (column)", fontsize=6.3, ha="center")
    fig.savefig(os.path.join(FIG, "fig4_matrix.png"), dpi=340)
    fig.savefig(os.path.join(FIG, "fig4_matrix.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------- fig 5
def fig5():
    fig = plt.figure(figsize=(7.35, 1.60))
    ax = fig.add_axes([0.058, 0.155, 0.930, 0.700])
    for n, a, b in STRESS:
        ax.axvspan(a, b, color=TRANSMIT, alpha=0.075, lw=0)
    series = [("US", TRANSMIT, "-"), ("VIX", "#B07D2B", "-"),
              ("EU", NAVY, "-"), ("JP", RECEIVE, "-"), ("IN", "#6A8EAE", "--")]
    for c, col, ls in series:
        ax.plot(roll.index, roll["NET_" + c], color=col, lw=0.9 if ls == "-" else 0.8,
                ls=ls, label=SHORT[c])
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_ylabel("NET connectedness (pp)", fontsize=7.0)
    ax.set_ylim(-56, 56)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, fontsize=5.4, frameon=False, loc="upper left",
              bbox_to_anchor=(-0.008, 1.015), handlelength=1.15, columnspacing=0.85,
              handletextpad=0.32)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6.5, length=2.2)
    fig.savefig(os.path.join(FIG, "fig5_arch.png"), dpi=340)
    fig.savefig(os.path.join(FIG, "fig5_arch.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    e = fig1(); fig2(); fig3(); fig4(); fig5()
    print("figures written")
    for src, dst, m in e[:16]:
        print(f"  {SHORT[src]:>10s} -> {SHORT[dst]:<10s} {m*100:5.2f} pp")
