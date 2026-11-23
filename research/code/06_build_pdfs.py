"""06_build_pdfs.py -- typeset the research brief.

Builds paper_1page.pdf (1 page) and paper_2page.pdf (2 pages) with ReportLab:
two-column body, measured frames (each block is wrapped before the frame is
placed, so nothing can overlap or run off the page), embedded figures from
research/output/figures, and the corrected text.

Content is driven by research/output/results.json; the body size is reduced
until each target page count is met exactly.
"""
from __future__ import annotations

import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.flowables import HRFlowable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
BUILD = os.path.join(OUT, "build")
os.makedirs(BUILD, exist_ok=True)

PAGE_W, PAGE_H = A4
M = 30.0                 # outer margin
GUT = 20.0              # gutter between columns
BODY_W = PAGE_W - 2 * M
COL_W = (BODY_W - GUT) / 2.0
FOOT = 18.0             # footer strip at the bottom

ACC, NAVY, INK = "#C0503C", "#1F3A5F", "#16222E"
MUT, LIGHT, RULE = "#6B7885", "#EFF3F6", "#C7D2DA"

# --------------------------------------------------------------------- fonts
MPL_FONTS = os.path.join(os.path.dirname(matplotlib.__file__),
                         "mpl-data", "fonts", "ttf")
_FONTS = {
    "Serif": "STIXGeneral.ttf",
    "Serif-Bold": "STIXGeneralBol.ttf",
    "Serif-Italic": "STIXGeneralItalic.ttf",
    "Serif-BoldItalic": "STIXGeneralBolIta.ttf",
    "Sans": "DejaVuSans.ttf",
    "Sans-Bold": "DejaVuSans-Bold.ttf",
}
for name, fname in _FONTS.items():
    pdfmetrics.registerFont(TTFont(name, os.path.join(MPL_FONTS, fname)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                              italic="Serif-Italic", boldItalic="Serif-BoldItalic")

# --------------------------------------------------------------------- data
R = json.load(open(os.path.join(OUT, "results.json")))
tbl = {row["index"]: row for row in R["static"]["table"]}
PER = R["persistence"]
EP = R["episodes"]
TCI = R["static"]["tci"]
SC = R["stress"]["comparison"]
DCC = R["dcc"]

SHORT = {"US": "S&amp;P 500", "EU": "STOXX 600", "JP": "Nikkei 225",
         "IN": "NIFTY 50", "UST": "US 7&ndash;10y", "GLD": "Gold",
         "OIL": "WTI", "VIX": "VIX"}
ORDER = ["US", "EU", "VIX", "GLD", "UST", "OIL", "IN", "JP"]

f1 = lambda v: f"{v:.1f}".replace("-", "\u2212")
sg = lambda v: f"{v:+.1f}".replace("-", "\u2212")


def pct_pos(c):
    s = PER[c]["share_positive"] * 100
    if s >= 99.5:
        return "100%"
    if s <= 0.5:
        return "&lt;1%"
    return f"{s:.0f}%"


# --------------------------------------------------------------------- math
def make_equations():
    """Render the estimator equations to PNG with mathtext."""
    eqs = [
        r"$Y_t = c + \sum_{i=1}^{p} A_i\,Y_{t-i} + \varepsilon_t,"
        r"\qquad \Sigma = \mathrm{E}[\varepsilon_t \varepsilon_t']$",
        r"$\theta_{ij}(H) = \frac{\sigma_{jj}^{-1}\sum_{h=0}^{H-1}"
        r"\left(e_i' \Psi_h \Sigma e_j\right)^2}"
        r"{\sum_{h=0}^{H-1} e_j' \Psi_h \Sigma \Psi_h' e_j},"
        r"\qquad \tilde{\theta}_{ij} = \frac{\theta_{ij}}{\sum_j \theta_{ij}}$",
        r"$\mathrm{TO}_i = \sum_{j \neq i} \tilde{\theta}_{ji}, \quad"
        r"\mathrm{FROM}_i = \sum_{j \neq i} \tilde{\theta}_{ij}, \quad"
        r"\mathrm{NET}_i = \mathrm{TO}_i - \mathrm{FROM}_i, \quad"
        r"\mathrm{TCI} = \frac{100}{N} \sum_{i \neq j} \tilde{\theta}_{ij}$",
    ]
    paths = []
    plt.rcParams["mathtext.fontset"] = "stix"
    for k, tex in enumerate(eqs):
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, tex, fontsize=11, color=INK)
        path = os.path.join(BUILD, f"eq{k + 1}.png")
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


# --------------------------------------------------------------------- styles
def styles(bs):
    lead = bs * 1.34
    return dict(
        kicker=ParagraphStyle("kicker", fontName="Serif-Bold", fontSize=bs * 0.92,
                              leading=bs * 1.1, textColor=colors.HexColor(ACC)),
        deck=ParagraphStyle("deck", fontName="Serif-Italic", fontSize=bs * 0.94,
                            leading=bs * 1.25, textColor=colors.HexColor(MUT)),
        title=ParagraphStyle("title", fontName="Serif-Bold", fontSize=bs * 3.05,
                             leading=bs * 3.25, textColor=colors.HexColor(INK)),
        sub=ParagraphStyle("sub", fontName="Serif-Italic", fontSize=bs * 1.42,
                           leading=bs * 1.7, textColor=colors.HexColor("#33414F")),
        byline=ParagraphStyle("byline", fontName="Serif", fontSize=bs * 0.92,
                              leading=bs * 1.2, textColor=colors.HexColor(INK)),
        body=ParagraphStyle("body", fontName="Serif", fontSize=bs, leading=lead,
                            textColor=colors.HexColor(INK), alignment=TA_JUSTIFY),
        bullet=ParagraphStyle("bullet", fontName="Serif", fontSize=bs,
                              leading=lead, textColor=colors.HexColor(INK),
                              alignment=TA_JUSTIFY, leftIndent=bs * 1.35,
                              bulletIndent=bs * 0.25, spaceBefore=bs * 0.55),
        head=ParagraphStyle("head", fontName="Serif-Bold", fontSize=bs * 1.08,
                            leading=bs * 1.3, textColor=colors.HexColor(NAVY),
                            spaceBefore=0, spaceAfter=0),
        abstract=ParagraphStyle("abstract", fontName="Serif", fontSize=bs * 1.05,
                                leading=bs * 1.42, textColor=colors.HexColor(INK),
                                alignment=TA_JUSTIFY),
        caption=ParagraphStyle("caption", fontName="Serif", fontSize=bs * 0.86,
                               leading=bs * 1.24, textColor=colors.HexColor(MUT),
                               alignment=TA_LEFT),
        tabhead=ParagraphStyle("tabhead", fontName="Serif-Bold", fontSize=bs * 0.82,
                               leading=bs * 1.0, textColor=colors.HexColor(MUT)),
        tabcell=ParagraphStyle("tabcell", fontName="Serif", fontSize=bs * 0.94,
                               leading=bs * 1.1, textColor=colors.HexColor(INK)),
        tabcellb=ParagraphStyle("tabcellb", fontName="Serif-Bold", fontSize=bs * 0.94,
                                leading=bs * 1.1, textColor=colors.HexColor(INK)),
        note=ParagraphStyle("note", fontName="Serif", fontSize=bs * 0.86,
                            leading=bs * 1.2, textColor=colors.HexColor(MUT)),
    )


# --------------------------------------------------------------------- pieces
def header_block(S, total):
    lines = [
        Table([[Paragraph("QUANTITATIVE FINANCE &middot; RESEARCH BRIEF", S["kicker"]),
                Paragraph(f"PAGE 1 OF {total}", ParagraphStyle(
                    "pg", parent=S["note"], alignment=TA_RIGHT))]],
              colWidths=[BODY_W * 0.6, BODY_W * 0.4],
              style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                ("TOPPADDING", (0, 0), (-1, -1), 0),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), S["body"].fontSize * 0.6),
                                ("VALIGN", (0, 0), (-1, -1), "BOTTOM")])),
        Paragraph("Cross-market transmission, state dependence and the persistence "
                  "of the shock hierarchy", S["deck"]),
        Spacer(1, S["body"].fontSize * 0.55),
        Paragraph("The Architecture of Financial Contagion", S["title"]),
        Spacer(1, S["body"].fontSize * 0.35),
        Paragraph("Who transmits shocks across global markets &mdash; and does the "
                  "hierarchy survive a crisis?", S["sub"]),
        Spacer(1, S["body"].fontSize * 0.5),
        Table([[Paragraph("<b>Buffy</b> &middot; Quantitative Research", S["byline"]),
                Paragraph("Daily data, 19 Sep 2007 &ndash; 19 Nov 2026 &middot; "
                          "N = 8 markets &middot; T = 4,195 observations",
                          ParagraphStyle("bl", parent=S["byline"], alignment=TA_RIGHT))]],
              colWidths=[BODY_W * 0.45, BODY_W * 0.55],
              style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                ("TOPPADDING", (0, 0), (-1, -1), 0),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 0)])),
        Spacer(1, S["body"].fontSize * 0.45),
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor(NAVY),
                   spaceBefore=0, spaceAfter=S["body"].fontSize * 0.7),
        abstract_box(S),
    ]
    return lines


def abstract_box(S):
    label = Paragraph("ABSTRACT", ParagraphStyle(
        "al", parent=S["kicker"], fontSize=S["body"].fontSize * 0.82))
    text = Paragraph(
        "A shock that begins in one market rarely stays there, but correlation "
        "cannot say who passes it on. We estimate a daily eight-market VAR and "
        "decompose forecast-error variance into directional spillovers. Average "
        "connectedness is 40.0% of total variance, peaks at 62.3% in March 2020, "
        "and is 3.6&nbsp;pp higher in stress than in calm periods (95% CI "
        "1.2&ndash;6.0). The level moves far more than the ordering: the S&amp;P "
        "500 transmits while Japan and India remain overwhelmingly net receivers. "
        "What reads as contagion is an amplification of a persistent hierarchy, "
        "not a rearrangement of it.", S["abstract"])
    inner = Table([[label], [text]], colWidths=[BODY_W - 3.2 - 11],
                  style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                                    ("BOTTOMPADDING", (0, 0), (0, 0), S["body"].fontSize * 0.45),
                                    ("BOTTOMPADDING", (0, 1), (0, 1), 2)]))
    return Table([["", inner]], colWidths=[3.2, BODY_W - 3.2],
                 style=TableStyle([
                     ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(ACC)),
                     ("LEFTPADDING", (0, 0), (-1, -1), 0),
                     ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                     ("LEFTPADDING", (1, 0), (1, 0), 11),
                     ("TOPPADDING", (0, 0), (-1, -1), 0),
                     ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                     ("VALIGN", (0, 0), (-1, -1), "TOP")]))


def head(S, num, text, space_before=0):
    return [Spacer(1, space_before),
            Paragraph(f"{num}&nbsp;&nbsp;&nbsp;{text}", S["head"]),
            HRFlowable(width="100%", thickness=0.9, color=colors.HexColor(NAVY),
                       spaceBefore=S["body"].fontSize * 0.3,
                       spaceAfter=S["body"].fontSize * 0.5)]


def figure(path, width):
    iw, ih = PILImage.open(path).size
    return Image(path, width=width, height=width * ih / iw)


def caption(S, tag, lead, text):
    return Paragraph(f'<font color="{NAVY}"><b>FIGURE {tag}</b></font> '
                     f'&middot; <b>{lead}</b> &mdash; {text}', S["caption"])


def market_table(S):
    head_row = [Paragraph(h, S["tabhead"]) for h in
                ["MARKET", "TO", "FROM", "NET", "NET&gt;0"]]
    data = [head_row]
    for c in ORDER:
        row = tbl[c]
        data.append([Paragraph(SHORT[c], S["tabcellb"]),
                     Paragraph(f1(row["TO"]), S["tabcell"]),
                     Paragraph(f1(row["FROM"]), S["tabcell"]),
                     Paragraph(sg(row["NET"]), S["tabcellb"]),
                     Paragraph(pct_pos(c), S["tabcell"])])
    cw = [COL_W * 0.40, COL_W * 0.145, COL_W * 0.16, COL_W * 0.16, COL_W * 0.135]
    st = [("LEFTPADDING", (0, 0), (-1, -1), 0),
          ("RIGHTPADDING", (0, 0), (-1, -1), 2),
          ("TOPPADDING", (0, 0), (-1, -1), S["body"].fontSize * 0.20),
          ("BOTTOMPADDING", (0, 0), (-1, -1), S["body"].fontSize * 0.20),
          ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(NAVY)),
          ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor(RULE)),
          ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.HexColor(NAVY))]
    return Table(data, colWidths=cw, style=TableStyle(st))


def stat_strip(S):
    vals = [("36.2%", "calm regime"), ("53.2%", "stress regime"),
            ("+17.0", "pp, regime VAR"), ("0.35&rarr;0.41", "DCC corr")]
    big = ParagraphStyle("big", fontName="Serif-Bold",
                         fontSize=S["body"].fontSize * 1.55,
                         leading=S["body"].fontSize * 1.7,
                         textColor=colors.HexColor(INK), alignment=TA_CENTER)
    small = ParagraphStyle("sm", fontName="Serif", fontSize=S["body"].fontSize * 0.76,
                           leading=S["body"].fontSize * 0.95,
                           textColor=colors.HexColor(MUT), alignment=TA_CENTER)
    n = len(vals)
    data = [[Paragraph(v, big) for v, _ in vals],
            [Paragraph(l, small) for _, l in vals]]
    t = Table(data, colWidths=[COL_W / n] * n,
              style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(LIGHT)),
                                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(RULE)),
                                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.white),
                                ("TOPPADDING", (0, 0), (-1, 0), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
                                ("TOPPADDING", (0, 1), (-1, 1), 1),
                                ("BOTTOMPADDING", (0, 1), (-1, 1), 5),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def takeaway(S, text, width=BODY_W):
    p = Paragraph(text, ParagraphStyle("tk", parent=S["body"],
                                       fontSize=S["body"].fontSize * 1.02))
    lab = Paragraph("KEY TAKEAWAY", ParagraphStyle(
        "tkl", parent=S["kicker"], fontSize=S["body"].fontSize * 0.82))
    inner = Table([[lab, p]], colWidths=[width * 0.16, width - 3.2 - width * 0.16 - 22],
                  style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return Table([["", inner]], colWidths=[3.2, width - 3.2],
                 style=TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor(NAVY)),
                                   ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(LIGHT)),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                   ("LEFTPADDING", (1, 0), (1, 0), 11),
                                   ("RIGHTPADDING", (1, 0), (1, 0), 11),
                                   ("TOPPADDING", (0, 0), (-1, -1), 7),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))


def measure(flows, width):
    h = 0.0
    for f in flows:
        _, fh = f.wrap(width, 10 ** 6)
        h += fh + f.getSpaceBefore() + f.getSpaceAfter()
    return h


# --------------------------------------------------------------------- page 1
def page1_blocks(S, total, eq_paths, fscale):
    H = header_block(S, total)
    fig1_w, fig2_w = BODY_W * fscale, BODY_W * fscale
    left = (head(S, "1", "WHY DIRECTION MATTERS") + [
        Paragraph(
            "The question about a financial shock is not whether markets are "
            "correlated but how far it travels and in which direction. Across our "
            "eight series the unconditional average pairwise correlation is 0.05, "
            "yet 40.0% of the average forecast-error variance is attributable to "
            "shocks from other markets. Both hold because dependence is signed: the "
            "four equity markets co-move (0.43), the VIX moves against them "
            "(&minus;0.74 with the S&amp;P 500) and gold and Treasuries hedge both. "
            "Averaging positive and negative correlations can obscure overall "
            "co-movement; a variance decomposition instead measures each market's "
            "contribution to other markets' forecast-error variance.", S["body"]),
    ] + head(S, "2", "DATA AND ESTIMATOR", space_before=S["body"].fontSize * 0.9) + [
        Paragraph(
            "Eight liquid markets, daily, 19 Sep 2007 to 19 Nov 2026: S&amp;P 500, "
            "STOXX Europe 600, Nikkei 225, NIFTY 50, 7&ndash;10 year US Treasuries "
            "(IEF), gold (GLD), WTI front-month and the VIX. Returns are log "
            "differences in percent (the VIX, of its logarithm); all are stationary "
            "(ADF p &lt; 0.001, KPSS not rejected at 5%). Equity returns are "
            "local-currency. 4,195 common trading days.", S["body"]),
        Spacer(1, S["body"].fontSize * 0.7),
        *[f for p in eq_paths
          for f in (figure(p, COL_W * 0.96), Spacer(1, S["body"].fontSize * 0.45))],
        Paragraph(
            "Order <i>p</i> = 1 by BIC, stable (max companion root 0.27), estimated "
            "by OLS; the generalized decomposition of Pesaran and Shin (1998) at "
            "<i>H</i> = 10 is order-invariant (2,000 random Cholesky orderings: "
            "23.1&ndash;26.0%). Connectedness is recomputed on a 250-day rolling "
            "window each trading day; stress contrasts use a 5,000-repetition "
            "circular block bootstrap (20-day blocks), since overlapping windows are "
            "strongly autocorrelated (AR(1) = 0.99). Residual autocorrelation "
            "persists: a Ljung&ndash;Box portmanteau rejects for all 8 univariate "
            "equations, so the lag order is a specification choice, not a settled "
            "one.", S["body"]),
    ])

    right = (head(S, "3", "THE SHOCK HIERARCHY") + [
        Paragraph(
            "The S&amp;P 500 has the highest TO connectedness (87.1), making it the "
            "dominant net transmitter in this system. Europe follows (NET +16.7). "
            "Japan is the mirror image: it sends 27.7% and receives 53.5% (NET "
            "&minus;25.8), and the three largest bilateral channels in the system "
            "all terminate there &mdash; S&amp;P 500 11.0 pp, VIX 7.6 pp, Europe "
            "7.4 pp. India is the second absorber (NET &minus;12.4), supplied "
            "mainly by the S&amp;P 500 (4.9 pp) and Europe (4.8 pp). Gold is nearly "
            "insular (6.6% sent); Treasuries and oil sit near balance, which makes "
            "them hedges rather than conduits.", S["body"]),
        Spacer(1, S["body"].fontSize * 0.55),
        market_table(S),
        Spacer(1, S["body"].fontSize * 0.6),
        Paragraph(
            "This ranking is not an artefact of one window. Across all 3,946 "
            "rolling windows the S&amp;P 500's NET is never negative "
            "(5th&ndash;95th percentile +18.3 to +42.8 pp) and the Nikkei's never "
            "positive (&minus;50.4 to &minus;13.9); the VIX is a net transmitter in "
            "94% of windows. The ordering is persistent; the intensity is state "
            "dependent.", S["body"]),
        Spacer(1, S["body"].fontSize * 0.9),
        stat_strip(S),
        Spacer(1, S["body"].fontSize * 0.4),
        Paragraph("Connectedness in stress: three independent measurements",
                  S["note"]),
        Spacer(1, S["body"].fontSize * 0.6),
        Paragraph(
            "Amplification is robust in direction but not in size. The rolling "
            "contrast is +3.6 pp. The regime VAR &mdash; separate coefficients on "
            "stress and calm days &mdash; gives 36.2% against 53.2%, a gap of "
            "17.0 pp. That is a different quantity, not a stronger estimate of the "
            "same one. Excluding the VIX, itself a price of risk, cuts the rolling "
            "gap to about 1.0 pp. All are variance shares from a predictive model, "
            "not causal effects.", S["body"]),
    ])

    band = [takeaway(S,
                     "Connectedness is state-dependent; the ordering is persistent. "
                     "The S&amp;P 500 is a net transmitter in 100% of 3,946 rolling "
                     "windows, Japan a net receiver in 100%.")]
    fig1 = [figure(os.path.join(FIG, "fig1_network.png"), fig1_w),
            Spacer(1, S["body"].fontSize * 0.35),
            caption(S, "1", "The shock hierarchy",
                    "Horizontal position encodes NET connectedness, node area total "
                    "directional connectedness (TO + FROM), arrow width the net "
                    "pairwise spillover. Left to right, the system runs from the US "
                    "and European equity blocks and the VIX into Japan, India, oil "
                    "and gold; the three thickest channels all terminate at the "
                    "Nikkei.")]
    fig2 = [figure(os.path.join(FIG, "fig2_tci.png"), fig2_w),
            Spacer(1, S["body"].fontSize * 0.35),
            caption(S, "2", "Connectedness is dynamic; the ordering is persistent",
                    "Total connectedness from 250-day rolling windows, with the four "
                    "pre-specified stress episodes shaded. It averages 45.5%, runs "
                    "from 31.5% (July 2024) to 62.3% (16 March 2020), and averages "
                    "46.8% in stress against 43.2% in calm windows (+3.6 pp, "
                    "p = 0.003, block bootstrap). The independent DCC measure rises "
                    "0.356 over the same contrast (95% CI 0.028&ndash;0.087).")]
    return H, left, right, band, fig1, fig2


def footer(canvas, doc, total):
    canvas.saveState()
    y = 12
    canvas.setStrokeColor(colors.HexColor(RULE))
    canvas.setLineWidth(0.5)
    canvas.line(M, y + 7.5, PAGE_W - M, y + 7.5)
    canvas.setFillColor(colors.HexColor(MUT))
    txt = ("Data: Yahoo Finance adjusted closes (S&P 500, STOXX Europe 600, "
           "Nikkei 225, NIFTY 50, IEF, GLD, WTI) and CBOE VIX history; code, raw "
           "data and outputs in the project repository.  Methods: Diebold & "
           "Yilmaz (2012, 2014); Pesaran & Shin (1998); Engle (2002).")
    label = f"page {canvas.getPageNumber()} of {total}"
    fs_label = 4.7
    canvas.setFont("Serif", fs_label)
    lw = pdfmetrics.stringWidth(label, "Serif", fs_label)
    avail = PAGE_W - 2 * M - lw - 14
    fs = 4.6
    while fs > 3.6 and pdfmetrics.stringWidth(txt, "Serif", fs) > avail:
        fs -= 0.1
    canvas.setFont("Serif", fs)
    canvas.drawString(M, y, txt)
    canvas.setFont("Serif", fs_label)
    canvas.drawRightString(PAGE_W - M, y, label)
    canvas.restoreState()


# --------------------------------------------------------------------- build
def layout_one(body, fscale, total_pages, eq_paths):
    S = styles(body)
    H, left, right, band, fig1, fig2 = page1_blocks(S, total_pages, eq_paths, fscale)

    hh = measure(H, BODY_W)
    lh, rh = measure(left, COL_W), measure(right, COL_W)
    ch = max(lh, rh)
    bh = measure(band, BODY_W)
    f1h = measure(fig1, BODY_W * fscale)
    f2h = measure(fig2, BODY_W * fscale)
    gap = body * 0.9

    total = hh + gap + ch + gap + bh + gap + f1h + gap + f2h
    avail = PAGE_H - M - FOOT - 4
    if total > avail:
        return None, total - avail

    top = PAGE_H - M
    y_h = top - hh
    y_c = y_h - gap - ch
    y_b = y_c - gap - bh
    y_f1 = y_b - gap - f1h
    y_f2 = y_f1 - gap - f2h

    frames = [
        Frame(M, y_h, BODY_W, hh, id="hdr", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M, y_c, COL_W, ch, id="c1", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M + COL_W + GUT, y_c, COL_W, ch, id="c2", leftPadding=0,
              rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(M, y_b, BODY_W, bh, id="band", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M, y_f1, BODY_W, f1h, id="fig1", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M, y_f2, BODY_W, f2h, id="fig2", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
    ]
    from reportlab.platypus import FrameBreak
    story = list(H) + [FrameBreak()] + list(left) + [FrameBreak()] + list(right) \
        + [FrameBreak()] + list(band) + [FrameBreak()] + list(fig1) + [FrameBreak()] \
        + list(fig2)
    return (frames, story), 0


def build_one(body, fscale, path, total_pages, eq_paths):
    res, over = layout_one(body, fscale, total_pages, eq_paths)
    if res is None:
        return None, over
    frames, story = res
    doc = new_doc(path, "The Architecture of Financial Contagion", FOOT)
    doc.addPageTemplates([PageTemplate(id="p1", frames=frames,
                                       onPage=lambda c, d: footer(c, d, total_pages))])
    doc.build(story)
    return path, 0


def page2_blocks(S, eq_paths, fscale):
    HM = header_block  # not reused; page 2 has a slimmer header
    kick = Table([[Paragraph("THE ARCHITECTURE OF FINANCIAL CONTAGION",
                             ParagraphStyle("p2t", parent=S["head"],
                                            fontSize=S["body"].fontSize * 1.05)),
                   Paragraph("PAGE 2 OF 2 &middot; CRISIS, ROBUSTNESS, VALIDATION",
                             ParagraphStyle("p2r", parent=S["note"],
                                            alignment=TA_RIGHT))]],
                 colWidths=[BODY_W * 0.6, BODY_W * 0.4],
                 style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                   ("TOPPADDING", (0, 0), (-1, -1), 0),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                                   ("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
    headblk = [kick,
               Paragraph("The sample, the estimates and the headline figures are "
                         "identical to the one-page brief; this page adds the regime "
                         "decomposition, the specification grid, the statistical "
                         "audit and the limitations.", S["deck"]),
               HRFlowable(width="100%", thickness=1.2,
                          color=colors.HexColor(NAVY), spaceBefore=4, spaceAfter=0)]

    l1 = head(S, "4", "CRISIS REGIMES", space_before=0) + [
        Paragraph(
            "Estimating the VAR separately on stress and on calm days, rather than "
            "averaging rolling windows, sharpens the contrast: total connectedness "
            "is 53.2% in the stress regime against 36.2% in the calm regime, a gap "
            "of 17.0 pp. Composition changes too. Own-variance shares fall in every "
            "equity market &mdash; the NIFTY's from 69% to 40%, the STOXX 600's from "
            "47% to 32% &mdash; while the US and European columns darken: the US "
            "explains 11% of Indian and 7% of Japanese variance in stress, against "
            "2% and 3% in calm markets. Safe assets move the other way: gold's own "
            "share rises and Treasuries and oil become more insular. Crises do not "
            "create new conduits; they load the existing ones.", S["body"]),
    ]

    rob_head = [Paragraph(h, S["tabhead"]) for h in
                ["SPECIFICATION", "TCI %", "&Delta; STRESS", "p"]]
    rob = [rob_head,
           ["Baseline (p=1, H=10, W=250)", "40.0", "+3.6", "0.003"],
           ["Horizon H = 5 / 20", "40.0 / 40.0", "&mdash;", "&mdash;"],
           ["Lag order p = 2 / 3 / 8", "40.2 / 40.4 / 41.1", "&mdash;", "&mdash;"],
           ["Window 150 days", "47.4", "+3.6", "0.85"],
           ["Window 400 days", "44.3", "+0.7", "0.84"],
           ["Excluding the VIX", "33.9", "+1.0", "&mdash;"],
           ["EM leg = EEM, not NIFTY", "46.6", "+0.4", "0.97"],
           ["Leave-one-out, range", "35.6&ndash;47.1", "+0.5 to +2.1", "&ge;0.94"]]
    rob_rows = [[c if not isinstance(c, str) else Paragraph(c, S["tabhead"])
                 for c in rob[0]]]
    for r in rob[1:]:
        rob_rows.append([Paragraph(r[0], S["tabcell"])] +
                        [Paragraph(x, S["tabcell"]) for x in r[1:]])
    cw = [COL_W * 0.50, COL_W * 0.20, COL_W * 0.17, COL_W * 0.13]
    rob_tbl = Table(rob_rows, colWidths=cw,
                    style=TableStyle([
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), S["body"].fontSize * 0.18),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), S["body"].fontSize * 0.18),
                        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(NAVY)),
                        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor(RULE)),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.HexColor(NAVY))]))

    r1 = head(S, "6", "ROBUSTNESS", space_before=0) + [
        Paragraph(
            "Connectedness is insensitive to the forecast horizon (H = 5, 10, 20 all "
            "give 40.0%) and barely moves with the VAR order (p = 2: 40.2%, p = 3: "
            "40.4%, AIC order p = 8: 41.1%). Window length matters more: means of "
            "47.4%, 45.5% and 44.3% at 150, 250 and 400 days, correlating "
            "0.84&ndash;0.85 with the baseline; leave-one-market-out indices stay "
            "between 35.6% and 47.1% (correlation &ge; 0.94).", S["body"]),
        Spacer(1, S["body"].fontSize * 0.6),
        rob_tbl,
        Spacer(1, S["body"].fontSize * 0.4),
        Paragraph("<b>The sign of stress amplification is robust; its magnitude is "
                  "not.</b>", S["note"]),
    ]

    ep_head = [Paragraph(h, S["tabhead"]) for h in
               ["EPISODE", "START", "DAYS", "TCI", "IN&larr;"]]
    ep_rows = [ep_head]
    for name, a, b in R["stress"]["windows"]:
        e = EP[name]
        ep_rows.append([Paragraph(name, S["tabcell"]),
                        Paragraph(a[:4], S["tabcell"]),
                        Paragraph(str(e["n"]), S["tabcell"]),
                        Paragraph(f1(e["tci"]), S["tabcell"]),
                        Paragraph(f1(e["from_in"]), S["tabcell"])])
    ep_rows.append([Paragraph("Calm windows", S["tabcellb"]),
                    Paragraph("&mdash;", S["tabcell"]),
                    Paragraph("2,503", S["tabcell"]),
                    Paragraph("43.2", S["tabcell"]),
                    Paragraph("37.8", S["tabcell"])])
    ep_tbl = Table(ep_rows,
                   colWidths=[COL_W * 0.40, COL_W * 0.145, COL_W * 0.145,
                              COL_W * 0.155, COL_W * 0.155],
                   style=TableStyle([
                       ("LEFTPADDING", (0, 0), (-1, -1), 0),
                       ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                       ("TOPPADDING", (0, 0), (-1, -1), S["body"].fontSize * 0.18),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), S["body"].fontSize * 0.18),
                       ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                       ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(NAVY)),
                       ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor(RULE)),
                       ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.HexColor(NAVY)),
                       ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor(LIGHT))]))

    l2 = head(S, "5", "STRESS EPISODES", space_before=S["body"].fontSize * 0.55) + [
        ep_tbl,
        Spacer(1, S["body"].fontSize * 0.55),
        Paragraph(
            "Rolling-window means; &ldquo;IN&larr;&rdquo; is total spillover received "
            "by India. India's incoming spillovers rise 7.0 pp in stress relative to "
            "calm windows (95% CI 3.1&ndash;10.4, p = 0.0004), but the bilateral "
            "US&rarr;India channel does not (&minus;0.54 pp, p = 0.20). The rise is "
            "broad, not a single US conduit.", S["body"]),
    ] + head(S, "8", "LIMITATIONS", space_before=S["body"].fontSize * 0.9) + [
        Paragraph(
            "Non-synchronous trading: Asia closes before Europe and the US, so a "
            "same-date VAR attributes some US news to the following day and "
            "understates same-day transmission into Asia &mdash; the receiver result "
            "is, if anything, conservative. Connectedness is predictive, not causal: "
            "a variance share is not a structural coefficient. The VIX is an "
            "endogenous market price and carries much of the stress result &mdash; "
            "without it the gap falls to 1.0 pp. Local-currency returns exclude the "
            "FX channel, and the bootstrap reflects sampling rather than "
            "specification uncertainty.", S["body"]),
    ]

    r2 = head(S, "7", "STATISTICAL VALIDATION", space_before=0) + [
        Paragraph("Monte Carlo on a synthetic three-variable VAR with diagonal "
                  "innovations: the analytic decomposition matches simulated "
                  "forecast-error variance shares to 0.0016 over 200,000 draws.",
                  S["bullet"], bulletText="\u2022"),
        Paragraph("Order invariance: the generalized measure is 40.0%; 2,000 random "
                  "Cholesky orderings span 23.1&ndash;26.0%.", S["bullet"],
                  bulletText="\u2022"),
        Paragraph("Independent estimator: coefficients, TCI and the identities "
                  "&Sigma;TO = &Sigma;FROM = N&middot;TCI = 319.9 (&Sigma;NET = 0) "
                  "reproduce exactly against statsmodels (&Delta;&beta; = 0.0000).",
                  S["bullet"], bulletText="\u2022"),
        Paragraph("Residuals: multivariate Ljung&ndash;Box Q = 193.8 (df 640, "
                  "p = 1.00), although the portmanteau rejects for all eight "
                  "univariate equations at 5%. Granger: 22 of 56 pairs reject at 5%, "
                  "14 after Bonferroni.", S["bullet"], bulletText="\u2022"),
    ] + head(S, "9", "CONCLUSION", space_before=S["body"].fontSize * 0.95) + [
        Paragraph(
            "Global equity, rate, commodity and volatility markets share a "
            "persistent hierarchy: the United States and Europe transmit, Japan and "
            "India receive, and the ordering survives every window. Stress raises the "
            "volume of transmission by 3.6 pp on rolling windows and 17.0 pp in a "
            "regime split, mainly by loading existing channels rather than creating "
            "new ones.", S["body"]),
    ]

    f3 = [figure(os.path.join(FIG, "fig4_matrix.png"), BODY_W * fscale),
          Spacer(1, S["body"].fontSize * 0.35),
          caption(S, "3", "Variance shares by regime",
                  "Rows receive, columns transmit; an entry is the share of the row "
                  "market's forecast-error variance attributed to the column market "
                  "(%), estimated separately on calm and on stress days. Blue cells "
                  "absorb more of their own shocks in stress; red cells import more. "
                  "The equity block loses its own-variance share while the US and "
                  "European columns gain, and the safe-asset block becomes more "
                  "self-contained.")]
    f4 = [figure(os.path.join(FIG, "fig5_arch.png"), BODY_W * fscale),
          Spacer(1, S["body"].fontSize * 0.35),
          caption(S, "4", "Rolling NET connectedness: levels move more than the "
                          "ordering",
                  "The S&amp;P 500 line stays above zero and the Nikkei and NIFTY "
                  "lines below it in every window; the VIX and Europe are "
                  "intermittent transmitters. Stress episodes shaded.")]
    refs = [HRFlowable(width="100%", thickness=0.7, color=colors.HexColor(NAVY),
                       spaceBefore=2, spaceAfter=4),
            Paragraph(
                '<font color="#1F3A5F"><b>REFERENCES</b></font>&nbsp;&nbsp; '
                "Barun\u00edk, J. and K\u0159ehl\u00edk, T. (2018). Measuring the "
                "frequency dynamics of financial connectedness and systemic risk. "
                "<i>Journal of Financial Econometrics</i> 16(2), 271&ndash;296. "
                "&middot; Diebold, F. X. and Y\u0131lmaz, K. (2012). Better to give "
                "than to receive: predictive directional measurement of volatility "
                "spillovers. <i>International Journal of Forecasting</i> 28(1), "
                "57&ndash;66. &middot; Diebold, F. X. and Y\u0131lmaz, K. (2014). On "
                "the network topology of variance decompositions: measuring the "
                "connectedness of financial firms. <i>Journal of Econometrics</i> "
                "182(1), 119&ndash;134. &middot; Engle, R. F. (2002). Dynamic "
                "conditional correlation. <i>Journal of Business and Economic "
                "Statistics</i> 20(3), 339&ndash;350. &middot; Pesaran, H. H. and "
                "Shin, Y. (1998). Generalized impulse response analysis in linear "
                "multivariate models. <i>Economics Letters</i> 58(1), 17&ndash;29.",
                S["note"])]
    return headblk, l1, r1, f3, l2, r2, f4, refs


def layout_two(body, fscale, eq_paths):
    from reportlab.platypus import FrameBreak
    S = styles(body)
    headblk, l1, r1, f3, l2, r2, f4, refs = page2_blocks(S, eq_paths, fscale)

    hh = measure(headblk, BODY_W)
    row1 = max(measure(l1, COL_W), measure(r1, COL_W))
    f3h = measure(f3, BODY_W * fscale)
    row2 = max(measure(l2, COL_W), measure(r2, COL_W))
    f4h = measure(f4, BODY_W * fscale)
    rf = measure(refs, BODY_W)
    gap = body * 0.85
    total = hh + gap + row1 + gap + f3h + gap + row2 + gap + f4h + gap + rf
    avail = PAGE_H - 2 * M - 4
    if total > avail:
        return None, total - avail

    top = PAGE_H - M
    y_h = top - hh
    y1 = y_h - gap - row1
    y3 = y1 - gap - f3h
    y2 = y3 - gap - row2
    y4 = y2 - gap - f4h
    y_r = y4 - gap - rf

    frames = [
        Frame(M, y_h, BODY_W, hh, id="h2", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M, y1, COL_W, row1, id="a1", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M + COL_W + GUT, y1, COL_W, row1, id="a2", leftPadding=0,
              rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(M, y3, BODY_W, f3h, id="f3", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M, y2, COL_W, row2, id="b1", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M + COL_W + GUT, y2, COL_W, row2, id="b2", leftPadding=0,
              rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(M, y4, BODY_W, f4h, id="f4", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(M, y_r, BODY_W, rf, id="rf", leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
    ]
    story = (list(headblk) + [FrameBreak()] + list(l1) + [FrameBreak()] + list(r1)
             + [FrameBreak()] + list(f3) + [FrameBreak()] + list(l2)
             + [FrameBreak()] + list(r2) + [FrameBreak()] + list(f4)
             + [FrameBreak()] + list(refs))
    return (frames, story), 0


def new_doc(path, title, bottom):
    return BaseDocTemplate(path, pagesize=A4, leftMargin=M, rightMargin=M,
                           topMargin=M, bottomMargin=bottom, title=title,
                           author="Buffy, Quantitative Research",
                           subject="Cross-market connectedness research brief",
                           creator="research/code/06_build_pdfs.py")


def build_combined(body, fscale, path, total_pages, eq_paths):
    r1, o1 = layout_one(body, fscale, total_pages, eq_paths)
    if r1 is None:
        return None, o1
    r2, o2 = layout_two(body, fscale, eq_paths)
    if r2 is None:
        return None, o2
    frames1, story1 = r1
    frames2, story2 = r2
    doc = new_doc(path, "The Architecture of Financial Contagion", M)
    doc.addPageTemplates([
        PageTemplate(id="p1", frames=frames1,
                     onPage=lambda c, d: footer(c, d, total_pages)),
        PageTemplate(id="p2", frames=frames2,
                     onPage=lambda c, d: footer(c, d, total_pages))])
    story = list(story1) + [NextPageTemplate("p2"), PageBreak()] + list(story2)
    doc.build(story)
    return path, 0


# --------------------------------------------------------------------- fit
def page_count(path):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    n = len(pdf)
    pdf.close()
    return n


def main():
    eq = make_equations()
    targets = [(1, os.path.join(OUT, "paper_1page.pdf"), build_one, 1),
               (2, os.path.join(OUT, "paper_2page.pdf"), build_combined, 2)]
    failures = []
    for _pages, out, fn, total in targets:
        chosen = None
        for fscale in (1.0, 0.94, 0.88, 0.82, 0.76):
            for body in (7.0, 6.8, 6.6, 6.4, 6.2, 6.0, 5.8, 5.6):
                res, over = fn(body, fscale, out, total, eq)
                if res is None:
                    continue
                n = page_count(out)
                if n == total:
                    chosen = (body, fscale, n)
                    break
                else:
                    print(f"  {os.path.basename(out)}: {n} pages at "
                          f"body={body} fig={fscale}")
            if chosen:
                break
        if not chosen:
            failures.append(out)
            print(f"FAILED to fit {out}")
        else:
            print(f"{os.path.basename(out)}: body={chosen[0]}pt "
                  f"fig-scale={chosen[1]} pages={chosen[2]}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
