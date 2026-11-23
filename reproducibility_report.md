# Reproducibility Report

## Analysis Version
- Date: November 21, 2026
- Python environment: .venv (Python 3.12)
- Packages: statsmodels 0.15.0, arch 9.0.0

## Changes Applied

### Change 1: S&P 500 TO connectedness clarification
- Text replaced: "The S&P 500 accounts for {f(TO['US'])}%..." 
- With: "The S&P 500 has the highest TO connectedness (87.1), making it the dominant net transmitter in this system."
- Location: research/code/05_paper.py, TO connectedness paragraph

### Change 2: Correlation explanation
- Text replaced: "Averaging correlations over opposite signs cancels transmission; a variance decomposition does not."
- With: "Averaging positive and negative correlations can obscure overall co-movement; a variance decomposition instead measures each market's contribution to other markets' forecast-error variance."
- Location: research/code/05_paper.py

### Change 3: Stable → Persistent hierarchy
- Text replaced: "stable hierarchy" 
- With: "persistent hierarchy" in §9 of the two-page note
- Location: research/code/05_paper.py, conclusion section

### Change 4: Dependency reinstall and validation rerun
- Installed: statsmodels 0.15.0, arch 9.0.0
- Rerun: research/code/02_analysis.py → research/output/results.json
- Rerun: research/code/03_validation.py → research/output/validation.json
- All outputs verified against stored values

## Verified Results Comparison

### Key metrics from rerun vs stored:

**TCI (Total Connectedness Index):**
- Stored: 39.98
- Rerun: 39.98 (verified match)
- Status: ✅ No discrepancy

**NET Connectivity values:**
- US: +28.32 (both stored and rerun)
- EU: +16.67 (both stored and rerun)
- JP: -25.79 (both stored and rerun)
- IN: -12.36 (both stored and rerun)
- UST: -5.56 (both stored and rerun)
- GLD: -2.07 (both stored and rerun)
- OIL: -6.54 (both stored and rerun)
- VIX: +7.33 (both stored and rerun)
- Status: ✅ All values match

**Stress vs Calm TCI comparison:**
- Stored diff: +3.6333140251260687
- Rerun diff: +3.6333140251260687
- CI: [1.1868634765279524, 6.005397542620749]
- p-value: 0.0032 (both stored and rerun)
- Status: ✅ No discrepancy

**DCC-GARCH parameters:**
- a: 0.176 (both)
- b: 0.811 (both)
- mean|ρ|: 0.356 (both)
- Status: ✅ No discrepancy

**Persistence (share of positive roots):**
- US: 100% (both)
- EU: 91.4% (both)
- JP: 0% (both)
- IN: 0.03% (both)
- VIX: 93.7% (both)
- Status: ✅ All values match

**Ljung-Box diagnostics:**
- All eight markets significant at 5% level (both stored and rerun)
- Status: ✅ No discrepancy

**FROM_IN (Forecast-from contribution):**
- Value: +6.96
- CI: [3.10, 10.45]
- p = 0.0004 (both)
- Status: ✅ No discrepancy

**US→IN (US to India):**
- Value: −0.54
- p = 0.199 (both)
- Status: ✅ No discrepancy

**VIX-excluded gap:** ≈1.0 percentage points (both)

## PDF Outputs

Both documents are typeset by `research/code/06_build_pdfs.py` (ReportLab, two-column
A4, frames measured from the wrapped content so no block can overlap or run off the
page). Figures come from `research/code/04_figures.py`, regenerated from the current
`results.json` and `rolling_main.csv`.

| File | Pages | Contents |
|---|---|---|
| `paper_1page.pdf` | 1 | Full brief: abstract, sections 1–3, market table, regime strip, Figures 1–2 |
| `paper_2page.pdf` | 2 | Page 1 as above, plus page 2: sections 4–9, robustness and stress tables, Figures 3–4, references |

Build check: body text 6.4 pt, figures at 100% width, page counts asserted as 1 and
2; text layer extracted and checked so that all three corrections are present and
none of the superseded sentences remain.

Corrections present in the PDF text layer:
1. "The S&P 500 has the highest TO connectedness (87.1), making it the dominant net transmitter in this system."
2. "Averaging positive and negative correlations can obscure overall co-movement; a variance decomposition instead measures each market's contribution to other markets' forecast-error variance."
3. "persistent hierarchy" (replacing "stable hierarchy" in §9)

## Discrepancies
- No discrepancies found between rerun outputs and stored results
- All numeric values verified match the stored results.json and validation.json
- All statistical diagnostics confirmed consistent across runs

## Conclusion
All four requested changes have been applied successfully. The dependency reinstall and full analysis/validation rerun confirmed that all numerical results are reproducible and consistent with the stored values. The three text corrections (Changes 1, 2, 3) have been verified in the paper code. The final PDFs deliver the revised documents with the scientific corrections.
