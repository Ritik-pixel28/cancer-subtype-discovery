# Verification record

Verified on 2026-11-12 against base commit `0f441443764cf0bf029c5a3a5c09d1c30adf24a9` plus the reproducibility changes delivered with this record.

## Executed checks

- A clean checkout initially had no `data/raw` directory. The offline reference setup successfully restored expression and label inputs from committed dashboard artifacts.
- Restored expression values and sample labels were compared against the original local raw CSVs and matched. Historical file-byte hashes are not claimed to match restored CSV formatting.
- Full test suite: **157 passed in 26.58 seconds**. Includes all dashboard sections via Streamlit AppTest, navigation/filter interactions, PDF generation, report export checksums, Phase 8 characterization, core analysis tests and the new setup/enrichment regression tests.
- A fresh Phase 6 calculation using restored inputs regenerated `strong_candidates.csv`. Its data frame matched the committed 183-row result, including statistics and expression direction. Verification outputs were written separately from saved research artifacts.
- `python run.py --check` passed with the complete application environment.
- Optional enrichment setup downloaded and parsed all four configured GMT libraries and loaded GPL80 annotation successfully. Setup recorded source hashes locally. The original historical GMT inputs were not recovered; no assertion of identical historical enrichment results is made.
- `git diff --check` passed.

## Runtime

Python 3.13 on macOS; NumPy 2.4.1, pandas 3.0.0, scikit-learn 1.9.1, SciPy 1.18.1, statsmodels 0.15.0, pytest 9.1.1, Streamlit 1.65.0, ReportLab 5.0.1, pypdf 6.19.0. The existing application environment was reused; its missing pypdf test dependency was installed into an isolated temporary dependency directory.

## Scope limits

This run did not repeat the entire Phase 5 resampling workflow, rerun historical pathway enrichment, inspect the application visually in a browser, or validate a public deployment. AppTest verifies application behavior but does not establish visual correctness of WebGL, CSS or responsive layouts. No saved scientific outputs, model settings or cluster assignments were changed. Independent cohort and clinical validation remain outside the completed research-prototype scope.
