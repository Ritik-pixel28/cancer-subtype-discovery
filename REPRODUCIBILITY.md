# Reproducing the reference project

The completed research scope is Phases 1–6 (discovery and candidate biomarkers), Phase 7 (pathway enrichment), Phase 8 (integrated subtype characterization), and Phase 9 (saved-results dashboard and reports). Optional surrogate-model/SHAP analysis and arbitrary dataset uploads are not implemented deliverables.

## Clean checkout: dashboard and all tests

From the repository root, using Python 3.13:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m scripts.setup_reference_data
python -m pytest -q
python run.py --check
python run.py
```

Open http://127.0.0.1:8509. The setup command is offline: it restores the public expression matrix from `reports/app/expression.csv.gz` and the known labels from `reports/app/pca.csv`. Those labels were saved after PCA; the setup command does not fit a model, change cluster assignments, or introduce labels into discovery. Expression and label values were compared with the original local CSVs during verification.

The setup checks sample alignment, dimensions, finite expression values, label counts, and existing inputs. It refuses to replace different local inputs and preserves matching CSVs. `data/raw/reference_setup.json` records source and restored-file checksums. CSV serialization can differ from the original raw-file hashes; the historical provenance is retained unchanged. Raw files remain ignored by Git.

The dashboard reads versioned results. Viewing it or running setup does not regenerate the scientific analysis. The automated suite includes Streamlit page interactions and PDF generation; this is distinct from a browser visual inspection or testing a hosted deployment.

## Re-running the analysis

Run these commands in a separate working copy: the analysis scripts replace generated results.

```sh
python -m scripts.setup_reference_data
python -m scripts.run_phase5_validation
python -m scripts.run_phase6_biomarkers
python -m scripts.annotate_phase6_probes
python -m scripts.finalize_phase6_biomarkers
python -m scripts.prepare_dashboard_artifacts
python -m scripts.setup_reference_data --download-enrichment
python -m scripts.run_phase7_enrichment
python -m scripts.run_phase8_characterization
```

Probe annotation requires GPL80 from NCBI GEO. The existing annotation loader downloads it through GEOparse when no local cache exists. It can also read a GPL80 SOFT file placed under `data/annotation/`.

The enrichment setup downloads missing GMT libraries from the [Enrichr library endpoint](https://maayanlab.cloud/Enrichr/#libraries) using the library names configured in `config.yaml`:

- `KEGG_2021_Human.gmt`
- `Reactome_2022.gmt`
- `MSigDB_Hallmark_2020.gmt`
- `GO_Biological_Process_2021.gmt`

Files go under `data/annotation/gene_sets/`. Existing caches are preserved; invalid or empty downloads are rejected. `setup_manifest.json` records hashes and whether each library was downloaded or supplied locally. Review each underlying database's terms before redistribution or commercial use.

**Historical limitation:** the original GMT files and their checksums were not committed with the recorded Phase 7 run. Newly downloaded libraries with the same names cannot be assumed byte-identical. Therefore the saved 254 significant terms are historical recorded results, not a promised result from a new download. Preserve the downloaded caches, configuration, environment, and checksums with any new run. Exact historical enrichment reproduction remains contingent on recovering the original libraries.

The exact original consensus matrix is also absent. The dashboard correctly leaves its heatmap unavailable; it does not manufacture pairwise probabilities from the final cluster labels. A new consensus run can produce a new matrix, but should not be represented as the missing original artifact.

## Scientific scope

Results are exploratory associations from the 72-sample Golub cohort. Candidate-marker testing reuses the expression data that defined the clusters; BH correction alone does not address that selection. These outputs describe candidate cluster signatures and require independent validation. Subtype characterization is complete within this exploratory scope; it is not clinical validation, a diagnostic tool, or evidence of treatment benefit.
