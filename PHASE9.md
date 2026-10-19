# Phase 9 — Cancer Subtype Discovery Explorer

Group G87 · Molecular Discovery · revised October 9, 2026

## Design

The landing experience takes its composition from the supplied White Desert reference: a full-screen scene, restrained navigation, oversized condensed typography, and spacious editorial sections. The scientific workspace uses warm off-white surfaces, dark slate text, teal/ochre/violet cluster colors, and large plots.

The landing scene is genuine WebGL geometry: 486 DNA atoms from the experimental **RCSB PDB 1BNA** structure. Atom coordinates are centered and rigidly rotated for display, with uniform scene scale. Bonds are inferred from covalent radii and distance. Lighting, colors, repeated background copies and motion are artistic. The molecule is a structural reference, not a cancer-specific gene sequence or a reconstruction from patient expression data.

Animation pauses off screen and when the browser tab is hidden, honors reduced-motion settings, and has an explicit pause control. The renderer caps pixel ratio and animation rate. A static coordinate-derived illustration is available if WebGL cannot initialize. All visual assets, fonts, coordinates and rendering code are bundled locally; runtime network access is not required.

Reference: https://white-desert.com/
Structure: https://www.rcsb.org/structure/1BNA
Rebuild the bundled display coordinates with `python scripts/prepare_molecular_scene.py`.
Drew et al. (1981), doi:10.1073/pnas.78.4.2179.

## Run locally

Python 3.11+ is required. The build was tested on Python 3.14.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m streamlit run app/streamlit_app.py
```

The historical `requirements.txt` remains the upstream pipeline environment. `requirements-app.txt` pins the tested application runtime, while `requirements-test.txt` adds test dependencies.

The entry page is the molecular landing experience. Choose **Enter explorer** to enter the analytical workspace. The section selector exposes all twelve modules. Focus cluster persists across in-session navigation; global gene search opens Biomarkers. The original header links and immediate tab switching are restored. The section selector retains filters within the current session; header links open a new session.

## Twelve research modules

1. **Research overview:** validated sample/probe/component/candidate counts, cluster composition, PCA and navigable method stages.
2. **Dataset explorer:** probe selector, literal sample search, expression preview, filtered CSV, descriptive statistics, true quality checks, original expression histogram, sample distributions, known-label composition.
3. **Preprocessing:** ranked MAD values, selection threshold, retained/excluded probes, per-probe before/after standardization and actual means/standard deviations.
4. **PCA:** adjustable component axes, 2D/3D projection, consensus/known-label coloring, scree plot, cumulative variance and CSV/figure exports.
5. **Cluster discovery:** sample lookup, cluster sizes, nearest samples by Euclidean distance in the saved 30-PC representation, ordered standardized-expression heatmaps, and algorithm comparison by k.
6. **Validation:** ARI/silhouette curves, Gap with calculated 1-SE selection, PAC minimum, empirical bootstrap intervals, and a validated optional consensus-matrix reader.
7. **Biomarkers:** all 6,000 probe-cluster tests or strong candidates, literal gene/probe search, cluster/FDR/effect/direction/annotation filters, significance-versus-effect plots, raw-expression boxplots, and deduplicated probe/gene overlap tables.
8. **Pathways:** pending without Phase 7 artifacts; filters, dot plots, overlap details and method/background metadata when supplied.
9. **Explainability:** pending without Phase 8 artifacts; global importance, per-sample contributions, feature-value coloring, model metadata, separately displayed surrogate predictions and consensus assignments, and evaluated fidelity/confusion matrices when supplied.
10. **Comparisons:** cluster summaries, shared mapped genes, PCA, specific probe expression distributions/scatterplots, exploratory Spearman correlation, optional pathway/importance comparisons, and JSON configuration export.
11. **Reports:** executive, full technical and cluster-specific PDF/HTML reports, selectable sections, CSVs, PNG/SVG figures, provenance JSON and ZIP bundle.
12. **Methods:** readable method explanations, actual YAML configuration, software versions, code version and working-tree state, artifact SHA-256 checksums, status and limitations.

## Data and artifact preparation

The app consumes saved analysis outputs. It never runs clustering on page changes. It validates schemas, finite values, sample IDs, integer cluster labels, probability ranges and cross-artifact alignment. An invalid artifact is withheld with a visible status rather than silently converted into a valid-looking result.

The included `reports/app/expression.csv.gz` contains the existing public Golub expression matrix copied from the user's local pipeline dataset. `mad.csv`, `pca.csv`, `pca_variance.csv` and `dataset_summary.json` are prepared using the existing preprocessing/PCA implementation. Known labels are joined only after PCA. Saved cluster assignments are not changed.

To regenerate presentation artifacts after a compatible upstream analysis:

```sh
# Original Golub expression and labels must exist in data/raw/.
python -m scripts.prepare_dashboard_artifacts
```

The preparation step stores source hashes and the source base commit. If input expression or analytical configuration changes, regenerate the upstream analysis too. Matching sample IDs alone cannot establish that separate result versions belong together.

The source annotation covers the strong-candidate mapping. Missing annotations in the all-tested-probes view do not mean those probes can never be mapped. Probe-level and gene-level overlap are kept separate.

The exact original consensus matrix is **not available** in the supplied artifacts. The application does not fabricate it from final labels. Export `reports/validation/consensus_k3_matrix.csv` with sample IDs on both axes to enable the matrix heatmap. It must be square, symmetric, finite, within [0,1], with a unit diagonal and exactly matching sample IDs.

## Phase 7 contract

`reports/enrichment/pathway_results.csv`:

Required: `cluster`, `database`, `pathway_id`, `pathway_name`, `adjusted_p_value`, `overlap_count`, `overlapping_genes`.

Optional: `raw_p_value`, `pathway_size`, `enrichment_score`.

`reports/enrichment/metadata.json` should record `method`, `enrichment_measure`, `database_version`, `background_gene_universe`, `gene_identifier_type`, `annotation_strategy`, `analysis_date` and `software_versions`. The chart names the supplied measure; without an enrichment score it explicitly plots -log10(adjusted p-value).

## Phase 8 contract

- `reports/explainability/global_importance.csv`: `cluster`, `feature_id`, `mean_abs_shap`; optional `gene_symbol`, `rank`.
- `reports/explainability/sample_explanations.csv`: `sample_id`, `target_cluster`, `feature_id`, `shap_value`, `feature_value`.
- `reports/explainability/surrogate_predictions.csv`: `sample_id`, `predicted_cluster`, `evaluation_split`. One prediction per sample within a split.
- `reports/explainability/model_metadata.json`: model type/version, training configuration, feature representation, evaluation strategy, train/evaluation sample information, fidelity metrics, SHAP explainer and `output_units`.

These files are optional. Missing outputs remain pending. Synthetic integration fixtures exist only in temporary test directories, never in research results.

## Reports and exports

Reports are built on demand from validated artifacts. They record selected sections and scope, source checksums, configuration, software versions, generation time in Asia/Kolkata, application base commit and modified-working-tree status. Cluster-specific reports filter memberships, PCA and candidate tables; validation remains correctly labeled as global algorithm evidence.

Executive summaries are four concise pages. The full report uses separate sections with static charts and wrapped tables. Full selected tables and PNG/SVG figures accompany the report in the ZIP. HTML escapes table contents. Scientific scope and limitations remain explicit even when optional sections are omitted.

Plotly charts provide PNG exports, interactive HTML downloads and an SVG export view. Report-generated scientific figures are also available as direct PNG and SVG downloads.

## Tests

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Verification: 125 automated tests passed on the rebuilt application.

Coverage includes existing pipeline tests plus every application section, gene filters and empty states, 3D PCA, raw-data alignment, standardization, malformed schemas, optional-phase contracts, selected report scope, actual PDF generation and source/export checksums. Browser checks additionally cover the molecular renderer, pause button, landing navigation, search, downloads, and responsive layout. Generated PDFs are rendered and visually inspected.

## Deployment

The local preview is the current review target. No public hosting or GitHub publication is assumed from a local preview.

A Dockerfile and `.dockerignore` are included for container deployment:

```sh
docker build -t molecular-discovery .
docker run --rm -p 8501:8501 molecular-discovery
```

The container recipe has not been built in this environment unless explicitly recorded in the delivery notes. It excludes ignored raw data and uses the versioned public presentation artifacts. For Streamlit Community Cloud, select `app/streamlit_app.py`; `app/requirements.txt` points to the pinned application dependencies. Python 3.13 is the container target.

Before publishing, validate the exact target runtime, make sure only the intended public reference data are included, and check the live deployment. No arbitrary patient uploads, accounts or clinical predictions are included.

## Third-party assets

- Three.js r160.1, MIT license: `app/assets/vendor/THREE-LICENSE`.
- Barlow Condensed Medium, SIL Open Font License: `app/assets/FONT-LICENSE`.
- DNA coordinates: RCSB PDB 1BNA; attribution above and in the Methods page.

No media from White Desert or YouTube is copied into the application.

## Navigation and figure visibility — October 13, 2026

The original header and eagerly rendered tabs are restored: switching tabs shows already-prepared figures immediately. Chart export drawers are also prepared without a separate page rerun. Standalone HTML is cached as bytes, compatible with Streamlit 1.51 as well as 1.65. Probe labels use an indexed lookup, and reporting libraries load only on report/method pages.

The report figure selector now displays the actual generated PNG above its PNG/SVG downloads. Figure selection runs in a Streamlit fragment, so changing it does not rebuild the whole Reports page or regenerate the PDF. Download buttons do not rerun the dashboard. The screenshot-reported Validation preview is covered by a regression test and visually checked in the browser.

## Reliable local launch

From the repository folder, run `python run.py`. This selects the complete project environment automatically and serves the application at http://127.0.0.1:8509. It avoids accidentally using a bare Anaconda Streamlit install without the report dependencies. `python run.py --check` reports the chosen interpreter without starting a server.

Compatibility checks cover Streamlit 1.51 (the Anaconda version shown in the reported traceback) and the pinned 1.65 environment. The old installation was tested in an isolated environment with report dependencies added; the global Anaconda installation was not changed. Figure downloads pass binary bytes on both versions. Overview statistics use equal-width columns in both rows, and the research navigation cards are laid out in sequential rows.

Latest verification: 33 application/service checks passed under Streamlit 1.51 and again under 1.65. Browser checks confirmed chart HTML download, visible charts, and aligned layouts at 1440px and 700px widths.
