# Cancer Subtype Discovery

> **An unsupervised machine-learning pipeline for discovering potential molecular subtypes from gene-expression data.**

Cancer patients with the same clinical diagnosis can have substantially different molecular profiles. This project aims to identify such hidden molecular patterns by applying reproducible unsupervised learning methods to high-dimensional gene-expression data.

The initial reference dataset is the **Golub et al. leukemia gene-expression dataset**, containing 72 samples and 7,129 gene-expression probes.

---

## Overview

The goal of this project is to build a end-to-end pipeline for discovering and characterizing molecular subgroups from gene-expression data.

The pipeline is designed to:

- Load and validate gene-expression data
- Perform data-quality checks
- Select informative genes using variability measures
- Standardize gene-expression values
- Reduce dimensionality using PCA and other methods
- Discover molecular clusters
- Evaluate cluster quality and stability
- Identify potential cluster-specific biomarkers
- Perform biological pathway enrichment
- Provide interpretable visualizations and reports

### Core principle

Cancer labels are **not used to create the clusters**.

The discovery process is unsupervised. The available cancer labels are retained separately for downstream interpretation and evaluation.

---

## Pipeline

```text
                 Gene Expression Data
                         │
                         ▼
              Data Loading & Validation
                         │
                         ▼
                  Quality Checks
                         │
                         ▼
                Variable Gene Selection
                         │
                         ▼
               Gene-wise Standardization
                         │
                         ▼
                 Dimensionality Reduction
                         │
                         ├── PCA
                         └── UMAP
                         │
                         ▼
                     Clustering
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           K-Means   Hierarchical   GMM
                         │
                         ▼
                Consensus Clustering
                         │
                         ▼
                 Cluster Validation
                         │
                         ▼
                  Stability Analysis
                         │
                         ▼
                 Biomarker Discovery
                         │
                         ▼
                Pathway Enrichment
                         │
                         ▼
                   Explainability
                         │
                         ▼
               Reports & Visualization
```

---

## Dataset

The initial reference dataset is the **Golub leukemia gene-expression dataset** from Golub et al. (1999), accessed through the Bioconductor `golubEsets` package.

### Dataset characteristics

| Property | Value |
|---|---:|
| Samples | 72 |
| Gene-expression probes | 7,129 |
| ALL samples | 47 |
| AML samples | 25 |
| Platform | Affymetrix Hgu6800 |
| Dataset type | Gene-expression |
| Learning setting | Unsupervised |

The project stores the expression matrix and labels separately:

```text
data/raw/
├── golub_expr.csv
└── golub_labels.csv
```

### Label usage

The `cancer_type` labels are **not used during feature selection or clustering**.

They are retained for:

- downstream cluster interpretation
- evaluation of discovered structure
- comparison with known leukemia classes
- biological analysis

This separation is important because using the known cancer labels to form clusters would turn the discovery process into a supervised task.

---

## Methodology

### 1. Data Loading and Validation

The data-loading layer validates:

- required columns
- missing sample identifiers
- duplicate sample identifiers
- empty datasets
- numeric expression features
- missing cancer labels
- sample alignment between expression and label data

Invalid input is rejected using project-specific validation exceptions.

---

### 2. Preprocessing

The preprocessing stage prepares the high-dimensional expression matrix for downstream analysis.

Current preprocessing includes:

- expression-value diagnostics
- variable-gene selection
- Median Absolute Deviation (MAD) based feature selection
- gene-wise standardization

The number of selected genes and preprocessing behavior are controlled through `config.yaml`.

---

### 3. Dimensionality Reduction

Principal Component Analysis (PCA) is used to transform the high-dimensional gene-expression matrix into a lower-dimensional representation.

PCA helps:

- reduce dimensionality
- capture major sources of variation
- make downstream clustering more stable
- visualize sample-level structure
- identify dominant expression patterns

The current configuration uses:

```yaml
dimensionality_reduction:
  pca_n_components: 30
  umap_n_neighbors: 15
  umap_min_dist: 0.1
  umap_random_state: 42
```

---

### 4. Clustering

The planned clustering stage compares multiple unsupervised approaches:

- K-Means
- Hierarchical clustering with Ward linkage
- Gaussian Mixture Models
- Consensus clustering

The clustering configuration currently explores:

```yaml
k_range: [2, 3, 4, 5, 6, 7, 8]
```

Using multiple clustering methods allows the project to compare whether discovered subgroups are consistent across different algorithms.

---

### 5. Cluster Validation

Cluster quality will be evaluated using quantitative validation and resampling-based stability analysis.

The project configuration includes:

```yaml
validation:
  bootstrap_n_iter: 100
  bootstrap_sample_frac: 0.8
  ari_threshold_pass: 0.70
  stability_threshold_pass: 0.60
```

The goal is not simply to find clusters, but to determine whether those clusters are reproducible and meaningful.

---

### 6. Biomarker Discovery

Once stable clusters have been identified, genes that differ between clusters will be investigated as potential molecular biomarkers.

The current configuration specifies:

```yaml
biomarkers:
  test: "mannwhitneyu"
  fdr_method: "fdr_bh"
  fdr_threshold: 0.05
  logfc_threshold: 1.0
  top_n_report: 20
```

Multiple-testing correction is applied using the Benjamini-Hochberg false-discovery-rate procedure.

---

### 7. Pathway Enrichment

Candidate biomarker genes will be analyzed for biological pathway enrichment.

The planned analysis includes:

- KEGG
- Reactome
- Hallmark gene sets
- Gene Ontology Biological Processes

This stage is intended to help translate statistical clusters into biologically interpretable patterns.

---

### 8. Explainability

The project plans to use interpretable machine-learning techniques to identify features that contribute most strongly to differences between discovered groups.

The current configuration includes a Random Forest surrogate model for this purpose.

---

## Current Progress

### Completed

- [x] Project structure and configuration
- [x] Python environment and dependencies
- [x] Reproducibility utilities
- [x] Golub dataset acquisition
- [x] Expression and label CSV generation
- [x] Data loading and validation
- [x] Sample alignment checks
- [x] Missing-value checks
- [x] Expression-value diagnostics
- [x] Variable-gene selection using MAD
- [x] Gene-wise standardization
- [x] Unit tests for data loading and preprocessing
- [x] PCA implementation
- [x] Basic PCA unit tests
- [x] K-Means clustering
- [x] Hierarchical clustering (Ward linkage)
- [x] Gaussian Mixture Models
- [x] Consensus clustering (resampling-based co-clustering matrix)
- [x] Unit tests for all clustering methods (input validation, reproducibility, output shape)

### In Progress

- [ ] PCA explained-variance analysis
- [ ] PCA visualization
- [ ] Final dimensionality-reduction workflow

### Planned

- [ ] Cluster validation
- [ ] Stability analysis
- [ ] Biomarker discovery
- [ ] Pathway enrichment
- [ ] Explainability
- [ ] Final reports
- [ ] Streamlit interface

---

## Project Structure

The repository is being developed incrementally. The current implementation is intentionally smaller than the final target architecture.

```text
cancer-subtype-discovery/
│
├── data/
│   └── raw/                    # Local dataset files (not committed)
│
├── src/
│   ├── clustering/             # Clustering algorithms (Phase 4)
│   │   ├── __init__.py
│   │   └── core.py             # KMeans, Hierarchical, GMM, Consensus
│   ├── dimensionality/         # PCA and dimensionality reduction
│   ├── preprocessing/          # Data preprocessing
│   ├── utils/                  # Reproducibility utilities
│   ├── config.py               # Configuration models
│   ├── data_loader.py          # Data loading and validation
│   ├── exceptions.py           # Project-specific exceptions
│   └── __init__.py
│
├── tests/
│   ├── test_clustering.py
│   ├── test_data_loader.py
│   ├── test_dimensionality.py
│   └── test_preprocessing.py
│
├── config.yaml                 # Pipeline configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # Project license
```

Additional modules such as clustering, validation, biomarkers, enrichment, reporting, and the Streamlit application will be added as the pipeline develops.

---

## Installation

### Requirements

- Python 3.11+
- Git
- `pip`
- Virtual environment support

### 1. Clone the repository

```bash
git clone <repository-url>
cd cancer-subtype-discovery
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The project expects the following local files:

```text
data/raw/
├── golub_expr.csv
└── golub_labels.csv
```

The dataset is intentionally excluded from Git version control because raw project data is listed in `.gitignore`.

After placing the dataset files in the expected location, verify the data-loading pipeline by running the test suite.

---

## Configuration

Pipeline parameters are centralized in:

```text
config.yaml
```

This includes configuration for:

- data paths
- feature selection
- preprocessing
- PCA
- UMAP
- clustering
- consensus clustering
- validation
- biomarker analysis
- enrichment
- explainability
- reporting
- application settings

Centralizing these parameters helps keep experiments reproducible and avoids hard-coding analysis choices throughout the codebase.

---

## Testing

The project uses `pytest` for automated testing.

Run the complete test suite with:

```bash
python -m pytest -v
```

Current test coverage includes:

- data loading
- data validation
- sample alignment
- preprocessing
- dimensionality reduction

A successful run currently reports:

```text
54 passed
```

Tests are intended to be run before committing changes to the repository.

---

## Reproducibility

Reproducibility is a core requirement of the project.

Random seeds are centralized through the project's reproducibility utilities and configuration.

The default random seed is:

```text
42
```

Random-state parameters are also explicitly configured for algorithms where applicable.

The project aims to ensure that the same:

- dataset
- preprocessing configuration
- random seed
- algorithm parameters

produce consistent analytical results.

---

## Design Principles

The project follows several principles throughout development:

### Reproducibility

Analysis parameters should be configurable and deterministic where possible.

### Separation of concerns

Data loading, preprocessing, dimensionality reduction, clustering, validation, and reporting are kept as separate components.

### Testability

Core functionality is covered by automated unit tests.

### Unsupervised discovery

Known cancer labels are kept separate from the clustering process.

### Statistical validation

Discovered clusters should be evaluated for both quality and stability rather than accepted solely based on visual separation.

### Biological interpretation

Statistical findings should ultimately be connected to biomarkers and biological pathways.

---

## Limitations

The current study is an initial prototype based on the Golub leukemia dataset.

Important limitations include:

- The dataset contains a relatively small number of samples compared with the number of gene-expression features.
- The initial dataset represents leukemia rather than multiple cancer types.
- Discovered clusters require independent validation before being considered clinically meaningful.
- Statistical association does not automatically imply biological or clinical relevance.
- The current implementation is still under active development.

The purpose of the current stage is therefore **method development and reproducible pipeline construction**, not clinical diagnosis.

---

## Roadmap

```text
[x] Data acquisition
[x] Data validation
[x] Preprocessing
[x] Variable-gene selection
[x] Standardization
[x] PCA implementation
[ ] PCA analysis and visualization
[ ] UMAP
[x] K-Means clustering
[x] Hierarchical clustering (Ward)
[x] Gaussian Mixture Models
[x] Consensus clustering
[ ] Cluster validation
[ ] Stability analysis
[ ] Biomarker discovery
[ ] Pathway enrichment
[ ] Explainability
[ ] Automated reporting
[ ] Streamlit application
[ ] End-to-end pipeline
```

---

## Scientific Context

This project is motivated by the observation that molecular heterogeneity can exist within broad disease categories.

Instead of starting with predefined cancer subtypes, the pipeline first analyzes gene-expression structure and attempts to identify naturally occurring molecular groups.

The resulting groups can then be compared with known clinical or biological labels as part of downstream interpretation.

This approach is intended to support exploratory cancer-subtype discovery rather than replace established clinical classification.

---

## License

This project is distributed under the license included in the repository's `LICENSE` file.

---

## Acknowledgements

The initial reference dataset is based on the leukemia gene-expression study by:

**Golub, T. R. et al. (1999).**  
*Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring.*

The dataset is accessed through the Bioconductor `golubEsets` package.

---

## Project Status

**Status:** 🚧 Active Development

The core data-loading, validation, preprocessing, reproducibility, and PCA components are currently implemented and tested. The remaining clustering, validation, biomarker, enrichment, explainability, reporting, and application components are under development.