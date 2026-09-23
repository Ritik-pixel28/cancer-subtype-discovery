# Cancer Subtype Discovery

> **An unsupervised machine-learning pipeline for discovering and characterizing potential molecular subgroups from gene-expression data.**

Cancer patients with the same broad clinical diagnosis can show substantially different molecular expression patterns. This project investigates whether hidden molecular structure can be discovered from high-dimensional gene-expression data using reproducible unsupervised machine-learning methods.

The initial reference dataset is the **Golub et al. leukemia gene-expression dataset**, containing **72 patient samples** and **7,129 gene-expression probes**.

---

## Overview

The goal of this project is to build an end-to-end and reproducible pipeline for discovering, validating, and biologically interpreting molecular subgroups from gene-expression data.

The pipeline is designed to:

- Load and validate gene-expression data
- Perform expression-data quality checks
- Select informative genes using robust variability measures
- Standardize gene-expression features
- Reduce dimensionality using PCA
- Discover patient clusters using multiple clustering algorithms
- Evaluate cluster separation and stability
- Compare discovered clusters with known labels only after clustering
- Identify candidate cluster-specific biomarkers
- Perform pathway-enrichment analysis
- Add explainability and biological interpretation
- Generate reproducible reports and visualizations

### Core principle

Known cancer labels are **not used to create the clusters**.

The discovery process is unsupervised. Available labels such as ALL and AML are kept separate from feature selection and clustering and are used only later for external evaluation and interpretation.

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
                  MAD Top 2000
                          │
                          ▼
             Gene-wise Standardization
                          │
                          ▼
             Dimensionality Reduction
                          │
                          ▼
                       PCA
                2000 genes → 30 PCs
                          │
                          ▼
                     Clustering
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           K-Means       Ward        GMM
              │           │           │
              └───────────┼───────────┘
                          ▼
                Consensus Clustering
                          │
                          ▼
                  Cluster Validation
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
        ARI          Silhouette      Bootstrap Stability
                                           │
                         ┌─────────────────┴──────────────┐
                         ▼                                ▼
                  Gap Statistic                         PAC
                         │                                │
                         └──────────────┬─────────────────┘
                                        ▼
                              Candidate k Selection
                                        │
                                        ▼
                          Exploratory Consensus Clusters
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
| Dataset type | Gene expression |
| Learning setting | Unsupervised |

The project stores expression data and known labels separately:

```text
data/raw/
├── golub_expr.csv
├── golub_labels.csv
└── golub_dataset.zip
```

### Label usage

The `cancer_type` labels are **not used during variable-gene selection, PCA, clustering, consensus clustering, Gap analysis, PAC analysis, or bootstrap clustering**.

Known labels are retained for:

- External evaluation using Adjusted Rand Index
- Downstream interpretation
- Comparison with known leukemia classes
- Biological context

This separation prevents the subtype-discovery stage from becoming a supervised classification task.

---

## Methodology

### 1. Data Loading and Validation

The `DataLoader` validates the structure and alignment of the input datasets.

Current checks include:

- Required columns
- Missing sample identifiers
- Duplicate sample identifiers
- Empty datasets
- Numeric expression features
- Missing cancer labels
- Sample alignment between expression and label files

Expression and label files must contain the same patient identifiers before the pipeline continues.

---

### 2. Preprocessing

The preprocessing stage prepares the high-dimensional gene-expression matrix for clustering.

Current preprocessing includes:

- Expression-value diagnostics
- Median Absolute Deviation based feature selection
- Selection of the top 2,000 variable genes
- Gene-wise standardization

#### Variable-gene selection

Median Absolute Deviation (MAD) is used to rank genes by variability across patients.

For a gene \(g\):

```text
MAD(g) = median(|x_i - median(x)|)
```

The 2,000 genes with the highest MAD values are retained.

This step is fully unsupervised and does not use cancer labels.

#### Standardization

Each selected gene is transformed to approximately:

```text
mean = 0
standard deviation = 1
```

This prevents genes with larger numerical scales from dominating PCA and distance-based clustering.

#### Log transformation

A standard `log2(x + 1)` transformation is not currently applied because the Golub expression matrix contains negative values.

The pipeline therefore avoids applying a transformation that would be mathematically invalid for the observed data representation.

---

### 3. Dimensionality Reduction

Principal Component Analysis is used to reduce the selected gene-expression matrix from:

```text
72 samples × 2000 genes
```

to:

```text
72 samples × 30 principal components
```

PCA helps:

- Reduce dimensionality
- Reduce noise
- Capture dominant patterns of variation
- Make clustering more computationally manageable
- Reduce problems caused by extremely high-dimensional feature space

### PCA reproducibility

The PCA implementation uses a fixed random state:

```text
random_state = 42
```

A dedicated reproducibility test verifies that identical input data and configuration produce identical PCA output.

UMAP remains planned for visualization and is not currently part of the clustering-validation workflow.

---

## 4. Clustering

The current implementation includes four clustering approaches.

### K-Means

K-Means partitions patients into `k` groups by minimizing within-cluster variation.

Current implementation includes:

- Configurable `k`
- Fixed random seed
- Multiple initializations
- Sample-index preservation
- Input validation

### Hierarchical clustering

Agglomerative hierarchical clustering is implemented using:

```text
Ward linkage
Euclidean geometry
```

Ward linkage attempts to minimize the increase in within-cluster variance during hierarchical merging.

### Gaussian Mixture Model

Gaussian Mixture Models provide a probabilistic clustering approach.

The implementation supports:

- Hard cluster assignments
- Soft membership probabilities
- Configurable number of components
- Fixed random state

### Consensus clustering

Consensus clustering evaluates how consistently pairs of patients are grouped together across repeated subsamples.

For each patient pair:

```text
consensus ≈ 0
```

means they are rarely grouped together, while:

```text
consensus ≈ 1
```

means they are consistently grouped together.

The consensus matrix can subsequently be converted into hard cluster assignments using average-linkage hierarchical clustering on:

```text
1 - consensus
```

---

## 5. Cluster Validation

Phase 5 evaluates clustering solutions using multiple complementary signals.

No single validation metric is treated as sufficient evidence by itself.

### 5a. Adjusted Rand Index

Adjusted Rand Index is used as **external evaluation** by comparing discovered clusters with the known ALL/AML labels.

The known labels are introduced only after clustering.

ARI is therefore used to ask:

> How much does the discovered partition agree with the known leukemia classes?

It is not used to generate or optimize the clusters.

---

### 5b. Silhouette Score

Silhouette Score measures internal cluster separation using the same PCA feature space used for clustering.

Higher values indicate that patients are, on average:

- Closer to patients in their own cluster
- Farther from patients in neighboring clusters

Unlike ARI, Silhouette Score does not require known cancer labels.

---

### 5c. Bootstrap Stability

Bootstrap-style stability analysis repeatedly takes an 80% subset of patients and reruns clustering.

Current configuration:

```text
resamples       = 100
sample fraction = 0.8
random state    = 42
```

The resampled clustering is compared with the corresponding subset of the full-data clustering using best-match **Jaccard membership similarity**.

Cluster numbers themselves are not assumed to remain identical between runs.

For example:

```text
Run 1: biological group → cluster 0
Run 2: same group       → cluster 2
```

is not treated as disagreement simply because the numeric label changed.

The analysis reports:

- Mean stability
- Lower 95% empirical interval
- Upper 95% empirical interval

---

### 5d. Gap Statistic

Gap Statistic compares the compactness of clusters in the observed data with clustering obtained from randomly generated reference datasets.

For candidate values:

```text
k = 2, 3, 4, 5, 6, 7, 8
```

the implementation calculates:

- Observed log within-cluster dispersion
- Mean reference log dispersion
- Gap value
- Reference variability

Candidate `k` is selected using the **Gap 1-SE rule** rather than simply choosing the largest raw Gap value.

The rule selects the smallest `k` satisfying approximately:

```text
Gap(k) >= Gap(k + 1) - uncertainty(k + 1)
```

---

### 5e. PAC

PAC stands for:

**Proportion of Ambiguous Clustering**

It summarizes how many sample pairs in the consensus matrix have uncertain co-clustering probabilities.

Current ambiguity interval:

```text
0.1 < consensus < 0.9
```

Lower PAC values indicate fewer ambiguous sample-pair relationships.

The matrix diagonal is excluded because every sample is trivially grouped with itself.

---

## Phase 5 Results on the Golub Dataset

The real-data Phase 5 workflow evaluates candidate clustering solutions for:

```text
k = 2 through 8
```

using K-Means, Ward hierarchical clustering, and Gaussian Mixture Models.

### k-selection signals

Both the corrected Gap Statistic rule and PAC analysis support:

```text
k = 3
```

Observed PAC values:

| k | PAC |
|---:|---:|
| 2 | 0.4100 |
| 3 | **0.2085** |
| 4 | 0.3513 |
| 5 | 0.3842 |
| 6 | 0.4092 |
| 7 | 0.4049 |
| 8 | 0.3858 |

The minimum PAC occurs at:

```text
k = 3
```

### k=3 algorithm comparison

| Method | ARI | Silhouette | Mean Stability | 95% Lower | 95% Upper |
|---|---:|---:|---:|---:|---:|
| K-Means | 0.0945 | 0.1415 | 0.7902 | 0.5885 | 0.9231 |
| Ward | 0.1733 | 0.1209 | 0.7819 | 0.5704 | 0.9700 |
| GMM | 0.3134 | 0.0848 | 0.4321 | 0.3081 | 0.5714 |

These metrics show that the different validation criteria do not all prefer the same clustering algorithm.

For example:

- GMM has stronger agreement with known ALL/AML labels at `k=3`
- K-Means shows stronger internal separation and higher average bootstrap stability
- Ward shows similar bootstrap stability to K-Means
- GMM is considerably less stable under resampling

For this reason, the project does not choose a clustering method based only on ARI.

---

## Exploratory Consensus Solution

A consensus clustering solution was generated using:

```text
k = 3
base clusterer = K-Means
consensus resamples = 100
sample fraction = 0.8
random state = 42
```

Observed evaluation:

```text
Consensus ARI        = 0.1741
Consensus Silhouette = 0.1548
```

Cluster sizes:

| Cluster | Samples |
|---:|---:|
| 0 | 14 |
| 1 | 35 |
| 2 | 23 |

These assignments are saved as:

```text
reports/validation/consensus_k3_clusters.csv
```

### Interpretation caution

The consensus `k=3` partition is currently treated as the **best-supported exploratory clustering solution for downstream analysis**.

It should not be described as a clinically validated set of leukemia subtypes.

The predefined stability and external-agreement goals were not fully achieved, so biological interpretation must remain cautious.

Independent datasets, biological validation, and domain expertise would be required before claiming that the discovered clusters represent true cancer subtypes.

---

## 6. Biomarker Discovery

**Status: In progress**

Phase 6 will investigate genes that differ between the exploratory consensus clusters.

The current planned comparison is:

```text
Cluster 0 vs all remaining samples
Cluster 1 vs all remaining samples
Cluster 2 vs all remaining samples
```

For each gene, the planned analysis includes:

- Mann-Whitney U test
- Raw p-value
- Benjamini-Hochberg FDR correction
- Median expression within the target cluster
- Median expression outside the target cluster
- Median expression difference
- Ranking of candidate marker genes

### Multiple-testing correction

Thousands of genes are tested simultaneously.

The project therefore uses:

```text
Benjamini-Hochberg False Discovery Rate
```

through `statsmodels`.

### Effect-size consideration

The Golub expression matrix contains negative values.

Therefore, the project will not blindly calculate a conventional expression ratio such as:

```python
log2(mean_cluster / mean_other)
```

because this can be undefined or misleading when means are zero or negative.

The initial implementation will instead report expression differences that are compatible with the observed data representation.

---

## 7. Pathway Enrichment

**Status: Planned**

Candidate biomarker genes will later be analyzed for biological pathway enrichment.

Planned resources include:

- KEGG
- Reactome
- Hallmark gene sets
- Gene Ontology Biological Processes

The goal is to determine whether statistically identified marker genes correspond to coherent biological pathways.

---

## 8. Explainability

**Status: Planned**

The project plans to use interpretable machine-learning methods to investigate which genes most strongly distinguish discovered clusters.

A Random Forest surrogate model and SHAP-based interpretation are planned for this stage.

---

## Current Progress

### Completed

- [x] Project structure and configuration
- [x] Python environment and dependencies
- [x] Reproducibility utilities
- [x] Golub dataset acquisition
- [x] Expression and label CSV generation
- [x] Data loading and structural validation
- [x] Sample-alignment validation
- [x] Expression-value diagnostics
- [x] MAD variable-gene selection
- [x] Top 2,000 gene selection
- [x] Gene-wise standardization
- [x] PCA dimensionality reduction
- [x] PCA reproducibility test
- [x] K-Means clustering
- [x] Ward hierarchical clustering
- [x] Gaussian Mixture Model clustering
- [x] Consensus clustering
- [x] External ARI evaluation
- [x] Silhouette analysis
- [x] Bootstrap/Jaccard stability analysis
- [x] Gap Statistic
- [x] Gap 1-SE candidate selection
- [x] PAC analysis
- [x] Consensus `k=3` evaluation
- [x] Phase 5 integration on the real Golub dataset
- [x] Phase 5 result export
- [x] 80 automated tests passing
- [x] Phase 5 results committed and pushed to GitHub
- [x] Biomarker module scaffold
- [x] `statsmodels` dependency added

### In Progress

- [ ] Biomarker discovery
- [ ] Cluster-vs-rest differential-expression testing
- [ ] Benjamini-Hochberg FDR correction
- [ ] Candidate biomarker ranking

### Planned

- [ ] PCA explained-variance analysis
- [ ] PCA visualization
- [ ] UMAP visualization
- [ ] Pathway enrichment
- [ ] Biological interpretation
- [ ] SHAP explainability
- [ ] Automated report generation
- [ ] Streamlit interface
- [ ] End-to-end production pipeline

---

## Project Structure

```text
cancer-subtype-discovery/
│
├── app/
│
├── data/
│   └── raw/
│       ├── golub_dataset.zip
│       ├── golub_expr.csv
│       └── golub_labels.csv
│
├── logs/
│
├── models/
│
├── notebooks/
│
├── reports/
│   └── validation/
│       ├── ari_silhouette_results.csv
│       ├── bootstrap_results.csv
│       ├── consensus_k3_clusters.csv
│       ├── gap_results.csv
│       └── pac_results.csv
│
├── scripts/
│   └── run_phase5_validation.py
│
├── src/
│   ├── biomarkers/
│   │   ├── __init__.py
│   │   └── core.py
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   └── core.py
│   │
│   ├── dimensionality/
│   │   ├── __init__.py
│   │   └── core.py
│   │
│   ├── enrichment/
│   ├── explainability/
│   ├── pipeline/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── core.py
│   │
│   ├── reporting/
│   ├── utils/
│   ├── validation/
│   │   ├── __init__.py
│   │   └── core.py
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   └── exceptions.py
│
├── templates/
│
├── tests/
│   ├── test_biomarkers.py
│   ├── test_clustering.py
│   ├── test_data_loader.py
│   ├── test_dimensionality.py
│   ├── test_preprocessing.py
│   └── test_validation.py
│
├── config.yaml
├── requirements.txt
├── README.md
└── LICENSE
```

The repository is intentionally modular so that statistical analysis, clustering, biological interpretation, reporting, and application code remain separated.

---

## Installation

### Requirements

- Python 3.11+
- Git
- `pip`
- Virtual-environment support

The current development environment has also been tested with Python 3.13.

### Clone the repository

```bash
git clone <repository-url>
cd cancer-subtype-discovery
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Dataset Setup

The pipeline expects:

```text
data/raw/
├── golub_expr.csv
└── golub_labels.csv
```

The expression file contains:

```text
sample_id + 7,129 gene-expression columns
```

The labels file contains:

```text
sample_id
cancer_type
```

Raw project data should not be used as a training signal for cluster formation through the known cancer labels.

---

## Running Phase 5 Validation

Run the complete real-data Phase 5 workflow from the repository root:

```bash
python -m scripts.run_phase5_validation
```

The script performs:

```text
Data loading
    ↓
MAD top-2000 gene selection
    ↓
Standardization
    ↓
PCA to 30 components
    ↓
K-Means / Ward / GMM comparison
    ↓
ARI + Silhouette
    ↓
Gap Statistic
    ↓
PAC
    ↓
Bootstrap stability
    ↓
Consensus k=3 evaluation
    ↓
Validation result export
```

Generated outputs are stored under:

```text
reports/validation/
```

---

## Testing

The project uses `pytest`.

Run the full suite with:

```bash
python -m pytest -v
```

Current automated testing covers:

- Data loading
- Input validation
- Sample alignment
- Preprocessing
- MAD feature selection
- Gene standardization
- PCA
- PCA reproducibility
- K-Means clustering
- Ward clustering
- Gaussian Mixture Models
- Consensus clustering
- ARI evaluation
- Silhouette Score
- Bootstrap stability
- Gap Statistic
- Gap candidate selection
- PAC
- PAC candidate selection
- Validation edge cases

Current passing suite:

```text
80 passed
```

Tests are run before changes are pushed to the main branch.

---

## Reproducibility

Reproducibility is a core requirement of the project.

The default random seed is:

```text
42
```

Fixed random states are used where applicable for:

- PCA
- K-Means
- GMM
- Consensus resampling
- Gap reference generation
- Bootstrap resampling

The Phase 5 runner has also been executed twice with identical configuration and compared to verify reproducible output.

The project aims for identical:

- Dataset
- Preprocessing
- Random seed
- Algorithm configuration

to produce consistent analytical results.

---

## Design Principles

### Unsupervised discovery

Known cancer labels must not influence the formation of discovered clusters.

### Reproducibility

Randomness should be explicitly controlled where possible.

### Separation of concerns

Data loading, preprocessing, dimensionality reduction, clustering, validation, biomarker analysis, enrichment, explainability, and reporting are maintained as separate modules.

### Testability

Core analytical components should have automated unit tests before being used on the real dataset.

### Multi-metric validation

A clustering solution should not be selected from one metric alone.

The project combines:

```text
ARI
Silhouette
Bootstrap Stability
Gap Statistic
PAC
Consensus analysis
```

### Statistical caution

Good clustering metrics do not automatically imply that discovered groups are biologically or clinically meaningful.

### Biological interpretation

Candidate subgroups must eventually be investigated through genes, pathways, external datasets, and domain expertise.

---

## Limitations

The current project is an exploratory prototype.

Important limitations include:

- Only 72 patient samples are available
- The number of genes is much larger than the number of patients
- The initial dataset focuses on leukemia
- Cluster-validation metrics do not all strongly agree
- Bootstrap lower confidence bounds do not fully meet the originally desired stability target
- External ARI agreement is modest
- The exploratory consensus clusters have not been independently validated
- No clinical conclusions should currently be drawn
- Biomarker discovery and pathway interpretation are still under development
- Independent datasets will be required for meaningful biological validation

The purpose of the current work is:

> **reproducible exploratory molecular subgroup discovery and methodology development, not clinical diagnosis.**

---

## Roadmap

```text
[x] Project setup
[x] Data acquisition
[x] Data validation
[x] Sample alignment
[x] Expression diagnostics
[x] MAD variable-gene selection
[x] Standardization
[x] PCA implementation
[x] PCA reproducibility
[x] K-Means clustering
[x] Ward hierarchical clustering
[x] Gaussian Mixture Models
[x] Consensus clustering
[x] ARI evaluation
[x] Silhouette analysis
[x] Bootstrap stability
[x] Gap Statistic
[x] Gap 1-SE selection
[x] PAC analysis
[x] Phase 5 real-data integration
[x] Consensus k=3 exploratory partition
[x] Phase 5 result export
[x] 80-test regression suite

[ ] Biomarker discovery
[ ] Mann-Whitney testing
[ ] BH-FDR correction
[ ] Candidate marker ranking
[ ] PCA visualization
[ ] UMAP
[ ] Pathway enrichment
[ ] Biological interpretation
[ ] Explainability / SHAP
[ ] Automated reporting
[ ] Streamlit application
[ ] End-to-end pipeline
```

---

## Scientific Context

This project is motivated by the observation that substantial molecular heterogeneity can exist within broad disease categories.

Instead of starting with predefined subtype labels, the pipeline first analyzes expression patterns and attempts to identify naturally occurring patient groups.

Only after those groups have been formed are known labels introduced for external comparison.

The resulting workflow therefore separates:

```text
Discovery
    ↓
Statistical validation
    ↓
External comparison
    ↓
Biological interpretation
```

This approach is intended to support exploratory cancer-subtype research rather than replace established clinical classification.

---

## License

This project is distributed under the license included in the repository's `LICENSE` file.

---

## Acknowledgements

The initial reference dataset is based on:

**Golub, T. R. et al. (1999).**  
*Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring.*

The dataset is accessed through the Bioconductor `golubEsets` package.

---

## Project Status

**Status:** 🚧 Active Development

### Current milestone

```text
Phase 1 — Data loading                  ✅
Phase 2 — Preprocessing                 ✅
Phase 3 — PCA                           ✅
Phase 4 — Clustering                    ✅
Phase 5 — Cluster validation            ✅
Phase 6 — Biomarker discovery           🚧 In Progress
Phase 7 — Pathway enrichment            ⏳ Planned
Phase 8 — Explainability                ⏳ Planned
Phase 9 — Reporting / application       ⏳ Planned
```

Phase 5 has produced an exploratory `k=3` consensus partition for downstream analysis.

Phase 6 is now beginning with statistical identification of genes that distinguish the discovered clusters.

The project remains an exploratory research pipeline and does not currently make clinical or diagnostic claims.