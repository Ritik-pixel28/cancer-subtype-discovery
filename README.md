# Cancer Subtype Discovery

> **An unsupervised machine-learning pipeline for discovering and characterizing potential molecular subgroups from gene-expression data.**

Cancer patients with the same broad clinical diagnosis can show substantially different molecular expression patterns. This project investigates whether hidden molecular structure can be discovered from high-dimensional gene-expression data using reproducible unsupervised machine-learning methods.

The initial reference dataset is the **Golub et al. leukemia gene-expression dataset**, containing **72 patient samples** and **7,129 gene-expression probes**.

---

## Phase 9 — Molecular Discovery Explorer

A cinematic molecular landing page leads into twelve research modules: dataset and preprocessing exploration, 2D/3D PCA, cluster discovery, validation, biomarkers, optional Phase 7/8 interpretation, comparisons, PDF/HTML reporting and reproducibility. The landing animation uses the published 1BNA DNA structure. Scientific values come from validated project artifacts.

```sh
python -m pip install -r requirements-app.txt
python -m streamlit run app/streamlit_app.py
```

See [PHASE9.md](PHASE9.md) for complete setup, contracts, limitations and deployment notes. Phase 7/8 analyses and the original consensus matrix are not fabricated when missing.

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

**Status: Phase 6 implementation and analysis completed — final Git commit and push pending**

Phase 6 moves the project from cluster discovery toward biological interpretation.

Phase 5 answered:

> How are the patient samples grouped?

Phase 6 now asks:

> Which genes are most strongly associated with each discovered cluster?

The exploratory consensus solution from Phase 5 contains three clusters:

```text
Cluster 0 → 14 samples
Cluster 1 → 35 samples
Cluster 2 → 23 samples
```

Biomarker discovery is performed at the **original gene-expression level**, not on PCA components.

PCA was useful for clustering because it reduced 2,000 selected genes into 30 principal components. However, a principal component is a combination of many genes and is not itself a biological gene.

Therefore, Phase 6 returns to the actual MAD-selected gene-expression features.

---

### Phase 6 Workflow

```text
                    Consensus k = 3 Clusters
                              │
                              ▼
                   Gene Expression Matrix
                     Top 2,000 MAD Genes
                              │
                              ▼
                     Sample ID Alignment
                              │
                              ▼
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        Cluster 0         Cluster 1         Cluster 2
          vs Rest           vs Rest           vs Rest
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                     Mann-Whitney U Test
                       for every gene
                              │
                              ▼
                         Raw p-values
                              │
                              ▼
                Benjamini-Hochberg FDR
                              │
                              ▼
                 Median Expression Difference
                              │
                              ▼
                Rank-Biserial Effect Size
                              │
                              ▼
                 Statistical Significance
                      FDR < 0.05
                              │
                              ▼
                 Effect-Size Investigation
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           ≥ 0.5           ≥ 0.7           ≥ 0.8
                                                │
                                                ▼
                                   Strong Candidate Filter
                                   |rank-biserial| ≥ 0.8
                                                │
                                                ▼
                                    Direction Analysis
                                       Higher / Lower
                                                │
                                                ▼
                                  Cross-Cluster Overlap
                                                │
                                                ▼
                                Affymetrix Probe Annotation
                                      using GEO GPL80
                                                │
                                                ▼
                                    Actual Gene Symbols
                                                │
                                                ▼
                              Top Annotated Candidate Genes
                                                │
                                                ▼
                              Input for Pathway Enrichment
```

---

### 6.1 Cluster-vs-Rest Design

Biomarker discovery is performed separately for each consensus cluster.

The comparisons are:

```text
Cluster 0
14 samples
     │
     ▼
vs
     │
     ▼
Cluster 1 + Cluster 2
58 samples
```

```text
Cluster 1
35 samples
     │
     ▼
vs
     │
     ▼
Cluster 0 + Cluster 2
37 samples
```

```text
Cluster 2
23 samples
     │
     ▼
vs
     │
     ▼
Cluster 0 + Cluster 1
49 samples
```

This one-vs-rest approach allows the pipeline to identify genes whose expression patterns are particularly associated with an individual cluster.

---

### 6.2 Input Alignment

Correct sample alignment is essential before statistical testing.

The expression matrix contains:

```text
72 expression samples
```

and the Phase 5 consensus result contains:

```text
72 cluster assignments
```

The pipeline verifies that:

```text
Expression samples = 72
Cluster labels     = 72
Missing IDs        = 0
Unknown IDs        = 0
```

Samples are matched using `sample_id`, rather than relying only on row order.

This prevents a patient's expression profile from accidentally being compared with another patient's cluster assignment.

---

### 6.3 Gene Selection for Biomarker Analysis

The original Golub dataset contains:

```text
7,129 expression probes
```

The same unsupervised MAD-based selection used earlier in the pipeline is applied to retain:

```text
Top 2,000 variable genes
```

Biomarker testing is then performed on these 2,000 gene-expression features.

Unlike clustering, Phase 6 does **not** perform biomarker analysis using the 30 PCA components.

---

### 6.4 Mann-Whitney U Test

For every selected gene, the target cluster is compared against all remaining samples using a two-sided:

```text
Mann-Whitney U Test
```

The Mann-Whitney U test is a non-parametric statistical test.

It is useful here because gene-expression values are not assumed to follow a normal distribution.

For a particular gene:

```text
Target Cluster Expression
          │
          │
          ▼
     [x1, x2, x3, ...]
          │
          │
          ├──────────────┐
          │              │
          ▼              ▼
    Mann-Whitney U   Remaining Samples
          │              │
          │              ▼
          │        [y1, y2, y3, ...]
          │
          ▼
      U Statistic
          │
          ▼
       p-value
```

A lower p-value indicates stronger statistical evidence that the two expression distributions differ.

---

### 6.5 Constant-Gene Handling

During unit testing, an important edge case was identified.

A gene can occasionally contain exactly the same expression value across every sample.

Example:

```text
Target Cluster → [5, 5, 5]

Rest           → [5, 5, 5]
```

Such a gene contains no information that can distinguish the groups.

The pipeline explicitly handles this case as:

```text
p-value            = 1.0
median difference  = 0
rank-biserial      = 0
```

This also prevents missing statistical values from affecting downstream FDR correction.

---

### 6.6 Multiple-Testing Correction

A major issue in gene-expression analysis is that thousands of statistical tests are performed simultaneously.

If 2,000 genes were tested using only:

```text
p < 0.05
```

some genes could appear significant purely by chance.

Therefore, Phase 6 applies:

```text
Benjamini-Hochberg
False Discovery Rate
```

using `statsmodels`.

The primary significance criterion is:

```text
FDR < 0.05
```

The workflow is:

```text
2,000 Gene Tests
       │
       ▼
2,000 Raw p-values
       │
       ▼
Benjamini-Hochberg Correction
       │
       ▼
Adjusted FDR Values
       │
       ▼
FDR < 0.05
       │
       ▼
Statistically Significant
Cluster-Associated Genes
```

---

### 6.7 Expression Difference

Statistical significance alone does not describe the direction of a gene-expression difference.

Therefore, Phase 6 also calculates:

```text
median_in_cluster
median_out_cluster
median_difference
```

where:

```text
median_difference
=
median_in_cluster
-
median_out_cluster
```

Interpretation:

```text
median_difference > 0
        │
        ▼
Gene expression is higher
in the target cluster
```

```text
median_difference < 0
        │
        ▼
Gene expression is lower
in the target cluster
```

Median values are used because they are less sensitive to extreme observations than means.

---

### 6.8 Why Conventional log2 Fold Change Is Not Used

The current Golub expression representation contains negative values.

Therefore, blindly calculating:

```python
log2(mean_cluster / mean_other)
```

could be undefined or misleading when group means are zero or negative.

For this reason, the current implementation reports:

```text
Median Expression Difference
+
Rank-Biserial Effect Size
```

instead of forcing a conventional log2 fold-change calculation onto an incompatible numerical representation.

---

### 6.9 Rank-Biserial Effect Size

Along with statistical significance, the pipeline calculates a rank-based effect size derived from the Mann-Whitney U statistic.

The rank-biserial value ranges approximately from:

```text
-1 ---------------- 0 ---------------- +1
 │                  │                   │
 ▼                  ▼                   ▼
Lower in         Weak / No          Higher in
Target Cluster   Separation         Target Cluster
```

Interpretation:

```text
rank_biserial > 0
→ target cluster generally has higher expression

rank_biserial < 0
→ target cluster generally has lower expression

|rank_biserial| closer to 1
→ stronger separation between the groups
```

This allows genes to be evaluated using both:

```text
Statistical Significance
          +
Effect Magnitude
```

rather than p-values alone.

---

## Phase 6 Results on the Golub Dataset

The Phase 6 pipeline was successfully executed on the real Golub dataset.

For each of the three consensus clusters:

```text
Genes tested per comparison = 2,000
```

The results were:

| Cluster | Samples | Genes Tested | FDR < 0.05 |
|---:|---:|---:|---:|
| 0 | 14 | 2,000 | 743 |
| 1 | 35 | 2,000 | 1,192 |
| 2 | 23 | 2,000 | 933 |

The large number of significant genes suggests that the discovered clusters differ across many expression features.

However, these genes should not automatically be interpreted as validated biomarkers.

The clusters and gene associations are discovered from the same exploratory dataset.

Therefore, the project refers to them as:

> **candidate cluster-associated genes**

rather than clinically validated cancer biomarkers.

---

### 6.10 Effect-Size Sensitivity Analysis

Because FDR significance alone retained hundreds of genes, several effect-size thresholds were investigated.

The number of candidates remaining at each threshold was:

| `|rank-biserial|` Threshold | Cluster 0 | Cluster 1 | Cluster 2 |
|---:|---:|---:|---:|
| ≥ 0.5 | 597 | 561 | 520 |
| ≥ 0.6 | 425 | 297 | 333 |
| ≥ 0.7 | 251 | 85 | 162 |
| ≥ 0.8 | 123 | 16 | 44 |

The pattern can be visualized as:

```text
Effect-size threshold becomes stricter
                    │
                    ▼

        0.5           0.6           0.7           0.8
         │             │             │             │
         ▼             ▼             ▼             ▼

C0      597    ───►    425    ───►   251    ───►   123

C1      561    ───►    297    ───►    85    ───►    16

C2      520    ───►    333    ───►   162    ───►    44
```

For a more conservative exploratory shortlist, the project currently uses:

```text
FDR < 0.05
      AND
|rank_biserial| >= 0.8
```

The `0.8` threshold is used as a **project-level exploratory strength filter**.

It is not presented as a universal biological cutoff.

---

### 6.11 Strong Candidate Results

After applying:

```text
FDR < 0.05
AND
|rank_biserial| >= 0.8
```

the strong candidate counts are:

```text
                Strong Candidates

Cluster 0             123
   │
   ├── Higher         119
   └── Lower            4


Cluster 1              16
   │
   ├── Higher           3
   └── Lower           13


Cluster 2              44
   │
   ├── Higher          24
   └── Lower           20
```

In table form:

| Cluster | Strong Candidates | Higher | Lower |
|---:|---:|---:|---:|
| 0 | 123 | 119 | 4 |
| 1 | 16 | 3 | 13 |
| 2 | 44 | 24 | 20 |

Across all comparisons:

```text
Strong candidate rows = 183
Unique strong probes  = 180
```

---

### 6.12 Cross-Cluster Overlap

Only three strong probes appear in more than one cluster comparison:

```text
D10326_s_at
U52112_rna5_at
X95190_at
```

Their direction changes between Cluster 1 and Cluster 2.

For example:

```text
                    X95190_at

              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼

          Cluster 1             Cluster 2

             Higher                Lower

         effect +0.844         effect -0.897
```

This is not automatically considered an error.

Because the analysis uses one-vs-rest comparisons, the same probe can distinguish different clusters in opposite directions.

---

## 6.13 Affymetrix Probe Annotation

The Golub dataset uses older Affymetrix probe identifiers such as:

```text
L20298_at
M94856_at
X56468_at
U57342_at
L12392_at
```

These identifiers are useful statistically but are difficult to interpret biologically.

Therefore, strong probes are mapped to gene annotations using:

```text
GEO Platform GPL80
Affymetrix Hgu6800
```

The project uses:

```text
GEOparse
```

to retrieve and process the GPL80 annotation table.

The annotation stage converts:

```text
Affymetrix Probe ID
        │
        ▼
Gene Symbol
        │
        ▼
Gene Title
        │
        ▼
Entrez Gene Information
        │
        ▼
Biologically Interpretable Candidate
```

---

### 6.14 Annotation Results

The strong-candidate annotation produced:

```text
Strong candidate rows : 183
Unique probes         : 180
Mapped rows           : 170 / 183
Unmapped rows         : 13
Mapping rate          : approximately 92.9%
```

Some older Affymetrix probes cannot be mapped cleanly to a current gene symbol.

These unmapped probes are retained separately rather than silently discarded.

They are saved in:

```text
reports/biomarkers/unmapped_strong_probes.csv
```

---

### 6.15 Example Probe-to-Gene Mappings

Examples from the annotation stage include:

| Probe ID | Gene Symbol | Gene Description |
|---|---|---|
| `L20298_at` | `CBFB` | core-binding factor, beta subunit |
| `M94856_at` | `FABP5` | fatty acid binding protein 5 |
| `X56468_at` | `YWHAQ` | 14-3-3 protein theta |
| `X65488_at` | `HNRNPU` | heterogeneous nuclear ribonucleoprotein U |
| `M13450_at` | `ESD` | esterase D |
| `U20998_at` | `SRP9` | signal recognition particle 9kDa |
| `X13238_at` | `COX6C` | cytochrome c oxidase subunit VIc |
| `D55654_at` | `MDH1` | malate dehydrogenase 1 |
| `D38551_at` | `RAD21` | RAD21 cohesin complex component |
| `D13370_at` | `APEX1` | DNA repair enzyme |
| `D63881_at` | `SUZ12` | polycomb repressive complex 2 subunit |

---

## 6.16 Final Candidate Ranking

After annotation, the pipeline produces the top mapped candidate genes for each cluster.

Candidates are ranked using:

```text
Lower FDR
    │
    ▼
Stronger Statistical Evidence
    │
    +
    ▼
Larger Absolute Rank-Biserial Effect
    │
    ▼
Top Exploratory Candidates
```

The final report keeps:

```text
Top 10 annotated candidates per cluster
```

---

### Cluster 0 — Example Top Candidates

Current mapped candidates include:

```text
CBFB
FABP5
YWHAQ
HNRNPU
ESD
H3F3A / H3F3AP4 / H3F3B
SRP9
SET / SETSIP
COX6C
MDH1
```

Cluster 0 is dominated by genes with:

```text
higher expression relative to the rest
```

which is consistent with the strong-candidate direction summary:

```text
119 higher
4 lower
```

---

### Cluster 1 — Top Candidates

The current final ranking includes:

| Rank | Gene | Probe | Direction | Rank-Biserial |
|---:|---|---|---|---:|
| 1 | MLF2 | `U57342_at` | Lower | -0.8888 |
| 2 | AAMP | `M95627_at` | Lower | -0.8641 |
| 3 | ARHGEF2 | `U72206_at` | Lower | -0.8579 |
| 4 | PKLR | `D10326_s_at` | Higher | 0.8471 |
| 5 | CHMP1A | `U58048_at` | Lower | -0.8440 |
| 6 | ACOX2 | `X95190_at` | Higher | 0.8440 |
| 7 | RENBP | `U52112_rna5_at` | Lower | -0.8286 |
| 8 | TAX1BP1 | `U33821_at` | Lower | -0.8278 |
| 9 | TLK2 | `AB004884_at` | Lower | -0.8224 |
| 10 | PTPN1 | `M31724_at` | Lower | -0.8193 |

Cluster 1 therefore shows a strong tendency toward:

```text
lower expression among its strongest candidate genes
```

with:

```text
3 higher
13 lower
```

strong candidates overall.

---

### Cluster 2 — Top Candidates

The current final ranking includes:

| Rank | Gene | Probe | Direction | Rank-Biserial |
|---:|---|---|---|---:|
| 1 | HTT | `L12392_at` | Higher | 0.9272 |
| 2 | GOLGA2 | `L06147_at` | Lower | -0.9130 |
| 3 | ADAM15 | `U41767_s_at` | Higher | 0.9104 |
| 4 | GAST | `M15958_at` | Lower | -0.9015 |
| 5 | PIK3R2 | `X80907_at` | Higher | 0.8988 |
| 6 | ACOX2 | `X95190_at` | Lower | -0.8971 |
| 7 | TERT | `AF015950_at` | Lower | -0.8864 |
| 8 | INHA | `X04445_rna1_s_at` | Lower | -0.8829 |
| 9 | CSN2 | `L10615_s_at` | Higher | 0.8829 |
| 10 | H3F3A / H3F3AP4 / H3F3B | `X05855_at` | Lower | -0.8776 |

Cluster 2 shows a more balanced strong-expression pattern:

```text
24 higher
20 lower
```

---

## 6.17 Phase 6 Output Files

The Phase 6 workflow generates:

```text
reports/
│
└── biomarkers/
    │
    ├── cluster_0_vs_rest.csv
    │
    ├── cluster_1_vs_rest.csv
    │
    ├── cluster_2_vs_rest.csv
    │
    ├── all_biomarker_results.csv
    │
    ├── significant_biomarkers.csv
    │
    ├── top20_candidates_per_cluster.csv
    │
    ├── biomarker_summary.csv
    │
    ├── strong_candidates.csv
    │
    ├── strong_candidates_annotated.csv
    │
    ├── top10_annotated_candidates_per_cluster.csv
    │
    └── unmapped_strong_probes.csv
```

The main final Phase 6 report is:

```text
reports/biomarkers/
top10_annotated_candidates_per_cluster.csv
```

---

## 6.18 Phase 6 Scripts

The Phase 6 workflow is separated into three executable stages.

```text
scripts/
│
├── run_phase6_biomarkers.py
│       │
│       └── Statistical biomarker discovery
│
├── annotate_phase6_probes.py
│       │
│       └── GPL80 probe → gene annotation
│
└── finalize_phase6_biomarkers.py
        │
        └── Final top candidate reports
```

Run the statistical analysis using:

```bash
python -m scripts.run_phase6_biomarkers
```

Run probe annotation using:

```bash
python -m scripts.annotate_phase6_probes
```

Create the final candidate report using:

```bash
python -m scripts.finalize_phase6_biomarkers
```

---

## 6.19 Phase 6 Testing

A dedicated biomarker unit-test suite was added.

Current Phase 6 tests verify:

```text
Biomarker result shape                    ✅
Expected output columns                   ✅
FDR range                                 ✅
Positive expression direction             ✅
Negative expression direction             ✅
Sample-ID alignment                       ✅
Sample mismatch rejection                 ✅
Unknown cluster rejection                 ✅
NaN rejection                             ✅
Constant-gene handling                    ✅
```

Phase 6 biomarker tests:

```text
10 passed
```

The full project regression suite reached:

```text
90 passed
```

A final full regression run will be performed immediately before the Phase 6 Git commit.

---

## 6.20 Phase 6 Dependencies

Phase 6 adds:

```text
statsmodels
GEOparse
```

`statsmodels` is used for:

```text
Benjamini-Hochberg FDR correction
```

and `GEOparse` is used for:

```text
GPL80 probe annotation
```

The installed GEOparse version used during development is:

```text
GEOparse 2.0.4
```

---

## 6.21 Scientific Interpretation

Phase 6 creates a bridge between unsupervised cluster discovery and biological interpretation.

```text
Phase 5
Cluster Discovery
      │
      ▼
Three Exploratory Groups
      │
      ▼
Phase 6
Gene-Level Comparison
      │
      ▼
Candidate Cluster-Associated Genes
      │
      ▼
Probe Annotation
      │
      ▼
Biologically Interpretable Gene Symbols
      │
      ▼
Phase 7
Pathway Enrichment
```

Phase 5 answered:

> **Which patients tend to group together?**

Phase 6 answers:

> **Which gene-expression differences characterize those groups?**

Phase 7 will investigate:

> **Do those candidate genes converge on meaningful biological pathways?**

---

### Interpretation Caution

The Phase 6 genes should **not** currently be described as clinically validated cancer biomarkers.

The reasons include:

- the Golub dataset contains only 72 samples
- clusters were discovered using the same exploratory dataset
- biomarker associations were then tested on that same dataset
- the clusters themselves require external validation
- the strong-effect threshold is an exploratory project choice
- some probes come from an older microarray annotation platform

The appropriate terminology is therefore:

```text
Candidate biomarkers
```

or preferably:

```text
Cluster-associated candidate genes
```

Independent cohorts and biological validation would be required before these genes could be considered validated leukemia biomarkers.

---

## Phase 6 Summary

```text
                     PHASE 6 SUMMARY

                Consensus k = 3 Clusters
                           │
                           ▼
                  2,000 Selected Genes
                           │
                           ▼
                 Cluster-vs-Rest Testing
                           │
                           ▼
                  Mann-Whitney U Test
                           │
                           ▼
                    BH-FDR Correction
                           │
                           ▼
          ┌────────────────┴────────────────┐
          ▼                                 ▼
 Median Expression Difference      Rank-Biserial Effect
                                            │
                                            ▼
                                 Effect Threshold Study
                                            │
                                            ▼
                              |rank-biserial| >= 0.8
                                            │
                                            ▼
                                  Strong Candidates
                                            │
                       ┌────────────────────┼────────────────────┐
                       ▼                    ▼                    ▼
                  Cluster 0            Cluster 1            Cluster 2
                     123                   16                   44
                       │                    │                    │
                       └────────────────────┼────────────────────┘
                                            │
                                            ▼
                                  GPL80 Annotation
                                            │
                                            ▼
                                  170 / 183 Mapped
                                            │
                                            ▼
                             Top 10 Candidates per Cluster
                                            │
                                            ▼
                              Ready for Pathway Analysis
```


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
- [x] Biomarker discovery
- [x] Cluster-vs-rest differential-expression testing
- [x] Benjamini-Hochberg FDR correction
- [x] Candidate biomarker ranking

### In Progress

- [ ] PCA explained-variance analysis
- [x] PCA visualization
- [ ] UMAP visualization
- [x] Pathway enrichment
- [x] Biological interpretation
- [ ] SHAP explainability
- [x] Automated report generation
- [x] Streamlit interface
- [ ] End-to-end production pipeline

---

## Project Structure

```text
cancer-subtype-discovery/
│
├── app/
│
├── data/
│   ├── raw/
│   │   ├── golub_dataset.zip
│   │   ├── golub_expr.csv
│   │   └── golub_labels.csv
│   │
│   └── annotation/
│       └── GPL80 annotation data/cache
│
├── logs/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── validation/
│   │   ├── ari_silhouette_results.csv
│   │   ├── bootstrap_results.csv
│   │   ├── consensus_k3_clusters.csv
│   │   ├── gap_results.csv
│   │   └── pac_results.csv
│   │
│   └── biomarkers/
│       ├── cluster_0_vs_rest.csv
│       ├── cluster_1_vs_rest.csv
│       ├── cluster_2_vs_rest.csv
│       │
│       ├── all_biomarker_results.csv
│       ├── significant_biomarkers.csv
│       ├── biomarker_summary.csv
│       │
│       ├── top20_candidates_per_cluster.csv
│       ├── strong_candidates.csv
│       ├── strong_candidates_annotated.csv
│       │
│       ├── top10_annotated_candidates_per_cluster.csv
│       └── unmapped_strong_probes.csv
│
├── scripts/
│   ├── run_phase5_validation.py
│   ├── run_phase6_biomarkers.py
│   ├── annotate_phase6_probes.py
│   └── finalize_phase6_biomarkers.py
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
│   ├── data/
│   │
│   ├── dimensionality/
│   │   ├── __init__.py
│   │   └── core.py
│   │
│   ├── enrichment/
│   │
│   ├── explainability/
│   │
│   ├── pipeline/
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── core.py
│   │
│   ├── reporting/
│   │
│   ├── utils/
│   │
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
├── .gitignore
├── config.yaml
├── LICENSE
├── README.md
└── requirements.txt
```

### Structure by Phase

```text
Phase 1
Data Loading & Validation
        │
        ├── src/data_loader.py
        ├── src/exceptions.py
        └── tests/test_data_loader.py


Phase 2
Preprocessing
        │
        ├── src/preprocessing/
        └── tests/test_preprocessing.py


Phase 3
Dimensionality Reduction
        │
        ├── src/dimensionality/
        └── tests/test_dimensionality.py


Phase 4
Clustering
        │
        ├── src/clustering/
        └── tests/test_clustering.py


Phase 5
Cluster Validation
        │
        ├── src/validation/
        ├── tests/test_validation.py
        ├── scripts/run_phase5_validation.py
        │
        └── reports/validation/
                │
                ├── ari_silhouette_results.csv
                ├── bootstrap_results.csv
                ├── gap_results.csv
                ├── pac_results.csv
                └── consensus_k3_clusters.csv


Phase 6
Biomarker Discovery
        │
        ├── src/biomarkers/
        │       │
        │       ├── __init__.py
        │       └── core.py
        │
        ├── tests/test_biomarkers.py
        │
        ├── scripts/
        │       │
        │       ├── run_phase6_biomarkers.py
        │       ├── annotate_phase6_probes.py
        │       └── finalize_phase6_biomarkers.py
        │
        └── reports/biomarkers/
                │
                ├── cluster_0_vs_rest.csv
                ├── cluster_1_vs_rest.csv
                ├── cluster_2_vs_rest.csv
                ├── all_biomarker_results.csv
                ├── significant_biomarkers.csv
                ├── strong_candidates.csv
                ├── strong_candidates_annotated.csv
                ├── top10_annotated_candidates_per_cluster.csv
                └── unmapped_strong_probes.csv


Phase 7
Pathway Enrichment
        │
        └── src/enrichment/
              ⏳ Next Phase


Phase 8
Explainability
        │
        └── src/explainability/
              ⏳ Planned


Final Reporting / Application
        │
        ├── src/reporting/
        ├── src/pipeline/
        ├── app/
        └── templates/
              ⏳ Planned
```


## Installation

### Requirements

- Python 3.11+
- Git
- `pip`
- Virtual-environment support

The current development environment has also been tested with Python 3.13.


### Main Python Dependencies

The project uses the following major libraries:

```text
pandas
numpy
scipy
scikit-learn
statsmodels
GEOparse
PyYAML
pytest
pytest-cov
```

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
[x] Biomarker discovery

[ ] Mann-Whitney testing
[ ] BH-FDR correction
[ ] Candidate marker ranking
[x] PCA visualization
[ ] UMAP
[ ] Pathway enrichment
[ ] Biological interpretation
[ ] Explainability / SHAP
[x] Automated reporting (HTML/ZIP)
[x] Streamlit application
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
Phase 6 — Biomarker discovery           ✅
Phase 7 — Pathway enrichment            ✅ Complete
Phase 8 — Subtype characterization      ✅ Complete
Phase 9 — Reporting / application       ✅ Local app available
```

Phase 5 has produced an exploratory `k=3` consensus partition for downstream analysis.

Phase 6 performed statistical identification of genes that distinguish the discovered clusters.

Phase 7 performed background-aware over-representation analysis (ORA) across four pathway libraries.

Phase 8 synthesized these findings into an integrated biological interpretation and subtype characterization layer.

The project remains an exploratory research pipeline and does not currently make clinical or diagnostic claims.
### Start the dashboard reliably

From this repository folder run:

```sh
python run.py
```

Open http://127.0.0.1:8509. The launcher selects the complete project environment; a bare `streamlit` command may select a different Anaconda installation without the PDF dependencies. `python run.py --check` checks the selected environment without starting the website.

---

## Phase 7: Biological Pathway Enrichment

Phase 7 performs exploratory over-representation analysis (ORA) using Phase 6 cluster-specific biomarkers.

### Method
- GPL80 probe annotation; ambiguous mappings and AFFX controls excluded.
- Four libraries: KEGG 2021 Human, Reactome 2022, MSigDB Hallmark 2020, and GO Biological Process 2021.
- One-sided hypergeometric testing with an explicit effective background.
- Pathway-size limits: 5–500 genes.
- Benjamini–Hochberg FDR correction separately for each cluster and library.
- Significance threshold: FDR <= 0.05.

### Recorded results
- 2,000 tested probes.
- 1,580 unique mapped background genes.
- 8,724 pathway tests.
- 254 significant pathway terms at FDR <= 0.05.
- Cluster 0: 254 significant terms.
- Cluster 1: 0 significant terms.
- Cluster 2: 0 significant terms.

These results are exploratory. Enriched pathways may share genes, and statistical enrichment does not establish causality or clinical validity. Biological interpretation and independent validation remain necessary.

### Outputs
Results are stored in `reports/enrichment/`, including all pathway results, significant pathways, summary counts, mapping information, and run metadata.

### Run
Run `python scripts/run_phase7_enrichment.py` from the project root. The script requires the Phase 6 biomarker results, GPL80 annotation, and four configured GMT files under `data/annotation/gene_sets/`.

---

## Phase 8: Integrated Biological Interpretation and Subtype Characterization

Phase 8 integrates discovered clusters (Phase 5), biomarker candidates (Phase 6), and pathway enrichment profiles (Phase 7) into a unified computational interpretation and characterization layer.

### Objectives
- Characterize the biological and phenotypic properties of the discovered Consensus $k=3$ subgroups.
- Integrate biomarker statistics (effect size, FDR, median difference, direction) with probe-to-gene annotations.
- Synthesize pathway enrichment findings and evaluate underlying cellular mechanisms.
- Conduct rigorous cross-cluster statistical comparisons (Fisher's exact tests for known ALL/AML composition, biomarker Jaccard overlap, PCA centroid distances).
- Formulate an evidence-based tiered scientific interpretation distinguishing directly observed data, statistical associations, biological hypotheses, and validation requirements.

### Scientific Findings by Subgroup
1. **Cluster 0 (n = 14, 19.4%): Pure ALL / Subgroup Associated with High Ribosomal & Metabolic Gene Expression**
   - **Phenotypic alignment:** 100% ALL (14/14), 0% AML. Complete absence of AML yields a zero cell in contingency testing ($p = 6.46 \times 10^{-6}$ vs Cluster 2).
   - **Biomarkers:** 743 FDR-significant probes, 123 strong candidates ($|r_{\text{rb}}| \ge 0.8$, 119 higher, 4 lower). Top markers include *CBFB*, *FABP5*, *YWHAQ*, *HNRNPU*, *ESD*.
   - **Pathways:** 254 significant terms (11 KEGG, 174 Reactome, 4 Hallmark, 65 GO BP). Statistically over-represented for ribosomal translation initiation, proteasome components (*PSMA*/*PSMB*), oxidative phosphorylation, and *MYC* targets.
2. **Cluster 1 (n = 35, 48.6%): Dominant Intermediate / Broad Transcriptional Variation Subgroup**
   - **Phenotypic alignment:** 27 ALL (77.1%), 8 AML (22.9%).
   - **Biomarkers:** 1,192 FDR-significant probes, 16 strong candidates (3 higher, 13 lower). Top markers include *MLF2*, *AAMP*, *ARHGEF2*, *PKLR*.
   - **Pathways:** 0 significant terms at library-wide FDR $\le 0.05$, reflecting broad transcriptional modulation across lymphoid and myeloid backgrounds without a singular dominant pathway signature.
3. **Cluster 2 (n = 23, 31.9%): Myeloid-Enriched / AML Predominant Subgroup**
   - **Phenotypic alignment:** 17 AML (73.9%), 6 ALL (26.1%). Highly significant enrichment for AML relative to Cluster 1 (Odds Ratio = 9.56, $p = 1.63 \times 10^{-4}$) and Cluster 0 ($p = 6.46 \times 10^{-6}$).
   - **Biomarkers:** 933 FDR-significant probes, 44 strong candidates (24 higher, 20 lower). Top markers include *HTT*, *GOLGA2*, *ADAM15*, *PIK3R2*, *ACOX2*.
   - **Pathways:** 0 significant terms at library-wide FDR $\le 0.05$ after background adjustment, despite clear differential expression of individual myeloid candidate genes.

### Cross-Cluster Comparison
- **ALL/AML Contingency:** Fisher's exact tests confirm robust separation of AML predominance in Cluster 2 versus pure ALL in Cluster 0. Zero AML in Cluster 0 produces an infinite odds ratio (+∞) due to a zero cell in the 2×2 table.
- **Biomarker Overlap:** Jaccard similarity of FDR-significant probes is 0.276 between Cluster 0 and 1, 0.142 between Cluster 0 and 2, and 0.481 between Cluster 1 and 2.
- **PCA Centroid Distances:** Euclidean distance in (PC1, PC2, PC3) space is 38.17 (C0 vs C1), 38.24 (C0 vs C2), and 33.41 (C1 vs C2).

### Evidentiary Tiers & Scientific Limitations
- **Tier 1 (Direct Observations):** Exact sample counts, label distributions, biomarker test results, pathway counts.
- **Tier 2 (Statistical Associations):** Significance under non-parametric tests (Mann-Whitney U), BH-FDR correction, Fisher's exact contingency tests, and hypergeometric ORA.
- **Tier 3 (Biological Interpretations):** Putative elevated translational and biosynthetic expression in Cluster 0 ALL blasts; expected lineage-associated divergence in Cluster 2.
- **Tier 4 (Validation Hypotheses):** Cohort generalizability in independent external datasets; cross-platform validation using modern RNA-seq; orthogonal molecular verification via RT-qPCR/proteomics.
- **Limitations:** Small historical cohort ($n = 72$), legacy Hu6800 platform, no longitudinal clinical survival or outcome annotations. The study does not infer patient prognosis, therapeutic response, or clinical utility.

### Output Artifacts (`reports/phase8/`)
- `subtype_characterization_summary.csv`: Cluster-level summary metrics.
- `cluster_biomarker_summary.csv`: Annotated candidate biomarkers with effect sizes and directions.
- `cluster_pathway_summary.csv`: Significant pathway enrichments per cluster.
- `cross_cluster_comparison.csv`: Formal statistical comparisons between clusters.
- `subtype_characterization_report.md`: Complete evidence-based scientific report.
- `phase8_run_metadata.json`: Provenance metadata, input checksums, environment parameters.
- `figures/`: High-resolution figures (`cluster_sample_distribution.png`, `biomarker_counts_by_cluster.png`, `pathway_enrichment_counts_by_cluster.png`, `top_biomarkers_effect_size.png`, `top_pathways_cluster0.png`, `cross_cluster_pca_distribution.png`).

### Execution
```sh
# Run Phase 8 analysis from repository root
python scripts/run_phase8_characterization.py
```
