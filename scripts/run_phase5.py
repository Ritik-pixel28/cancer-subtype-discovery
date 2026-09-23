"""
Phase 5 integration script: Cluster Validation and Stability on Golub data.

Usage
-----
    python scripts/run_phase5.py

This script runs the full Phase 5 validation pipeline on the actual Golub
leukemia gene-expression dataset.  It does NOT use cancer labels (ALL/AML)
for any of the validation calculations — all evaluation is purely unsupervised.

Pipeline
--------
  1. Load golub_expr.csv
  2. Select top 2,000 variable genes (MAD)
  3. Standardize gene-wise
  4. Reduce to 30 PCA components
  5. For each method × k in the configured k_range:
       - compute silhouette score
       - estimate bootstrap stability (mean/std ARI)
  6. For each k in k_range:
       - build the consensus matrix via ConsensusClusterer
       - validate the matrix structure
       - summarize co-clustering statistics
  7. Print a combined results table
  8. Save results to reports/validation_results.csv

All thresholds and parameters are read from config.yaml — nothing is
hard-coded in this script.

Important
---------
Passing thresholds (ari_threshold_pass, silhouette_threshold_pass) are
project-defined criteria for stability assessment, not claims about
biological correctness.  A high ARI means the clustering is reproducible
under resampling; it does not mean the discovered groups are true cancer
subtypes.
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running from the project root with `python scripts/run_phase5.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import (
    ConsensusClusterer,
    GMMClusterer,
    HierarchicalClusterer,
    KMeansClusterer,
)
from src.config import PipelineConfig
from src.dimensionality import PCAProcessor
from src.preprocessing import Preprocessor
from src.utils.reproducibility import set_global_seed
from src.validation import (
    BootstrapStability,
    ConsensusValidator,
    SilhouetteAnalyzer,
    ValidationSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_preprocess(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Load Golub expression data and run Phases 1-3 preprocessing.

    Returns the PCA-reduced matrix (72 samples × 30 components).
    Cancer labels are intentionally NOT loaded here.
    """
    expr_path = Path(cfg.data.raw_expr_path)
    if not expr_path.exists():
        raise FileNotFoundError(
            f"Expression file not found: {expr_path}\n"
            "Place golub_expr.csv in data/raw/ before running this script."
        )

    print(f"Loading expression data from {expr_path} …")
    expression = pd.read_csv(expr_path)

    # Set sample_id as index for clean downstream handling
    if "sample_id" in expression.columns:
        expression = expression.set_index("sample_id")

    print(f"  Shape: {expression.shape}  (samples × genes)")

    # --- Feature selection: top 2,000 MAD genes ---
    preprocessor = Preprocessor(n_top_genes=cfg.data.n_top_genes)
    # select_variable_genes expects sample_id as a column; reset to match API.
    # copy() de-fragments the DataFrame first to avoid PerformanceWarning.
    expr_with_id = expression.copy().reset_index()
    selected_genes = preprocessor.select_variable_genes(expr_with_id)
    selected_expr = expression[selected_genes]
    print(f"  Selected {len(selected_genes)} variable genes (MAD)")

    # --- Standardize ---
    # standardize_genes expects sample_id column too
    selected_with_id = selected_expr.copy().reset_index()
    standardized = preprocessor.standardize_genes(selected_with_id)
    standardized.index = expression.index
    print(f"  Standardized to zero mean, unit variance")

    # --- PCA ---
    pca = PCAProcessor(n_components=cfg.dimensionality_reduction.pca_n_components)
    pca_matrix = pca.fit_transform(standardized)
    print(
        f"  PCA: {standardized.shape[1]} genes → {pca_matrix.shape[1]} components"
    )
    print(f"  Final matrix: {pca_matrix.shape[0]} samples × {pca_matrix.shape[1]} PCs\n")

    return pca_matrix


def make_kmeans_factory(k: int, seed: int):
    """Return a zero-arg factory for KMeansClusterer(k)."""
    return lambda: KMeansClusterer(n_clusters=k, random_state=seed)


def make_ward_factory(k: int):
    """Return a zero-arg factory for HierarchicalClusterer(k)."""
    return lambda: HierarchicalClusterer(n_clusters=k)


def make_gmm_factory(k: int, seed: int):
    """Return a zero-arg factory for GMMClusterer(k)."""
    return lambda: GMMClusterer(n_components=k, random_state=seed)


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def run_phase5(config_path: str = "config.yaml") -> None:
    cfg = PipelineConfig.from_yaml(config_path)

    set_global_seed(cfg.seed)

    k_range = cfg.clustering.k_range
    val_cfg = cfg.validation
    seed = cfg.seed

    print("=" * 60)
    print("PHASE 5: CLUSTER VALIDATION AND STABILITY")
    print("=" * 60)
    print(f"  k range           : {k_range}")
    print(f"  bootstrap n_iter  : {val_cfg.bootstrap_n_iter}")
    print(f"  bootstrap frac    : {val_cfg.bootstrap_sample_frac}")
    print(f"  ARI pass threshold: {val_cfg.ari_threshold_pass}  (project criterion)")
    print(f"  Sil pass threshold: {val_cfg.silhouette_threshold_pass}  (project criterion)")
    print()

    # Step 1: Preprocess
    X = load_and_preprocess(cfg)

    # Step 2: Build clustering results for all methods × k
    # We run clustering here so validation receives labels, not raw data.
    print("Running clustering for all methods × k values …")
    clustering_results: dict[tuple[str, int], pd.Series] = {}
    factory_map: dict[tuple[str, int], object] = {}

    for k in k_range:
        clustering_results[("kmeans", k)] = KMeansClusterer(
            n_clusters=k, random_state=seed
        ).fit_predict(X)
        factory_map[("kmeans", k)] = make_kmeans_factory(k, seed)

        clustering_results[("ward", k)] = HierarchicalClusterer(
            n_clusters=k
        ).fit_predict(X)
        factory_map[("ward", k)] = make_ward_factory(k)

        clustering_results[("gmm", k)] = GMMClusterer(
            n_components=k, random_state=seed
        ).fit_predict(X)
        factory_map[("gmm", k)] = make_gmm_factory(k, seed)

    print(f"  Done. {len(clustering_results)} clustering solutions computed.\n")

    # Step 3: Silhouette scores
    print("Computing silhouette scores …")
    sil_analyzer = SilhouetteAnalyzer()
    silhouette_df = sil_analyzer.compute_all(X, clustering_results)
    print(f"  Done. {len(silhouette_df)} scores computed.\n")

    # Step 4: Bootstrap stability
    print(
        f"Computing bootstrap stability "
        f"({val_cfg.bootstrap_n_iter} iterations × {len(factory_map)} configs) …"
    )
    bs = BootstrapStability(
        n_iter=val_cfg.bootstrap_n_iter,
        sample_frac=val_cfg.bootstrap_sample_frac,
        random_state=seed,
    )
    stability_df = bs.compute_all(X, factory_map)
    print(f"  Done. {len(stability_df)} stability scores computed.\n")

    # Step 5: Combined summary table
    vs = ValidationSummary(
        ari_threshold=val_cfg.ari_threshold_pass,
        silhouette_threshold=val_cfg.silhouette_threshold_pass,
    )
    summary = vs.build(silhouette_df, stability_df)

    print("=" * 60)
    print("RESULTS: Silhouette + Bootstrap Stability")
    print("  (methods: kmeans, ward, gmm | k = 2..8)")
    print("  'ari_passes' and 'silhouette_passes' are project-defined")
    print("  criteria, not claims about biological validity.")
    print("=" * 60)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
    print()

    # Step 6: Consensus matrix validation
    print("Building and validating consensus matrices …")
    print(
        f"  (n_resamples={cfg.clustering.consensus_n_resamples}, "
        f"p_item={cfg.clustering.consensus_p_item})"
    )
    cv = ConsensusValidator()
    consensus_rows = []

    for k in k_range:
        base = KMeansClusterer(n_clusters=k, random_state=seed)
        cc = ConsensusClusterer(
            base_clusterer=base,
            n_resamples=cfg.clustering.consensus_n_resamples,
            p_item=cfg.clustering.consensus_p_item,
            p_feature=cfg.clustering.consensus_p_feature,
            random_state=cfg.clustering.consensus_random_state,
        )
        M = cc.fit_predict(X)
        checks = cv.validate(M)
        stats = cv.summarize(M)

        all_valid = all(checks.values())
        consensus_rows.append(
            {
                "k": k,
                "matrix_valid": all_valid,
                "mean_consensus": stats["mean_consensus"],
                "median_consensus": stats["median_consensus"],
                "frac_high_consensus": stats["frac_high_consensus"],
            }
        )
        if not all_valid:
            print(f"  WARNING: consensus matrix for k={k} failed checks: {checks}")

    consensus_df = pd.DataFrame(consensus_rows)
    print()
    print("=" * 60)
    print("RESULTS: Consensus Matrix Validation (KMeans base)")
    print("  'frac_high_consensus' = fraction of sample pairs with")
    print("  consensus score >= 0.8 (off-diagonal pairs only).")
    print("=" * 60)
    print(consensus_df.to_string(index=False))
    print()

    # Step 7: Save results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    results_path = reports_dir / "validation_results.csv"
    summary.to_csv(results_path, index=False)

    consensus_path = reports_dir / "consensus_validation.csv"
    consensus_df.to_csv(consensus_path, index=False)

    print(f"Results saved to {results_path} and {consensus_path}")
    print()
    print("Phase 5 complete.")
    print()
    print("NOTE: These metrics describe statistical properties of the")
    print("clustering solutions. They do not constitute biological")
    print("validation of cancer subtypes.")


if __name__ == "__main__":
    run_phase5()
