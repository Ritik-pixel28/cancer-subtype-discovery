from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.clustering import (
    ConsensusClusterer,
    GMMClusterer,
    HierarchicalClusterer,
    KMeansClusterer,
)
from src.data_loader import DataLoader
from src.dimensionality import PCAProcessor
from src.preprocessing import Preprocessor
from src.validation import (
    bootstrap_stability,
    compute_ari,
    compute_gap_statistic,
    compute_pac_curve,
    compute_silhouette,
    select_k_by_gap,
    select_k_by_pac,
)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load Golub expression data and known labels
    # ---------------------------------------------------------

    loader = DataLoader(
        expression_path="data/raw/golub_expr.csv",
        labels_path="data/raw/golub_labels.csv",
    )

    expression, labels = loader.load()

    print("Expression shape:", expression.shape)
    print("Labels shape:", labels.shape)

    # ---------------------------------------------------------
    # 2. Select top 2000 variable genes using MAD
    # ---------------------------------------------------------

    preprocessor = Preprocessor(n_top_genes=2000)

    selected_genes = preprocessor.select_variable_genes(
        expression
    )

    print("Selected genes:", len(selected_genes))

    selected_expression = expression[
        ["sample_id", *selected_genes]
    ].copy()

    # ---------------------------------------------------------
    # 3. Standardize selected genes
    # ---------------------------------------------------------

    sample_ids = selected_expression["sample_id"].copy()

    standardized = preprocessor.standardize_genes(
        selected_expression
    )

    # Preserve patient IDs through PCA and clustering.
    standardized.index = sample_ids

    if not np.isfinite(standardized.to_numpy()).all():
        raise ValueError(
            "Standardized expression contains NaN or infinite values."
        )

    print("Standardized shape:", standardized.shape)

    # ---------------------------------------------------------
    # 4. PCA: 2000 genes -> 30 components
    # ---------------------------------------------------------

    pca = PCAProcessor(n_components=30)

    pca_data = pca.fit_transform(standardized)

    print("PCA shape:", pca_data.shape)

    # ---------------------------------------------------------
    # 5. Prepare reference ALL / AML labels
    # ---------------------------------------------------------

    reference_labels = (
        labels
        .set_index("sample_id")["cancer_type"]
        .reindex(pca_data.index)
    )

    # ---------------------------------------------------------
    # 6. Compare clustering methods across k = 2..8
    # ---------------------------------------------------------

    results: list[dict[str, float | int | str]] = []

    for k in range(2, 9):
        clusterers = {
            "KMeans": KMeansClusterer(
                n_clusters=k,
                random_state=42,
            ),
            "Ward": HierarchicalClusterer(
                n_clusters=k,
            ),
            "GMM": GMMClusterer(
                n_components=k,
                random_state=42,
            ),
        }

        for method_name, clusterer in clusterers.items():
            cluster_labels = clusterer.fit_predict(
                pca_data
            )

            ari = compute_ari(
                reference_labels,
                cluster_labels,
            )

            silhouette = compute_silhouette(
                pca_data,
                cluster_labels,
            )

            results.append(
                {
                    "method": method_name,
                    "k": k,
                    "ari": ari,
                    "silhouette": silhouette,
                }
            )

    results_df = pd.DataFrame(results)

    print()
    print("Clustering validation: k = 2..8")
    print("--------------------------------")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ---------------------------------------------------------
    # 7. Gap Statistic for k = 2..8
    # ---------------------------------------------------------

    gap_results = compute_gap_statistic(
        X=pca_data,
        clusterer_factory=lambda k: KMeansClusterer(
            n_clusters=k,
            random_state=42,
        ),
        k_values=(2, 3, 4, 5, 6, 7, 8),
        n_reference=20,
        random_state=42,
    )

    gap_candidate_k = select_k_by_gap(
        gap_results
    )

    print()
    print("Gap Statistic: KMeans k = 2..8")
    print("--------------------------------")

    print(
        gap_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"Gap candidate k: {gap_candidate_k}"
    )



    # ---------------------------------------------------------
    # 8. PAC across k = 2..8
    # ---------------------------------------------------------

    pac_results = compute_pac_curve(
        X=pca_data,
        consensus_factory=lambda k: ConsensusClusterer(
            base_clusterer=KMeansClusterer(
                n_clusters=k,
                random_state=42,
            ),
            n_resamples=100,
            p_item=0.8,
            p_feature=1.0,
            random_state=42,
        ),
        k_values=(2, 3, 4, 5, 6, 7, 8),
        lower_bound=0.1,
        upper_bound=0.9,
    )

    pac_candidate_k = select_k_by_pac(
        pac_results
    )

    print()
    print("PAC: Consensus KMeans k = 2..8")
    print("--------------------------------")

    print(
        pac_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"PAC candidate k: {pac_candidate_k}"
    )


        # ---------------------------------------------------------
    # 9. Bootstrap stability for k = 3
    # ---------------------------------------------------------

    bootstrap_results = []

    bootstrap_clusterers = {
        "KMeans": lambda: KMeansClusterer(
            n_clusters=3,
            random_state=42,
        ),
        "Ward": lambda: HierarchicalClusterer(
            n_clusters=3,
        ),
        "GMM": lambda: GMMClusterer(
            n_components=3,
            random_state=42,
        ),
    }

    for method_name, clusterer_factory in bootstrap_clusterers.items():

        stability = bootstrap_stability(
            X=pca_data,
            clusterer_factory=clusterer_factory,
            n_resamples=100,
            sample_fraction=0.8,
            random_state=42,
        )

        bootstrap_results.append(
            {
                "method": method_name,
                "k": 3,
                "stability_mean": stability.mean,
                "ci_lower": stability.ci_lower,
                "ci_upper": stability.ci_upper,
            }
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_results
    )

    print()
    print("Bootstrap stability: k = 3")
    print("--------------------------")

    print(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # ---------------------------------------------------------
    # 10. Evaluate Consensus clustering at k = 3
    # ---------------------------------------------------------

    consensus_clusterer = ConsensusClusterer(
        base_clusterer=KMeansClusterer(
            n_clusters=3,
            random_state=42,
        ),
        n_resamples=100,
        p_item=0.8,
        p_feature=1.0,
        random_state=42,
    )

    consensus_matrix = consensus_clusterer.fit_predict(
        pca_data
    )

    consensus_labels = (
        ConsensusClusterer.cluster_from_consensus(
            consensus_matrix,
            n_clusters=3,
        )
    )

    consensus_ari = compute_ari(
        reference_labels,
        consensus_labels,
    )

    consensus_silhouette = compute_silhouette(
        pca_data,
        consensus_labels,
    )

    print()
    print("Consensus clustering: k = 3")
    print("---------------------------")
    print(f"ARI:        {consensus_ari:.4f}")
    print(f"Silhouette: {consensus_silhouette:.4f}")

    print()
    print("Consensus cluster sizes:")
    print(
        consensus_labels
        .value_counts()
        .sort_index()
    )


        # ---------------------------------------------------------
    # 11. Save exploratory consensus cluster assignments
    # ---------------------------------------------------------

    output_dir = Path("reports/validation")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_assignments = pd.DataFrame(
        {
            "sample_id": consensus_labels.index,
            "cluster": consensus_labels.to_numpy(),
        }
    )

    cluster_assignments.to_csv(
        output_dir / "consensus_k3_clusters.csv",
        index=False,
    )

    results_df.to_csv(
        output_dir / "ari_silhouette_results.csv",
        index=False,
    )

    gap_results.to_csv(
        output_dir / "gap_results.csv",
        index=False,
    )

    pac_results.to_csv(
        output_dir / "pac_results.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        output_dir / "bootstrap_results.csv",
        index=False,
    )

    print()
    print(
        "Saved Phase 5 results to:",
        output_dir,
    )


if __name__ == "__main__":
    main()