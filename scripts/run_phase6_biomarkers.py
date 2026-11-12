from pathlib import Path

import pandas as pd

from src.biomarkers import discover_biomarkers
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor


DATA_PATH = "data/raw/golub_expr.csv"
LABEL_PATH = "data/raw/golub_labels.csv"

CLUSTER_PATH = "reports/validation/consensus_k3_clusters.csv"

OUTPUT_DIR = Path("reports/biomarkers")

N_TOP_GENES = 2000
TOP_N_REPORT = 20


def select_strong_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Apply the recorded Phase 6 exploratory candidate filter."""
    candidates = results.loc[
        (results["fdr"] < 0.05) & (results["rank_biserial"].abs() >= 0.8)
    ].copy()
    candidates["direction"] = candidates["median_difference"].map(
        lambda value: "higher" if value > 0 else "lower" if value < 0 else "unchanged"
    )
    return candidates


def main():
    print("=" * 70)
    print("PHASE 6 — BIOMARKER DISCOVERY")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load original Golub expression data
    # ---------------------------------------------------------

    loader = DataLoader(
        DATA_PATH,
        LABEL_PATH,
    )

    expression, _ = loader.load()

    print()
    print("Expression data loaded")
    print(f"Samples: {len(expression)}")
    print(
        f"Gene columns: "
        f"{expression.shape[1] - 1}"
    )

    # ---------------------------------------------------------
    # 2. Select same top-variable genes used upstream
    # ---------------------------------------------------------

    preprocessor = Preprocessor(
        n_top_genes=N_TOP_GENES
    )

    selected_genes = (
        preprocessor.select_variable_genes(
            expression
        )
    )

    selected_expression = (
        expression[
            ["sample_id", *selected_genes]
        ]
        .copy()
        .set_index("sample_id")
    )

    print()
    print(
        f"Selected genes using MAD: "
        f"{len(selected_genes)}"
    )

    # ---------------------------------------------------------
    # 3. Load consensus k=3 cluster assignments
    # ---------------------------------------------------------

    cluster_df = pd.read_csv(
        CLUSTER_PATH
    )

    if "sample_id" not in cluster_df.columns:
        raise ValueError(
            "Cluster file must contain "
            "'sample_id'."
        )

    # Prefer the expected column name.
    # If there is exactly one other column,
    # use that as the cluster-label column.
    if "cluster" in cluster_df.columns:
        cluster_column = "cluster"

    else:
        possible_columns = [
            column
            for column in cluster_df.columns
            if column != "sample_id"
        ]

        if len(possible_columns) != 1:
            raise ValueError(
                "Could not identify cluster "
                "label column. Columns found: "
                f"{cluster_df.columns.tolist()}"
            )

        cluster_column = possible_columns[0]

    cluster_labels = (
        cluster_df
        .set_index("sample_id")[
            cluster_column
        ]
        .astype(int)
    )

    print()
    print("Consensus clusters loaded")

    print(
        cluster_labels
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # 4. Confirm sample alignment
    # ---------------------------------------------------------

    expression_samples = set(
        selected_expression.index
    )

    cluster_samples = set(
        cluster_labels.index
    )

    if expression_samples != cluster_samples:
        missing_clusters = (
            expression_samples
            - cluster_samples
        )

        unknown_clusters = (
            cluster_samples
            - expression_samples
        )

        raise ValueError(
            "Expression/cluster sample mismatch.\n"
            f"Missing cluster labels: "
            f"{missing_clusters}\n"
            f"Unknown samples: "
            f"{unknown_clusters}"
        )

    print()
    print(
        "Sample alignment: OK "
        f"({len(expression_samples)} samples)"
    )

    # ---------------------------------------------------------
    # 5. Create output directory
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    significant_results = []
    top_candidates = []

    # ---------------------------------------------------------
    # 6. Cluster-vs-rest biomarker analysis
    # ---------------------------------------------------------

    for cluster in sorted(
        cluster_labels.unique()
    ):
        print()
        print("-" * 70)
        print(
            f"Cluster {cluster} vs Rest"
        )
        print("-" * 70)

        result = discover_biomarkers(
            expression=selected_expression,
            cluster_labels=cluster_labels,
            target_cluster=int(cluster),
        )

        result.to_csv(
            OUTPUT_DIR
            / f"cluster_{cluster}_vs_rest.csv",
            index=False,
        )

        all_results.append(result)

        significant = result[
            result["significant"]
        ].copy()

        significant_results.append(
            significant
        )

        print(
            f"Genes tested: {len(result)}"
        )

        print(
            "FDR < 0.05: "
            f"{len(significant)}"
        )

        # Top candidates:
        # FDR first, then stronger effect.
        ranked = result.copy()

        ranked[
            "abs_rank_biserial"
        ] = ranked[
            "rank_biserial"
        ].abs()

        ranked = ranked.sort_values(
            by=[
                "fdr",
                "abs_rank_biserial",
            ],
            ascending=[
                True,
                False,
            ],
        )

        top = ranked.head(
            TOP_N_REPORT
        ).copy()

        top_candidates.append(top)

        print()
        print("Top 5 candidate genes:")

        print(
            top[
                [
                    "gene",
                    "fdr",
                    "median_difference",
                    "rank_biserial",
                    "significant",
                ]
            ]
            .head(5)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # 7. Combine results
    # ---------------------------------------------------------

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "all_biomarker_results.csv",
        index=False,
    )

    # This file is required by the annotation/finalization stages. Previously
    # it existed only as a saved artifact and was not regenerated by this runner.
    select_strong_candidates(combined).to_csv(
        OUTPUT_DIR / "strong_candidates.csv", index=False,
    )

    significant_combined = pd.concat(
        significant_results,
        ignore_index=True,
    )

    significant_combined.to_csv(
        OUTPUT_DIR
        / "significant_biomarkers.csv",
        index=False,
    )

    top_combined = pd.concat(
        top_candidates,
        ignore_index=True,
    )

    top_combined.to_csv(
        OUTPUT_DIR
        / "top20_candidates_per_cluster.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------

    summary_rows = []

    for cluster in sorted(
        cluster_labels.unique()
    ):
        cluster_result = combined[
            combined["cluster"] == cluster
        ]

        summary_rows.append(
            {
                "cluster": int(cluster),
                "samples": int(
                    (
                        cluster_labels
                        == cluster
                    ).sum()
                ),
                "genes_tested": len(
                    cluster_result
                ),
                "fdr_significant_genes": int(
                    cluster_result[
                        "significant"
                    ].sum()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "biomarker_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("PHASE 6 BIOMARKER DISCOVERY COMPLETE")
    print("=" * 70)

    print()
    print(summary.to_string(index=False))

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()