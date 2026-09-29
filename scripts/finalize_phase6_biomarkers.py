from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "reports/biomarkers/strong_candidates_annotated.csv"
)

OUTPUT_DIR = Path(
    "reports/biomarkers"
)

TOP_N = 10


def main():
    print("=" * 70)
    print("PHASE 6 — FINAL BIOMARKER CANDIDATES")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    # ---------------------------------------------------------
    # 1. Separate mapped and unmapped probes
    # ---------------------------------------------------------

    mapped = df[
        df["gene_symbol"].notna()
        & (df["gene_symbol"].astype(str).str.strip() != "")
    ].copy()

    unmapped = df[
        ~df.index.isin(mapped.index)
    ].copy()

    print()
    print(f"Total strong rows : {len(df)}")
    print(f"Mapped rows       : {len(mapped)}")
    print(f"Unmapped rows     : {len(unmapped)}")

    # Save unmapped probes for transparency
    unmapped.to_csv(
        OUTPUT_DIR / "unmapped_strong_probes.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Calculate absolute effect size
    # ---------------------------------------------------------

    mapped["abs_rank_biserial"] = (
        mapped["rank_biserial"].abs()
    )

    # ---------------------------------------------------------
    # 3. Rank candidates within each cluster
    # ---------------------------------------------------------

    mapped = mapped.sort_values(
        by=[
            "cluster",
            "fdr",
            "abs_rank_biserial",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    final_candidates = (
        mapped
        .groupby(
            "cluster",
            group_keys=False,
        )
        .head(TOP_N)
        .copy()
    )

    # Add rank within cluster
    final_candidates["candidate_rank"] = (
        final_candidates
        .groupby("cluster")
        .cumcount()
        + 1
    )

    # ---------------------------------------------------------
    # 4. Keep clean report columns
    # ---------------------------------------------------------

    columns = [
        "candidate_rank",
        "cluster",
        "gene",
        "gene_symbol",
        "gene_title",
        "direction",
        "fdr",
        "median_in_cluster",
        "median_out_cluster",
        "median_difference",
        "rank_biserial",
    ]

    columns = [
        column
        for column in columns
        if column in final_candidates.columns
    ]

    final_candidates = final_candidates[
        columns
    ]

    # ---------------------------------------------------------
    # 5. Save final candidate report
    # ---------------------------------------------------------

    final_candidates.to_csv(
        OUTPUT_DIR
        / "top10_annotated_candidates_per_cluster.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Print candidates
    # ---------------------------------------------------------

    for cluster in sorted(
        final_candidates["cluster"].unique()
    ):
        cluster_df = final_candidates[
            final_candidates["cluster"] == cluster
        ]

        print()
        print("-" * 70)
        print(f"Cluster {cluster}")
        print("-" * 70)

        print(
            cluster_df[
                [
                    "candidate_rank",
                    "gene_symbol",
                    "gene",
                    "direction",
                    "fdr",
                    "rank_biserial",
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------

    summary = (
        mapped
        .groupby(
            ["cluster", "direction"]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print()
    print("=" * 70)
    print("STRONG MAPPED CANDIDATE SUMMARY")
    print("=" * 70)
    print()
    print(summary)

    print()
    print(
        "Saved:"
        " reports/biomarkers/"
        "top10_annotated_candidates_per_cluster.csv"
    )

    print(
        "Saved:"
        " reports/biomarkers/"
        "unmapped_strong_probes.csv"
    )


if __name__ == "__main__":
    main()