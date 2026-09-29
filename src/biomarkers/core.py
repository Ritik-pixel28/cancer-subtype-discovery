from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def discover_biomarkers(
    expression: pd.DataFrame,
    cluster_labels: pd.Series,
    target_cluster: int,
) -> pd.DataFrame:
    """
    Discover candidate biomarkers for one cluster versus all other samples.

    expression:
        Rows = samples
        Columns = genes
        Index = sample IDs

    cluster_labels:
        Cluster label for every sample.
        Index must contain sample IDs.

    target_cluster:
        Cluster to compare against all remaining samples.
    """

    if expression.empty:
        raise ValueError(
            "Expression matrix must not be empty."
        )

    if cluster_labels.empty:
        raise ValueError(
            "Cluster labels must not be empty."
        )

    if not expression.index.is_unique:
        raise ValueError(
            "Expression matrix contains duplicate sample IDs."
        )

    if not cluster_labels.index.is_unique:
        raise ValueError(
            "Cluster labels contain duplicate sample IDs."
        )

    if set(expression.index) != set(cluster_labels.index):
        raise ValueError(
            "Expression matrix and cluster labels must contain the same samples."
        )

    non_numeric = expression.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Expression matrix must be numeric. "
            f"Non-numeric columns: {non_numeric[:5]}"
        )

    if expression.isnull().any(axis=None):
        raise ValueError(
            "Expression matrix contains missing values."
        )

    if not np.isfinite(expression.to_numpy()).all():
        raise ValueError(
            "Expression matrix contains infinite values."
        )

    # Align labels using sample IDs
    aligned_labels = cluster_labels.reindex(
        expression.index
    )

    if target_cluster not in aligned_labels.unique():
        raise ValueError(
            f"Target cluster {target_cluster} does not exist."
        )

    target_mask = aligned_labels == target_cluster
    rest_mask = ~target_mask

    n_target = int(target_mask.sum())
    n_rest = int(rest_mask.sum())

    if n_target == 0 or n_rest == 0:
        raise ValueError(
            "Both target cluster and rest group must contain samples."
        )

    results = []

    for gene in expression.columns:

        target_values = expression.loc[
            target_mask,
            gene
        ].to_numpy()

        rest_values = expression.loc[
            rest_mask,
            gene
        ].to_numpy()

        # Handle genes with exactly the same value in every sample.
        # Such genes contain no information for distinguishing the groups.
        combined_values = np.concatenate(
            [target_values, rest_values]
        )
        if np.all(combined_values == combined_values[0]):
            statistic = (
                n_target * n_rest
            ) / 2.0

            p_value = 1.0

        else:
            statistic, p_value = mannwhitneyu(
                target_values,
                rest_values,
                alternative="two-sided",
            )

        median_target = float(
            np.median(target_values)
        )

        median_rest = float(
            np.median(rest_values)
        )

        median_difference = (
            median_target - median_rest
        )

        # Effect size based on Mann-Whitney U
        rank_biserial = (
            2 * float(statistic)
            / (n_target * n_rest)
            - 1
        )

        results.append(
            {
                "gene": gene,
                "cluster": target_cluster,
                "mann_whitney_u": float(statistic),
                "p_value": float(p_value),
                "median_in_cluster": median_target,
                "median_out_cluster": median_rest,
                "median_difference": float(
                    median_difference
                ),
                "rank_biserial": float(
                    rank_biserial
                ),
            }
        )

    results_df = pd.DataFrame(results)

    _, corrected_pvalues, _, _ = multipletests(
        results_df["p_value"],
        alpha=0.05,
        method="fdr_bh",
    )

    results_df["fdr"] = corrected_pvalues

    results_df["significant"] = (
        results_df["fdr"] < 0.05
    )

    results_df = results_df.sort_values(
        by=["fdr", "p_value"],
        ascending=True,
    ).reset_index(drop=True)

    return results_df