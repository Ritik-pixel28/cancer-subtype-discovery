"""
Cluster validation utilities for the Cancer Subtype Discovery pipeline.

Known cancer-type labels are used only for external validation.
They must not be used during clustering or feature selection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score


class ClustererProtocol(Protocol):
    """Minimal interface required by bootstrap stability."""

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        ...


@dataclass(frozen=True)
class BootstrapStabilityResult:
    """Summary of bootstrap cluster stability."""

    mean: float
    ci_lower: float
    ci_upper: float
    scores: tuple[float, ...]


def compute_ari(
    reference_labels: pd.Series,
    cluster_labels: pd.Series,
) -> float:
    """
    Compute Adjusted Rand Index between known labels and discovered clusters.

    Parameters
    ----------
    reference_labels:
        Known reference labels, for example ALL / AML.
        Index must contain sample IDs.

    cluster_labels:
        Cluster assignments produced by the clustering stage.
        Index must contain the same sample IDs.

    Returns
    -------
    float
        Adjusted Rand Index.

    Raises
    ------
    ValueError
        If inputs are empty, contain missing values, have duplicate sample IDs,
        or do not contain the same samples.
    """

    if reference_labels.empty or cluster_labels.empty:
        raise ValueError("Reference labels and cluster labels must not be empty.")

    if not reference_labels.index.is_unique:
        raise ValueError("Reference labels contain duplicate sample IDs.")

    if not cluster_labels.index.is_unique:
        raise ValueError("Cluster labels contain duplicate sample IDs.")

    if reference_labels.isna().any():
        raise ValueError("Reference labels contain missing values.")

    if cluster_labels.isna().any():
        raise ValueError("Cluster labels contain missing values.")

    if set(reference_labels.index) != set(cluster_labels.index):
        raise ValueError(
            "Reference labels and cluster labels must contain the same samples."
        )

    # Align known labels to the exact order of the clustering output.
    aligned_reference = reference_labels.reindex(cluster_labels.index)

    return float(
        adjusted_rand_score(
            aligned_reference.to_numpy(),
            cluster_labels.to_numpy(),
        )
    )

def compute_silhouette(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
) -> float:
    """
    Compute the Silhouette Score for discovered clusters.

    Parameters
    ----------
    X:
        Feature matrix used for clustering, typically PCA output.
        Rows are samples and columns are features/components.

    cluster_labels:
        Cluster assignment for each sample.
        Index must contain the same sample IDs as X.

    Returns
    -------
    float
        Mean Silhouette Score.

    Raises
    ------
    ValueError
        If inputs are invalid, samples do not match, or the number
        of clusters is not suitable for Silhouette calculation.
    """

    if X.empty:
        raise ValueError("Feature matrix must not be empty.")

    if cluster_labels.empty:
        raise ValueError("Cluster labels must not be empty.")

    if not X.index.is_unique:
        raise ValueError("Feature matrix contains duplicate sample IDs.")

    if not cluster_labels.index.is_unique:
        raise ValueError("Cluster labels contain duplicate sample IDs.")

    if set(X.index) != set(cluster_labels.index):
        raise ValueError(
            "Feature matrix and cluster labels must contain the same samples."
        )

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Feature matrix must be entirely numeric. "
            f"Non-numeric columns: {non_numeric[:5]}"
        )

    if X.isnull().any(axis=None):
        raise ValueError("Feature matrix contains missing values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature matrix contains infinite values.")

    if cluster_labels.isna().any():
        raise ValueError("Cluster labels contain missing values.")

    # Align labels to the exact sample order of X.
    aligned_labels = cluster_labels.reindex(X.index)

    n_samples = len(X)
    n_clusters = aligned_labels.nunique()

    if n_clusters < 2:
        raise ValueError(
            "Silhouette Score requires at least 2 clusters."
        )

    if n_clusters >= n_samples:
        raise ValueError(
            "Silhouette Score requires fewer clusters than samples."
        )

    return float(
        silhouette_score(
            X.to_numpy(),
            aligned_labels.to_numpy(),
            metric="euclidean",
        )
    )


def _mean_cluster_jaccard(
    reference_labels: pd.Series,
    candidate_labels: pd.Series,
) -> float:
    """
    Compare two clusterings using best-match Jaccard similarity.

    Cluster numbers themselves are arbitrary, so cluster 0 in one run
    does not have to be cluster 0 in another run.
    """

    reference = reference_labels.reindex(candidate_labels.index)

    scores: list[float] = []

    for reference_cluster in reference.unique():
        reference_members = set(
            reference.index[reference == reference_cluster]
        )

        best_score = 0.0

        for candidate_cluster in candidate_labels.unique():
            candidate_members = set(
                candidate_labels.index[
                    candidate_labels == candidate_cluster
                ]
            )

            intersection = len(reference_members & candidate_members)
            union = len(reference_members | candidate_members)

            if union > 0:
                score = intersection / union
                best_score = max(best_score, score)

        scores.append(best_score)

    return float(np.mean(scores))


def bootstrap_stability(
    X: pd.DataFrame,
    clusterer_factory: Callable[[], ClustererProtocol],
    n_resamples: int = 100,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> BootstrapStabilityResult:
    """
    Estimate clustering stability using repeated patient resampling.

    Parameters
    ----------
    X:
        Feature matrix used for clustering, normally PCA output.

    clusterer_factory:
        Function that returns a fresh clustering object with a
        fit_predict() method.

    n_resamples:
        Number of repeated resampling runs.

    sample_fraction:
        Fraction of patients used in each resample.

    random_state:
        Seed for reproducibility.

    Returns
    -------
    BootstrapStabilityResult
        Mean Jaccard stability, 95% confidence interval,
        and individual resampling scores.
    """

    if X.empty:
        raise ValueError("Feature matrix must not be empty.")

    if not X.index.is_unique:
        raise ValueError("Feature matrix contains duplicate sample IDs.")

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Feature matrix must be entirely numeric. "
            f"Non-numeric columns: {non_numeric[:5]}"
        )

    if X.isnull().any(axis=None):
        raise ValueError("Feature matrix contains missing values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature matrix contains infinite values.")

    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")

    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be between 0 and 1.")

    # First cluster the complete dataset.
    baseline_clusterer = clusterer_factory()
    baseline_labels = baseline_clusterer.fit_predict(X)

    if set(baseline_labels.index) != set(X.index):
        raise ValueError(
            "Clusterer output must contain the same samples as X."
        )

    n_samples = len(X)

    sample_size = max(
        2,
        int(np.floor(sample_fraction * n_samples)),
    )

    n_baseline_clusters = baseline_labels.nunique()

    if sample_size < n_baseline_clusters:
        raise ValueError(
            "Resample size is smaller than the number of clusters."
        )

    rng = np.random.default_rng(random_state)

    scores: list[float] = []

    for _ in range(n_resamples):
        sample_positions = rng.choice(
            n_samples,
            size=sample_size,
            replace=False,
        )

        X_resampled = X.iloc[sample_positions]

        clusterer = clusterer_factory()
        resampled_labels = clusterer.fit_predict(X_resampled)

        reference_subset = baseline_labels.reindex(
            X_resampled.index
        )

        score = _mean_cluster_jaccard(
            reference_subset,
            resampled_labels,
        )

        scores.append(score)

    scores_array = np.asarray(scores, dtype=float)

    return BootstrapStabilityResult(
        mean=float(scores_array.mean()),
        ci_lower=float(np.quantile(scores_array, 0.025)),
        ci_upper=float(np.quantile(scores_array, 0.975)),
        scores=tuple(float(score) for score in scores_array),
    )


def _within_cluster_dispersion(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
) -> float:
    """
    Compute total within-cluster squared Euclidean dispersion.

    Smaller values mean samples are more compact around
    their cluster centroids.
    """

    aligned_labels = cluster_labels.reindex(X.index)

    total = 0.0

    for cluster_id in aligned_labels.unique():
        mask = aligned_labels == cluster_id
        cluster_data = X.loc[mask].to_numpy()

        if len(cluster_data) == 0:
            continue

        centroid = cluster_data.mean(axis=0)

        squared_distances = np.sum(
            (cluster_data - centroid) ** 2,
            axis=1,
        )

        total += float(squared_distances.sum())

    return total



def compute_gap_statistic(
    X: pd.DataFrame,
    clusterer_factory: Callable[[int], ClustererProtocol],
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    n_reference: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute Gap Statistic across candidate numbers of clusters.

    Parameters
    ----------
    X:
        Feature matrix used for clustering, normally PCA output.

    clusterer_factory:
        Function receiving k and returning a fresh clusterer.

        Example:
        lambda k: KMeansClusterer(
            n_clusters=k,
            random_state=42,
        )

    k_values:
        Candidate numbers of clusters.

    n_reference:
        Number of random reference datasets generated per k.

    random_state:
        Seed used to generate reproducible reference datasets.

    Returns
    -------
    pd.DataFrame
        One row per candidate k with:
        - k
        - log_wk
        - reference_log_wk_mean
        - gap
        - gap_std
    """

    if X.empty:
        raise ValueError("Feature matrix must not be empty.")

    if not X.index.is_unique:
        raise ValueError(
            "Feature matrix contains duplicate sample IDs."
        )

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Feature matrix must be entirely numeric. "
            f"Non-numeric columns: {non_numeric[:5]}"
        )

    if X.isnull().any(axis=None):
        raise ValueError(
            "Feature matrix contains missing values."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Feature matrix contains infinite values."
        )

    if not k_values:
        raise ValueError(
            "k_values must contain at least one candidate."
        )

    if any(k < 2 for k in k_values):
        raise ValueError(
            "Every k value must be at least 2."
        )

    if any(k >= len(X) for k in k_values):
        raise ValueError(
            "Every k must be smaller than the number of samples."
        )

    if n_reference < 1:
        raise ValueError(
            "n_reference must be at least 1."
        )

    rng = np.random.default_rng(random_state)

    data = X.to_numpy()

    feature_min = data.min(axis=0)
    feature_max = data.max(axis=0)

    results: list[dict[str, float | int]] = []

    for k in k_values:

        # ---------------------------------
        # Real dataset
        # ---------------------------------

        real_clusterer = clusterer_factory(k)
        real_labels = real_clusterer.fit_predict(X)

        real_dispersion = _within_cluster_dispersion(
            X,
            real_labels,
        )

        # Avoid log(0)
        real_dispersion = max(
            real_dispersion,
            np.finfo(float).tiny,
        )

        log_wk = float(np.log(real_dispersion))

        # ---------------------------------
        # Random reference datasets
        # ---------------------------------

        reference_logs: list[float] = []

        for _ in range(n_reference):

            reference_array = rng.uniform(
                low=feature_min,
                high=feature_max,
                size=X.shape,
            )

            reference_df = pd.DataFrame(
                reference_array,
                index=X.index,
                columns=X.columns,
            )

            reference_clusterer = clusterer_factory(k)

            reference_labels = (
                reference_clusterer.fit_predict(reference_df)
            )

            reference_dispersion = (
                _within_cluster_dispersion(
                    reference_df,
                    reference_labels,
                )
            )

            reference_dispersion = max(
                reference_dispersion,
                np.finfo(float).tiny,
            )

            reference_logs.append(
                float(np.log(reference_dispersion))
            )

        reference_array_logs = np.asarray(
            reference_logs,
            dtype=float,
        )

        reference_mean = float(
            reference_array_logs.mean()
        )

        gap = reference_mean - log_wk

        gap_std = float(
            reference_array_logs.std(ddof=1)
            if len(reference_array_logs) > 1
            else 0.0
        )

        results.append(
            {
                "k": k,
                "log_wk": log_wk,
                "reference_log_wk_mean": reference_mean,
                "gap": gap,
                "gap_std": gap_std,
            }
        )

    return pd.DataFrame(results)



def select_k_by_pac(
    pac_results: pd.DataFrame,
) -> int:
    """
    Select the candidate k with the lowest PAC score.
    """

    required_columns = {"k", "pac"}

    if not required_columns.issubset(
        pac_results.columns
    ):
        raise ValueError(
            "PAC results must contain 'k' and 'pac' columns."
        )

    if pac_results.empty:
        raise ValueError(
            "PAC results must not be empty."
        )

    best_row = pac_results.loc[
        pac_results["pac"].idxmin()
    ]

    return int(best_row["k"])



def compute_pac_score(
    consensus_matrix: pd.DataFrame,
    lower_bound: float = 0.1,
    upper_bound: float = 0.9,
) -> float:
    """
    Compute the Proportion of Ambiguous Clustering (PAC).

    PAC is the proportion of pairwise consensus values that fall
    between lower_bound and upper_bound.

    Lower PAC means less ambiguity and generally more stable clustering.

    Parameters
    ----------
    consensus_matrix:
        Square sample-by-sample consensus matrix with values in [0, 1].

    lower_bound:
        Lower ambiguity threshold.

    upper_bound:
        Upper ambiguity threshold.

    Returns
    -------
    float
        PAC score between 0 and 1.
    """

    if consensus_matrix.empty:
        raise ValueError(
            "Consensus matrix must not be empty."
        )

    if consensus_matrix.shape[0] != consensus_matrix.shape[1]:
        raise ValueError(
            "Consensus matrix must be square."
        )

    if list(consensus_matrix.index) != list(consensus_matrix.columns):
        raise ValueError(
            "Consensus matrix index and columns must match."
        )

    non_numeric = consensus_matrix.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            "Consensus matrix must be entirely numeric."
        )

    values = consensus_matrix.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Consensus matrix contains non-finite values."
        )

    if ((values < 0) | (values > 1)).any():
        raise ValueError(
            "Consensus values must be between 0 and 1."
        )

    if not 0 <= lower_bound < upper_bound <= 1:
        raise ValueError(
            "Bounds must satisfy 0 <= lower_bound < upper_bound <= 1."
        )

    if len(consensus_matrix) < 2:
        raise ValueError(
            "Consensus matrix must contain at least 2 samples."
        )
    

    pair_values = values[
        np.triu_indices_from(values, k=1)
    ]

    ambiguous = (
        (pair_values > lower_bound)
        & (pair_values < upper_bound)
    )

    return float(ambiguous.mean())


def compute_pac_curve(
    X: pd.DataFrame,
    consensus_factory: Callable[[int], object],
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    lower_bound: float = 0.1,
    upper_bound: float = 0.9,
) -> pd.DataFrame:
    """
    Compute PAC score across candidate k values.

    Parameters
    ----------
    X:
        Feature matrix, normally PCA output.

    consensus_factory:
        Function receiving k and returning a ConsensusClusterer.

    k_values:
        Candidate numbers of clusters.

    lower_bound, upper_bound:
        PAC ambiguity interval.

    Returns
    -------
    pd.DataFrame
        Columns: k, pac
    """

    if X.empty:
        raise ValueError(
            "Feature matrix must not be empty."
        )

    if not k_values:
        raise ValueError(
            "k_values must contain at least one candidate."
        )

    if any(k < 2 for k in k_values):
        raise ValueError(
            "Every k must be at least 2."
        )

    if any(k >= len(X) for k in k_values):
        raise ValueError(
            "Every k must be smaller than the number of samples."
        )

    results: list[dict[str, float | int]] = []

    for k in k_values:
        consensus_clusterer = consensus_factory(k)

        consensus_matrix = (
            consensus_clusterer.fit_predict(X)
        )

        pac = compute_pac_score(
            consensus_matrix,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        results.append(
            {
                "k": k,
                "pac": pac,
            }
        )

    return pd.DataFrame(results)






def select_k_by_gap(
    gap_results: pd.DataFrame,
) -> int:
    """
    Select k using the standard Gap Statistic 1-SE rule.

    Choose the smallest k such that:

        Gap(k) >= Gap(k + 1) - s(k + 1)

    where s is the uncertainty of the reference Gap estimate.
    """

    required_columns = {
        "k",
        "gap",
        "gap_std",
    }

    if not required_columns.issubset(
        gap_results.columns
    ):
        raise ValueError(
            "Gap results must contain 'k', 'gap', and 'gap_std' columns."
        )

    if gap_results.empty:
        raise ValueError(
            "Gap results must not be empty."
        )

    ordered = (
        gap_results
        .sort_values("k")
        .reset_index(drop=True)
    )

    for i in range(len(ordered) - 1):
        current_gap = float(
            ordered.loc[i, "gap"]
        )

        next_gap = float(
            ordered.loc[i + 1, "gap"]
        )

        next_std = float(
            ordered.loc[i + 1, "gap_std"]
        )

        if current_gap >= next_gap - next_std:
            return int(
                ordered.loc[i, "k"]
            )

    # If no earlier k satisfies the rule,
    # fall back to the largest candidate.
    return int(
        ordered.iloc[-1]["k"]
    )