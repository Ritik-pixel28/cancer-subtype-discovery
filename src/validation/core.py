"""
Cluster validation for the Cancer Subtype Discovery pipeline.

This module evaluates whether discovered clusters are well-separated,
stable under resampling, and consistent in their consensus structure.
It answers four questions:

  1. How well separated are the clusters? → SilhouetteAnalyzer
  2. How stable are they under resampling? → BootstrapStability
  3. How consistent is the consensus structure? → ConsensusValidator
  4. How do these properties change with k? → ValidationSummary

Important: cancer-type labels (ALL / AML) are NEVER used here.
All validation is purely unsupervised.

The thresholds used to flag results as "passing" are project-defined
criteria for stability, not statements about biological correctness.

Classes
-------
SilhouetteAnalyzer
    Measures average intra- vs. inter-cluster distance via silhouette score.
BootstrapStability
    Measures how consistently samples are co-clustered under resampling,
    using Adjusted Rand Index (ARI) to compare partitions.
ConsensusValidator
    Validates and summarizes a pre-built consensus co-clustering matrix.
ValidationSummary
    Combines silhouette and stability results into a single reporting table.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared input guard
# ---------------------------------------------------------------------------

def _check_feature_matrix(X: pd.DataFrame) -> None:
    """Raise ValueError if X is not a valid numeric feature matrix."""
    if X.empty:
        raise ValueError("Feature matrix X is empty.")

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Feature matrix must be entirely numeric. "
            f"Non-numeric columns: {non_numeric[:5]}"
        )

    if X.isnull().any(axis=None):
        raise ValueError("Feature matrix contains NaN values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature matrix contains infinite values.")


def _check_labels(labels: pd.Series, X: pd.DataFrame) -> None:
    """Raise ValueError for labels that cannot be used for silhouette scoring."""
    n_unique = labels.nunique()
    if n_unique < 2:
        raise ValueError(
            f"Silhouette score requires at least 2 distinct clusters; "
            f"got {n_unique}."
        )
    if n_unique == len(X):
        raise ValueError(
            "Every sample is in its own cluster; silhouette score is undefined."
        )


# ---------------------------------------------------------------------------
# Silhouette analysis
# ---------------------------------------------------------------------------

class SilhouetteAnalyzer:
    """
    Compute silhouette scores for clustering results.

    The silhouette score measures how similar each sample is to its own
    cluster compared to the nearest other cluster.  It ranges from −1
    (wrong cluster) to +1 (tight, well-separated cluster).  A score near
    0 indicates overlapping clusters.

    sklearn's ``silhouette_score`` is used throughout; no manual formula.

    Parameters
    ----------
    metric:
        Distance metric passed to ``silhouette_score``.  Defaults to
        ``"euclidean"``, which is appropriate for PCA-reduced data.
    """

    def __init__(self, metric: str = "euclidean") -> None:
        self.metric = metric

    def compute(self, X: pd.DataFrame, labels: pd.Series) -> float:
        """
        Calculate the mean silhouette score for one clustering solution.

        Parameters
        ----------
        X:
            PCA-reduced feature matrix (samples × components).  Must not
            contain cancer-type labels.
        labels:
            Integer cluster assignments aligned with ``X.index``.

        Returns
        -------
        float
            Mean silhouette coefficient across all samples.

        Raises
        ------
        ValueError
            If X contains NaN / inf, is empty, is non-numeric, has fewer
            than 2 clusters, or every sample is its own cluster.
        """
        _check_feature_matrix(X)
        _check_labels(labels, X)

        score = silhouette_score(
            X.to_numpy(),
            labels.to_numpy(),
            metric=self.metric,
        )
        return float(score)

    def compute_all(
        self,
        X: pd.DataFrame,
        results: dict[tuple[str, int], pd.Series],
    ) -> pd.DataFrame:
        """
        Compute silhouette scores for multiple clustering solutions.

        Parameters
        ----------
        X:
            PCA-reduced feature matrix shared by all solutions.
        results:
            Mapping from ``(method_name, k)`` to a ``pd.Series`` of cluster
            labels (as returned by the Phase-4 clusterers).

        Returns
        -------
        pd.DataFrame
            Columns: ``method``, ``k``, ``silhouette_score``.
            One row per (method, k) pair.  If a score cannot be computed
            (e.g. degenerate clustering), the row is omitted and a warning
            is logged.
        """
        rows = []
        for (method, k), labels in results.items():
            try:
                score = self.compute(X, labels)
                rows.append({"method": method, "k": k, "silhouette_score": score})
            except ValueError as exc:
                logger.warning(
                    "Skipping silhouette for method=%s k=%d: %s", method, k, exc
                )
        return pd.DataFrame(rows, columns=["method", "k", "silhouette_score"])


# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

class BootstrapStability:
    """
    Measure clustering stability using repeated sub-sampling and ARI.

    Procedure (per method, per k)
    ------------------------------
    1. Cluster the full dataset once to get reference labels.
    2. Repeat ``n_iter`` times:
       a. Draw ``floor(sample_frac × n)`` sample indices without replacement.
       b. Cluster that subset with a fresh clusterer instance.
       c. Extract the reference labels for those same sample indices.
       d. Compute ARI between the two label arrays.
          ARI handles label permutations correctly, so cluster-0 vs
          cluster-1 flipping does not penalise an otherwise identical
          partition.
    3. Return mean and std of the ARI values.

    Why ARI and not direct label comparison?
    ----------------------------------------
    Cluster labels are arbitrary integers.  Run 1 might call a group
    "cluster 0" and Run 2 might call the same group "cluster 1".  ARI
    measures partition agreement regardless of how the labels are numbered.

    Parameters
    ----------
    n_iter:
        Number of bootstrap iterations.
    sample_frac:
        Fraction of samples to include in each bootstrap draw (without
        replacement).  Corresponds to ``bootstrap_sample_frac`` in config.
    random_state:
        Master seed; each iteration uses a deterministically derived
        sub-seed so the full procedure is reproducible.
    """

    def __init__(
        self,
        n_iter: int = 100,
        sample_frac: float = 0.8,
        random_state: int = 42,
    ) -> None:
        if sample_frac <= 0.0 or sample_frac > 1.0:
            raise ValueError(
                f"sample_frac must be in (0, 1]; got {sample_frac}."
            )
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1; got {n_iter}.")

        self.n_iter = n_iter
        self.sample_frac = sample_frac
        self.random_state = random_state

    def compute(
        self,
        X: pd.DataFrame,
        clusterer_factory: Callable[[], object],
    ) -> dict[str, float]:
        """
        Estimate stability for one clustering configuration.

        Parameters
        ----------
        X:
            PCA-reduced feature matrix (samples × components).
        clusterer_factory:
            Zero-argument callable that returns a **new** clusterer instance
            each time it is called.  The returned object must implement
            ``fit_predict(X: pd.DataFrame) -> pd.Series``.

            Example::

                factory = lambda: KMeansClusterer(n_clusters=3, random_state=42)

        Returns
        -------
        dict
            Keys: ``mean_ari``, ``std_ari``.
            ARI values live in [−1, 1]; higher is more stable.

        Raises
        ------
        ValueError
            If X fails basic validation or has fewer than 4 samples
            (too small to meaningfully subsample).
        """
        _check_feature_matrix(X)

        n_samples = len(X)
        n_sub = max(2, int(np.floor(self.sample_frac * n_samples)))

        if n_samples < 4:
            raise ValueError(
                f"Bootstrap stability requires at least 4 samples; "
                f"got {n_samples}."
            )

        # Reference clustering on the full dataset
        ref_labels = clusterer_factory().fit_predict(X)

        rng = np.random.default_rng(self.random_state)
        ari_values: list[float] = []

        for _ in range(self.n_iter):
            # Draw a subset of samples (positional indices)
            pos_idx = rng.choice(n_samples, size=n_sub, replace=False)
            pos_idx_sorted = np.sort(pos_idx)

            sub_X = X.iloc[pos_idx_sorted]

            try:
                sub_labels = clusterer_factory().fit_predict(sub_X)
            except ValueError:
                # Skip iterations where clustering fails (e.g. k > subset size)
                logger.debug("Bootstrap iteration skipped due to clustering error.")
                continue

            # Reference labels for the same sample positions
            ref_sub = ref_labels.iloc[pos_idx_sorted]

            ari = adjusted_rand_score(
                ref_sub.to_numpy(),
                sub_labels.to_numpy(),
            )
            ari_values.append(ari)

        if not ari_values:
            # Every iteration failed — return NaN rather than crash
            return {"mean_ari": float("nan"), "std_ari": float("nan")}

        return {
            "mean_ari": float(np.mean(ari_values)),
            "std_ari": float(np.std(ari_values, ddof=1) if len(ari_values) > 1 else 0.0),
        }

    def compute_all(
        self,
        X: pd.DataFrame,
        factory_map: dict[tuple[str, int], Callable[[], object]],
    ) -> pd.DataFrame:
        """
        Compute stability for multiple (method, k) configurations.

        Parameters
        ----------
        X:
            PCA-reduced feature matrix.
        factory_map:
            Mapping from ``(method_name, k)`` to a zero-argument clusterer
            factory.

        Returns
        -------
        pd.DataFrame
            Columns: ``method``, ``k``, ``mean_ari``, ``std_ari``.
        """
        rows = []
        for (method, k), factory in factory_map.items():
            try:
                stats = self.compute(X, factory)
                rows.append(
                    {
                        "method": method,
                        "k": k,
                        "mean_ari": stats["mean_ari"],
                        "std_ari": stats["std_ari"],
                    }
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping stability for method=%s k=%d: %s", method, k, exc
                )
        return pd.DataFrame(rows, columns=["method", "k", "mean_ari", "std_ari"])


# ---------------------------------------------------------------------------
# Consensus matrix validation
# ---------------------------------------------------------------------------

class ConsensusValidator:
    """
    Validate and summarize a pre-built consensus co-clustering matrix.

    This class does NOT re-run consensus clustering.  It only inspects
    the matrix returned by ``ConsensusClusterer.fit_predict()``.

    A well-formed consensus matrix ``M`` satisfies:
    - Square shape (n_samples × n_samples)
    - Index and columns match the original sample IDs
    - All values are in [0, 1]
    - M is symmetric (M[i, j] == M[j, i])
    - Diagonal entries are all 1.0 (a sample is always co-clustered with itself)

    Parameters
    ----------
    high_consensus_threshold:
        Pairs above this value are considered "consistently co-clustered".
        Used only in ``summarize``; has no impact on validation checks.
        Default is 0.8, a commonly used cut-off for tight consensus.
    """

    def __init__(self, high_consensus_threshold: float = 0.8) -> None:
        if not 0.0 <= high_consensus_threshold <= 1.0:
            raise ValueError(
                f"high_consensus_threshold must be in [0, 1]; "
                f"got {high_consensus_threshold}."
            )
        self.high_consensus_threshold = high_consensus_threshold

    def validate(self, M: pd.DataFrame) -> dict[str, bool]:
        """
        Check structural properties of a consensus matrix.

        Parameters
        ----------
        M:
            Consensus matrix as returned by ``ConsensusClusterer.fit_predict``.

        Returns
        -------
        dict
            Keys: ``shape_is_square``, ``values_in_unit_interval``,
            ``is_symmetric``, ``diagonal_is_one``.
            All values are ``bool``.
        """
        arr = M.to_numpy()
        n_rows, n_cols = arr.shape

        shape_ok = n_rows == n_cols
        values_ok = bool((arr >= 0.0).all() and (arr <= 1.0).all())

        # Symmetry and diagonal checks only make sense on a square matrix.
        # np.allclose raises a broadcast error when shapes are incompatible.
        if shape_ok:
            symmetric_ok = bool(np.allclose(arr, arr.T, atol=1e-8))
            diagonal_ok = bool(np.allclose(np.diag(arr), 1.0, atol=1e-8))
        else:
            symmetric_ok = False
            diagonal_ok = False

        return {
            "shape_is_square": shape_ok,
            "values_in_unit_interval": values_ok,
            "is_symmetric": symmetric_ok,
            "diagonal_is_one": diagonal_ok,
        }

    def summarize(self, M: pd.DataFrame) -> dict[str, float]:
        """
        Compute summary statistics for a consensus matrix.

        Only off-diagonal entries are included (a sample with itself is
        trivially co-clustered and would inflate the mean).

        Statistics reported
        -------------------
        - ``mean_consensus``: mean of all off-diagonal entries
        - ``median_consensus``: median of all off-diagonal entries
        - ``frac_high_consensus``: fraction of off-diagonal pairs above
          ``high_consensus_threshold``

        Parameters
        ----------
        M:
            Consensus matrix.

        Returns
        -------
        dict with float values.
        """
        arr = M.to_numpy().astype(float)
        n = arr.shape[0]

        if n < 2:
            return {
                "mean_consensus": float("nan"),
                "median_consensus": float("nan"),
                "frac_high_consensus": float("nan"),
            }

        # Extract upper triangle (excluding diagonal) to avoid double-counting
        upper_idx = np.triu_indices(n, k=1)
        off_diag = arr[upper_idx]

        return {
            "mean_consensus": float(np.mean(off_diag)),
            "median_consensus": float(np.median(off_diag)),
            "frac_high_consensus": float(
                np.mean(off_diag >= self.high_consensus_threshold)
            ),
        }


# ---------------------------------------------------------------------------
# Validation summary table
# ---------------------------------------------------------------------------

class ValidationSummary:
    """
    Combine silhouette and bootstrap stability results into one table.

    This class does not rank or pick a "best" k.  It simply joins the
    two result DataFrames into a single reporting table so that all
    metrics can be read together.

    The thresholds ``ari_threshold_pass`` and ``silhouette_threshold_pass``
    are used only to add human-readable ``ari_passes`` / ``silhouette_passes``
    boolean flags; they do not determine any automatic selection.

    Parameters
    ----------
    ari_threshold:
        Project-defined ARI threshold (from ``ValidationConfig``).
        Mean ARI >= this value is flagged as passing.
        This is a project criterion, not a guarantee of biological validity.
    silhouette_threshold:
        Project-defined silhouette threshold (from ``ValidationConfig``).
        Silhouette >= this value is flagged as passing.
    """

    def __init__(
        self,
        ari_threshold: float = 0.70,
        silhouette_threshold: float = 0.60,
    ) -> None:
        self.ari_threshold = ari_threshold
        self.silhouette_threshold = silhouette_threshold

    def build(
        self,
        silhouette_df: pd.DataFrame,
        stability_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge silhouette and stability results into a single summary table.

        Parameters
        ----------
        silhouette_df:
            Output of ``SilhouetteAnalyzer.compute_all``.
            Expected columns: ``method``, ``k``, ``silhouette_score``.
        stability_df:
            Output of ``BootstrapStability.compute_all``.
            Expected columns: ``method``, ``k``, ``mean_ari``, ``std_ari``.

        Returns
        -------
        pd.DataFrame
            Columns: ``method``, ``k``, ``silhouette_score``, ``mean_ari``,
            ``std_ari``, ``ari_passes``, ``silhouette_passes``.
            Rows are sorted by method then k.
        """
        merged = pd.merge(
            silhouette_df,
            stability_df,
            on=["method", "k"],
            how="outer",
        )
        merged = merged.sort_values(["method", "k"]).reset_index(drop=True)

        # Flag columns — these are project-defined criteria, not biological truth
        merged["ari_passes"] = merged["mean_ari"] >= self.ari_threshold
        merged["silhouette_passes"] = (
            merged["silhouette_score"] >= self.silhouette_threshold
        )

        return merged
