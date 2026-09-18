"""
Clustering components for the Cancer Subtype Discovery pipeline.

All clusterers operate on a numeric feature matrix (e.g. PCA-reduced gene
expression) and MUST NOT receive cancer-type labels as input.  Labels are
reserved for downstream interpretation and evaluation (Phase 5).

Classes
-------
KMeansClusterer
    K-Means clustering via scikit-learn.
HierarchicalClusterer
    Agglomerative (Ward-linkage) hierarchical clustering via scikit-learn.
GMMClusterer
    Gaussian Mixture Model clustering via scikit-learn.
ConsensusClusterer
    Resampling-based consensus / co-clustering matrix builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _validate_input(X: pd.DataFrame, k: int | None = None) -> None:
    """
    Validate a feature matrix before clustering.

    Parameters
    ----------
    X:
        Feature matrix.  Rows are samples; columns are features.
    k:
        Number of clusters / components requested.  When provided the
        function also checks ``2 <= k <= n_samples``.

    Raises
    ------
    ValueError
        On any validation failure (empty input, non-numeric columns,
        NaN / infinite values, invalid k).
    """
    if X.empty:
        raise ValueError("Input feature matrix is empty.")

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Input must be entirely numeric.  "
            f"Non-numeric columns: {non_numeric[:5]}"
        )

    if X.isnull().any(axis=None):
        raise ValueError("Input contains NaN values.  Remove or impute before clustering.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input contains infinite values.  Remove or clip before clustering.")

    if k is not None:
        n_samples = len(X)
        if k < 2:
            raise ValueError(
                f"n_clusters must be >= 2, got {k}."
            )
        if k > n_samples:
            raise ValueError(
                f"n_clusters ({k}) must not exceed the number of samples ({n_samples})."
            )


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

class KMeansClusterer:
    """
    K-Means clustering wrapper.

    Parameters
    ----------
    n_clusters:
        Number of clusters (k).  Must satisfy ``2 <= k <= n_samples``.
    random_state:
        Random seed for reproducibility.
    n_init:
        Number of K-Means initializations to run; the best result is kept.
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 42,
        n_init: int = 10,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Fit K-Means and return one integer cluster label per sample.

        Parameters
        ----------
        X:
            Numeric feature matrix (samples × features).

        Returns
        -------
        pd.Series
            Integer cluster assignments, indexed by ``X.index``, named
            ``"cluster"``.
        """
        _validate_input(X, k=self.n_clusters)

        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = model.fit_predict(X.to_numpy())

        return pd.Series(labels, index=X.index, name="cluster", dtype=int)


# ---------------------------------------------------------------------------
# Hierarchical (Ward)
# ---------------------------------------------------------------------------

class HierarchicalClusterer:
    """
    Agglomerative hierarchical clustering with Ward linkage.

    Ward linkage minimises the total within-cluster variance and requires
    Euclidean distances; it is therefore hardcoded here.

    Parameters
    ----------
    n_clusters:
        Number of clusters.  Must satisfy ``2 <= k <= n_samples``.
    """

    LINKAGE: str = "ward"

    def __init__(self, n_clusters: int) -> None:
        self.n_clusters = n_clusters

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Fit Ward hierarchical clustering and return cluster labels.

        Parameters
        ----------
        X:
            Numeric feature matrix (samples × features).

        Returns
        -------
        pd.Series
            Integer cluster assignments, indexed by ``X.index``, named
            ``"cluster"``.
        """
        _validate_input(X, k=self.n_clusters)

        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.LINKAGE,
        )
        labels = model.fit_predict(X.to_numpy())

        return pd.Series(labels, index=X.index, name="cluster", dtype=int)


# ---------------------------------------------------------------------------
# Gaussian Mixture Model
# ---------------------------------------------------------------------------

class GMMClusterer:
    """
    Gaussian Mixture Model clustering wrapper.

    Parameters
    ----------
    n_components:
        Number of mixture components (analogous to k).
        Must satisfy ``2 <= n_components <= n_samples``.
    random_state:
        Random seed for reproducibility.
    covariance_type:
        Covariance structure for each component.
        One of ``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``.
        Defaults to ``"full"`` (each component has its own full covariance
        matrix), which is the most flexible and commonly used option.
    """

    def __init__(
        self,
        n_components: int,
        random_state: int = 42,
        covariance_type: str = "full",
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.covariance_type = covariance_type
        self._model: GaussianMixture | None = None

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Fit GMM and return hard cluster assignments (argmax of posteriors).

        Parameters
        ----------
        X:
            Numeric feature matrix (samples × features).

        Returns
        -------
        pd.Series
            Integer cluster assignments, indexed by ``X.index``, named
            ``"cluster"``.
        """
        _validate_input(X, k=self.n_components)

        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._model.fit(X.to_numpy())
        labels = self._model.predict(X.to_numpy())

        return pd.Series(labels, index=X.index, name="cluster", dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return soft membership probabilities for each component.

        ``fit_predict`` must be called before this method.

        Parameters
        ----------
        X:
            Numeric feature matrix (samples × features).

        Returns
        -------
        pd.DataFrame
            Shape ``(n_samples, n_components)``.  Each row sums to 1.
            Columns are named ``"component_0"``, ``"component_1"``, …
        """
        if self._model is None:
            raise RuntimeError(
                "Call fit_predict before predict_proba."
            )

        _validate_input(X)
        proba = self._model.predict_proba(X.to_numpy())

        columns = [f"component_{i}" for i in range(self.n_components)]
        return pd.DataFrame(proba, index=X.index, columns=columns)


# ---------------------------------------------------------------------------
# Consensus clusterer
# ---------------------------------------------------------------------------

class ConsensusClusterer:
    """
    Resampling-based consensus / co-clustering matrix builder.

    Procedure
    ---------
    Given a feature matrix ``X`` with ``n`` samples and ``p`` features:

    1. Initialise two ``n × n`` matrices of zeros:
       - ``C[i, j]``: number of resamples where samples *i* and *j* were
         assigned to the **same** cluster.
       - ``I[i, j]``: number of resamples where **both** *i* and *j* were
         included in the subsample.

    2. Repeat ``n_resamples`` times:

       a. Draw ``floor(p_item × n)`` sample indices **without replacement**
          using a seeded NumPy ``default_rng``.
       b. Draw ``floor(p_feature × p)`` feature column indices **without
          replacement** from the same RNG.
       c. Cluster the resulting sub-matrix using ``base_clusterer``.
       d. For every ordered pair ``(i, j)`` in the subsample:
          - Increment ``I[i, j]`` and ``I[j, i]``.
          - If ``label[i] == label[j]``, also increment ``C[i, j]``
            and ``C[j, i]``.
       e. Increment diagonal of both matrices by 1 (a sample is trivially
          co-clustered with itself whenever it is selected).

    3. Compute ``M = C / I`` element-wise.  Pairs that were never
       simultaneously selected receive a value of 0 (their ``I`` entry is 0;
       division is handled safely via ``np.divide``).

    4. Return ``M`` as a ``pd.DataFrame`` indexed and columned by
       ``X.index``.

    The consensus matrix ``M[i, j]`` represents the empirical probability
    that samples *i* and *j* belong to the same cluster across resampled
    datasets.  Values range in ``[0, 1]``; higher values indicate a more
    stable co-clustering relationship.

    Hard cluster assignments can be obtained from the consensus matrix using
    ``cluster_from_consensus``.

    Parameters
    ----------
    base_clusterer:
        A fitted or unfitted clusterer instance that exposes a
        ``fit_predict(X: pd.DataFrame) -> pd.Series`` method.
        Typically one of ``KMeansClusterer``, ``HierarchicalClusterer``, or
        ``GMMClusterer``.
    n_resamples:
        Number of resampling iterations.  Corresponds to
        ``consensus_n_resamples`` in ``config.yaml``.
    p_item:
        Fraction of samples to include in each resample.  Corresponds to
        ``consensus_p_item`` (default 0.8).
    p_feature:
        Fraction of features to include in each resample.  Corresponds to
        ``consensus_p_feature`` (default 1.0, i.e. all features).
    random_state:
        Master random seed.  Each resample uses a deterministically derived
        sub-seed so the full procedure is reproducible.
    """

    def __init__(
        self,
        base_clusterer: KMeansClusterer | HierarchicalClusterer | GMMClusterer,
        n_resamples: int = 100,
        p_item: float = 0.8,
        p_feature: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.base_clusterer = base_clusterer
        self.n_resamples = n_resamples
        self.p_item = p_item
        self.p_feature = p_feature
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Build and return the consensus co-clustering matrix.

        Parameters
        ----------
        X:
            Numeric feature matrix (samples × features).  Cancer-type labels
            must **not** be included.

        Returns
        -------
        pd.DataFrame
            Square ``(n_samples × n_samples)`` consensus matrix ``M`` where
            ``M[i, j]`` is the proportion of resamples in which samples *i*
            and *j* were assigned to the same cluster (given that both were
            present in the resample).  Index and columns match ``X.index``.
        """
        _validate_input(X)

        n_samples, n_features = X.shape
        n_items = max(2, int(np.floor(self.p_item * n_samples)))
        n_feats = max(1, int(np.floor(self.p_feature * n_features)))

        data = X.to_numpy()

        C = np.zeros((n_samples, n_samples), dtype=np.float64)
        I = np.zeros((n_samples, n_samples), dtype=np.float64)

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_resamples):
            # --- subsample samples and features ---
            sample_idx = rng.choice(n_samples, size=n_items, replace=False)
            feature_idx = rng.choice(n_features, size=n_feats, replace=False)

            sub_data = data[np.ix_(sample_idx, feature_idx)]
            sub_df = pd.DataFrame(
                sub_data,
                index=X.index[sample_idx],
                columns=X.columns[feature_idx],
            )

            # --- cluster the subsample ---
            labels_series = self.base_clusterer.fit_predict(sub_df)
            labels = labels_series.to_numpy()

            # --- update co-occurrence matrices ---
            # Diagonal: a sample is always co-clustered with itself
            for pos_i, si in enumerate(sample_idx):
                I[si, si] += 1
                C[si, si] += 1

            for pos_i in range(len(sample_idx)):
                for pos_j in range(pos_i + 1, len(sample_idx)):
                    si = sample_idx[pos_i]
                    sj = sample_idx[pos_j]
                    I[si, sj] += 1
                    I[sj, si] += 1
                    if labels[pos_i] == labels[pos_j]:
                        C[si, sj] += 1
                        C[sj, si] += 1

        # --- compute consensus matrix ---
        M = np.divide(C, I, out=np.zeros_like(C), where=I > 0)

        return pd.DataFrame(M, index=X.index, columns=X.index)

    @staticmethod
    def cluster_from_consensus(
        consensus_matrix: pd.DataFrame,
        n_clusters: int,
    ) -> pd.Series:
        """
        Derive hard cluster assignments from a consensus matrix.

        Applies average-linkage hierarchical clustering to
        ``(1 - consensus_matrix)``, treating it as a dissimilarity matrix.
        This is a standard approach for extracting a partition from a
        consensus co-clustering matrix (Monti et al., 2003).

        Parameters
        ----------
        consensus_matrix:
            Square consensus matrix returned by ``fit_predict``.
        n_clusters:
            Number of desired clusters.

        Returns
        -------
        pd.Series
            Integer cluster assignments, indexed by ``consensus_matrix.index``,
            named ``"cluster"``.
        """
        dissimilarity = 1.0 - consensus_matrix.to_numpy()
        # Clip small floating-point negatives caused by rounding
        dissimilarity = np.clip(dissimilarity, 0.0, None)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dissimilarity)

        return pd.Series(
            labels,
            index=consensus_matrix.index,
            name="cluster",
            dtype=int,
        )
