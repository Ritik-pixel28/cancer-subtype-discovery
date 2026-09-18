"""
Unit tests for the clustering module (Phase 4).

Design decisions
----------------
- Cluster labels are arbitrary integers; numeric identity has no meaning
  across runs or between algorithms.  Where reproducibility is tested we
  compare full partitions using Adjusted Rand Index (ARI = 1.0 ⟺ identical
  partition, regardless of label permutation).
- Small synthetic datasets are used so tests run quickly without real data.
- The consensus clusterer is tested on a tiny dataset with few resamples to
  keep runtime acceptable in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from src.clustering import (
    ConsensusClusterer,
    GMMClusterer,
    HierarchicalClusterer,
    KMeansClusterer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def well_separated_data() -> pd.DataFrame:
    """
    Two clearly separated clusters (10 samples × 4 features).

    Samples 0-4 cluster around (10, 10, 10, 10);
    samples 5-9 cluster around (-10, -10, -10, -10).
    """
    rng = np.random.default_rng(0)
    group_a = rng.normal(loc=10.0, scale=0.5, size=(5, 4))
    group_b = rng.normal(loc=-10.0, scale=0.5, size=(5, 4))
    data = np.vstack([group_a, group_b])
    return pd.DataFrame(
        data,
        index=[f"S{i}" for i in range(10)],
        columns=["PC1", "PC2", "PC3", "PC4"],
    )


@pytest.fixture()
def small_data() -> pd.DataFrame:
    """Simple 6-sample, 3-feature numeric DataFrame."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((6, 3))
    return pd.DataFrame(
        data,
        index=[f"S{i}" for i in range(6)],
        columns=["PC1", "PC2", "PC3"],
    )


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

class TestKMeansClusterer:
    def test_returns_one_label_per_sample(self, well_separated_data):
        result = KMeansClusterer(n_clusters=2).fit_predict(well_separated_data)
        assert isinstance(result, pd.Series)
        assert result.shape == (len(well_separated_data),)

    def test_index_is_preserved(self, well_separated_data):
        result = KMeansClusterer(n_clusters=2).fit_predict(well_separated_data)
        assert list(result.index) == list(well_separated_data.index)

    def test_series_is_named_cluster(self, well_separated_data):
        result = KMeansClusterer(n_clusters=2).fit_predict(well_separated_data)
        assert result.name == "cluster"

    def test_labels_are_integers(self, well_separated_data):
        result = KMeansClusterer(n_clusters=2).fit_predict(well_separated_data)
        assert result.dtype == int

    def test_reproducibility_with_fixed_seed(self, well_separated_data):
        r1 = KMeansClusterer(n_clusters=2, random_state=42).fit_predict(well_separated_data)
        r2 = KMeansClusterer(n_clusters=2, random_state=42).fit_predict(well_separated_data)
        ari = adjusted_rand_score(r1.to_numpy(), r2.to_numpy())
        assert ari == pytest.approx(1.0)

    def test_different_seeds_may_differ(self, well_separated_data):
        # This is a sanity check that we are actually using the seed;
        # with very well separated data the partitions should be identical,
        # so we just verify both runs produce valid outputs.
        r1 = KMeansClusterer(n_clusters=2, random_state=0).fit_predict(well_separated_data)
        r2 = KMeansClusterer(n_clusters=2, random_state=99).fit_predict(well_separated_data)
        assert r1.shape == r2.shape

    def test_k_equals_n_samples_is_valid(self, small_data):
        n = len(small_data)
        result = KMeansClusterer(n_clusters=n).fit_predict(small_data)
        assert result.shape == (n,)

    def test_multiple_k_values(self, well_separated_data):
        for k in range(2, 6):
            result = KMeansClusterer(n_clusters=k).fit_predict(well_separated_data)
            assert result.shape == (len(well_separated_data),)
            assert set(result.unique()).issubset(set(range(k)))


# ---------------------------------------------------------------------------
# Hierarchical Ward
# ---------------------------------------------------------------------------

class TestHierarchicalClusterer:
    def test_returns_one_label_per_sample(self, well_separated_data):
        result = HierarchicalClusterer(n_clusters=2).fit_predict(well_separated_data)
        assert isinstance(result, pd.Series)
        assert result.shape == (len(well_separated_data),)

    def test_index_is_preserved(self, well_separated_data):
        result = HierarchicalClusterer(n_clusters=2).fit_predict(well_separated_data)
        assert list(result.index) == list(well_separated_data.index)

    def test_series_is_named_cluster(self, well_separated_data):
        result = HierarchicalClusterer(n_clusters=2).fit_predict(well_separated_data)
        assert result.name == "cluster"

    def test_labels_are_integers(self, well_separated_data):
        result = HierarchicalClusterer(n_clusters=2).fit_predict(well_separated_data)
        assert result.dtype == int

    def test_deterministic(self, well_separated_data):
        """Hierarchical clustering is deterministic (no random state needed)."""
        r1 = HierarchicalClusterer(n_clusters=2).fit_predict(well_separated_data)
        r2 = HierarchicalClusterer(n_clusters=2).fit_predict(well_separated_data)
        ari = adjusted_rand_score(r1.to_numpy(), r2.to_numpy())
        assert ari == pytest.approx(1.0)

    def test_multiple_k_values(self, well_separated_data):
        for k in range(2, 6):
            result = HierarchicalClusterer(n_clusters=k).fit_predict(well_separated_data)
            assert result.shape == (len(well_separated_data),)


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------

class TestGMMClusterer:
    def test_returns_one_label_per_sample(self, well_separated_data):
        result = GMMClusterer(n_components=2).fit_predict(well_separated_data)
        assert isinstance(result, pd.Series)
        assert result.shape == (len(well_separated_data),)

    def test_index_is_preserved(self, well_separated_data):
        result = GMMClusterer(n_components=2).fit_predict(well_separated_data)
        assert list(result.index) == list(well_separated_data.index)

    def test_series_is_named_cluster(self, well_separated_data):
        result = GMMClusterer(n_components=2).fit_predict(well_separated_data)
        assert result.name == "cluster"

    def test_labels_are_integers(self, well_separated_data):
        result = GMMClusterer(n_components=2).fit_predict(well_separated_data)
        assert result.dtype == int

    def test_reproducibility_with_fixed_seed(self, well_separated_data):
        r1 = GMMClusterer(n_components=2, random_state=42).fit_predict(well_separated_data)
        r2 = GMMClusterer(n_components=2, random_state=42).fit_predict(well_separated_data)
        ari = adjusted_rand_score(r1.to_numpy(), r2.to_numpy())
        assert ari == pytest.approx(1.0)

    def test_predict_proba_shape(self, well_separated_data):
        clusterer = GMMClusterer(n_components=2, random_state=42)
        clusterer.fit_predict(well_separated_data)
        proba = clusterer.predict_proba(well_separated_data)
        assert proba.shape == (len(well_separated_data), 2)

    def test_predict_proba_rows_sum_to_one(self, well_separated_data):
        clusterer = GMMClusterer(n_components=2, random_state=42)
        clusterer.fit_predict(well_separated_data)
        proba = clusterer.predict_proba(well_separated_data)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_raises_before_fit(self, well_separated_data):
        clusterer = GMMClusterer(n_components=2)
        with pytest.raises(RuntimeError, match="fit_predict"):
            clusterer.predict_proba(well_separated_data)

    def test_multiple_k_values(self, well_separated_data):
        for k in range(2, 5):
            result = GMMClusterer(n_components=k).fit_predict(well_separated_data)
            assert result.shape == (len(well_separated_data),)


# ---------------------------------------------------------------------------
# Input validation (shared across all clusterers)
# ---------------------------------------------------------------------------

class TestInputValidation:
    """
    Each clusterer delegates to the same ``_validate_input`` function,
    so testing on KMeansClusterer covers all three.
    """

    def test_empty_dataframe_raises(self):
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            KMeansClusterer(n_clusters=2).fit_predict(empty)

    def test_nan_values_raise(self, small_data):
        bad = small_data.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            KMeansClusterer(n_clusters=2).fit_predict(bad)

    def test_infinite_values_raise(self, small_data):
        bad = small_data.copy()
        bad.iloc[1, 1] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            KMeansClusterer(n_clusters=2).fit_predict(bad)

    def test_negative_infinite_values_raise(self, small_data):
        bad = small_data.copy()
        bad.iloc[2, 2] = -np.inf
        with pytest.raises(ValueError, match="infinite"):
            KMeansClusterer(n_clusters=2).fit_predict(bad)

    def test_non_numeric_column_raises(self, small_data):
        bad = small_data.copy()
        bad["label"] = "ALL"
        with pytest.raises(ValueError, match="numeric"):
            KMeansClusterer(n_clusters=2).fit_predict(bad)

    def test_k_less_than_2_raises(self, small_data):
        with pytest.raises(ValueError, match=">="):
            KMeansClusterer(n_clusters=1).fit_predict(small_data)

    def test_k_zero_raises(self, small_data):
        with pytest.raises(ValueError, match=">="):
            KMeansClusterer(n_clusters=0).fit_predict(small_data)

    def test_k_greater_than_n_samples_raises(self, small_data):
        n = len(small_data)
        with pytest.raises(ValueError, match="n_clusters"):
            KMeansClusterer(n_clusters=n + 1).fit_predict(small_data)

    def test_hierarchical_k_less_than_2_raises(self, small_data):
        with pytest.raises(ValueError, match=">="):
            HierarchicalClusterer(n_clusters=1).fit_predict(small_data)

    def test_gmm_k_less_than_2_raises(self, small_data):
        with pytest.raises(ValueError, match=">="):
            GMMClusterer(n_components=1).fit_predict(small_data)

    def test_hierarchical_nan_raises(self, small_data):
        bad = small_data.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            HierarchicalClusterer(n_clusters=2).fit_predict(bad)

    def test_gmm_nan_raises(self, small_data):
        bad = small_data.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            GMMClusterer(n_components=2).fit_predict(bad)


# ---------------------------------------------------------------------------
# Consensus clusterer
# ---------------------------------------------------------------------------

class TestConsensusClusterer:
    """
    Consensus tests use very few resamples (n_resamples=5) to keep runtime
    short.  The full 100-resample configuration is used in the integration
    check (test_real_data_check.py or the standalone script).
    """

    def test_returns_square_dataframe(self, well_separated_data):
        base = KMeansClusterer(n_clusters=2, random_state=0)
        consensus = ConsensusClusterer(
            base_clusterer=base,
            n_resamples=5,
            p_item=0.8,
            p_feature=1.0,
            random_state=42,
        )
        M = consensus.fit_predict(well_separated_data)
        n = len(well_separated_data)
        assert isinstance(M, pd.DataFrame)
        assert M.shape == (n, n)

    def test_index_and_columns_match_input(self, well_separated_data):
        base = KMeansClusterer(n_clusters=2, random_state=0)
        consensus = ConsensusClusterer(base_clusterer=base, n_resamples=5)
        M = consensus.fit_predict(well_separated_data)
        assert list(M.index) == list(well_separated_data.index)
        assert list(M.columns) == list(well_separated_data.index)

    def test_values_in_unit_interval(self, well_separated_data):
        base = KMeansClusterer(n_clusters=2, random_state=0)
        consensus = ConsensusClusterer(base_clusterer=base, n_resamples=10)
        M = consensus.fit_predict(well_separated_data)
        assert (M.to_numpy() >= 0.0).all()
        assert (M.to_numpy() <= 1.0).all()

    def test_matrix_is_symmetric(self, well_separated_data):
        base = KMeansClusterer(n_clusters=2, random_state=0)
        consensus = ConsensusClusterer(base_clusterer=base, n_resamples=10)
        M = consensus.fit_predict(well_separated_data)
        assert np.allclose(M.to_numpy(), M.to_numpy().T)

    def test_diagonal_is_one(self, well_separated_data):
        base = KMeansClusterer(n_clusters=2, random_state=0)
        consensus = ConsensusClusterer(base_clusterer=base, n_resamples=10)
        M = consensus.fit_predict(well_separated_data)
        assert np.allclose(np.diag(M.to_numpy()), 1.0)

    def test_reproducible_with_fixed_seed(self, well_separated_data):
        base = KMeansClusterer(n_clusters=2, random_state=0)
        M1 = ConsensusClusterer(
            base_clusterer=base, n_resamples=10, random_state=7
        ).fit_predict(well_separated_data)
        M2 = ConsensusClusterer(
            base_clusterer=base, n_resamples=10, random_state=7
        ).fit_predict(well_separated_data)
        assert np.allclose(M1.to_numpy(), M2.to_numpy())

    def test_cluster_from_consensus_returns_series(self, well_separated_data):
        base = KMeansClusterer(n_clusters=2, random_state=0)
        consensus = ConsensusClusterer(base_clusterer=base, n_resamples=10)
        M = consensus.fit_predict(well_separated_data)
        labels = ConsensusClusterer.cluster_from_consensus(M, n_clusters=2)
        assert isinstance(labels, pd.Series)
        assert labels.shape == (len(well_separated_data),)
        assert labels.name == "cluster"

    def test_cluster_from_consensus_index_preserved(self, well_separated_data):
        base = KMeansClusterer(n_clusters=2, random_state=0)
        consensus = ConsensusClusterer(base_clusterer=base, n_resamples=10)
        M = consensus.fit_predict(well_separated_data)
        labels = ConsensusClusterer.cluster_from_consensus(M, n_clusters=2)
        assert list(labels.index) == list(well_separated_data.index)

    def test_empty_input_raises(self):
        base = KMeansClusterer(n_clusters=2)
        consensus = ConsensusClusterer(base_clusterer=base, n_resamples=5)
        with pytest.raises(ValueError, match="empty"):
            consensus.fit_predict(pd.DataFrame())

    def test_nan_input_raises(self, small_data):
        bad = small_data.copy()
        bad.iloc[0, 0] = np.nan
        base = KMeansClusterer(n_clusters=2)
        consensus = ConsensusClusterer(base_clusterer=base, n_resamples=5)
        with pytest.raises(ValueError, match="NaN"):
            consensus.fit_predict(bad)
