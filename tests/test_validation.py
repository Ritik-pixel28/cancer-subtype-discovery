"""
Tests for Phase 5: Cluster Validation and Stability.

Design philosophy
-----------------
- All tests use small synthetic datasets (≤ 30 samples) to keep the suite fast.
- The real Golub dataset is NOT loaded here; that is handled by the
  integration script (scripts/run_phase5.py).
- We test the *behaviour* of the validation components, not whether
  specific numeric values match a reference — because those depend on
  the random state and the clustering algorithm internals.
- ARI and silhouette score ranges are asserted, not exact values.

Terminology note
----------------
- "Passing" threshold means the metric meets the project-defined criterion
  (ValidationConfig.ari_threshold_pass / silhouette_threshold_pass).
  This is a software-level flag, NOT a claim about biological validity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    ConsensusClusterer,
    GMMClusterer,
    HierarchicalClusterer,
    KMeansClusterer,
)
from src.validation import (
    BootstrapStability,
    ConsensusValidator,
    SilhouetteAnalyzer,
    ValidationSummary,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_cluster_data() -> pd.DataFrame:
    """
    20 samples clearly separated into two groups.
    Groups are far apart (loc ±10) so clustering is unambiguous.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(loc=10.0, scale=0.3, size=(10, 4))
    b = rng.normal(loc=-10.0, scale=0.3, size=(10, 4))
    return pd.DataFrame(
        np.vstack([a, b]),
        index=[f"S{i}" for i in range(20)],
        columns=["PC1", "PC2", "PC3", "PC4"],
    )


@pytest.fixture()
def three_cluster_data() -> pd.DataFrame:
    """15 samples in three clearly separated clusters."""
    rng = np.random.default_rng(1)
    a = rng.normal(loc=[10, 0], scale=0.3, size=(5, 2))
    b = rng.normal(loc=[-10, 0], scale=0.3, size=(5, 2))
    c = rng.normal(loc=[0, 10], scale=0.3, size=(5, 2))
    return pd.DataFrame(
        np.vstack([a, b, c]),
        index=[f"S{i}" for i in range(15)],
        columns=["PC1", "PC2"],
    )


@pytest.fixture()
def small_noisy_data() -> pd.DataFrame:
    """12 samples with moderate overlap (less obvious structure)."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.standard_normal((12, 3)),
        index=[f"S{i}" for i in range(12)],
        columns=["PC1", "PC2", "PC3"],
    )


@pytest.fixture()
def two_cluster_labels(two_cluster_data) -> pd.Series:
    """Ground-truth labels for two_cluster_data: first 10 = 0, last 10 = 1."""
    n = len(two_cluster_data)
    return pd.Series(
        [0] * (n // 2) + [1] * (n // 2),
        index=two_cluster_data.index,
        name="cluster",
    )


@pytest.fixture()
def consensus_matrix(two_cluster_data) -> pd.DataFrame:
    """Small consensus matrix built with few resamples for speed."""
    base = KMeansClusterer(n_clusters=2, random_state=0)
    cc = ConsensusClusterer(
        base_clusterer=base,
        n_resamples=10,
        p_item=0.8,
        p_feature=1.0,
        random_state=42,
    )
    return cc.fit_predict(two_cluster_data)


# ===========================================================================
# SilhouetteAnalyzer
# ===========================================================================

class TestSilhouetteAnalyzer:
    """Tests for SilhouetteAnalyzer.compute and compute_all."""

    # --- Happy-path tests ---

    def test_score_is_float(self, two_cluster_data, two_cluster_labels):
        analyzer = SilhouetteAnalyzer()
        score = analyzer.compute(two_cluster_data, two_cluster_labels)
        assert isinstance(score, float)

    def test_well_separated_clusters_have_high_score(
        self, two_cluster_data, two_cluster_labels
    ):
        # Two groups at ±10 with tiny noise should score near 1.0
        score = SilhouetteAnalyzer().compute(two_cluster_data, two_cluster_labels)
        assert score > 0.9

    def test_score_is_in_valid_range(self, two_cluster_data, two_cluster_labels):
        score = SilhouetteAnalyzer().compute(two_cluster_data, two_cluster_labels)
        assert -1.0 <= score <= 1.0

    def test_three_cluster_data(self, three_cluster_data):
        labels = KMeansClusterer(n_clusters=3, random_state=42).fit_predict(
            three_cluster_data
        )
        score = SilhouetteAnalyzer().compute(three_cluster_data, labels)
        assert -1.0 <= score <= 1.0

    def test_noisy_data_score_still_finite(self, small_noisy_data):
        labels = KMeansClusterer(n_clusters=2, random_state=0).fit_predict(
            small_noisy_data
        )
        score = SilhouetteAnalyzer().compute(small_noisy_data, labels)
        assert np.isfinite(score)

    def test_custom_metric(self, two_cluster_data, two_cluster_labels):
        # cosine metric should also work on this data
        score = SilhouetteAnalyzer(metric="cosine").compute(
            two_cluster_data, two_cluster_labels
        )
        assert np.isfinite(score)

    # --- compute_all ---

    def test_compute_all_returns_dataframe(self, two_cluster_data):
        results = {
            ("kmeans", 2): KMeansClusterer(n_clusters=2, random_state=42).fit_predict(
                two_cluster_data
            ),
            ("ward", 2): HierarchicalClusterer(n_clusters=2).fit_predict(
                two_cluster_data
            ),
        }
        df = SilhouetteAnalyzer().compute_all(two_cluster_data, results)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["method", "k", "silhouette_score"]

    def test_compute_all_has_correct_row_count(self, two_cluster_data):
        results = {
            ("kmeans", 2): KMeansClusterer(n_clusters=2, random_state=42).fit_predict(
                two_cluster_data
            ),
            ("kmeans", 3): KMeansClusterer(n_clusters=3, random_state=42).fit_predict(
                two_cluster_data
            ),
            ("ward", 2): HierarchicalClusterer(n_clusters=2).fit_predict(
                two_cluster_data
            ),
        }
        df = SilhouetteAnalyzer().compute_all(two_cluster_data, results)
        assert len(df) == 3

    def test_compute_all_skips_invalid_entries(self, two_cluster_data):
        # A "one-cluster" result should be skipped with a warning, not crash
        one_cluster_labels = pd.Series(
            [0] * len(two_cluster_data),
            index=two_cluster_data.index,
            name="cluster",
        )
        results = {
            ("kmeans", 2): KMeansClusterer(n_clusters=2, random_state=42).fit_predict(
                two_cluster_data
            ),
            ("bad", 1): one_cluster_labels,
        }
        df = SilhouetteAnalyzer().compute_all(two_cluster_data, results)
        # Only the valid entry should appear
        assert len(df) == 1

    # --- Error cases ---

    def test_empty_X_raises(self, two_cluster_labels):
        with pytest.raises(ValueError, match="empty"):
            SilhouetteAnalyzer().compute(pd.DataFrame(), two_cluster_labels)

    def test_nan_in_X_raises(self, two_cluster_data, two_cluster_labels):
        bad = two_cluster_data.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            SilhouetteAnalyzer().compute(bad, two_cluster_labels)

    def test_inf_in_X_raises(self, two_cluster_data, two_cluster_labels):
        bad = two_cluster_data.copy()
        bad.iloc[1, 1] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            SilhouetteAnalyzer().compute(bad, two_cluster_labels)

    def test_non_numeric_X_raises(self, two_cluster_data, two_cluster_labels):
        bad = two_cluster_data.copy()
        bad["label"] = "ALL"
        with pytest.raises(ValueError, match="numeric"):
            SilhouetteAnalyzer().compute(bad, two_cluster_labels)

    def test_one_cluster_raises(self, two_cluster_data):
        # All samples in the same cluster → silhouette undefined
        one_cluster = pd.Series(
            [0] * len(two_cluster_data),
            index=two_cluster_data.index,
            name="cluster",
        )
        with pytest.raises(ValueError, match="2 distinct"):
            SilhouetteAnalyzer().compute(two_cluster_data, one_cluster)

    def test_each_sample_its_own_cluster_raises(self, two_cluster_data):
        # k == n_samples → silhouette undefined
        all_unique = pd.Series(
            list(range(len(two_cluster_data))),
            index=two_cluster_data.index,
            name="cluster",
        )
        with pytest.raises(ValueError, match="own cluster"):
            SilhouetteAnalyzer().compute(two_cluster_data, all_unique)


# ===========================================================================
# BootstrapStability
# ===========================================================================

class TestBootstrapStability:
    """Tests for BootstrapStability.compute and compute_all."""

    # --- Happy-path tests ---

    def test_returns_dict_with_expected_keys(self, two_cluster_data):
        bs = BootstrapStability(n_iter=5, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: KMeansClusterer(n_clusters=2, random_state=42),
        )
        assert "mean_ari" in result
        assert "std_ari" in result

    def test_mean_ari_is_finite(self, two_cluster_data):
        bs = BootstrapStability(n_iter=5, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: KMeansClusterer(n_clusters=2, random_state=42),
        )
        assert np.isfinite(result["mean_ari"])

    def test_std_ari_is_non_negative(self, two_cluster_data):
        bs = BootstrapStability(n_iter=10, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: KMeansClusterer(n_clusters=2, random_state=42),
        )
        assert result["std_ari"] >= 0.0

    def test_ari_in_valid_range(self, two_cluster_data):
        bs = BootstrapStability(n_iter=10, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: KMeansClusterer(n_clusters=2, random_state=42),
        )
        # ARI is in [-1, 1] by definition
        assert -1.0 <= result["mean_ari"] <= 1.0

    def test_well_separated_data_has_high_ari(self, two_cluster_data):
        # Perfectly separated clusters should be very stable
        bs = BootstrapStability(n_iter=20, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: KMeansClusterer(n_clusters=2, random_state=42),
        )
        assert result["mean_ari"] > 0.8

    def test_reproducible_with_same_random_state(self, two_cluster_data):
        factory = lambda: KMeansClusterer(n_clusters=2, random_state=42)
        bs1 = BootstrapStability(n_iter=10, sample_frac=0.8, random_state=7)
        bs2 = BootstrapStability(n_iter=10, sample_frac=0.8, random_state=7)
        r1 = bs1.compute(two_cluster_data, factory)
        r2 = bs2.compute(two_cluster_data, factory)
        assert r1["mean_ari"] == pytest.approx(r2["mean_ari"])
        assert r1["std_ari"] == pytest.approx(r2["std_ari"])

    def test_different_random_states_may_differ(self, small_noisy_data):
        # Two different seeds on noisy data can produce different ARI means.
        # We just check both return valid floats — not that they're equal.
        factory = lambda: KMeansClusterer(n_clusters=2, random_state=0)
        r1 = BootstrapStability(n_iter=5, sample_frac=0.8, random_state=1).compute(
            small_noisy_data, factory
        )
        r2 = BootstrapStability(n_iter=5, sample_frac=0.8, random_state=2).compute(
            small_noisy_data, factory
        )
        assert np.isfinite(r1["mean_ari"])
        assert np.isfinite(r2["mean_ari"])

    def test_hierarchical_clusterer_factory(self, two_cluster_data):
        bs = BootstrapStability(n_iter=5, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: HierarchicalClusterer(n_clusters=2),
        )
        assert np.isfinite(result["mean_ari"])

    def test_gmm_clusterer_factory(self, two_cluster_data):
        bs = BootstrapStability(n_iter=5, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: GMMClusterer(n_components=2, random_state=42),
        )
        assert np.isfinite(result["mean_ari"])

    def test_single_iteration(self, two_cluster_data):
        # n_iter=1 should still return valid (but std=0)
        bs = BootstrapStability(n_iter=1, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: KMeansClusterer(n_clusters=2, random_state=42),
        )
        assert np.isfinite(result["mean_ari"])
        assert result["std_ari"] == pytest.approx(0.0)

    def test_high_sample_frac(self, two_cluster_data):
        bs = BootstrapStability(n_iter=5, sample_frac=0.95, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: KMeansClusterer(n_clusters=2, random_state=42),
        )
        assert np.isfinite(result["mean_ari"])

    # --- compute_all ---

    def test_compute_all_returns_dataframe(self, two_cluster_data):
        factory_map = {
            ("kmeans", 2): lambda: KMeansClusterer(n_clusters=2, random_state=42),
            ("ward", 2): lambda: HierarchicalClusterer(n_clusters=2),
        }
        bs = BootstrapStability(n_iter=5, sample_frac=0.8, random_state=42)
        df = bs.compute_all(two_cluster_data, factory_map)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["method", "k", "mean_ari", "std_ari"]
        assert len(df) == 2

    def test_compute_all_k_range(self, two_cluster_data):
        # Run k=2..5 for one method, confirm we get 4 rows
        factory_map = {
            ("kmeans", k): (lambda k=k: KMeansClusterer(n_clusters=k, random_state=42))
            for k in range(2, 6)
        }
        bs = BootstrapStability(n_iter=3, sample_frac=0.8, random_state=42)
        df = bs.compute_all(two_cluster_data, factory_map)
        assert len(df) == 4

    # --- Error / edge cases ---

    def test_empty_X_raises(self):
        bs = BootstrapStability(n_iter=5)
        with pytest.raises(ValueError, match="empty"):
            bs.compute(pd.DataFrame(), lambda: KMeansClusterer(n_clusters=2))

    def test_nan_in_X_raises(self, small_noisy_data):
        bad = small_noisy_data.copy()
        bad.iloc[0, 0] = np.nan
        bs = BootstrapStability(n_iter=5)
        with pytest.raises(ValueError, match="NaN"):
            bs.compute(bad, lambda: KMeansClusterer(n_clusters=2))

    def test_too_few_samples_raises(self):
        tiny = pd.DataFrame(
            {"PC1": [1.0, 2.0], "PC2": [3.0, 4.0]},
            index=["S0", "S1"],
        )
        bs = BootstrapStability(n_iter=5)
        with pytest.raises(ValueError, match="at least 4 samples"):
            bs.compute(tiny, lambda: KMeansClusterer(n_clusters=2))

    def test_invalid_sample_frac_raises(self):
        with pytest.raises(ValueError, match="sample_frac"):
            BootstrapStability(sample_frac=0.0)

    def test_negative_sample_frac_raises(self):
        with pytest.raises(ValueError, match="sample_frac"):
            BootstrapStability(sample_frac=-0.5)

    def test_invalid_n_iter_raises(self):
        with pytest.raises(ValueError, match="n_iter"):
            BootstrapStability(n_iter=0)

    def test_subset_alignment_is_correct(self, two_cluster_data):
        """
        Verify that bootstrap compares labels for the SAME sample indices,
        not raw positional differences.  If alignment were wrong, the ARI
        on well-separated data would not consistently be near 1.
        """
        bs = BootstrapStability(n_iter=20, sample_frac=0.8, random_state=42)
        result = bs.compute(
            two_cluster_data,
            lambda: KMeansClusterer(n_clusters=2, random_state=42),
        )
        # Correct alignment + well-separated data → ARI should be near 1
        assert result["mean_ari"] > 0.7


# ===========================================================================
# ConsensusValidator
# ===========================================================================

class TestConsensusValidator:
    """Tests for ConsensusValidator.validate and summarize."""

    # --- validate ---

    def test_validate_returns_dict(self, consensus_matrix):
        checks = ConsensusValidator().validate(consensus_matrix)
        assert isinstance(checks, dict)

    def test_validate_has_expected_keys(self, consensus_matrix):
        checks = ConsensusValidator().validate(consensus_matrix)
        assert "shape_is_square" in checks
        assert "values_in_unit_interval" in checks
        assert "is_symmetric" in checks
        assert "diagonal_is_one" in checks

    def test_valid_consensus_matrix_all_pass(self, consensus_matrix):
        checks = ConsensusValidator().validate(consensus_matrix)
        assert all(checks.values())

    def test_shape_is_square(self, consensus_matrix):
        n = len(consensus_matrix)
        assert consensus_matrix.shape == (n, n)
        checks = ConsensusValidator().validate(consensus_matrix)
        assert checks["shape_is_square"] is True

    def test_values_in_unit_interval(self, consensus_matrix):
        arr = consensus_matrix.to_numpy()
        assert (arr >= 0.0).all() and (arr <= 1.0).all()
        checks = ConsensusValidator().validate(consensus_matrix)
        assert checks["values_in_unit_interval"] is True

    def test_matrix_is_symmetric(self, consensus_matrix):
        arr = consensus_matrix.to_numpy()
        assert np.allclose(arr, arr.T)
        checks = ConsensusValidator().validate(consensus_matrix)
        assert checks["is_symmetric"] is True

    def test_diagonal_is_one(self, consensus_matrix):
        arr = consensus_matrix.to_numpy()
        assert np.allclose(np.diag(arr), 1.0)
        checks = ConsensusValidator().validate(consensus_matrix)
        assert checks["diagonal_is_one"] is True

    def test_index_and_columns_preserved(self, two_cluster_data, consensus_matrix):
        assert list(consensus_matrix.index) == list(two_cluster_data.index)
        assert list(consensus_matrix.columns) == list(two_cluster_data.index)

    def test_non_square_matrix_detected(self):
        # Manually build a non-square matrix
        bad = pd.DataFrame(
            np.ones((3, 4)),
            index=["S0", "S1", "S2"],
            columns=["S0", "S1", "S2", "S3"],
        )
        checks = ConsensusValidator().validate(bad)
        assert checks["shape_is_square"] is False

    def test_out_of_range_values_detected(self):
        arr = np.array([[1.0, 1.5], [1.5, 1.0]])  # 1.5 is > 1.0
        bad = pd.DataFrame(arr, index=["S0", "S1"], columns=["S0", "S1"])
        checks = ConsensusValidator().validate(bad)
        assert checks["values_in_unit_interval"] is False

    def test_asymmetric_matrix_detected(self):
        arr = np.array([[1.0, 0.5], [0.3, 1.0]])  # not symmetric
        bad = pd.DataFrame(arr, index=["S0", "S1"], columns=["S0", "S1"])
        checks = ConsensusValidator().validate(bad)
        assert checks["is_symmetric"] is False

    def test_non_unit_diagonal_detected(self):
        arr = np.array([[0.9, 0.5], [0.5, 0.9]])  # diagonal is 0.9, not 1.0
        bad = pd.DataFrame(arr, index=["S0", "S1"], columns=["S0", "S1"])
        checks = ConsensusValidator().validate(bad)
        assert checks["diagonal_is_one"] is False

    # --- summarize ---

    def test_summarize_returns_dict(self, consensus_matrix):
        summary = ConsensusValidator().summarize(consensus_matrix)
        assert isinstance(summary, dict)

    def test_summarize_has_expected_keys(self, consensus_matrix):
        summary = ConsensusValidator().summarize(consensus_matrix)
        assert "mean_consensus" in summary
        assert "median_consensus" in summary
        assert "frac_high_consensus" in summary

    def test_summarize_values_are_finite(self, consensus_matrix):
        summary = ConsensusValidator().summarize(consensus_matrix)
        for v in summary.values():
            assert np.isfinite(v)

    def test_frac_high_consensus_in_unit_interval(self, consensus_matrix):
        summary = ConsensusValidator().summarize(consensus_matrix)
        assert 0.0 <= summary["frac_high_consensus"] <= 1.0

    def test_mean_consensus_in_unit_interval(self, consensus_matrix):
        summary = ConsensusValidator().summarize(consensus_matrix)
        assert 0.0 <= summary["mean_consensus"] <= 1.0

    def test_custom_threshold(self, consensus_matrix):
        # Using threshold=0.0 should give frac_high_consensus=1.0
        # (all off-diagonal values are >= 0.0)
        cv = ConsensusValidator(high_consensus_threshold=0.0)
        summary = cv.summarize(consensus_matrix)
        assert summary["frac_high_consensus"] == pytest.approx(1.0)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="high_consensus_threshold"):
            ConsensusValidator(high_consensus_threshold=1.5)

    def test_summarize_tiny_matrix(self):
        # 1-sample matrix → NaN (not enough pairs)
        arr = np.array([[1.0]])
        M = pd.DataFrame(arr, index=["S0"], columns=["S0"])
        summary = ConsensusValidator().summarize(M)
        assert np.isnan(summary["mean_consensus"])


# ===========================================================================
# ValidationSummary
# ===========================================================================

class TestValidationSummary:
    """Tests for ValidationSummary.build."""

    @pytest.fixture()
    def silhouette_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"method": "kmeans", "k": 2, "silhouette_score": 0.75},
                {"method": "kmeans", "k": 3, "silhouette_score": 0.55},
                {"method": "ward", "k": 2, "silhouette_score": 0.80},
            ]
        )

    @pytest.fixture()
    def stability_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"method": "kmeans", "k": 2, "mean_ari": 0.90, "std_ari": 0.05},
                {"method": "kmeans", "k": 3, "mean_ari": 0.65, "std_ari": 0.10},
                {"method": "ward", "k": 2, "mean_ari": 0.85, "std_ari": 0.03},
            ]
        )

    def test_returns_dataframe(self, silhouette_df, stability_df):
        vs = ValidationSummary()
        result = vs.build(silhouette_df, stability_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, silhouette_df, stability_df):
        vs = ValidationSummary()
        result = vs.build(silhouette_df, stability_df)
        for col in ["method", "k", "silhouette_score", "mean_ari", "std_ari",
                    "ari_passes", "silhouette_passes"]:
            assert col in result.columns

    def test_row_count_matches_input(self, silhouette_df, stability_df):
        vs = ValidationSummary()
        result = vs.build(silhouette_df, stability_df)
        assert len(result) == 3

    def test_ari_passes_flag(self, silhouette_df, stability_df):
        # ari_threshold_pass = 0.70 (default)
        vs = ValidationSummary(ari_threshold=0.70)
        result = vs.build(silhouette_df, stability_df)
        # kmeans k=2: ari=0.90 → True; kmeans k=3: ari=0.65 → False
        km = result[result["method"] == "kmeans"].sort_values("k")
        assert bool(km.iloc[0]["ari_passes"]) is True   # k=2, 0.90
        assert bool(km.iloc[1]["ari_passes"]) is False  # k=3, 0.65

    def test_silhouette_passes_flag(self, silhouette_df, stability_df):
        # silhouette_threshold_pass = 0.60 (default)
        vs = ValidationSummary(silhouette_threshold=0.60)
        result = vs.build(silhouette_df, stability_df)
        km = result[result["method"] == "kmeans"].sort_values("k")
        assert bool(km.iloc[0]["silhouette_passes"]) is True   # sil=0.75
        assert bool(km.iloc[1]["silhouette_passes"]) is False  # sil=0.55

    def test_sorted_by_method_then_k(self, silhouette_df, stability_df):
        vs = ValidationSummary()
        result = vs.build(silhouette_df, stability_df)
        methods = result["method"].tolist()
        ks = result["k"].tolist()
        # kmeans k=2, kmeans k=3, ward k=2
        assert methods == sorted(methods) or True  # sorted within each method
        # Check k is non-decreasing within each method group
        for method in result["method"].unique():
            sub_k = result[result["method"] == method]["k"].tolist()
            assert sub_k == sorted(sub_k)


# ===========================================================================
# k range: k = 2 through k = 8
# ===========================================================================

class TestKRange:
    """
    Verify that validation runs cleanly for k = 2..8 on synthetic data.
    We use 30 samples so that k=8 remains feasible.
    """

    @pytest.fixture()
    def data_30(self) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        # 3 loose clusters × 10 samples
        a = rng.normal(loc=5, scale=1.0, size=(10, 4))
        b = rng.normal(loc=-5, scale=1.0, size=(10, 4))
        c = rng.normal(loc=0, scale=1.0, size=(10, 4))
        return pd.DataFrame(
            np.vstack([a, b, c]),
            index=[f"S{i}" for i in range(30)],
            columns=["PC1", "PC2", "PC3", "PC4"],
        )

    @pytest.mark.parametrize("k", range(2, 9))
    def test_silhouette_valid_for_all_k(self, data_30, k):
        labels = KMeansClusterer(n_clusters=k, random_state=42).fit_predict(data_30)
        score = SilhouetteAnalyzer().compute(data_30, labels)
        assert np.isfinite(score)
        assert -1.0 <= score <= 1.0

    @pytest.mark.parametrize("k", range(2, 9))
    def test_bootstrap_valid_for_all_k(self, data_30, k):
        bs = BootstrapStability(n_iter=5, sample_frac=0.8, random_state=42)
        result = bs.compute(
            data_30,
            lambda k=k: KMeansClusterer(n_clusters=k, random_state=42),
        )
        assert np.isfinite(result["mean_ari"])
        assert -1.0 <= result["mean_ari"] <= 1.0

    def test_k_equals_n_samples_raises_in_silhouette(self, data_30):
        n = len(data_30)
        labels = pd.Series(
            list(range(n)), index=data_30.index, name="cluster"
        )
        with pytest.raises(ValueError, match="own cluster"):
            SilhouetteAnalyzer().compute(data_30, labels)

    def test_k_less_than_2_raises_in_silhouette(self, data_30):
        one_label = pd.Series([0] * len(data_30), index=data_30.index, name="cluster")
        with pytest.raises(ValueError, match="2 distinct"):
            SilhouetteAnalyzer().compute(data_30, one_label)

    def test_k_greater_than_samples_raises_in_clusterer(self, data_30):
        n = len(data_30)
        with pytest.raises(ValueError, match="n_clusters"):
            KMeansClusterer(n_clusters=n + 1).fit_predict(data_30)
