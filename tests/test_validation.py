import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    ConsensusClusterer,
    KMeansClusterer,
)

from src.validation import (
    bootstrap_stability,
    compute_ari,
    compute_gap_statistic,
    compute_pac_curve,
    compute_pac_score,
    compute_silhouette,
    select_k_by_gap,
    select_k_by_pac,
)


def test_ari_perfect_match():
    reference = pd.Series(
        ["ALL", "ALL", "AML", "AML"],
        index=["s1", "s2", "s3", "s4"],
    )

    clusters = pd.Series(
        [0, 0, 1, 1],
        index=["s1", "s2", "s3", "s4"],
        name="cluster",
    )

    score = compute_ari(reference, clusters)

    assert score == pytest.approx(1.0)


def test_ari_low_for_different_partition():
    reference = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1],
        index=[f"s{i}" for i in range(8)],
    )

    clusters = pd.Series(
        [0, 0, 1, 1, 0, 0, 1, 1],
        index=[f"s{i}" for i in range(8)],
    )

    score = compute_ari(reference, clusters)

    assert score < 0.5


def test_ari_aligns_samples_by_index():
    reference = pd.Series(
        ["ALL", "ALL", "AML", "AML"],
        index=["s1", "s2", "s3", "s4"],
    )

    clusters = pd.Series(
        [1, 0, 1, 0],
        index=["s3", "s1", "s4", "s2"],
    )

    score = compute_ari(reference, clusters)

    assert score == pytest.approx(1.0)


def test_ari_rejects_sample_mismatch():
    reference = pd.Series(
        ["ALL", "AML"],
        index=["s1", "s2"],
    )

    clusters = pd.Series(
        [0, 1],
        index=["s1", "s3"],
    )

    with pytest.raises(ValueError, match="same samples"):
        compute_ari(reference, clusters)


def test_ari_rejects_missing_values():
    reference = pd.Series(
        ["ALL", None],
        index=["s1", "s2"],
    )

    clusters = pd.Series(
        [0, 1],
        index=["s1", "s2"],
    )

    with pytest.raises(ValueError, match="missing"):
        compute_ari(reference, clusters)


def test_silhouette_high_for_well_separated_clusters():
    X = pd.DataFrame(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [10.0, 10.0],
            [10.0, 11.0],
            [11.0, 10.0],
        ],
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )

    clusters = pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )

    score = compute_silhouette(X, clusters)

    assert score > 0.8


def test_silhouette_aligns_labels_by_sample_id():
    X = pd.DataFrame(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0],
        ],
        index=["s1", "s2", "s3", "s4"],
    )

    clusters = pd.Series(
        [1, 0, 1, 0],
        index=["s3", "s1", "s4", "s2"],
    )

    score = compute_silhouette(X, clusters)

    assert score > 0.8


def test_silhouette_rejects_single_cluster():
    X = pd.DataFrame(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        index=["s1", "s2", "s3"],
    )

    clusters = pd.Series(
        [0, 0, 0],
        index=["s1", "s2", "s3"],
    )

    with pytest.raises(ValueError, match="at least 2 clusters"):
        compute_silhouette(X, clusters)


def test_silhouette_rejects_one_cluster_per_sample():
    X = pd.DataFrame(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        index=["s1", "s2", "s3"],
    )

    clusters = pd.Series(
        [0, 1, 2],
        index=["s1", "s2", "s3"],
    )

    with pytest.raises(ValueError, match="fewer clusters than samples"):
        compute_silhouette(X, clusters)


def test_silhouette_rejects_sample_mismatch():
    X = pd.DataFrame(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
        ],
        index=["s1", "s2", "s3"],
    )

    clusters = pd.Series(
        [0, 0, 1],
        index=["s1", "s2", "s4"],
    )

    with pytest.raises(ValueError, match="same samples"):
        compute_silhouette(X, clusters)


def _well_separated_data():
    return pd.DataFrame(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [0.3, 0.2],
            [0.1, 0.3],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.2],
            [10.2, 10.1],
            [10.3, 10.2],
            [10.1, 10.3],
            [10.2, 10.2],
        ],
        index=[f"s{i}" for i in range(12)],
    )


def test_bootstrap_stability_high_for_clear_clusters():
    X = _well_separated_data()

    result = bootstrap_stability(
        X,
        clusterer_factory=lambda: KMeansClusterer(
            n_clusters=2,
            random_state=42,
        ),
        n_resamples=20,
        sample_fraction=0.8,
        random_state=42,
    )

    assert result.mean > 0.9


def test_bootstrap_stability_returns_requested_number_of_scores():
    X = _well_separated_data()

    result = bootstrap_stability(
        X,
        clusterer_factory=lambda: KMeansClusterer(
            n_clusters=2,
            random_state=42,
        ),
        n_resamples=20,
        random_state=42,
    )

    assert len(result.scores) == 20


def test_bootstrap_stability_scores_are_between_zero_and_one():
    X = _well_separated_data()

    result = bootstrap_stability(
        X,
        clusterer_factory=lambda: KMeansClusterer(
            n_clusters=2,
            random_state=42,
        ),
        n_resamples=20,
        random_state=42,
    )

    assert 0.0 <= result.mean <= 1.0
    assert 0.0 <= result.ci_lower <= 1.0
    assert 0.0 <= result.ci_upper <= 1.0
    assert result.ci_lower <= result.ci_upper


def test_bootstrap_stability_is_reproducible():
    X = _well_separated_data()

    kwargs = dict(
        X=X,
        clusterer_factory=lambda: KMeansClusterer(
            n_clusters=2,
            random_state=42,
        ),
        n_resamples=20,
        random_state=42,
    )

    result_1 = bootstrap_stability(**kwargs)
    result_2 = bootstrap_stability(**kwargs)

    assert result_1 == result_2


def test_bootstrap_stability_rejects_invalid_sample_fraction():
    X = _well_separated_data()

    with pytest.raises(ValueError, match="sample_fraction"):
        bootstrap_stability(
            X,
            clusterer_factory=lambda: KMeansClusterer(
                n_clusters=2,
            ),
            sample_fraction=1.5,
        )



def test_gap_statistic_returns_one_row_per_k():
    X = _well_separated_data()

    result = compute_gap_statistic(
        X,
        clusterer_factory=lambda k: KMeansClusterer(
            n_clusters=k,
            random_state=42,
        ),
        k_values=(2, 3),
        n_reference=5,
        random_state=42,
    )

    assert len(result) == 2
    assert result["k"].tolist() == [2, 3]


def test_gap_statistic_has_expected_columns():
    X = _well_separated_data()

    result = compute_gap_statistic(
        X,
        clusterer_factory=lambda k: KMeansClusterer(
            n_clusters=k,
            random_state=42,
        ),
        k_values=(2, 3),
        n_reference=5,
        random_state=42,
    )

    assert {
        "k",
        "log_wk",
        "reference_log_wk_mean",
        "gap",
        "gap_std",
    }.issubset(result.columns)


def test_gap_statistic_values_are_finite():
    X = _well_separated_data()

    result = compute_gap_statistic(
        X,
        clusterer_factory=lambda k: KMeansClusterer(
            n_clusters=k,
            random_state=42,
        ),
        k_values=(2, 3),
        n_reference=5,
        random_state=42,
    )

    assert np.isfinite(
        result[
            [
                "log_wk",
                "reference_log_wk_mean",
                "gap",
                "gap_std",
            ]
        ].to_numpy()
    ).all()


def test_gap_statistic_is_reproducible():
    X = _well_separated_data()

    kwargs = dict(
        X=X,
        clusterer_factory=lambda k: KMeansClusterer(
            n_clusters=k,
            random_state=42,
        ),
        k_values=(2, 3),
        n_reference=5,
        random_state=42,
    )

    result_1 = compute_gap_statistic(**kwargs)
    result_2 = compute_gap_statistic(**kwargs)

    pd.testing.assert_frame_equal(
        result_1,
        result_2,
    )


def test_select_k_by_gap_uses_one_se_rule():
    results = pd.DataFrame(
        {
            "k": [2, 3, 4, 5],
            "gap": [
                0.79,
                0.82,
                0.835,
                0.84,
            ],
            "gap_std": [
                0.02,
                0.02,
                0.02,
                0.02,
            ],
        }
    )

    best_k = select_k_by_gap(results)

    assert best_k == 3



def test_pac_is_zero_for_unambiguous_consensus():
    matrix = pd.DataFrame(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        index=["s1", "s2", "s3", "s4"],
        columns=["s1", "s2", "s3", "s4"],
    )

    score = compute_pac_score(matrix)

    assert score == pytest.approx(0.0)


def test_pac_detects_ambiguous_pairs():
    matrix = pd.DataFrame(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ],
        index=["s1", "s2", "s3"],
        columns=["s1", "s2", "s3"],
    )

    score = compute_pac_score(matrix)

    assert score == pytest.approx(2 / 3)


def test_pac_rejects_invalid_bounds():
    matrix = pd.DataFrame(
        [
            [1.0, 0.5],
            [0.5, 1.0],
        ],
        index=["s1", "s2"],
        columns=["s1", "s2"],
    )

    with pytest.raises(ValueError, match="Bounds"):
        compute_pac_score(
            matrix,
            lower_bound=0.9,
            upper_bound=0.1,
        )


def test_pac_curve_returns_one_row_per_k():
    X = _well_separated_data()

    result = compute_pac_curve(
        X,
        consensus_factory=lambda k: ConsensusClusterer(
            base_clusterer=KMeansClusterer(
                n_clusters=k,
                random_state=42,
            ),
            n_resamples=10,
            p_item=0.8,
            p_feature=1.0,
            random_state=42,
        ),
        k_values=(2, 3),
    )

    assert result["k"].tolist() == [2, 3]
    assert len(result) == 2

    assert (
        (result["pac"] >= 0)
        & (result["pac"] <= 1)
    ).all()


def test_select_k_by_pac_returns_lowest_pac():
    results = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "pac": [0.10, 0.35, 0.20],
        }
    )

    best_k = select_k_by_pac(results)

    assert best_k == 2