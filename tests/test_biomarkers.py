import numpy as np
import pandas as pd
import pytest

from src.biomarkers import discover_biomarkers


def _example_expression():
    return pd.DataFrame(
        {
            "gene_a": [10, 11, 12, 1, 2, 1],
            "gene_b": [5, 5, 5, 5, 5, 5],
            "gene_c": [1, 2, 1, 10, 11, 12],
        },
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )


def _example_clusters():
    return pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
        name="cluster",
    )


def test_returns_one_row_per_gene():
    result = discover_biomarkers(
        _example_expression(),
        _example_clusters(),
        target_cluster=0,
    )

    assert len(result) == 3
    assert set(result["gene"]) == {
        "gene_a",
        "gene_b",
        "gene_c",
    }


def test_output_contains_expected_columns():
    result = discover_biomarkers(
        _example_expression(),
        _example_clusters(),
        target_cluster=0,
    )

    expected_columns = {
        "gene",
        "cluster",
        "mann_whitney_u",
        "p_value",
        "median_in_cluster",
        "median_out_cluster",
        "median_difference",
        "rank_biserial",
        "fdr",
        "significant",
    }

    assert expected_columns.issubset(result.columns)


def test_fdr_values_are_between_zero_and_one():
    result = discover_biomarkers(
        _example_expression(),
        _example_clusters(),
        target_cluster=0,
    )

    assert (
        (result["fdr"] >= 0)
        & (result["fdr"] <= 1)
    ).all()


def test_gene_a_is_higher_in_cluster_zero():
    result = discover_biomarkers(
        _example_expression(),
        _example_clusters(),
        target_cluster=0,
    )

    gene_a = result.loc[
        result["gene"] == "gene_a"
    ].iloc[0]

    assert gene_a["median_difference"] > 0
    assert gene_a["rank_biserial"] > 0


def test_gene_c_is_lower_in_cluster_zero():
    result = discover_biomarkers(
        _example_expression(),
        _example_clusters(),
        target_cluster=0,
    )

    gene_c = result.loc[
        result["gene"] == "gene_c"
    ].iloc[0]

    assert gene_c["median_difference"] < 0
    assert gene_c["rank_biserial"] < 0


def test_sample_order_is_aligned_by_index():
    expression = _example_expression()

    shuffled_clusters = pd.Series(
        [1, 0, 1, 0, 1, 0],
        index=["s4", "s1", "s5", "s2", "s6", "s3"],
    )

    result = discover_biomarkers(
        expression,
        shuffled_clusters,
        target_cluster=0,
    )

    gene_a = result.loc[
        result["gene"] == "gene_a"
    ].iloc[0]

    assert gene_a["median_difference"] > 0


def test_rejects_sample_mismatch():
    clusters = pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=[
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "wrong_sample",
        ],
    )

    with pytest.raises(
        ValueError,
        match="same samples",
    ):
        discover_biomarkers(
            _example_expression(),
            clusters,
            target_cluster=0,
        )


def test_rejects_unknown_target_cluster():
    with pytest.raises(
        ValueError,
        match="does not exist",
    ):
        discover_biomarkers(
            _example_expression(),
            _example_clusters(),
            target_cluster=99,
        )


def test_rejects_nan_values():
    expression = _example_expression()
    expression.loc["s1", "gene_a"] = np.nan

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        discover_biomarkers(
            expression,
            _example_clusters(),
            target_cluster=0,
        )

def test_constant_gene_has_no_effect():
    result = discover_biomarkers(
        _example_expression(),
        _example_clusters(),
        target_cluster=0,
    )

    gene_b = result.loc[
        result["gene"] == "gene_b"
    ].iloc[0]

    assert gene_b["p_value"] == 1.0
    assert gene_b["median_difference"] == 0
    assert gene_b["rank_biserial"] == 0