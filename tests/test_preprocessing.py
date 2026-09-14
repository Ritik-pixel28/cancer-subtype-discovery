import numpy as np
import pandas as pd

from src.preprocessing import Preprocessor


def test_labels_are_not_used_for_feature_selection() -> None:
    expression = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "gene_1": [1.0, 2.0, 3.0, 4.0],
            "gene_2": [10.0, 10.0, 10.0, 10.0],
            "gene_3": [1.0, 8.0, 2.0, 9.0],
        }
    )

    labels = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "cancer_type": ["ALL", "ALL", "AML", "AML"],
        }
    )

    preprocessor = Preprocessor(n_top_genes=2)

    selected = preprocessor.select_variable_genes(expression)

    assert "cancer_type" not in selected
    assert len(selected) == 2


def test_variable_gene_selection_prefers_high_variability() -> None:
        expression = pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3", "S4"],
                "gene_constant": [5.0, 5.0, 5.0, 5.0],
                "gene_variable": [1.0, 8.0, 2.0, 9.0],
            }
        )
        preprocessor = Preprocessor(n_top_genes=1)
        selected = preprocessor.select_variable_genes(expression)
        assert selected == ["gene_variable"]


def test_standardize_genes_has_zero_mean_and_unit_variance() -> None:
    expression = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "gene_1": [1.0, 2.0, 3.0, 4.0],
            "gene_2": [10.0, 20.0, 30.0, 40.0],
        }
    )

    preprocessor = Preprocessor()

    standardized = preprocessor.standardize_genes(expression)

    assert list(standardized.columns) == ["gene_1", "gene_2"]
    assert np.allclose(standardized.mean(axis=0), 0.0)
    assert np.allclose(standardized.std(axis=0, ddof=0), 1.0)

def test_standardize_genes_can_use_selected_genes_only() -> None:
    expression = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "gene_constant": [5.0, 5.0, 5.0, 5.0],
            "gene_variable": [1.0, 8.0, 2.0, 9.0],
            "gene_variable_2": [10.0, 30.0, 20.0, 40.0],
        }
    )

    preprocessor = Preprocessor(n_top_genes=2)

    selected_genes = preprocessor.select_variable_genes(expression)
    selected_expression = expression[["sample_id", *selected_genes]]

    standardized = preprocessor.standardize_genes(selected_expression)

    assert list(standardized.columns) == selected_genes
    assert "gene_constant" not in standardized.columns
    assert np.allclose(standardized.mean(axis=0), 0.0)
    assert np.allclose(standardized.std(axis=0, ddof=0), 1.0)