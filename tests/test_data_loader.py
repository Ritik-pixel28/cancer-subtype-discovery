import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.exceptions import DataValidationError, SampleAlignmentError


EXPRESSION_PATH = "data/raw/golub_expr.csv"
LABELS_PATH = "data/raw/golub_labels.csv"


def test_valid_data_loads_successfully() -> None:
    """Valid Golub data should load and align successfully."""
    loader = DataLoader(EXPRESSION_PATH, LABELS_PATH)

    expression, labels = loader.load()

    assert expression.shape == (72, 7130)
    assert labels.shape == (72, 2)
    assert set(expression["sample_id"]) == set(labels["sample_id"])


def test_missing_sample_id_is_rejected(tmp_path) -> None:
    """Expression data without sample_id should be rejected."""
    expression = pd.DataFrame(
        {
            "gene_1": [1.0, 2.0],
            "gene_2": [3.0, 4.0],
        }
    )

    labels = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "cancer_type": ["ALL", "AML"],
        }
    )

    expression_path = tmp_path / "expression.csv"
    labels_path = tmp_path / "labels.csv"

    expression.to_csv(expression_path, index=False)
    labels.to_csv(labels_path, index=False)

    loader = DataLoader(expression_path, labels_path)

    with pytest.raises(DataValidationError, match="sample_id"):
        loader.load()


def test_duplicate_sample_ids_are_rejected(tmp_path) -> None:
    """Duplicate sample IDs should be rejected."""
    expression = pd.DataFrame(
        {
            "sample_id": ["S1", "S1"],
            "gene_1": [1.0, 2.0],
        }
    )

    labels = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "cancer_type": ["ALL", "AML"],
        }
    )

    expression_path = tmp_path / "expression.csv"
    labels_path = tmp_path / "labels.csv"

    expression.to_csv(expression_path, index=False)
    labels.to_csv(labels_path, index=False)

    loader = DataLoader(expression_path, labels_path)

    with pytest.raises(DataValidationError, match="duplicate"):
        loader.load()


def test_sample_mismatch_is_rejected(tmp_path) -> None:
    """Expression and label sample IDs must match."""
    expression = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "gene_1": [1.0, 2.0],
        }
    )

    labels = pd.DataFrame(
        {
            "sample_id": ["S1", "S3"],
            "cancer_type": ["ALL", "AML"],
        }
    )

    expression_path = tmp_path / "expression.csv"
    labels_path = tmp_path / "labels.csv"

    expression.to_csv(expression_path, index=False)
    labels.to_csv(labels_path, index=False)

    loader = DataLoader(expression_path, labels_path)

    with pytest.raises(SampleAlignmentError, match="do not match"):
        loader.load()