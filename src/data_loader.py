from pathlib import Path

import pandas as pd

from src.exceptions import DataValidationError, SampleAlignmentError


class DataLoader:
    """Load and validate gene-expression data and sample metadata."""

    def __init__(
        self,
        expression_path: str | Path,
        labels_path: str | Path,
    ) -> None:
        self.expression_path = Path(expression_path)
        self.labels_path = Path(labels_path)

    def load_expression(self) -> pd.DataFrame:
        """Load the gene-expression matrix from CSV."""
        if not self.expression_path.exists():
            raise FileNotFoundError(
                f"Expression file not found: {self.expression_path}"
            )

        expression = pd.read_csv(self.expression_path)

        self._validate_expression(expression)

        return expression

    def load_labels(self) -> pd.DataFrame:
        """Load sample-level labels from CSV."""
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}"
            )

        labels = pd.read_csv(self.labels_path)

        self._validate_labels(labels)

        return labels

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load both datasets and verify sample alignment."""
        expression = self.load_expression()
        labels = self.load_labels()

        self._validate_sample_alignment(expression, labels)

        return expression, labels

    @staticmethod
    def _validate_expression(expression: pd.DataFrame) -> None:
        """Validate the expression matrix structure."""
        if expression.empty:
            raise DataValidationError("Expression dataset is empty.")

        if "sample_id" not in expression.columns:
            raise DataValidationError(
                "Expression dataset must contain a 'sample_id' column."
            )

        if expression["sample_id"].isna().any():
            raise DataValidationError(
                "Expression dataset contains missing sample IDs."
            )

        if expression["sample_id"].duplicated().any():
            raise DataValidationError(
                "Expression dataset contains duplicate sample IDs."
            )

        feature_columns = [
            column for column in expression.columns if column != "sample_id"
        ]

        if not feature_columns:
            raise DataValidationError(
                "Expression dataset contains no gene-expression features."
            )

        non_numeric = expression[feature_columns].select_dtypes(
            exclude="number"
        ).columns.tolist()

        if non_numeric:
            raise DataValidationError(
                f"Expression features must be numeric. "
                f"Non-numeric columns: {non_numeric[:5]}"
            )

    @staticmethod
    def _validate_labels(labels: pd.DataFrame) -> None:
        """Validate the labels dataset structure."""
        if labels.empty:
            raise DataValidationError("Labels dataset is empty.")

        required_columns = {"sample_id", "cancer_type"}

        missing_columns = required_columns - set(labels.columns)

        if missing_columns:
            raise DataValidationError(
                f"Labels dataset is missing columns: {sorted(missing_columns)}"
            )

        if labels["sample_id"].isna().any():
            raise DataValidationError(
                "Labels dataset contains missing sample IDs."
            )

        if labels["sample_id"].duplicated().any():
            raise DataValidationError(
                "Labels dataset contains duplicate sample IDs."
            )

        if labels["cancer_type"].isna().any():
            raise DataValidationError(
                "Labels dataset contains missing cancer types."
            )

    @staticmethod
    def _validate_sample_alignment(
        expression: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> None:
        """Ensure expression and label samples are identical."""
        expression_ids = set(expression["sample_id"])
        label_ids = set(labels["sample_id"])

        if expression_ids != label_ids:
            missing_labels = expression_ids - label_ids
            missing_expression = label_ids - expression_ids

            raise SampleAlignmentError(
                "Expression and label sample IDs do not match. "
                f"Missing labels: {sorted(missing_labels)}; "
                f"Missing expression samples: {sorted(missing_expression)}"
            )