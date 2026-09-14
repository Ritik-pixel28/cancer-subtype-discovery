import pandas as pd
from sklearn.decomposition import PCA


class PCAProcessor:
    """Reduce gene-expression dimensions using PCA."""

    def __init__(self, n_components: int = 30) -> None:
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def fit_transform(self, expression: pd.DataFrame) -> pd.DataFrame:
        """Fit PCA and return transformed samples."""
        transformed = self.model.fit_transform(expression)

        columns = [
            f"PC{i + 1}"
            for i in range(self.n_components)
        ]

        return pd.DataFrame(
            transformed,
            index=expression.index,
            columns=columns,
        )