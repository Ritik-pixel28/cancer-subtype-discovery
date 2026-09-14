import numpy as np
import pandas as pd

from src.dimensionality import PCAProcessor


def test_pca_reduces_dimensions() -> None:
    expression = pd.DataFrame(
        {
            "gene_1": [1.0, 2.0, 3.0, 4.0],
            "gene_2": [2.0, 4.0, 6.0, 8.0],
            "gene_3": [4.0, 3.0, 2.0, 1.0],
        }
    )

    processor = PCAProcessor(n_components=2)

    result = processor.fit_transform(expression)

    assert result.shape == (4, 2)
    assert np.isfinite(result.to_numpy()).all()