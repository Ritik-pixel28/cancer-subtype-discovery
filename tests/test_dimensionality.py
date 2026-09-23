import numpy as np
import pandas as pd

from src.dimensionality import PCAProcessor


def test_pca_is_reproducible():
    rng = np.random.default_rng(42)

    expression = pd.DataFrame(
        rng.normal(size=(72, 100))
    )

    result_1 = PCAProcessor(
        n_components=10,
        random_state=42,
    ).fit_transform(expression)

    result_2 = PCAProcessor(
        n_components=10,
        random_state=42,
    ).fit_transform(expression)

    np.testing.assert_allclose(
        result_1.to_numpy(),
        result_2.to_numpy(),
    )


def test_pca_reduces_dimensions():
    rng = np.random.default_rng(42)

    expression = pd.DataFrame(
        rng.normal(size=(72, 100))
    )

    processor = PCAProcessor(
        n_components=30,
        random_state=42,
    )

    result = processor.fit_transform(expression)

    assert result.shape == (72, 30)
    assert list(result.columns) == [
        f"PC{i}" for i in range(1, 31)
    ]