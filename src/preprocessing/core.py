from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    """Preprocess gene-expression data for unsupervised subtype discovery."""

    def __init__(self, n_top_genes: int = 2000) -> None:
        self.n_top_genes = n_top_genes

    def select_variable_genes(self, expression: pd.DataFrame) -> list[str]:
        """
        Select genes with the highest variability across samples.

        `sample_id` is metadata and is never considered a gene.
        """
        gene_data = expression.drop(columns=["sample_id"], errors="ignore")      
        gene_data = gene_data.select_dtypes(include=[np.number])

        if gene_data.empty:
            raise ValueError("No numeric gene-expression columns found.")
        
        medians = gene_data.median(axis=0)
        mad = (gene_data - medians).abs().median(axis=0)

        selected = mad.sort_values(ascending=False).head(self.n_top_genes)
        return selected.index.tolist()
    
    def standardize_genes(self, expression: pd.DataFrame) -> pd.DataFrame:
        """Standardize each gene to zero mean and unit variance."""
        features = expression.drop(columns=["sample_id"])

        means = features.mean(axis=0)
        stds = features.std(axis=0, ddof=0)
        standardized = (features - means) / stds

        return standardized