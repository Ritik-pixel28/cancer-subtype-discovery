"""Export presentation coordinates without rerunning clustering or using labels in PCA."""
from pathlib import Path
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.dimensionality import PCAProcessor
from src.reporting.data import read_table

ROOT = Path(__file__).resolve().parents[1]

def export(root=ROOT):
    expression, labels = DataLoader(root / "data/raw/golub_expr.csv", root / "data/raw/golub_labels.csv").load()
    preprocessor = Preprocessor(n_top_genes=2000)
    genes = preprocessor.select_variable_genes(expression)
    selected = expression[["sample_id", *genes]]
    standardized = preprocessor.standardize_genes(selected)
    standardized.index = expression.sample_id.astype(str)
    pca = PCAProcessor(n_components=30, random_state=42)
    coordinates = pca.fit_transform(standardized).reset_index()
    clusters = read_table(root, "clusters")
    labels["sample_id"] = labels.sample_id.astype(str)
    if set(coordinates.sample_id) != set(clusters.sample_id):
        raise ValueError("Saved cluster sample IDs do not match expression sample IDs")
    coordinates = coordinates.merge(clusters, on="sample_id", validate="one_to_one").merge(labels, on="sample_id", validate="one_to_one")
    output = root / "reports/app"
    output.mkdir(parents=True, exist_ok=True)
    coordinates.to_csv(output / "pca.csv", index=False)
    pd.DataFrame({"component": range(1, 31), "explained_variance_ratio": pca.model.explained_variance_ratio_}).to_csv(output / "pca_variance.csv", index=False)
    print(f"Exported {len(coordinates)} aligned samples to {output}")

if __name__ == "__main__":
    export()
