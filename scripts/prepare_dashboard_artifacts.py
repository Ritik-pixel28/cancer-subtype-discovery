"""Prepare read-only presentation artifacts with the existing preprocessing implementation."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import yaml
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from scripts.export_phase9_pca import export

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT):
    expr, labels = DataLoader(
        root / "data/raw/golub_expr.csv", root / "data/raw/golub_labels.csv"
    ).load()
    config = yaml.safe_load((root / "config.yaml").read_text())
    n = config["data"]["n_top_genes"]
    pre = Preprocessor(n_top_genes=n)
    genes = pre.select_variable_genes(expr)
    numeric = expr.drop(columns="sample_id")
    mad = (numeric - numeric.median()).abs().median().sort_values(ascending=False)
    ranking = mad.rename("mad").rename_axis("probe_id").reset_index()
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected"] = ranking.probe_id.isin(genes)
    out = root / "reports/app"
    out.mkdir(exist_ok=True, parents=True)
    expr.to_csv(
        out / "expression.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    ranking.to_csv(out / "mad.csv", index=False)
    export(root)
    summary = {
        "dataset": "Golub leukemia",
        "group": "G87",
        "samples": len(expr),
        "probes": len(numeric.columns),
        "selected_probes": len(genes),
        "pca_components": config["dimensionality_reduction"]["pca_n_components"],
        "seed": config["seed"],
        "known_labels": labels.cancer_type.value_counts().to_dict(),
        "quality": {
            "missing_values": int(numeric.isna().sum().sum()),
            "duplicate_sample_ids": int(expr.sample_id.duplicated().sum()),
            "non_numeric_columns": int(
                len(numeric.columns)
                - len(numeric.select_dtypes(include="number").columns)
            ),
            "constant_probes": int((numeric.nunique() <= 1).sum()),
            "negative_values": int((numeric < 0).sum().sum()),
            "aligned_samples": set(expr.sample_id) == set(labels.sample_id),
        },
    }
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    paths = [
        "data/raw/golub_expr.csv",
        "data/raw/golub_labels.csv",
        "config.yaml",
        "reports/validation/consensus_k3_clusters.csv",
    ]
    summary["source_sha256"] = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"Prepared {len(expr)} samples, {len(numeric.columns)} probes; no clustering was rerun."
    )


if __name__ == "__main__":
    prepare()
