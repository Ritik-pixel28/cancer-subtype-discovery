"""Restore the public reference inputs; optionally download enrichment resources.

Run from the repository root: python -m scripts.setup_reference_data
Existing inputs are never overwritten. No research analysis is run here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_reference(root: Path = ROOT) -> dict:
    root = Path(root)
    expression_source = root / "reports/app/expression.csv.gz"
    labels_source = root / "reports/app/pca.csv"
    expression = pd.read_csv(expression_source)
    labels = pd.read_csv(labels_source, usecols=["sample_id", "cancer_type"])
    summary = json.loads((root / "reports/app/dataset_summary.json").read_text())
    if expression.shape != (summary["samples"], summary["probes"] + 1):
        raise ValueError("Bundled expression shape does not match the dataset summary.")
    for frame in (expression, labels):
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Reference inputs contain missing or duplicate sample IDs.")
    if set(expression.sample_id) != set(labels.sample_id):
        raise ValueError("Reference expression and labels have different sample IDs.")
    numeric = expression.drop(columns="sample_id")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Reference expression contains non-finite values.")
    if labels.cancer_type.isna().any() or labels.cancer_type.value_counts().to_dict() != summary["known_labels"]:
        raise ValueError("Reference labels do not match the dataset summary.")
    labels = labels.set_index("sample_id").loc[expression.sample_id].reset_index()
    outputs = {
        root / "data/raw/golub_expr.csv": expression,
        root / "data/raw/golub_labels.csv": labels,
    }
    # Validate every existing destination before creating any missing input.
    for path, frame in outputs.items():
        if path.exists():
            try:
                pd.testing.assert_frame_equal(
                    pd.read_csv(path).set_index("sample_id").sort_index(),
                    frame.set_index("sample_id").sort_index(),
                    check_dtype=False, check_exact=True,
                )
            except (AssertionError, ValueError, KeyError) as exc:
                raise ValueError(f"Existing input differs from the bundled reference; refusing to overwrite: {path}") from exc
    for path, frame in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                frame.to_csv(stream, index=False)
    manifest = {
        "source": "Versioned public Golub dashboard artifacts",
        "note": "Values and sample labels restored; CSV formatting may differ from historical raw-file hashes. Labels are not used to fit PCA or clusters.",
        "sources_sha256": {str(p.relative_to(root)): sha256(p) for p in (expression_source, labels_source)},
        "outputs_sha256": {str(p.relative_to(root)): sha256(p) for p in outputs},
    }
    (root / "data/raw/reference_setup.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def download_enrichment(root: Path = ROOT) -> None:
    from src.annotation import load_gpl80_annotation
    from scripts.run_phase7_enrichment import load_gmt

    root = Path(root)
    config = yaml.safe_load((root / "config.yaml").read_text())
    destination = root / "data/annotation/gene_sets"
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for library in config["enrichment"]["gene_sets"]:
        if not isinstance(library, str) or Path(library).name != library:
            raise ValueError("Gene-set library must be a plain library name.")
        path = destination / f"{library}.gmt"
        url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?" + urlencode({"mode": "text", "libraryName": library})
        downloaded = not path.exists()
        if downloaded:
            import tempfile
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            with tempfile.NamedTemporaryFile(dir=destination, suffix=".gmt", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            try:
                load_gmt(temporary)  # Reject empty/error responses before installation.
                with path.open("xb") as stream:
                    stream.write(payload)
            finally:
                temporary.unlink(missing_ok=True)
        load_gmt(path)
        entries.append({"library": library, "sha256": sha256(path), "downloaded_this_run": downloaded, "download_url": url if downloaded else None})
    load_gpl80_annotation(root / "data/annotation")
    manifest = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Downloaded libraries are not verified copies of the original Phase 7 inputs. Preserve this cache and manifest for future reproduction.",
        "libraries": entries,
        "annotation_sha256": {p.name: sha256(p) for p in (root / "data/annotation").glob("GPL80*") if p.is_file()},
    }
    (destination / "setup_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-enrichment", action="store_true", help="Download missing GMT libraries and GPL80 annotation (requires internet).")
    args = parser.parse_args()
    restore_reference()
    print("Reference expression and labels ready. Existing matching inputs preserved.")
    if args.download_enrichment:
        download_enrichment()
        print("Enrichment inputs ready; checksums recorded under data/annotation/gene_sets/.")


if __name__ == "__main__":
    main()
