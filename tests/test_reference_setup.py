"""The offline bootstrap must preserve data and refuse conflicting local inputs."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.setup_reference_data import restore_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference_root(tmp_path):
    target = tmp_path / "reports/app"
    target.mkdir(parents=True)
    for name in ("expression.csv.gz", "pca.csv", "dataset_summary.json"):
        shutil.copy2(ROOT / "reports/app" / name, target / name)
    return tmp_path


def test_restore_preserves_expression_and_label_alignment(reference_root):
    manifest = restore_reference(reference_root)
    expr = pd.read_csv(reference_root / "data/raw/golub_expr.csv")
    labels = pd.read_csv(reference_root / "data/raw/golub_labels.csv")
    pd.testing.assert_frame_equal(expr, pd.read_csv(reference_root / "reports/app/expression.csv.gz"))
    assert expr.sample_id.tolist() == labels.sample_id.tolist()
    assert labels.cancer_type.value_counts().to_dict() == {"ALL": 47, "AML": 25}
    assert len(manifest["sources_sha256"]) == 2


def test_restore_does_not_rewrite_existing_matching_inputs(reference_root):
    restore_reference(reference_root)
    paths = list((reference_root / "data/raw").glob("*.csv"))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    restore_reference(reference_root)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}


def test_conflict_is_rejected_before_any_missing_input_is_written(reference_root):
    raw = reference_root / "data/raw"
    raw.mkdir(parents=True)
    labels = raw / "golub_labels.csv"
    labels.write_text("sample_id,cancer_type\n1,OTHER\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        restore_reference(reference_root)
    assert labels.read_text() == "sample_id,cancer_type\n1,OTHER\n"
    assert not (raw / "golub_expr.csv").exists()


def test_misaligned_source_is_rejected(reference_root):
    path = reference_root / "reports/app/pca.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "sample_id"] = 99999
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different sample IDs"):
        restore_reference(reference_root)
    assert not (reference_root / "data/raw").exists()
