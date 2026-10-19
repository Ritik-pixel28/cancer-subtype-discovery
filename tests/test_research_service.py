from pathlib import Path
import shutil
import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from app.services.research import ResearchData, same_samples, integer_labels
from src.reporting.engine import ReportOptions, build_report, selected_tables

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return ResearchData(ROOT)


def test_real_artifact_integrity(data):
    assert not data.errors
    assert data.get("expression").shape == (72, 7129)
    assert len(data.get("all_biomarkers")) == 6000
    assert data.get("mad").selected.sum() == 2000
    assert data.get("biomarkers").cluster.value_counts().to_dict() == {
        0: 123,
        2: 44,
        1: 16,
    }


def test_missing_and_duplicate_samples_rejected():
    with pytest.raises(ValueError):
        same_samples(pd.Series(["a", "a"]), pd.Series(["a", "b"]), "test")
    with pytest.raises(ValueError):
        same_samples(pd.Series(["a"]), pd.Series(["a", "b"]), "test")
    with pytest.raises(ValueError):
        integer_labels(pd.Series([0, 1.5]), "test")


def test_expression_is_joined_by_id(data):
    probe = data.get("biomarkers").iloc[0].gene
    long = data.expression_long([probe])
    assignment = data.get("clusters").set_index("sample_id")
    assert len(long) == 72
    assert all(
        row.cluster == assignment.loc[row.sample_id, "cluster"]
        for row in long.itertuples()
    )
    z = data.expression_long([probe], True)
    assert abs(z.expression.mean()) < 1e-12
    assert np.isclose(z.expression.std(ddof=0), 1)


def test_cluster_report_scope(data):
    tables = selected_tables(data, ReportOptions("Cluster-specific report", 0))
    assert len(tables["clusters"]) == 14
    assert len(tables["biomarkers"]) == 123
    assert set(tables["pca"].cluster) == {0}


def test_report_pdf_and_exports(data):
    result = build_report(
        data,
        ReportOptions(
            "Executive summary",
            None,
            (
                "Dataset",
                "Clusters",
                "Validation",
                "Biomarkers",
                "Limitations",
                "Provenance",
            ),
        ),
    )
    reader = PdfReader(io.BytesIO(result["pdf"]))
    assert len(reader.pages) == 4
    text = " ".join(p.extract_text() for p in reader.pages)
    assert "G87" in text and "183" in text and "diagnosis" in text
    with zipfile.ZipFile(io.BytesIO(result["zip"])) as z:
        assert "figures/clusters.svg" in z.namelist()
        metadata = json.loads(z.read("provenance.json"))
        assert metadata["artifact_sha256"]
        assert len(pd.read_csv(z.open("tables/clusters.csv"))) == 72
    assert b"<h2>PCA</h2>" not in result["html"]


def test_invalid_required_report_data_fails(data):
    original = data.errors.copy()
    data.errors["biomarkers"] = "invalid test fixture"
    try:
        with pytest.raises(ValueError, match="invalid biomarkers"):
            selected_tables(data, ReportOptions())
    finally:
        data.errors = original


def fixture_root(tmp_path):
    (tmp_path / "reports").mkdir()
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    shutil.copytree(ROOT / "reports/validation", tmp_path / "reports/validation")
    return tmp_path


def test_optional_phase7_and_phase8_contracts(tmp_path):
    root = fixture_root(tmp_path)
    (root / "reports/enrichment").mkdir()
    (root / "reports/explainability").mkdir()
    pd.DataFrame(
        [
            {
                "cluster": 0,
                "database": "TEST DATABASE",
                "pathway_id": "TEST:1",
                "pathway_name": "SYNTHETIC FIXTURE ONLY",
                "adjusted_p_value": 0.03,
                "overlap_count": 2,
                "overlapping_genes": "TESTA;TESTB",
            }
        ]
    ).to_csv(root / "reports/enrichment/pathway_results.csv", index=False)
    pd.DataFrame(
        [{"cluster": 0, "feature_id": "TEST_PROBE", "mean_abs_shap": 0.2}]
    ).to_csv(root / "reports/explainability/global_importance.csv", index=False)
    d = ResearchData(root)
    assert len(d.get("pathways")) == 1
    assert len(d.get("importance")) == 1
    pd.DataFrame(
        [{"cluster": 8, "feature_id": "TEST_PROBE", "mean_abs_shap": -0.2}]
    ).to_csv(root / "reports/explainability/global_importance.csv", index=False)
    d = ResearchData(root)
    assert "importance" in d.errors
    assert d.get("importance").empty


def test_missing_consensus_not_fabricated(data):
    assert data.get("consensus").empty
    assert data.get("importance").empty


def test_native_phase7_results_keep_original_statistics(data):
    raw = pd.read_csv(ROOT / "reports/enrichment/enrichment_all_results.csv")
    loaded = data.get("pathways")
    assert data.status["pathways"] == "Available"
    assert len(loaded) == len(raw) == 8724
    assert np.array_equal(loaded.adjusted_p_value, raw["Adjusted P-value"])
    assert np.array_equal(loaded.raw_p_value, raw["P-value"])
    assert loaded.database.equals(raw.gene_set_library)
    assert (loaded.adjusted_p_value <= 0.05).sum() == 254
    assert "enrichment_score" not in loaded
    assert data.enrichment_meta["test"] == "One-sided hypergeometric test"


def test_invalid_native_phase7_statistics_rejected(tmp_path):
    root = fixture_root(tmp_path)
    target = root / "reports/enrichment"
    target.mkdir()
    frame = pd.read_csv(ROOT / "reports/enrichment/enrichment_all_results.csv").head(1)
    frame.loc[0, "Adjusted P-value"] = 1.5
    frame.to_csv(target / "enrichment_all_results.csv", index=False)
    result = ResearchData(root)
    assert result.get("pathways").empty
    assert result.status["pathways"] == "Invalid artifact"
