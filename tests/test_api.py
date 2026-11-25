"""Comprehensive integration tests for the Cancer Subtype Discovery Vercel API."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient
from api.index import app, DISCLAIMER

client = TestClient(app)


def test_health_endpoint():
    """Verify health check returns status ok and research disclaimer."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["disclaimer"] == DISCLAIMER


def test_overview_endpoint():
    """Verify overview endpoint returns research metrics and artifact statuses."""
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 72
    assert data["n_clusters"] == 3
    assert data["cluster_counts"] == {"0": 14, "1": 35, "2": 23}
    assert data["variance_retained"] is not None
    assert data["variance_retained"] > 0
    assert data["strong_biomarker_entries"] > 0
    assert data["annotated_biomarker_entries"] > 0
    assert data["phases_available"]["clustering"] is True
    assert data["phases_available"]["pca"] is True
    assert data["phases_available"]["pathways"] is True
    assert data["phases_available"]["phase8"] is True
    assert data["phases_available"]["explainability"] is False
    assert data["disclaimer"] == DISCLAIMER


def test_dataset_endpoint():
    """Verify dataset summary and quality check flags."""
    response = client.get("/api/dataset")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    summary = data["summary"]
    assert summary["dataset"] == "Golub leukemia"
    assert summary["samples"] == 72
    assert summary["probes"] == 7129
    assert summary["selected_probes"] == 2000
    assert summary["pca_components"] == 30
    assert summary["known_labels"]["ALL"] == 47
    assert summary["known_labels"]["AML"] == 25
    assert summary["quality"]["missing_values"] == 0
    assert summary["quality"]["duplicate_sample_ids"] == 0
    assert data["disclaimer"] == DISCLAIMER


def test_pca_endpoint():
    """Verify PCA coordinates and explained variance."""
    response = client.get("/api/pca?limit=50")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert len(data["data"]) == 50
    assert data["count"] == 72
    first = data["data"][0]
    assert "sample_id" in first
    assert "PC1" in first
    assert "PC2" in first
    assert "cluster" in first
    assert len(data["variance"]) == 30
    assert data["disclaimer"] == DISCLAIMER


def test_clusters_endpoint():
    """Verify cluster assignments and summary table."""
    response = client.get("/api/clusters")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["total_samples"] == 72
    assert data["n_clusters"] == 3
    assert len(data["assignments"]) == 72
    assert len(data["summary"]) == 3
    # Check known label distributions across clusters
    c0 = next(s for s in data["summary"] if s["cluster"] == 0)
    assert c0["count"] == 14
    assert c0["known_ALL"] == 14
    assert c0["known_AML"] == 0
    assert data["disclaimer"] == DISCLAIMER


def test_validation_endpoint():
    """Verify validation metrics (ARI, silhouette, bootstrap, gap, PAC)."""
    response = client.get("/api/validation")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert len(data["ari_silhouette"]) > 0
    assert len(data["bootstrap"]) > 0
    assert len(data["gap"]) > 0
    assert len(data["pac"]) > 0
    # Check metric ranges
    for row in data["ari_silhouette"]:
        assert -1.0 <= row["ari"] <= 1.0
        assert -1.0 <= row["silhouette"] <= 1.0
    for row in data["bootstrap"]:
        assert 0.0 <= row["stability_mean"] <= 1.0
    assert data["disclaimer"] == DISCLAIMER


def test_biomarkers_filtering():
    """Verify biomarker query filtering and aliases."""
    # Test standard filters
    res = client.get("/api/biomarkers?max_fdr=0.01&min_effect=0.85")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    for row in data["data"]:
        assert row["fdr"] <= 0.01
        assert abs(row["rank_biserial"]) >= 0.85

    # Test cluster-specific filter
    res_c0 = client.get("/api/biomarkers?cluster=0")
    assert res_c0.status_code == 200
    data_c0 = res_c0.json()
    for row in data_c0["data"]:
        assert row["cluster"] == 0

    # Test parameter aliases (fdr and effect)
    res_alias = client.get("/api/biomarkers?fdr=0.05&effect=0.8&limit=10")
    assert res_alias.status_code == 200
    data_alias = res_alias.json()
    assert len(data_alias["data"]) <= 10
    assert "results" in data_alias
    assert data_alias["results"] == data_alias["data"]


def test_pathways_normalized():
    """Verify pathway enrichment normalization from ORA results."""
    response = client.get("/api/pathways?limit=20")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert len(data["data"]) <= 20
    assert data["count"] > 0
    # Check normalized schema
    first = data["data"][0]
    assert "pathway_name" in first
    assert "database" in first
    assert "pathway_id" in first
    assert "adjusted_p_value" in first
    assert "overlapping_genes" in first
    assert "cluster" in first
    assert "overlap_count" in first
    # Check metadata
    assert data["metadata"]["phase"] == 7
    assert data["metadata"]["method"] == "Over-representation analysis (ORA)"
    assert data["disclaimer"] == DISCLAIMER


def test_phase8_characterization():
    """Verify Phase 8 subtype characterization and cross-cluster comparison."""
    response = client.get("/api/phase8")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert len(data["summary"]) == 3
    assert len(data["comparison"]) > 0
    assert "metadata" in data
    # Check primary biological themes are preserved
    c0 = next(s for s in data["summary"] if s["cluster"] == 0)
    assert "ribosomal" in c0["primary_biological_theme"].lower()
    assert data["disclaimer"] == DISCLAIMER


def test_reports_inventory():
    """Verify report inventory lists available research artifacts."""
    response = client.get("/api/reports")
    assert response.status_code == 200
    data = response.json()
    assert "artifacts" in data
    names = [a["name"] for a in data["artifacts"]]
    assert "PCA coordinates" in names
    assert "Cluster assignments" in names
    assert "Validation results" in names
    assert "Strong biomarker candidates" in names
    assert "Pathway enrichment" in names
    assert "Phase 8 characterization" in names
    assert all(a["available"] for a in data["artifacts"])
    assert data["disclaimer"] == DISCLAIMER


def test_methods_documentation():
    """Verify methods endpoint returns complete methodology and limitations."""
    response = client.get("/api/methods")
    assert response.status_code == 200
    data = response.json()
    assert "dataset" in data
    assert "preprocessing" in data
    assert "clustering" in data
    assert "validation" in data
    assert "biomarkers" in data
    assert "pathway_enrichment" in data
    assert len(data["limitations"]) >= 5
    assert any("not a clinical diagnostic tool" in l.lower() or "clinical" in l.lower() for l in data["limitations"])
    assert data["disclaimer"] == DISCLAIMER


def test_invalid_query_parameters():
    """Verify input validation handles invalid limits and out-of-range floats."""
    # Limit exceeds max allowed
    res_bad_limit = client.get("/api/pca?limit=9999")
    assert res_bad_limit.status_code == 422

    # FDR out of range [0, 1]
    res_bad_fdr = client.get("/api/biomarkers?max_fdr=5.0")
    assert res_bad_fdr.status_code == 422


def test_static_frontend_root():
    """Verify that root URL / serves the responsive frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in response.text
    assert "Cancer Subtype Discovery" in response.text
    assert "MOLECULAR DISCOVERY" in response.text


def test_static_frontend_index_html():
    """Verify that /index.html serves the static frontend HTML directly."""
    response = client.get("/index.html")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Cancer Subtype Discovery" in response.text


def test_static_mount_does_not_mask_api_routes():
    """Ensure static files mount does not intercept or mask /api/* endpoints."""
    # API route returns JSON
    res_api = client.get("/api/health")
    assert res_api.status_code == 200
    assert "application/json" in res_api.headers.get("content-type", "")
    assert res_api.json()["status"] == "ok"

    # Root returns HTML
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
