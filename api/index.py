"""FastAPI research data API for Cancer Subtype Discovery — Vercel deployment."""

from pathlib import Path
import sys

# Ensure project root is on the path so src/ and app/ modules resolve correctly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

DISCLAIMER = (
    "This application is intended for research and educational purposes. "
    "It is not a clinical diagnostic tool and should not be used to make medical decisions."
)

app = FastAPI(
    title="Cancer Subtype Discovery API",
    description="Research data API for the Golub leukemia molecular subtype discovery project.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data loading — module-level singleton, loaded once per cold start
# ---------------------------------------------------------------------------

class _AppData:
    """Loads and caches all research artifacts at startup."""

    def __init__(self, root: Path):
        self.root = root
        self.errors: dict[str, str] = {}
        self._load_all()

    def _csv(self, relative: str) -> pd.DataFrame:
        path = self.root / "reports" / relative
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            self.errors[relative] = str(exc)
            return pd.DataFrame()

    def _json(self, relative: str) -> dict:
        path = self.root / relative
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            self.errors[relative] = str(exc)
            return {}

    def _load_all(self):
        self.dataset_summary = self._json("reports/app/dataset_summary.json")
        self.pca = self._csv("app/pca.csv")
        self.pca_variance = self._csv("app/pca_variance.csv")
        self.clusters = self._csv("validation/consensus_k3_clusters.csv")
        self.validation = self._csv("validation/ari_silhouette_results.csv")
        self.bootstrap = self._csv("validation/bootstrap_results.csv")
        self.gap = self._csv("validation/gap_results.csv")
        self.pac = self._csv("validation/pac_results.csv")
        self.biomarkers = self._csv("biomarkers/strong_candidates_annotated.csv")
        self.all_biomarkers = self._csv("biomarkers/all_biomarker_results.csv")
        self.phase8_summary = self._csv("phase8/subtype_characterization_summary.csv")
        self.phase8_comparison = self._csv("phase8/cross_cluster_comparison.csv")
        self.phase8_meta = self._json("reports/phase8/phase8_run_metadata.json")
        self.enrichment_meta = self._json("reports/enrichment/enrichment_run_metadata.json")
        self._load_pathways()

    def _load_pathways(self):
        """Load and normalize the enrichment significant results."""
        raw = self._csv("enrichment/enrichment_significant_results.csv")
        if raw.empty:
            self.pathways = pd.DataFrame()
            return
        # Column normalization
        rename = {
            "Term": "pathway_name",
            "gene_set_library": "database",
            "Gene_set": "pathway_id",
            "Adjusted P-value": "adjusted_p_value",
            "FDR_BH": "fdr_bh",
            "Genes": "overlapping_genes",
            "overlap_count": "overlap_count",
            "Odds Ratio": "odds_ratio",
        }
        raw = raw.rename(columns=rename)
        # Ensure required columns exist
        for col in ["pathway_name", "database", "pathway_id", "adjusted_p_value", "overlapping_genes", "cluster", "overlap_count"]:
            if col not in raw.columns:
                raw[col] = None
        self.pathways = raw


def _safe_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe records without NaN or infinite values."""
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    cleaned = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if v is None or pd.isna(v):
                clean_row[k] = None
            elif isinstance(v, (float, np.floating)):
                clean_row[k] = float(v) if np.isfinite(v) else None
            elif isinstance(v, (int, np.integer)):
                clean_row[k] = int(v)
            elif isinstance(v, (bool, np.bool_)):
                clean_row[k] = bool(v)
            else:
                clean_row[k] = v
        cleaned.append(clean_row)
    return cleaned


_data: _AppData | None = None


def get_data() -> _AppData:
    global _data
    if _data is None:
        _data = _AppData(ROOT)
    return _data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0", "disclaimer": DISCLAIMER}


@app.get("/api/overview")
def overview():
    d = get_data()
    clusters_df = d.clusters
    pca_var = d.pca_variance
    variance_retained = None
    if not pca_var.empty and "explained_variance_ratio" in pca_var.columns:
        variance_retained = round(float(pca_var["explained_variance_ratio"].sum()), 4)
    cluster_counts = {}
    if not clusters_df.empty:
        cluster_counts = clusters_df["cluster"].value_counts().sort_index().to_dict()
        cluster_counts = {str(int(k)): int(v) for k, v in cluster_counts.items()}
    biomarkers_df = d.biomarkers
    strong_count = len(biomarkers_df) if not biomarkers_df.empty else 0
    annotated = int(biomarkers_df["gene_symbol"].notna().sum()) if not biomarkers_df.empty and "gene_symbol" in biomarkers_df.columns else 0
    return {
        "disclaimer": DISCLAIMER,
        "dataset": d.dataset_summary,
        "cluster_counts": cluster_counts,
        "total_samples": len(clusters_df) if not clusters_df.empty else None,
        "n_clusters": clusters_df["cluster"].nunique() if not clusters_df.empty else None,
        "variance_retained": variance_retained,
        "strong_biomarker_entries": strong_count,
        "annotated_biomarker_entries": annotated,
        "phases_available": {
            "pca": not d.pca.empty,
            "clustering": not d.clusters.empty,
            "validation": not d.validation.empty,
            "biomarkers": not d.biomarkers.empty,
            "pathways": not d.pathways.empty,
            "phase8": not d.phase8_summary.empty,
            "explainability": False,
        },
    }


@app.get("/api/dataset")
def dataset():
    d = get_data()
    return {
        "disclaimer": DISCLAIMER,
        "available": bool(d.dataset_summary),
        "summary": d.dataset_summary,
        "errors": {k: v for k, v in d.errors.items() if "summary" in k},
    }


@app.get("/api/pca")
def pca(limit: int = Query(default=200, le=500)):
    d = get_data()
    pca_df = d.pca
    if pca_df.empty:
        return {"available": False, "data": [], "variance": [], "disclaimer": DISCLAIMER}
    # Cap response size
    records = _safe_records(pca_df.head(limit))
    variance = _safe_records(d.pca_variance)
    return {
        "available": True,
        "count": len(pca_df),
        "data": records,
        "variance": variance,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/clusters")
def clusters():
    d = get_data()
    df = d.clusters
    if df.empty:
        return {"available": False, "data": [], "summary": [], "disclaimer": DISCLAIMER}
    records = _safe_records(df)
    # Build summary per cluster
    summary = []
    pca_df = d.pca
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        entry = {
            "cluster": int(cluster_id),
            "count": len(subset),
            "percentage": round(len(subset) / len(df) * 100, 1),
        }
        # Merge ALL/AML info from PCA if available
        if not pca_df.empty and "cancer_type" in pca_df.columns:
            pca_subset = pca_df[pca_df["cluster"] == cluster_id]
            if not pca_subset.empty:
                ct = pca_subset["cancer_type"].value_counts().to_dict()
                entry["known_ALL"] = int(ct.get("ALL", 0))
                entry["known_AML"] = int(ct.get("AML", 0))
        summary.append(entry)
    return {
        "available": True,
        "total_samples": len(df),
        "n_clusters": df["cluster"].nunique(),
        "assignments": records,
        "summary": summary,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/validation")
def validation():
    d = get_data()
    return {
        "available": not d.validation.empty,
        "ari_silhouette": _safe_records(d.validation),
        "bootstrap": _safe_records(d.bootstrap),
        "gap": _safe_records(d.gap),
        "pac": _safe_records(d.pac),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/biomarkers")
def biomarkers(
    cluster: int | None = Query(default=None),
    max_fdr: float | None = Query(default=None, ge=0.0, le=1.0),
    min_effect: float | None = Query(default=None, ge=0.0, le=1.0),
    fdr: float | None = Query(default=None, ge=0.0, le=1.0),
    effect: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=500, le=2000),
):
    d = get_data()
    df = d.biomarkers
    if df.empty:
        return {"available": False, "data": [], "results": [], "count": 0, "disclaimer": DISCLAIMER}
    f = df.copy()
    effective_fdr = max_fdr if max_fdr is not None else (fdr if fdr is not None else 0.05)
    effective_effect = min_effect if min_effect is not None else (effect if effect is not None else 0.8)
    if "fdr" in f.columns:
        f = f[f["fdr"] <= effective_fdr]
    if "rank_biserial" in f.columns:
        f = f[f["rank_biserial"].abs() >= effective_effect]
    if cluster is not None and "cluster" in f.columns:
        f = f[f["cluster"] == cluster]
    if "gene" in f.columns and "probe_id" not in f.columns:
        f["probe_id"] = f["gene"]
    if "median_difference" in f.columns and "median_diff" not in f.columns:
        f["median_diff"] = f["median_difference"]
    f = f.head(limit)
    records = _safe_records(f)
    return {
        "available": True,
        "count": len(f),
        "total_before_filter": len(df),
        "data": records,
        "results": records,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/pathways")
def pathways(
    cluster: int | None = Query(default=None),
    limit: int = Query(default=100, le=500),
):
    d = get_data()
    df = d.pathways
    if df.empty:
        return {
            "available": False,
            "data": [],
            "count": 0,
            "metadata": d.enrichment_meta,
            "disclaimer": DISCLAIMER,
        }
    f = df.copy()
    if cluster is not None and "cluster" in f.columns:
        f = f[f["cluster"] == cluster]
    # Sort by adjusted p-value
    if "adjusted_p_value" in f.columns:
        f = f.sort_values("adjusted_p_value")
    f = f.head(limit)
    return {
        "available": True,
        "count": len(f),
        "data": _safe_records(f),
        "metadata": d.enrichment_meta,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/phase8")
def phase8():
    d = get_data()
    return {
        "available": not d.phase8_summary.empty,
        "summary": _safe_records(d.phase8_summary),
        "comparison": _safe_records(d.phase8_comparison),
        "metadata": d.phase8_meta,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/reports")
def reports():
    d = get_data()
    artifacts = {
        "PCA coordinates": (d.pca, "app/pca.csv"),
        "PCA variance": (d.pca_variance, "app/pca_variance.csv"),
        "Cluster assignments": (d.clusters, "validation/consensus_k3_clusters.csv"),
        "Validation results": (d.validation, "validation/ari_silhouette_results.csv"),
        "Bootstrap stability": (d.bootstrap, "validation/bootstrap_results.csv"),
        "Gap statistic": (d.gap, "validation/gap_results.csv"),
        "PAC ambiguity": (d.pac, "validation/pac_results.csv"),
        "Strong biomarker candidates": (d.biomarkers, "biomarkers/strong_candidates_annotated.csv"),
        "Pathway enrichment": (d.pathways, "enrichment/enrichment_significant_results.csv"),
        "Phase 8 characterization": (d.phase8_summary, "phase8/subtype_characterization_summary.csv"),
    }
    result = []
    for name, (df, path) in artifacts.items():
        result.append({
            "name": name,
            "path": path,
            "available": not df.empty,
            "rows": len(df) if not df.empty else 0,
        })
    return {
        "artifacts": result,
        "errors": d.errors,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/methods")
def methods():
    return {
        "disclaimer": DISCLAIMER,
        "dataset": {
            "name": "Golub leukemia gene-expression dataset",
            "source": "Public repository (GEO)",
            "samples": 72,
            "probes": 7129,
            "known_labels": "ALL (47) and AML (25) — used only for external evaluation",
        },
        "preprocessing": {
            "feature_selection": "Median Absolute Deviation (MAD) — top 2,000 probes retained",
            "standardization": "Gene-wise centering and population standard deviation (ddof=0)",
            "note": "Known labels did not enter preprocessing or clustering",
        },
        "dimensionality_reduction": {
            "method": "PCA",
            "components": 30,
            "note": "PCA is applied to standardized selected probes only",
        },
        "clustering": {
            "methods": ["K-Means", "Ward hierarchical", "GMM", "Consensus"],
            "k_range": "2–8",
            "selected_k": 3,
            "consensus": "100 resampling iterations, 80% item subsampling",
        },
        "validation": {
            "internal": "Silhouette coefficient",
            "external": "Adjusted Rand Index (ARI) vs ALL/AML labels",
            "stability": "Bootstrap Jaccard similarity (100 iterations)",
            "gap": "Gap statistic with 1-SE rule",
            "pac": "Proportion of Ambiguous Clustering",
        },
        "biomarkers": {
            "test": "Mann-Whitney U (cluster vs rest)",
            "correction": "Benjamini-Hochberg FDR",
            "effect_size": "Rank-biserial correlation",
            "annotation": "GPL80 Affymetrix HuGeneFL platform",
            "strong_candidate_threshold": "FDR < 0.05, |rank-biserial| ≥ 0.8",
        },
        "pathway_enrichment": {
            "method": "Over-representation analysis (ORA)",
            "test": "One-sided hypergeometric test",
            "databases": ["KEGG_2021_Human", "Reactome_2022", "MSigDB_Hallmark_2020", "GO_Biological_Process_2021"],
            "correction": "Benjamini-Hochberg separately per cluster and library",
        },
        "limitations": [
            "Small cohort (n=72) limits statistical power",
            "Consensus partition is exploratory — not established clinical subtypes",
            "Same-cohort biomarker discovery requires independent biological validation",
            "External ARI agreement is modest; cluster boundaries may not align with clinical classes",
            "This application does not perform clinical diagnosis or predict patient outcomes",
        ],
    }
