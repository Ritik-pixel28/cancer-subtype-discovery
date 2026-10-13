from pathlib import Path
from html import escape
from datetime import datetime, timezone
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd

TABLES = {
    "clusters": ("validation/consensus_k3_clusters.csv", ["sample_id", "cluster"]),
    "validation": ("validation/ari_silhouette_results.csv", ["method", "k", "ari", "silhouette"]),
    "bootstrap": ("validation/bootstrap_results.csv", ["method", "k", "stability_mean", "ci_lower", "ci_upper"]),
    "gap": ("validation/gap_results.csv", ["k", "gap", "gap_std"]),
    "pac": ("validation/pac_results.csv", ["k", "pac"]),
    "biomarkers": ("biomarkers/strong_candidates_annotated.csv", ["gene", "cluster", "gene_symbol", "fdr", "rank_biserial", "median_difference", "direction"]),
    "summary": ("biomarkers/biomarker_summary.csv", ["cluster", "samples", "genes_tested", "fdr_significant_genes"]),
    "pca": ("app/pca.csv", ["sample_id", "PC1", "PC2", "PC3", "cluster"]),
    "variance": ("app/pca_variance.csv", ["component", "explained_variance_ratio"]),
    "enrichment": ("enrichment/pathway_results.csv", ["cluster", "pathway", "gene_set", "fdr", "overlap_genes"]),
    "shap": ("explainability/shap_importance.csv", ["cluster", "gene", "mean_abs_shap"]),
}

def read_table(root: Path, name: str) -> pd.DataFrame:
    relative, required = TABLES[name]
    path = root / "reports" / relative
    if not path.exists():
        return pd.DataFrame(columns=required)
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{relative}: missing columns {', '.join(sorted(missing))}")
    numeric = set(required) & {"cluster", "k", "PC1", "PC2", "PC3", "ari", "silhouette", "stability_mean", "ci_lower", "ci_upper", "gap", "gap_std", "pac", "fdr", "rank_biserial", "median_difference", "mean_abs_shap", "explained_variance_ratio", "samples", "genes_tested", "fdr_significant_genes"}
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{relative}: {column} contains missing or non-finite values")
    for column in set(frame.columns) & {"fdr", "pac", "explained_variance_ratio"}:
        if not frame[column].between(0, 1).all():
            raise ValueError(f"{relative}: {column} must be between 0 and 1")
    if "mean_abs_shap" in frame and (frame.mean_abs_shap < 0).any():
        raise ValueError(f"{relative}: mean_abs_shap must be nonnegative")
    if "sample_id" in required:
        if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError(f"{relative}: sample IDs must be present and unique")
        frame["sample_id"] = frame.sample_id.astype(str)
    return frame

def filter_biomarkers(frame, clusters, query="", max_fdr=0.05, min_effect=0.8):
    result = frame[frame.cluster.isin(clusters) & (frame.fdr <= max_fdr) & (frame.rank_biserial.abs() >= min_effect)]
    if query.strip():
        searchable = result.reindex(columns=["gene", "gene_symbol", "gene_title"]).fillna("").astype(str)
        mask = searchable.apply(lambda col: col.str.contains(query.strip(), case=False, regex=False)).any(axis=1)
        result = result[mask]
    return result.sort_values(["cluster", "fdr", "gene"])

def report_html(tables: dict[str, pd.DataFrame]) -> str:
    sections = []
    for name, frame in tables.items():
        if frame.empty:
            sections.append(f"<h2>{escape(name.title())}</h2><p>Results not available yet.</p>")
        else:
            sections.append(f"<h2>{escape(name.title())}</h2>" + frame.to_html(index=False, escape=True))
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Cancer Subtype Discovery Report</title>
<style>body{font:14px system-ui;color:#152d3c;max-width:1200px;margin:40px auto;padding:24px}h1{font-size:36px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:11px}td,th{border:1px solid #dae2e6;padding:7px;text-align:left}th{background:#edf6f5}@media print{body{margin:0}thead{display:table-header-group}tr{break-inside:avoid}}</style>
<h1>Cancer Subtype Discovery</h1><p>Phase 9 · Reproducible research report</p>
<p>Exploratory consensus k=3 partition. Known ALL/AML labels are used only for external evaluation, never to form clusters. These results do not establish clinical subtypes or diagnostic biomarkers. Small sample size, modest external agreement and uncertain stability require independent validation.</p>
<h2>Methods</h2><p>Golub leukemia expression data → MAD top 2,000 probes → gene-wise standardization → 30-component PCA. Saved consensus assignments and validation results are read without rerunning clustering. Biomarkers use Mann–Whitney U, BH-FDR and rank-biserial effect size. Strong candidate tables are preselected at |effect| ≥ 0.8. Phase 7 and Phase 8 are included only if their result files exist.</p>
<p>Generated: ''' + datetime.now(timezone.utc).isoformat() + "</p>" + "".join(sections) + "</html>"

def report_bundle(root: Path, tables) -> bytes:
    buffer = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.html", report_html(tables))
        for name, (relative, _) in TABLES.items():
            path = root / "reports" / relative
            if path.exists():
                data = path.read_bytes()
                archive.writestr("results/" + relative, data)
                manifest[relative] = hashlib.sha256(data).hexdigest()
        archive.writestr("manifest.json", json.dumps({"sha256": manifest, "scope": "All available saved results; dashboard filters are not applied."}, indent=2))
    return buffer.getvalue()
