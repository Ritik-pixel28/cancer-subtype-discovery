"""Validated repository boundary. No clustering is performed by the application."""

from pathlib import Path
from functools import cached_property
import hashlib
import importlib.metadata
import json
import subprocess
import numpy as np
import pandas as pd
import yaml
from src.reporting.data import TABLES, read_table

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL = {
    "pathways": (
        "enrichment/pathway_results.csv",
        [
            "cluster",
            "database",
            "pathway_id",
            "pathway_name",
            "adjusted_p_value",
            "overlap_count",
            "overlapping_genes",
        ],
    ),
    "importance": (
        "explainability/global_importance.csv",
        ["cluster", "feature_id", "mean_abs_shap"],
    ),
    "explanations": (
        "explainability/sample_explanations.csv",
        ["sample_id", "target_cluster", "feature_id", "shap_value", "feature_value"],
    ),
    "predictions": (
        "explainability/surrogate_predictions.csv",
        ["sample_id", "predicted_cluster", "evaluation_split"],
    ),
}


def fingerprint(root=ROOT):
    files = (
        list((root / "reports").rglob("*.csv"))
        + list((root / "reports/app").glob("*.gz"))
        + list((root / "reports").rglob("*.json"))
        + [root / "config.yaml"]
    )
    return tuple(
        (str(p.relative_to(root)), p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(files)
        if p.exists()
    )


def numeric(frame, cols, source):
    for col in cols:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            if not np.isfinite(frame[col]).all():
                raise ValueError(
                    f"{source}: {col} contains missing or non-finite values"
                )
    return frame


def integer_labels(values, source):
    if ((values % 1) != 0).any() or (values < 0).any():
        raise ValueError(f"{source}: cluster IDs must be nonnegative integers")


def same_samples(a, b, source):
    if a.isna().any() or a.duplicated().any():
        raise ValueError(f"{source}: missing or duplicate sample IDs")
    if set(a.astype(str)) != set(b.astype(str)):
        raise ValueError(f"{source}: sample IDs do not exactly match saved assignments")


class ResearchData:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.errors = {}
        self.tables = {}
        self.status = {}
        for name in TABLES:
            if name in ["enrichment", "shap"]:
                continue
            self._load(name, lambda n=name: read_table(self.root, n))
        self.config = self._json_or_yaml("config.yaml", yaml.safe_load)
        self.summary = self._json_or_yaml(
            "reports/app/dataset_summary.json", json.loads
        )
        self._load("expression", self._expression)
        self._load("mad", lambda: pd.read_csv(self.root / "reports/app/mad.csv"))
        self._load("all_biomarkers", self._all_biomarkers)
        for name, (relative, required) in OPTIONAL.items():
            if name == "pathways":
                self._load(name, self._pathways)
            else:
                self._load(name, lambda p=relative, r=required: self._optional(p, r))
        self.model = self._json_or_yaml(
            "reports/explainability/model_metadata.json", json.loads
        )
        metadata_path = "reports/enrichment/metadata.json"
        if not (self.root / metadata_path).exists():
            metadata_path = "reports/enrichment/enrichment_run_metadata.json"
        self.enrichment_meta = self._json_or_yaml(metadata_path, json.loads)
        self._load("characterization", self._characterization_summary)
        self._load("consensus", self._consensus)
        self._validate_alignment()

    def _load(self, name, fn):
        try:
            self.tables[name] = fn()
            self.status[name] = (
                "Available" if not self.tables[name].empty else "Not available"
            )
        except FileNotFoundError:
            self.tables[name] = pd.DataFrame()
            self.status[name] = "Not available"
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
            self.tables[name] = pd.DataFrame()
            self.errors[name] = str(exc)
            self.status[name] = "Invalid artifact"

    def _json_or_yaml(self, path, parser):
        full = self.root / path
        if not full.exists():
            return {}
        try:
            value = parser(full.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError("Metadata must be a mapping")
            return value
        except (ValueError, yaml.YAMLError) as exc:
            self.errors[path] = str(exc)
            return {}

    def _expression(self):
        frame = pd.read_csv(
            self.root / "reports/app/expression.csv.gz", dtype={"sample_id": str}
        )
        if "sample_id" not in frame:
            raise ValueError("Expression matrix needs sample_id")
        if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError("Expression sample IDs must be present and unique")
        numeric(frame, list(frame.drop(columns="sample_id").columns), "Expression")
        return frame.set_index("sample_id")

    def _all_biomarkers(self):
        frame = pd.read_csv(self.root / "reports/biomarkers/all_biomarker_results.csv")
        required = {
            "gene",
            "cluster",
            "fdr",
            "p_value",
            "median_difference",
            "rank_biserial",
        }
        if required - set(frame):
            raise ValueError(
                "All biomarker results: required statistical columns missing"
            )
        numeric(
            frame,
            ["cluster", "fdr", "p_value", "rank_biserial", "median_difference"],
            "Biomarkers",
        )
        if (
            not frame.fdr.between(0, 1).all()
            or not frame.p_value.between(0, 1).all()
            or not frame.rank_biserial.between(-1, 1).all()
        ):
            raise ValueError("Biomarker statistics outside their valid ranges")
        if frame.duplicated(["gene", "cluster"]).any():
            raise ValueError("Duplicate probe-cluster result")
        frame["direction"] = np.where(
            frame.median_difference > 0,
            "higher",
            np.where(frame.median_difference < 0, "lower", "equal"),
        )
        annotation = read_table(self.root, "biomarkers")
        if not annotation.empty:
            names = annotation[["gene", "gene_symbol", "gene_title"]].drop_duplicates(
                "gene"
            )
            frame = frame.merge(names, on="gene", how="left", validate="many_to_one")
        else:
            frame["gene_symbol"] = None
            frame["gene_title"] = None
        return frame

    def _pathways(self):
        relative, required = OPTIONAL["pathways"]
        if (self.root / "reports" / relative).exists():
            return self._optional(relative, required)
        # Phase 7's native ORA export contains all tests, including nonsignificant rows.
        source = "enrichment/enrichment_all_results.csv"
        frame = pd.read_csv(self.root / "reports" / source)
        frame = frame.rename(columns={
            "Term": "pathway_name",
            "gene_set_library": "database",
            "Adjusted P-value": "adjusted_p_value",
            "P-value": "raw_p_value",
            "Genes": "overlapping_genes",
            "pathway_size_in_background": "pathway_size",
        })
        # The native export has term names, not accession IDs. Keep those labels
        # as lookup keys without claiming that an accession was supplied.
        if "pathway_id" not in frame and "pathway_name" in frame:
            frame["pathway_id"] = frame["pathway_name"]
        return self._validate_optional(frame, required, source)

    def _optional(self, relative, required):
        frame = pd.read_csv(
            self.root / "reports" / relative,
            dtype={"sample_id": str, "feature_id": str},
        )
        return self._validate_optional(frame, required, relative)

    def _validate_optional(self, frame, required, relative):
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(f"{relative}: missing {sorted(missing)}")
        numeric(
            frame,
            [
                "cluster",
                "target_cluster",
                "predicted_cluster",
                "adjusted_p_value",
                "raw_p_value",
                "overlap_count",
                "pathway_size",
                "enrichment_score",
                "mean_abs_shap",
                "rank",
                "shap_value",
                "feature_value",
            ],
            relative,
        )
        for field in ["cluster", "target_cluster", "predicted_cluster"]:
            if field in frame:
                integer_labels(frame[field], relative)
        for field in ["adjusted_p_value", "raw_p_value"]:
            if field in frame and not frame[field].between(0, 1).all():
                raise ValueError(f"{relative}: invalid probability")
        for field in ["mean_abs_shap", "overlap_count", "pathway_size"]:
            if field in frame and (frame[field] < 0).any():
                raise ValueError(f"{relative}: negative {field}")
        for field in ["pathway_name", "database", "pathway_id", "feature_id"]:
            if field in frame and (
                frame[field].isna().any()
                or frame[field].astype(str).str.strip().eq("").any()
            ):
                raise ValueError(f"{relative}: empty {field}")
        if "sample_id" in frame and frame.sample_id.isna().any():
            raise ValueError(f"{relative}: missing sample ID")
        if "evaluation_split" in frame:
            if (
                frame.evaluation_split.isna().any()
                or frame.evaluation_split.astype(str).str.strip().eq("").any()
            ):
                raise ValueError(f"{relative}: missing evaluation split")
            if frame.duplicated(["sample_id", "evaluation_split"]).any():
                raise ValueError(
                    f"{relative}: duplicate sample within evaluation split"
                )
        return frame

    def _consensus(self):
        frame = pd.read_csv(
            self.root / "reports/validation/consensus_k3_matrix.csv", index_col=0
        )
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        if frame.shape[0] != frame.shape[1] or set(frame.index) != set(frame.columns):
            raise ValueError(
                "Consensus matrix must be square with matching sample axes"
            )
        frame = frame.loc[frame.index, frame.index]
        v = frame.to_numpy(dtype=float)
        if (
            not np.isfinite(v).all()
            or (v < 0).any()
            or (v > 1).any()
            or not np.allclose(v, v.T)
            or not np.allclose(np.diag(v), 1)
        ):
            raise ValueError(
                "Consensus matrix must be symmetric, finite, in [0,1], with unit diagonal"
            )
        return frame

    def _validate_alignment(self):
        clusters = self.get("clusters")
        if clusters.empty:
            return
        try:
            integer_labels(clusters.cluster, "Clusters")
        except ValueError as exc:
            self.errors["clusters"] = str(exc)
            self.tables["clusters"] = pd.DataFrame()
            return
        ids = clusters.sample_id
        valid = set(clusters.cluster)
        for name in ["pca", "expression", "consensus"]:
            frame = self.get(name)
            if frame.empty:
                continue
            try:
                samples = frame.sample_id if name == "pca" else pd.Series(frame.index)
                same_samples(samples, ids, name)
                if name == "pca":
                    a = frame.set_index("sample_id").cluster.sort_index()
                    b = clusters.set_index("sample_id").cluster.sort_index()
                    if not a.equals(b):
                        raise ValueError(
                            "PCA cluster labels differ from saved consensus labels"
                        )
            except ValueError as exc:
                self.errors[name] = str(exc)
                self.tables[name] = pd.DataFrame()
                self.status[name] = "Invalid artifact"
        for name in [
            "biomarkers",
            "all_biomarkers",
            "pathways",
            "importance",
            "explanations",
            "predictions",
        ]:
            frame = self.get(name)
            if frame.empty:
                continue
            try:
                for col in ["cluster", "target_cluster", "predicted_cluster"]:
                    if col in frame:
                        integer_labels(frame[col], name)
                        if not set(frame[col]).issubset(valid):
                            raise ValueError(f"{name}: unknown cluster IDs")
                if "sample_id" in frame and not set(frame.sample_id).issubset(set(ids)):
                    raise ValueError(f"{name}: unknown sample IDs")
            except ValueError as exc:
                self.errors[name] = str(exc)
                self.tables[name] = pd.DataFrame()
                self.status[name] = "Invalid artifact"
        if self.summary and self.summary.get("samples") != len(clusters):
            self.errors["summary"] = (
                "Dataset summary sample count differs from saved assignments"
            )
            self.summary = {}

    def get(self, name):
        return self.tables.get(name, pd.DataFrame())

    def expression_long(self, probes, standardized=False, clusters=None):
        expr = self.get("expression")
        assignment = self.get("clusters")
        if expr.empty or assignment.empty:
            return pd.DataFrame()
        values = expr.loc[:, list(probes)].copy()
        if standardized:
            values = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
        values = values.reset_index().merge(
            assignment, on="sample_id", validate="one_to_one"
        )
        if clusters is not None:
            values = values[values.cluster.isin(clusters)]
        return values.melt(
            id_vars=["sample_id", "cluster"],
            var_name="probe_id",
            value_name="expression",
        )

    @cached_property
    def probe_labels(self):
        markers = self.get("biomarkers")
        if markers.empty:
            return {}
        return {
            row.gene: f"{row.gene_symbol} · {row.gene}"
            for row in markers.drop_duplicates("gene").itertuples()
            if pd.notna(row.gene_symbol)
        }

    def probe_label(self, probe):
        return self.probe_labels.get(probe, probe)

    def provenance(self):
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=self.root, text=True
                )
            )
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
            dirty = None
        versions = {}
        for package in [
            "streamlit",
            "pandas",
            "numpy",
            "plotly",
            "scikit-learn",
            "reportlab",
            "matplotlib",
        ]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unavailable"
        files = {
            str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((self.root / "reports").rglob("*"))
            if p.is_file()
            and p.suffix in [".csv", ".gz", ".json"]
            and "generated" not in p.parts
        }
        files["config.yaml"] = hashlib.sha256(
            (self.root / "config.yaml").read_bytes()
        ).hexdigest()
        return {
            "dataset": self.summary,
            "config": self.config,
            "application_commit": commit,
            "working_tree_modified": dirty,
            "software": versions,
            "artifact_sha256": files,
            "phase7": self.status.get("pathways"),
            "phase8": self.status.get("characterization"),
            "model_explainability": self.status.get("importance"),
        }

    def _characterization_summary(self):
        source = "phase8/subtype_characterization_summary.csv"
        frame = pd.read_csv(self.root / "reports" / source)
        required = {"cluster", "sample_count", "known_ALL_count", "known_AML_count",
                    "strong_candidate_count", "significant_pathways_total"}
        if required - set(frame):
            raise ValueError("Subtype characterization summary is missing required columns")
        numeric(frame, sorted(required), source)
        for field in required:
            integer_labels(frame[field], source)
        if frame.cluster.duplicated().any():
            raise ValueError("Subtype characterization summary has duplicate clusters")
        counts = self.get("clusters").cluster.value_counts().sort_index()
        actual = frame.set_index("cluster").sample_count.sort_index()
        if not actual.equals(counts):
            raise ValueError("Subtype characterization sample counts do not match saved clusters")
        if not (frame.known_ALL_count + frame.known_AML_count == frame.sample_count).all():
            raise ValueError("Subtype characterization reference label counts do not match sample counts")
        return frame

    def phase8_characterization(self):
        """Load Subtype characterization and cross-cluster comparisons if available."""
        path = self.root / "reports/phase8/subtype_characterization_summary.csv"
        if not path.is_file():
            return None
        try:
            summary = self.get("characterization")
            if summary.empty:
                return None
            cross_path = self.root / "reports/phase8/cross_cluster_comparison.csv"
            cross = pd.read_csv(cross_path) if cross_path.is_file() else pd.DataFrame()
            meta_path = self.root / "reports/phase8/phase8_run_metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            return {
                "summary": summary,
                "comparison": cross,
                "metadata": meta,
            }
        except Exception:
            return None
