"""
Phase 7: Biological Pathway Enrichment

Performs over-representation analysis (ORA) using:
- Phase 6 significant biomarker probes as the query
- Unique, unambiguous, non-control gene symbols from tested probes as background
- Locally cached GMT gene-set libraries
- Hypergeometric over-representation p-values
- Benjamini-Hochberg correction separately per cluster and library

Outputs are written to reports/enrichment/.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from src.annotation import load_gpl80_annotation

BIOMARKER_FILE = ROOT / "reports/biomarkers/all_biomarker_results.csv"
CONFIG_FILE = ROOT / "config.yaml"
OUTPUT_DIR = ROOT / "reports/enrichment"
ANNOTATION_DIR = ROOT / "data/annotation"
GENE_SET_DIR = ANNOTATION_DIR / "gene_sets"

LOGGER = logging.getLogger("phase7")


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_symbols(value: object) -> list[str]:
    """Return a symbol only when the probe maps unambiguously to one gene."""
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text in {"---", "nan", "None"} or "///" in text:
        return []

    return [text]


def load_gmt(path: Path) -> dict[str, set[str]]:
    """Load a GMT file into a term-to-gene-set dictionary."""
    if not path.is_file():
        raise FileNotFoundError(f"Gene-set file not found: {path}")

    gene_sets: dict[str, set[str]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.rstrip("\n\r").split("\t")

            if len(fields) < 3:
                LOGGER.warning(
                    "Skipping malformed GMT line %d in %s",
                    line_number,
                    path.name,
                )
                continue

            term = fields[0].strip()
            genes = {gene.strip() for gene in fields[2:] if gene.strip()}

            if term and genes:
                gene_sets[term] = genes

    if not gene_sets:
        raise ValueError(f"No valid gene sets found in {path}")

    return gene_sets


def adjust_pvalues(results: pd.DataFrame) -> pd.DataFrame:
    """Apply BH correction across the tested pathways in one cluster/library."""
    results = results.copy()

    if results.empty:
        results["FDR_BH"] = pd.Series(dtype=float)
        return results

    results["FDR_BH"] = multipletests(
        results["P-value"].astype(float).to_numpy(),
        method="fdr_bh",
    )[1]

    results["Adjusted P-value"] = results["FDR_BH"]
    return results


def run_ora(
    query_genes: set[str],
    background_genes: set[str],
    gene_sets: dict[str, set[str]],
    min_size: int,
    max_size: int,
) -> pd.DataFrame:
    """
    Run one-sided hypergeometric ORA.

    Pathway size and query overlap are calculated within the effective
    background universe. Pathways outside the configured size range are
    excluded before multiple-testing correction.
    """
    effective_background = background_genes.intersection(
        set().union(*gene_sets.values())
    )

    if not effective_background:
        raise ValueError("No background genes overlap the gene-set library.")

    query = query_genes.intersection(effective_background)

    if not query:
        LOGGER.warning(
            "No query genes overlap the effective background; "
            "no ORA results can be computed."
        )
        return pd.DataFrame()

    N = len(effective_background)
    n = len(query)
    rows = []

    for term, library_genes in gene_sets.items():
        pathway = library_genes.intersection(effective_background)
        K = len(pathway)

        if K < min_size or K > max_size:
            continue

        overlap = query.intersection(pathway)
        k = len(overlap)

        # Upper-tail probability: observing k or more query genes in the
        # pathway, given the query size and effective background.
        p_value = float(hypergeom.sf(k - 1, N, K, n)) if k else 1.0

        # Two-by-two table for an interpretable odds ratio.
        a = k
        b = n - k
        c = K - k
        d = N - a - b - c

        numerator = a * d
        denominator = b * c

        if denominator == 0:
            odds_ratio = float("inf") if numerator > 0 else float("nan")
        else:
            odds_ratio = numerator / denominator

        rows.append(
            {
                "Term": term,
                "P-value": p_value,
                "Adjusted P-value": float("nan"),
                "FDR_BH": float("nan"),
                "Odds Ratio": odds_ratio,
                "Genes": ";".join(sorted(overlap)),
                "overlap_count": k,
                "pathway_size_in_background": K,
                "query_gene_count": n,
                "background_gene_count": N,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    settings = config["enrichment"]

    libraries = settings["gene_sets"]
    min_size = int(settings["min_size"])
    max_size = int(settings["max_size"])
    fdr_threshold = float(settings["fdr_threshold"])
    organism = settings.get("organism", "Human")

    if min_size < 1 or max_size < min_size:
        raise ValueError("Invalid pathway-size limits in config.yaml.")

    LOGGER.info("Loading Phase 6 biomarker results...")
    biomarkers = pd.read_csv(BIOMARKER_FILE)

    required_columns = {"gene", "cluster", "p_value", "fdr", "significant"}
    missing_columns = required_columns - set(biomarkers.columns)

    if missing_columns:
        raise ValueError(
            f"Biomarker file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if biomarkers["gene"].isna().any():
        raise ValueError("Biomarker file contains missing probe IDs.")

    # Check that each probe has one consistent significance decision.
    if biomarkers.duplicated(["cluster", "gene"]).any():
        raise ValueError(
            "Duplicate cluster/probe rows found in biomarker results."
        )

    tested_probes = sorted(biomarkers["gene"].astype(str).unique())
    LOGGER.info("Unique tested probes: %d", len(tested_probes))

    if not tested_probes:
        raise ValueError("No tested probes found in Phase 6 results.")

    LOGGER.info("Loading probe annotation...")
    annotation = load_gpl80_annotation(ANNOTATION_DIR)

    if annotation["probe_id"].duplicated().any():
        raise ValueError("Annotation table contains duplicate probe IDs.")

    universe = pd.DataFrame({"gene": tested_probes}).merge(
        annotation,
        left_on="gene",
        right_on="probe_id",
        how="left",
        validate="one_to_one",
    )

    universe["is_control"] = universe["gene"].str.startswith("AFFX-")
    universe["usable_symbols"] = universe["gene_symbol"].apply(split_symbols)

    universe["mapping_status"] = "unmapped"
    universe.loc[
        universe["usable_symbols"].map(bool), "mapping_status"
    ] = "mapped"
    universe.loc[
        universe["gene_symbol"].fillna("").str.contains("///"),
        "mapping_status",
    ] = "ambiguous"
    universe.loc[universe["is_control"], "mapping_status"] = "control"

    universe.to_csv(
        OUTPUT_DIR / "annotated_tested_probe_universe.csv",
        index=False,
    )

    background_genes = {
        symbol
        for symbols, is_control in zip(
            universe["usable_symbols"], universe["is_control"]
        )
        if not is_control
        for symbol in symbols
    }

    if len(background_genes) < min_size:
        raise ValueError(
            f"Too few mapped background genes: {len(background_genes)}"
        )

    LOGGER.info("Unique mapped background genes: %d", len(background_genes))

    mapping_summary = {
        "tested_probe_count": len(tested_probes),
        "mapped_unique_gene_count": len(background_genes),
        "unmapped_probe_count": int(
            (universe["mapping_status"] == "unmapped").sum()
        ),
        "ambiguous_probe_count": int(
            (universe["mapping_status"] == "ambiguous").sum()
        ),
        "control_probe_count": int(
            (universe["mapping_status"] == "control").sum()
        ),
        "annotation_platform": "GPL80",
    }

    (OUTPUT_DIR / "mapping_summary.json").write_text(
        json.dumps(mapping_summary, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Loading local gene-set libraries...")
    loaded_libraries: dict[str, dict[str, set[str]]] = {}

    for library in libraries:
        path = GENE_SET_DIR / f"{library}.gmt"
        loaded_libraries[library] = load_gmt(path)
        LOGGER.info(
            "%s: loaded %d gene sets",
            library,
            len(loaded_libraries[library]),
        )

    # Map every tested probe to its usable symbols once.
    probe_to_symbols = dict(
        zip(universe["gene"], universe["usable_symbols"])
    )
    control_probes = set(
        universe.loc[universe["is_control"], "gene"].astype(str)
    )

    all_results: list[pd.DataFrame] = []
    run_errors: list[dict] = []

    for cluster in sorted(biomarkers["cluster"].unique()):
        LOGGER.info("Processing cluster %s...", cluster)

        cluster_df = biomarkers.loc[
            biomarkers["cluster"] == cluster
        ].copy()

        significant = cluster_df.loc[
            cluster_df["significant"].map(
                lambda value: str(value).strip().lower()
                in {"true", "1", "yes"}
                if not pd.isna(value)
                else False
            )
        ]

        query_genes = {
            symbol
            for probe in significant["gene"].astype(str)
            if probe not in control_probes
            for symbol in probe_to_symbols.get(probe, [])
        }

        LOGGER.info(
            "Cluster %s: %d significant probes, %d unique query genes",
            cluster,
            len(significant),
            len(query_genes),
        )

        if not query_genes:
            LOGGER.warning("No usable query genes for cluster %s.", cluster)
            continue

        for library in libraries:
            LOGGER.info(
                "Cluster %s — gene set %s", cluster, library
            )

            try:
                result = run_ora(
                    query_genes=query_genes,
                    background_genes=background_genes,
                    gene_sets=loaded_libraries[library],
                    min_size=min_size,
                    max_size=max_size,
                )

                if result.empty:
                    LOGGER.warning(
                        "No eligible pathway results for cluster %s, %s",
                        cluster,
                        library,
                    )
                    continue

                result = adjust_pvalues(result)
                result["cluster"] = cluster
                result["gene_set_library"] = library
                result["Gene_set"] = library
                result["Old P-value"] = float("nan")
                result["Old adjusted P-value"] = float("nan")
                result["Combined Score"] = float("nan")

                all_results.append(result)

                LOGGER.info(
                    "Cluster %s, %s: tested %d pathways; %d significant",
                    cluster,
                    library,
                    len(result),
                    int((result["FDR_BH"] <= fdr_threshold).sum()),
                )

            except Exception as exc:
                LOGGER.exception(
                    "Enrichment failed for cluster %s, library %s",
                    cluster,
                    library,
                )
                run_errors.append(
                    {
                        "cluster": str(cluster),
                        "gene_set_library": library,
                        "error": str(exc),
                    }
                )

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        combined.to_csv(
            OUTPUT_DIR / "enrichment_all_results.csv",
            index=False,
        )

        significant_results = combined.loc[
            pd.to_numeric(combined["FDR_BH"], errors="coerce")
            <= fdr_threshold
        ].copy()

        significant_results.to_csv(
            OUTPUT_DIR / "enrichment_significant_results.csv",
            index=False,
        )

        # Include every cluster/library combination, including zero counts.
        summary = (
            combined.groupby(
                ["cluster", "gene_set_library"], sort=False
            )["FDR_BH"]
            .apply(lambda values: int((values <= fdr_threshold).sum()))
            .rename("significant_pathway_count")
            .reset_index()
        )

        summary.to_csv(
            OUTPUT_DIR / "enrichment_summary.csv",
            index=False,
        )

        LOGGER.info("Total pathways tested: %d", len(combined))
        LOGGER.info(
            "Significant pathways at FDR <= %.3f: %d",
            fdr_threshold,
            len(significant_results),
        )
    else:
        LOGGER.warning("No enrichment results were generated.")

    metadata = {
        "phase": 7,
        "method": "Over-representation analysis (ORA)",
        "test": "One-sided hypergeometric test",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "organism": organism,
        "gene_set_libraries": libraries,
        "gene_set_source": "Locally cached GMT files",
        "min_size": min_size,
        "max_size": max_size,
        "pathway_size_definition": (
            "Number of pathway genes overlapping the effective background "
            "universe, after filtering to the configured size range."
        ),
        "fdr_threshold": fdr_threshold,
        "multiple_testing_correction": (
            "Benjamini-Hochberg separately for each cluster and library"
        ),
        "background_definition": (
            "Unique mapped gene symbols from probes tested in Phase 6; "
            "ambiguous mappings and AFFX controls excluded. For each library, "
            "the effective background is further restricted to genes present "
            "in at least one gene set in that library."
        ),
        "mapping_summary": mapping_summary,
        "errors": run_errors,
    }

    (OUTPUT_DIR / "enrichment_run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    errors_path = OUTPUT_DIR / "enrichment_errors.csv"
    if run_errors:
        pd.DataFrame(run_errors).to_csv(errors_path, index=False)
    elif errors_path.exists():
        errors_path.unlink()

    LOGGER.info("Phase 7 outputs saved under %s", OUTPUT_DIR)

    if run_errors:
        raise RuntimeError(
            f"Phase 7 completed with {len(run_errors)} errors. "
            f"Review {errors_path}."
        )


if __name__ == "__main__":
    main()
