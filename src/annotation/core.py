"""
Probe annotation for the Cancer Subtype Discovery pipeline.

This module maps Affymetrix Hgu6800 (GPL80) probe IDs to HGNC gene
symbols, gene titles, and Entrez Gene IDs using the NCBI GEO platform
annotation table for GPL80.

Platform context
----------------
The Golub leukemia dataset was profiled on the Affymetrix Hu6800
microarray, registered in NCBI GEO as **GPL80**.

Column names retrieved from GPL80:
    - ``ID``              → original probe set identifier
    - ``Gene Symbol``     → HGNC gene symbol (may contain ``///`` for
                            probes that cross-hybridise to multiple genes)
    - ``Gene Title``      → full gene name
    - ``ENTREZ_GENE_ID``  → Entrez Gene ID (may contain ``///`` for
                            multi-mapping probes)

Mapping status categories
--------------------------
``mapped``
    Probe matched the GPL80 table AND has a non-empty gene symbol.
``unmapped``
    Probe not found in the GPL80 table (e.g. HG-prefix composite
    probes, ``hum_alu_at``).
``no_symbol``
    Probe found in the GPL80 table but the ``Gene Symbol`` field is
    blank (deprecated probes or probes targeting non-coding sequence).
``multi_gene``
    Probe found in GPL80 and has a gene symbol that contains ``///``,
    indicating cross-hybridisation to two or more genes.
``control``
    Probe whose ID begins with ``AFFX-`` or is exactly ``hum_alu_at``
    (Affymetrix spike-in and housekeeping controls; no human gene
    mapping exists or is meaningful).

Multi-gene handling
-------------------
When a probe maps to multiple gene symbols separated by ``///``, the
mapping is preserved verbatim at the probe level (``gene_symbol`` keeps
the ``///`` form).  For downstream enrichment analysis, the caller must
decide how to expand or deduplicate multi-gene probes.  This module
documents and counts them; it does not silently resolve ambiguity.

Background universe
-------------------
The enrichment background must be derived from the complete set of
probes tested in Phase 6 (all 2 000 MAD-selected probes), NOT only
the significant or strong-candidate subsets.  Call
``annotate_probes(all_results_df, annotation_df)`` where
``all_results_df`` contains one row per probe per cluster from
``reports/biomarkers/all_biomarker_results.csv``, or pass the unique
list of 2 000 probe IDs directly.

Caching
-------
GEOparse is used to download the GPL80 soft file on the first run.
The file is saved to ``data/annotation/`` and reused on subsequent
runs, so repeated executions do not incur a network round-trip.

Classes
-------
AnnotationResult
    Structured container for probe annotation outputs and provenance.
ProbeAnnotator
    Core annotation engine; wraps loading, merging, and classification.

Functions
---------
load_gpl80_annotation(annotation_dir)
    Load (or download) the GPL80 annotation table and return the
    cleaned DataFrame.
annotate_probes(probe_df, annotation_df)
    Merge a probe-containing DataFrame with the annotation table and
    return an AnnotationResult.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLATFORM_ACCESSION: str = "GPL80"
"""GEO platform accession for the Affymetrix Hu6800 array."""

PROBE_ANNOTATION_SOURCE: str = "NCBI GEO GPL80 (Affymetrix Hu6800)"
"""Human-readable description of the annotation source."""

_REQUIRED_GPL80_COLUMNS: tuple[str, ...] = (
    "ID",
    "Gene Symbol",
    "Gene Title",
    "ENTREZ_GENE_ID",
)

_CONTROL_PREFIXES: tuple[str, ...] = ("AFFX-",)
_CONTROL_EXACT: frozenset[str] = frozenset({"hum_alu_at"})


# ---------------------------------------------------------------------------
# Mapping-status constants
# ---------------------------------------------------------------------------

STATUS_MAPPED: str = "mapped"
STATUS_UNMAPPED: str = "unmapped"
STATUS_NO_SYMBOL: str = "no_symbol"
STATUS_MULTI_GENE: str = "multi_gene"
STATUS_CONTROL: str = "control"


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def _is_control_probe(probe_id: str) -> bool:
    """Return True if *probe_id* is an Affymetrix spike-in/control probe."""
    if probe_id in _CONTROL_EXACT:
        return True
    return any(probe_id.startswith(pfx) for pfx in _CONTROL_PREFIXES)


def _has_multi_gene_symbol(gene_symbol: str) -> bool:
    """Return True if *gene_symbol* contains the ``///`` multi-mapping delimiter."""
    return "///" in str(gene_symbol)


# ---------------------------------------------------------------------------
# AnnotationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnnotationResult:
    """
    Container for probe annotation outputs and provenance.

    Attributes
    ----------
    annotated_df : pd.DataFrame
        Input probe DataFrame with annotation columns appended:
        ``gene_symbol``, ``gene_title``, ``entrez_gene_id``,
        ``mapping_status``, ``annotation_source``,
        ``platform_accession``.
    universe_df : pd.DataFrame
        Unique-probe-level annotation for the background gene universe
        (one row per unique probe ID, no duplicates across clusters).
    unmapped_df : pd.DataFrame
        Rows from ``annotated_df`` where ``mapping_status`` is
        ``unmapped`` or ``no_symbol``.
    control_df : pd.DataFrame
        Rows from ``annotated_df`` where ``mapping_status`` is
        ``control``.
    multi_gene_df : pd.DataFrame
        Rows from ``annotated_df`` where ``mapping_status`` is
        ``multi_gene``.
    statistics : dict
        Mapping summary counts:
        ``total_input_rows``, ``unique_probe_ids``,
        ``mapped``, ``unmapped``, ``no_symbol``,
        ``multi_gene``, ``control``,
        ``universe_probes``, ``universe_mapped_probes``,
        ``universe_unique_gene_symbols``.
    platform_accession : str
        GEO platform accession used.
    annotation_source : str
        Human-readable description of the annotation source.
    """

    annotated_df: pd.DataFrame
    universe_df: pd.DataFrame
    unmapped_df: pd.DataFrame
    control_df: pd.DataFrame
    multi_gene_df: pd.DataFrame
    statistics: dict = field(default_factory=dict)
    platform_accession: str = PLATFORM_ACCESSION
    annotation_source: str = PROBE_ANNOTATION_SOURCE


# ---------------------------------------------------------------------------
# GPL80 loader
# ---------------------------------------------------------------------------

def load_gpl80_annotation(
    annotation_dir: str | Path,
    *,
    silent: bool = True,
) -> pd.DataFrame:
    """
    Load the GPL80 annotation table, downloading it if necessary.

    The downloaded SOFT file is cached in *annotation_dir*.  On
    subsequent calls the cached file is used without network access.

    Parameters
    ----------
    annotation_dir :
        Directory where the GPL80 SOFT file will be saved / is cached.
    silent :
        Suppress GEOparse download progress messages.

    Returns
    -------
    pd.DataFrame
        Annotation table with columns renamed to:
        ``probe_id``, ``gene_symbol``, ``gene_title``,
        ``entrez_gene_id``.
        Missing values are represented as ``pd.NA`` (not empty strings).

    Raises
    ------
    ImportError
        If the ``GEOparse`` package is not installed and no cached
        annotation file exists.
    FileNotFoundError
        If GEOparse fails and no local cache is available.
    ValueError
        If the downloaded GPL80 table is missing required columns.
    """
    annotation_dir = Path(annotation_dir)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    # Check for an already-parsed parquet cache (fast path)
    parquet_cache = annotation_dir / "gpl80_annotation.parquet"
    if parquet_cache.exists():
        logger.info(
            "Loading GPL80 annotation from local cache: %s",
            parquet_cache,
        )
        return pd.read_parquet(parquet_cache)

    # Check for a GEOparse-cached SOFT file (medium path)
    soft_candidates = list(annotation_dir.glob("GPL80*.txt")) + list(
        annotation_dir.glob("GPL80*.soft")
    )

    raw_table: Optional[pd.DataFrame] = None

    if soft_candidates:
        logger.info(
            "Parsing existing GPL80 SOFT file: %s",
            soft_candidates[0],
        )
        try:
            import GEOparse  # type: ignore[import]

            gpl = GEOparse.get_GEO(
                geo=PLATFORM_ACCESSION,
                destdir=str(annotation_dir),
                silent=silent,
            )
            raw_table = gpl.table.copy()
        except Exception as exc:
            logger.warning(
                "GEOparse failed to parse cached SOFT file: %s", exc
            )

    if raw_table is None:
        # Try to download via GEOparse (slow path — requires network)
        try:
            import GEOparse  # type: ignore[import]

            logger.info(
                "Downloading GPL80 annotation from NCBI GEO (this "
                "may take a moment)…"
            )
            gpl = GEOparse.get_GEO(
                geo=PLATFORM_ACCESSION,
                destdir=str(annotation_dir),
                silent=silent,
            )
            raw_table = gpl.table.copy()
        except ImportError:
            raise ImportError(
                "The 'GEOparse' package is required to download GPL80 "
                "annotation.\n"
                "Install it with:  pip install GEOparse\n"
                "Alternatively, place the GPL80 SOFT file in "
                f"{annotation_dir}."
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not retrieve GPL80 annotation: {exc}\n"
                f"Checked cache directory: {annotation_dir}"
            )

    annotation = _clean_gpl80_table(raw_table)

    # Write parquet cache for future fast loads
    try:
        annotation.to_parquet(parquet_cache, index=False)
        logger.info("GPL80 annotation cached to %s", parquet_cache)
    except Exception as exc:
        logger.warning(
            "Could not write parquet cache (non-fatal): %s", exc
        )

    return annotation


def _clean_gpl80_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a raw GPL80 annotation table.

    Performs the following:
    - Checks that required columns are present.
    - Selects and renames relevant columns.
    - Strips leading/trailing whitespace from string fields.
    - Converts empty strings and ``"---"`` placeholders to ``pd.NA``.
    - Ensures the ``probe_id`` column is unique (GPL80 should have
      one row per probe).

    Parameters
    ----------
    raw_table :
        Raw DataFrame as returned by GEOparse.

    Returns
    -------
    pd.DataFrame with columns:
        ``probe_id``, ``gene_symbol``, ``gene_title``,
        ``entrez_gene_id``.
    """
    available = [c for c in _REQUIRED_GPL80_COLUMNS if c in raw_table.columns]
    missing = [c for c in _REQUIRED_GPL80_COLUMNS if c not in raw_table.columns]

    if "ID" not in raw_table.columns:
        raise ValueError(
            f"GPL80 table is missing the required 'ID' column. "
            f"Available columns: {raw_table.columns.tolist()}"
        )

    if missing:
        logger.warning(
            "GPL80 table is missing columns %s; they will be absent "
            "from the annotation output.",
            missing,
        )

    subset = raw_table[available].copy()

    rename_map = {
        "ID": "probe_id",
        "Gene Symbol": "gene_symbol",
        "Gene Title": "gene_title",
        "ENTREZ_GENE_ID": "entrez_gene_id",
    }
    subset = subset.rename(columns={k: v for k, v in rename_map.items() if k in subset.columns})

    # Normalise string columns
    for col in ["probe_id", "gene_symbol", "gene_title", "entrez_gene_id"]:
        if col in subset.columns:
            subset[col] = (
                subset[col]
                .astype(str)
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "---": pd.NA})
            )

    if subset["probe_id"].duplicated().any():
        n_dups = subset["probe_id"].duplicated().sum()
        logger.warning(
            "GPL80 annotation contains %d duplicate probe IDs; "
            "keeping first occurrence.",
            n_dups,
        )
        subset = subset.drop_duplicates(subset="probe_id", keep="first")

    subset = subset.reset_index(drop=True)
    logger.info(
        "GPL80 annotation loaded: %d probe entries.", len(subset)
    )
    return subset


# ---------------------------------------------------------------------------
# Probe annotator
# ---------------------------------------------------------------------------

def annotate_probes(
    probe_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    *,
    probe_col: str = "gene",
) -> AnnotationResult:
    """
    Annotate a probe-containing DataFrame using the GPL80 annotation table.

    This function performs a **left merge** on the probe identifier so
    that every input row is preserved (no silent row loss).  It then
    assigns a ``mapping_status`` to each row.

    Row multiplication check
    ------------------------
    If the merge produces more rows than the input (which should not
    happen because GPL80 has unique probe IDs), a ``ValueError`` is
    raised.

    Parameters
    ----------
    probe_df :
        DataFrame containing at least one column with probe IDs.
        This is typically ``strong_candidates.csv`` or
        ``all_biomarker_results.csv``.
    annotation_df :
        GPL80 annotation table as returned by ``load_gpl80_annotation``.
    probe_col :
        Name of the column in *probe_df* that contains probe IDs.

    Returns
    -------
    AnnotationResult

    Raises
    ------
    ValueError
        If *probe_col* is not found in *probe_df*.
    ValueError
        If the merge unexpectedly multiplies rows.
    """
    if probe_col not in probe_df.columns:
        raise ValueError(
            f"Column '{probe_col}' not found in probe_df. "
            f"Available columns: {probe_df.columns.tolist()}"
        )

    if "probe_id" not in annotation_df.columns:
        raise ValueError(
            "annotation_df must contain a 'probe_id' column. "
            "Use load_gpl80_annotation() to obtain a correctly "
            "formatted annotation table."
        )

    n_input = len(probe_df)

    # Perform left merge preserving all input rows
    merged = probe_df.merge(
        annotation_df,
        left_on=probe_col,
        right_on="probe_id",
        how="left",
    )

    # Guard: no row multiplication
    if len(merged) != n_input:
        raise ValueError(
            f"Annotation merge produced {len(merged)} rows from "
            f"{n_input} input rows. GPL80 annotation must have unique "
            f"probe IDs. Check the annotation table for duplicates."
        )

    # Drop the redundant 'probe_id' column from the right side
    if "probe_id" in merged.columns and "probe_id" != probe_col:
        merged = merged.drop(columns=["probe_id"])

    # Assign mapping status
    merged["mapping_status"] = merged.apply(
        _assign_mapping_status, axis=1, probe_col=probe_col
    )

    # Add provenance columns
    merged["annotation_source"] = PROBE_ANNOTATION_SOURCE
    merged["platform_accession"] = PLATFORM_ACCESSION

    # Build the four result sub-DataFrames
    unmapped_mask = merged["mapping_status"].isin(
        [STATUS_UNMAPPED, STATUS_NO_SYMBOL]
    )
    control_mask = merged["mapping_status"] == STATUS_CONTROL
    multi_gene_mask = merged["mapping_status"] == STATUS_MULTI_GENE

    unmapped_df = merged[unmapped_mask].copy().reset_index(drop=True)
    control_df = merged[control_mask].copy().reset_index(drop=True)
    multi_gene_df = merged[multi_gene_mask].copy().reset_index(drop=True)

    # Build the background gene universe (unique probe-level view)
    # One row per unique probe ID, regardless of how many clusters it
    # appeared in. This is the correct denominator for ORA.
    universe_df = (
        merged[[probe_col, "gene_symbol", "gene_title", "entrez_gene_id",
                "mapping_status", "annotation_source", "platform_accession"]]
        .drop_duplicates(subset=probe_col)
        .reset_index(drop=True)
    )

    # Extract unique mapped gene symbols for the universe count.
    # Explode multi-gene symbols separated by /// and count distinct.
    mapped_universe = universe_df[
        universe_df["mapping_status"].isin([STATUS_MAPPED, STATUS_MULTI_GENE])
    ]
    all_symbols = (
        mapped_universe["gene_symbol"]
        .dropna()
        .astype(str)
        .apply(lambda s: [x.strip() for x in s.split("///")])
        .explode()
        .str.strip()
    )
    unique_symbols = set(all_symbols[all_symbols != ""])

    status_counts = merged["mapping_status"].value_counts().to_dict()

    statistics = {
        "total_input_rows": n_input,
        "unique_probe_ids": int(merged[probe_col].nunique()),
        "mapped": int(status_counts.get(STATUS_MAPPED, 0)),
        "multi_gene": int(status_counts.get(STATUS_MULTI_GENE, 0)),
        "unmapped": int(status_counts.get(STATUS_UNMAPPED, 0)),
        "no_symbol": int(status_counts.get(STATUS_NO_SYMBOL, 0)),
        "control": int(status_counts.get(STATUS_CONTROL, 0)),
        "universe_probes": int(len(universe_df)),
        "universe_mapped_probes": int(
            len(mapped_universe)
        ),
        "universe_unique_gene_symbols": int(len(unique_symbols)),
    }

    _log_statistics(statistics)

    return AnnotationResult(
        annotated_df=merged,
        universe_df=universe_df,
        unmapped_df=unmapped_df,
        control_df=control_df,
        multi_gene_df=multi_gene_df,
        statistics=statistics,
    )


def _assign_mapping_status(row: pd.Series, probe_col: str) -> str:
    """Classify a single merged row into a mapping status category."""
    probe_id = str(row[probe_col])

    if _is_control_probe(probe_id):
        return STATUS_CONTROL

    gene_symbol = row.get("gene_symbol", pd.NA)

    # Probe not found in GPL80 (all annotation columns are NA)
    if pd.isna(gene_symbol):
        return STATUS_UNMAPPED

    symbol_str = str(gene_symbol).strip()

    if symbol_str == "":
        return STATUS_NO_SYMBOL

    if _has_multi_gene_symbol(symbol_str):
        return STATUS_MULTI_GENE

    return STATUS_MAPPED


def _log_statistics(stats: dict) -> None:
    """Emit annotation summary to the logger at INFO level."""
    logger.info(
        "Annotation summary:\n"
        "  Total input rows       : %d\n"
        "  Unique probe IDs       : %d\n"
        "  Status: mapped         : %d\n"
        "  Status: multi_gene     : %d\n"
        "  Status: unmapped       : %d\n"
        "  Status: no_symbol      : %d\n"
        "  Status: control        : %d\n"
        "  Universe probes        : %d\n"
        "  Universe mapped probes : %d\n"
        "  Universe unique symbols: %d",
        stats["total_input_rows"],
        stats["unique_probe_ids"],
        stats["mapped"],
        stats["multi_gene"],
        stats["unmapped"],
        stats["no_symbol"],
        stats["control"],
        stats["universe_probes"],
        stats["universe_mapped_probes"],
        stats["universe_unique_gene_symbols"],
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class ProbeAnnotator:
    """
    High-level interface for GPL80 probe annotation.

    This class wraps ``load_gpl80_annotation`` and ``annotate_probes``
    in a single object that can be reused across multiple calls without
    re-loading the annotation table.

    Parameters
    ----------
    annotation_dir :
        Path to the directory where GPL80 annotation data is cached.
    silent :
        Suppress GEOparse download messages.
    """

    def __init__(
        self,
        annotation_dir: str | Path = "data/annotation",
        *,
        silent: bool = True,
    ) -> None:
        self.annotation_dir = Path(annotation_dir)
        self.silent = silent
        self._annotation_df: Optional[pd.DataFrame] = None

    @property
    def annotation_df(self) -> pd.DataFrame:
        """
        Lazily loaded GPL80 annotation table.

        The table is loaded once and cached in memory.
        """
        if self._annotation_df is None:
            self._annotation_df = load_gpl80_annotation(
                self.annotation_dir,
                silent=self.silent,
            )
        return self._annotation_df

    def annotate(
        self,
        probe_df: pd.DataFrame,
        *,
        probe_col: str = "gene",
    ) -> AnnotationResult:
        """
        Annotate *probe_df* using the cached GPL80 annotation table.

        Parameters
        ----------
        probe_df :
            DataFrame containing probe IDs in *probe_col*.
        probe_col :
            Name of the probe-ID column.

        Returns
        -------
        AnnotationResult
        """
        return annotate_probes(
            probe_df,
            self.annotation_df,
            probe_col=probe_col,
        )
