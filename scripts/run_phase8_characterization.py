"""
Phase 8: Integrated Biological Interpretation and Subtype Characterization Runner

Executes the Phase 8 workflow:
- Validates inputs across Phase 5, 6, and 7
- Generates cluster-level summaries
- Integrates biomarker statistics and annotations
- Integrates pathway enrichment results
- Performs cross-cluster statistical comparisons
- Generates publication-ready figures
- Generates a comprehensive markdown report and execution metadata

Outputs are saved in reports/phase8/.
"""

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interpretation import run_phase8_analysis

LOGGER = logging.getLogger("phase8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 8: Integrated Biological Interpretation and Subtype Characterization"
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=ROOT / "reports/validation/consensus_k3_clusters.csv",
        help="Path to consensus cluster assignments",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/raw/golub_labels.csv",
        help="Path to sample leukemia labels",
    )
    parser.add_argument(
        "--pca",
        type=Path,
        default=ROOT / "reports/app/pca.csv",
        help="Path to PCA coordinates",
    )
    parser.add_argument(
        "--biomarkers",
        type=Path,
        default=ROOT / "reports/biomarkers/all_biomarker_results.csv",
        help="Path to all biomarker results",
    )
    parser.add_argument(
        "--strong-candidates",
        type=Path,
        default=ROOT / "reports/biomarkers/strong_candidates_annotated.csv",
        help="Path to annotated strong biomarker candidates",
    )
    parser.add_argument(
        "--enrichment-all",
        type=Path,
        default=ROOT / "reports/enrichment/enrichment_all_results.csv",
        help="Path to all pathway enrichment results",
    )
    parser.add_argument(
        "--enrichment-sig",
        type=Path,
        default=ROOT / "reports/enrichment/enrichment_significant_results.csv",
        help="Path to significant pathway enrichment results",
    )
    parser.add_argument(
        "--enrichment-summary",
        type=Path,
        default=ROOT / "reports/enrichment/enrichment_summary.csv",
        help="Path to pathway enrichment summary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/phase8",
        help="Directory to save Phase 8 reports and figures",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    LOGGER.info("Starting Phase 8 pipeline execution...")

    try:
        result = run_phase8_analysis(
            clusters_path=args.clusters,
            labels_path=args.labels,
            pca_path=args.pca,
            biomarkers_path=args.biomarkers,
            strong_candidates_path=args.strong_candidates,
            enrichment_all_path=args.enrichment_all,
            enrichment_sig_path=args.enrichment_sig,
            enrichment_summary_path=args.enrichment_summary,
            output_dir=args.output_dir,
        )
        LOGGER.info("Phase 8 successfully finished!")
        LOGGER.info("Subtype summary shape: %s", result.subtype_summary.shape)
        LOGGER.info("Cross-cluster comparison shape: %s", result.cross_cluster_comparison.shape)
        LOGGER.info("Figures generated: %d", len(result.figure_paths))
    except Exception as e:
        LOGGER.error("Phase 8 failed with error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
