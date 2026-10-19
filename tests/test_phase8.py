"""
Unit tests for Phase 8: Integrated Biological Interpretation and Subtype Characterization
"""

import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from app.services.research import ResearchData
from src.interpretation import SubtypeCharacterizer, run_phase8_analysis

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def valid_inputs():
    """Return dictionary of real valid input file paths."""
    return {
        "clusters_path": ROOT / "reports/validation/consensus_k3_clusters.csv",
        "labels_path": ROOT / "data/raw/golub_labels.csv",
        "pca_path": ROOT / "reports/app/pca.csv",
        "biomarkers_path": ROOT / "reports/biomarkers/all_biomarker_results.csv",
        "strong_candidates_path": ROOT / "reports/biomarkers/strong_candidates_annotated.csv",
        "enrichment_all_path": ROOT / "reports/enrichment/enrichment_all_results.csv",
        "enrichment_sig_path": ROOT / "reports/enrichment/enrichment_significant_results.csv",
        "enrichment_summary_path": ROOT / "reports/enrichment/enrichment_summary.csv",
    }


def test_missing_input_file_raises_filenotfound(tmp_path, valid_inputs):
    """Ensure missing input files raise FileNotFoundError."""
    invalid_inputs = valid_inputs.copy()
    invalid_inputs["clusters_path"] = tmp_path / "nonexistent_clusters.csv"
    characterizer = SubtypeCharacterizer(output_dir=tmp_path / "out", **invalid_inputs)
    with pytest.raises(FileNotFoundError, match="Required Phase input missing"):
        characterizer.validate_inputs()


def test_invalid_schema_missing_columns(tmp_path, valid_inputs):
    """Ensure tables with missing required columns raise ValueError."""
    bad_clusters = tmp_path / "bad_clusters.csv"
    pd.DataFrame({"wrong_col": [1, 2]}).to_csv(bad_clusters, index=False)

    inputs = valid_inputs.copy()
    inputs["clusters_path"] = bad_clusters
    characterizer = SubtypeCharacterizer(output_dir=tmp_path / "out", **inputs)
    with pytest.raises(ValueError, match="Clusters table missing required columns"):
        characterizer.validate_inputs()


def test_duplicate_sample_ids_rejected(tmp_path, valid_inputs):
    """Ensure duplicate sample IDs in cluster assignments raise ValueError."""
    bad_clusters = tmp_path / "dup_clusters.csv"
    clusters_df = pd.read_csv(valid_inputs["clusters_path"])
    # Duplicate first sample
    dup_df = pd.concat([clusters_df.iloc[[0]], clusters_df], ignore_index=True)
    dup_df.to_csv(bad_clusters, index=False)

    inputs = valid_inputs.copy()
    inputs["clusters_path"] = bad_clusters
    characterizer = SubtypeCharacterizer(output_dir=tmp_path / "out", **inputs)
    with pytest.raises(ValueError, match="missing or duplicate sample IDs"):
        characterizer.validate_inputs()


def test_sample_id_mismatch_rejected(tmp_path, valid_inputs):
    """Ensure mismatch between cluster sample IDs and label sample IDs raises ValueError."""
    bad_labels = tmp_path / "mismatched_labels.csv"
    labels_df = pd.read_csv(valid_inputs["labels_path"]).iloc[:-5]  # remove 5 samples
    labels_df.to_csv(bad_labels, index=False)

    inputs = valid_inputs.copy()
    inputs["labels_path"] = bad_labels
    characterizer = SubtypeCharacterizer(output_dir=tmp_path / "out", **inputs)
    with pytest.raises(ValueError, match="Sample ID mismatch"):
        characterizer.validate_inputs()


def test_cluster_level_aggregation_and_proportions(tmp_path, valid_inputs):
    """Verify correct cluster sample counts, percentages, and total proportions."""
    out_dir = tmp_path / "reports_phase8"
    characterizer = SubtypeCharacterizer(output_dir=out_dir, **valid_inputs)
    characterizer.validate_inputs()
    summary = characterizer.build_cluster_summaries()

    assert len(summary) == 3
    assert set(summary["cluster"]) == {0, 1, 2}

    # Total samples should be 72
    assert summary["sample_count"].sum() == 72
    assert np.isclose(summary["cluster_proportion"].sum(), 1.0, atol=1e-3)

    # Exact expected cluster sample sizes
    c_counts = summary.set_index("cluster")["sample_count"].to_dict()
    assert c_counts == {0: 14, 1: 35, 2: 23}

    # Known label totals (47 ALL, 25 AML)
    assert summary["known_ALL_count"].sum() == 47
    assert summary["known_AML_count"].sum() == 25

    # Cluster 0 is 100% ALL
    assert summary.loc[summary["cluster"] == 0, "ALL_percentage"].iloc[0] == 100.0
    assert summary.loc[summary["cluster"] == 0, "AML_percentage"].iloc[0] == 0.0

    # Cluster 2 is predominantly AML
    assert summary.loc[summary["cluster"] == 2, "AML_percentage"].iloc[0] > 70.0


def test_biomarker_integration_schema_and_counts(tmp_path, valid_inputs):
    """Verify biomarker integration preserves all strong candidates and computes status."""
    out_dir = tmp_path / "reports_phase8"
    characterizer = SubtypeCharacterizer(output_dir=out_dir, **valid_inputs)
    characterizer.validate_inputs()
    bio_integrated = characterizer.build_biomarker_integration()

    assert len(bio_integrated) == 183
    assert "annotation_status" in bio_integrated.columns
    assert set(bio_integrated["annotation_status"]).issubset({"unambiguous", "ambiguous", "unmapped"})
    assert set(bio_integrated["direction"]).issubset({"higher", "lower"})
    assert (bio_integrated["rank_biserial"].abs() >= 0.8).all()
    assert (bio_integrated["fdr"] <= 0.05).all()


def test_cross_cluster_comparison_statistics(tmp_path, valid_inputs):
    """Verify pairwise cross-cluster comparisons, Fisher exact test, and PCA distances."""
    out_dir = tmp_path / "reports_phase8"
    characterizer = SubtypeCharacterizer(output_dir=out_dir, **valid_inputs)
    characterizer.validate_inputs()
    cross = characterizer.build_cross_cluster_comparison()

    assert len(cross) == 3  # (0 vs 1), (0 vs 2), (1 vs 2)
    for row in cross.itertuples():
        assert 0.0 <= row.biomarker_jaccard_similarity <= 1.0
        assert 0.0 <= row.fisher_exact_p_value <= 1.0
        assert row.pca_centroid_euclidean_distance > 0.0

    # Cluster 1 vs 2 has significant AML enrichment difference
    c1_c2 = cross[cross["comparison"] == "Cluster 1 vs Cluster 2"].iloc[0]
    assert c1_c2.fisher_exact_p_value < 0.001


def test_empty_significant_pathways_handled_gracefully(tmp_path, valid_inputs):
    """Verify that an empty significant pathway file does not crash analysis."""
    empty_sig = tmp_path / "empty_sig.csv"
    pd.DataFrame(columns=[
        "Term", "P-value", "Adjusted P-value", "FDR_BH", "Odds Ratio", "Genes",
        "overlap_count", "pathway_size_in_background", "query_gene_count",
        "background_gene_count", "cluster", "gene_set_library"
    ]).to_csv(empty_sig, index=False)

    inputs = valid_inputs.copy()
    inputs["enrichment_sig_path"] = empty_sig
    out_dir = tmp_path / "out"
    characterizer = SubtypeCharacterizer(output_dir=out_dir, **inputs)
    characterizer.validate_inputs()

    summary = characterizer.build_cluster_summaries()
    assert (summary["significant_pathways_total"] == 0).all()

    pathway_df = characterizer.build_pathway_integration()
    assert pathway_df.empty


def test_deterministic_reproducibility(tmp_path, valid_inputs):
    """Verify that two identical runs produce identical output checksums."""
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"

    res1 = run_phase8_analysis(output_dir=run1_dir, **valid_inputs)
    res2 = run_phase8_analysis(output_dir=run2_dir, **valid_inputs)

    pd.testing.assert_frame_equal(res1.subtype_summary, res2.subtype_summary)
    pd.testing.assert_frame_equal(res1.biomarker_summary, res2.biomarker_summary)
    pd.testing.assert_frame_equal(res1.pathway_summary, res2.pathway_summary)
    pd.testing.assert_frame_equal(res1.cross_cluster_comparison, res2.cross_cluster_comparison)


def test_generated_artifacts_integrity():
    """Verify the generated artifacts in reports/phase8 meet required schemas and files."""
    phase8_dir = ROOT / "reports/phase8"
    assert phase8_dir.is_dir()

    expected_files = [
        "subtype_characterization_summary.csv",
        "cluster_biomarker_summary.csv",
        "cluster_pathway_summary.csv",
        "cross_cluster_comparison.csv",
        "subtype_characterization_report.md",
        "phase8_run_metadata.json",
    ]
    for filename in expected_files:
        filepath = phase8_dir / filename
        assert filepath.is_file(), f"Missing Phase 8 artifact: {filepath}"
        assert filepath.stat().st_size > 0, f"Artifact is empty: {filepath}"

    # Verify figures directory
    figures_dir = phase8_dir / "figures"
    assert figures_dir.is_dir()
    expected_figures = [
        "cluster_sample_distribution.png",
        "biomarker_counts_by_cluster.png",
        "pathway_enrichment_counts_by_cluster.png",
        "top_biomarkers_effect_size.png",
        "top_pathways_cluster0.png",
        "cross_cluster_pca_distribution.png",
    ]
    for fig in expected_figures:
        fig_path = figures_dir / fig
        assert fig_path.is_file(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure unexpectedly small: {fig_path}"

    # Verify metadata JSON
    meta_path = phase8_dir / "phase8_run_metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["status"] == "COMPLETED"
    assert meta["phase"] == 8
    assert "inputs" in meta
    assert "outputs" in meta


def test_expected_phenotype_counts_and_fisher_exact_p_values():
    """Verify expected phenotype counts and independently reproduced Fisher exact p-values."""
    cross_path = ROOT / "reports/phase8/cross_cluster_comparison.csv"
    assert cross_path.is_file()
    cross_df = pd.read_csv(cross_path)

    # 1. Independent calculation of 2x2 contingency tables
    # Cluster 0: ALL=14, AML=0; Cluster 1: ALL=27, AML=8; Cluster 2: ALL=6, AML=17
    contingencies = {
        "Cluster 0 vs Cluster 1": [[14, 27], [0, 8]],
        "Cluster 0 vs Cluster 2": [[14, 6], [0, 17]],
        "Cluster 1 vs Cluster 2": [[27, 6], [8, 17]],
    }

    for comp_name, table in contingencies.items():
        expected_or, expected_p = fisher_exact(table)
        row = cross_df[cross_df["comparison"] == comp_name].iloc[0]

        # Verify p-value match
        assert np.isclose(row["fisher_exact_p_value"], expected_p, rtol=1e-5), (
            f"P-value mismatch for {comp_name}: expected {expected_p}, got {row['fisher_exact_p_value']}"
        )

        # Verify odds ratio match (including inf for zero cell)
        if np.isinf(expected_or):
            assert np.isinf(row["fisher_odds_ratio"]), f"Expected infinite odds ratio for {comp_name}"
        else:
            assert np.isclose(row["fisher_odds_ratio"], expected_or, rtol=1e-4), (
                f"Odds ratio mismatch for {comp_name}: expected {expected_or}, got {row['fisher_odds_ratio']}"
            )


def test_zero_cell_odds_ratio_handling_and_reporting():
    """Verify correct handling and transparent reporting of zero-cell odds ratios."""
    cross_path = ROOT / "reports/phase8/cross_cluster_comparison.csv"
    cross_df = pd.read_csv(cross_path)

    # In CSV: zero AML in cluster 0 must yield float inf, NOT NaN or arbitrary pseudocounts
    c0_c1 = cross_df[cross_df["comparison"] == "Cluster 0 vs Cluster 1"].iloc[0]
    assert np.isinf(c0_c1["fisher_odds_ratio"])

    c0_c2 = cross_df[cross_df["comparison"] == "Cluster 0 vs Cluster 2"].iloc[0]
    assert np.isinf(c0_c2["fisher_odds_ratio"])

    # Finite comparison has numeric odds ratio
    c1_c2 = cross_df[cross_df["comparison"] == "Cluster 1 vs Cluster 2"].iloc[0]
    assert np.isclose(c1_c2["fisher_odds_ratio"], 9.5625, rtol=1e-4)

    # In Markdown report: report must distinguish zero-cell infinite OR from N/A
    report_path = ROOT / "reports/phase8/subtype_characterization_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    assert "+∞ (zero cell)" in report_text
    assert "N/A" not in report_text or "N/A (undefined)" not in report_text


def test_markdown_cross_cluster_table_cell_counts():
    """Verify that all rows in the rendered Markdown cross-cluster comparison table contain exactly 8 columns."""
    report_path = ROOT / "reports/phase8/subtype_characterization_report.md"
    assert report_path.is_file()
    lines = report_path.read_text(encoding="utf-8").splitlines()

    # Locate the cross-cluster comparison table
    in_table = False
    table_rows = []
    for line in lines:
        if "## 4. Cross-Cluster Comparison" in line:
            in_table = True
            continue
        if in_table:
            stripped = line.strip()
            if stripped.startswith("|"):
                table_rows.append(stripped)
            elif in_table and stripped.startswith("##"):
                break

    assert len(table_rows) >= 5  # header + separator + 3 data rows

    # Header row
    header_cols = [c.strip() for c in table_rows[0].split("|")[1:-1]]
    assert len(header_cols) == 8, f"Expected 8 header columns, got {len(header_cols)}: {header_cols}"

    # Separator row
    sep_cols = [c.strip() for c in table_rows[1].split("|")[1:-1]]
    assert len(sep_cols) == 8

    # Data rows
    for i, data_row in enumerate(table_rows[2:5], start=1):
        cells = [c.strip() for c in data_row.split("|")[1:-1]]
        assert len(cells) == 8, (
            f"Row {i} has {len(cells)} cells instead of 8: {cells}\nRaw line: {data_row}"
        )


def test_csv_schemas_and_phase9_compatibility():
    """Verify preservation of existing CSV schemas and backward compatibility with Phase 9."""
    phase8_dir = ROOT / "reports/phase8"

    # 1. Summary table schema
    summary_df = pd.read_csv(phase8_dir / "subtype_characterization_summary.csv")
    expected_summary_cols = {
        "cluster", "sample_count", "cluster_proportion", "known_ALL_count", "known_AML_count",
        "ALL_percentage", "AML_percentage", "mean_PC1", "mean_PC2", "mean_PC3",
        "total_tested_probes", "fdr_significant_biomarkers", "strong_candidate_count",
        "significant_pathways_total", "primary_biological_theme"
    }
    assert expected_summary_cols.issubset(summary_df.columns)

    # 2. Cross-cluster comparison schema
    cross_df = pd.read_csv(phase8_dir / "cross_cluster_comparison.csv")
    expected_cross_cols = {
        "comparison", "cluster_a", "cluster_b", "samples_a", "samples_b",
        "phenotype_contingency", "fisher_odds_ratio", "fisher_exact_p_value",
        "shared_significant_probes", "union_significant_probes",
        "biomarker_jaccard_similarity", "pca_centroid_euclidean_distance"
    }
    assert expected_cross_cols.issubset(cross_df.columns)

    # 3. Phase 9 service compatibility: ResearchData must load Phase 8 cleanly
    rd = ResearchData(ROOT)
    p8_data = rd.phase8_characterization()
    assert p8_data is not None
    assert "summary" in p8_data
    assert "comparison" in p8_data
    assert len(p8_data["summary"]) == 3
    assert len(p8_data["comparison"]) == 3

    # 4. Phase 9 optional missing contracts must remain unaffected
    assert rd.get("consensus").empty
    assert not rd.get("pathways").empty
    assert rd.get("importance").empty
