from pathlib import Path
import pytest
from scripts import run_phase7_enrichment as phase7


def test_missing_libraries_do_not_modify_saved_results(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    saved = output / "enrichment_summary.csv"
    saved.write_text("existing scientific results\n")
    monkeypatch.setattr(phase7, "OUTPUT_DIR", output)
    monkeypatch.setattr(phase7, "GENE_SET_DIR", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="setup_reference_data"):
        phase7.main()
    assert saved.read_text() == "existing scientific results\n"
    assert list(output.iterdir()) == [saved]


def test_ora_includes_zero_overlap_terms_in_testing_family():
    result = phase7.run_ora({"A"}, {"A", "B", "C", "D"}, {"hit": {"A", "B"}, "no_hit": {"C", "D"}}, 1, 10)
    assert len(result) == 2
    assert result.set_index("Term").loc["no_hit", "P-value"] == 1
    assert result.set_index("Term").loc["hit", "P-value"] == pytest.approx(0.5)
