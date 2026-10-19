from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


def app(page):
    at = AppTest.from_file(str(APP), default_timeout=40)
    at.query_params["view"] = page
    return at.run()


@pytest.mark.parametrize(
    "page",
    [
        "Home",
        "Overview",
        "Dataset",
        "Preprocessing",
        "PCA",
        "Clusters",
        "Validation",
        "Biomarkers",
        "Pathways",
        "Characterization",
        "Explainability",
        "Compare",
        "Reports",
        "Methods",
    ],
)
def test_every_section_renders(page):
    at = app(page)
    assert not at.exception


def test_overview_counts():
    at = app("Overview")
    assert [m.value for m in at.metric] == [
        "72",
        "7,129",
        "2,000",
        "30",
        "3",
        "183",
        "170",
        "82.2%",
    ]


def test_literal_gene_filter_and_empty_state():
    at = app("Biomarkers")
    next(x for x in at.text_input if x.label == "Gene symbol or probe ID").set_value(
        "CBFB"
    ).run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 1
    next(x for x in at.text_input if x.label == "Gene symbol or probe ID").set_value(
        "NO_MATCH_GENE"
    ).run()
    assert not at.exception
    assert any("No results match" in i.value for i in at.info)


def test_3d_and_external_coloring():
    at = app("PCA")
    next(x for x in at.selectbox if x.label == "Projection").set_value("3D").run()
    assert not at.exception
    at.radio[0].set_value("cancer_type").run()
    assert not at.exception


def test_dataset_empty_probe_selection():
    at = app("Dataset")
    at.multiselect[0].set_value([]).run()
    assert not at.exception


def test_actual_pdf_generation():
    at = app("Reports")
    next(b for b in at.button if "Generate research report" in b.label).click().run(
        timeout=60
    )
    assert not at.exception
    assert at.session_state["report_result"]["pdf"].startswith(b"%PDF-")


def test_global_search_navigation():
    at = app("Overview")
    next(x for x in at.text_input if x.label == "Find a gene or probe").set_value("MPO")
    next(x for x in at.button if x.label == "Find").click().run()
    assert not at.exception
    assert at.query_params["view"] in ("Biomarkers", ["Biomarkers"])
    assert (
        next(x for x in at.text_input if x.label == "Gene symbol or probe ID").value
        == "MPO"
    )


def test_section_navigation_and_focus_persistence():
    at = app("Overview")
    next(x for x in at.selectbox if x.label == "Focus cluster").set_value(0).run()
    next(x for x in at.selectbox if x.label == "Research sections").set_value(
        "PCA"
    ).run()
    assert not at.exception
    assert next(x for x in at.selectbox if x.label == "Focus cluster").value == 0
    assert at.query_params["view"] in ("PCA", ["PCA"])


@pytest.mark.parametrize(
    "page,charts", [("Overview", 4), ("Dataset", 6), ("Clusters", 6), ("Biomarkers", 4)]
)
def test_figures_are_prepared_for_all_tabs(page, charts):
    at = app(page)
    assert not at.exception
    assert len(at.get("plotly_chart")) == charts
    assert len(at.get("download_button")) >= charts // 2


def test_report_figure_preview_changes_with_selection():
    at = app("Reports")
    next(b for b in at.button if "Generate research report" in b.label).click().run(
        timeout=60
    )
    assert not at.exception
    assert len((at.get("image") or at.get("imgs"))) == 1
    first = (at.get("image") or at.get("imgs"))[0].proto.imgs[0].url
    next(x for x in at.selectbox if x.label == "Figure").set_value("Validation").run()
    assert not at.exception
    image = (at.get("image") or at.get("imgs"))[0].proto.imgs[0]
    assert image.caption.startswith("Validation")
    assert image.url != first
    assert len((at.get("image") or at.get("imgs"))) == 1


def test_native_phase7_page_and_empty_cluster():
    at = app("Pathways")
    assert not at.exception
    assert not any("awaiting" in x.value for x in at.warning)
    assert len(at.dataframe[0].value) == 254
    next(x for x in at.selectbox if x.label == "Focus cluster").set_value(1).run()
    assert not at.exception
    assert any("results are loaded" in x.value for x in at.info)


def test_phase8_page_filters_and_figures():
    at = app("Characterization")
    assert not at.exception
    assert [m.value for m in at.metric] == ["3", "72", "183", "254"]
    next(x for x in at.selectbox if x.label == "Focus cluster").set_value(2).run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["1", "23", "44", "0"]
    assert len(at.dataframe[0].value) == 1
    assert len(at.dataframe[1].value) == 2
    next(x for x in at.selectbox if x.label == "Subtype characterization figure").set_value("PCA distribution").run()
    assert not at.exception


def test_explainability_links_to_characterization():
    at = app("Explainability")
    next(b for b in at.button if b.label == "View Subtype Characterization ↗").click().run()
    assert not at.exception
    assert at.metric[0].value == "3"
