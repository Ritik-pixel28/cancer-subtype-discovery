"""Cancer Subtype Discovery Explorer — Group G87."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streamlit as st
from app.services.research import ResearchData, fingerprint
from app.components.layout import style, shell, landing, footer, PAGES

st.set_page_config(
    page_title="Molecular Discovery · G87",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
style()


@st.cache_data(show_spinner="Loading and validating research artifacts…")
def load_research(version):
    return ResearchData(ROOT)


data = load_research(fingerprint(ROOT))
page = st.query_params.get("view", "Home")
if page not in [*PAGES, "Home"]:
    page = "Home"
if page == "Home":
    landing(data)
else:
    shell(page, data)
    from importlib import import_module

    routes = {
        "Overview": ("research", "overview"),
        "Dataset": ("research", "dataset"),
        "Preprocessing": ("research", "preprocessing"),
        "PCA": ("research", "pca"),
        "Clusters": ("research", "clusters"),
        "Validation": ("research", "validation"),
        "Biomarkers": ("discovery", "biomarkers"),
        "Compare": ("discovery", "compare"),
        "Pathways": ("discovery", "pathways"),
        "Characterization": ("discovery", "characterization"),
        "Explainability": ("discovery", "explainability"),
        "Reports": ("reporting", "reports"),
        "Methods": ("reporting", "methods"),
    }
    module, view = routes[page]
    getattr(import_module("app.views." + module), view)(data)
    footer()
