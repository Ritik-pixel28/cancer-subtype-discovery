from pathlib import Path
import base64
import html
import json
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PAGES = {
    "Overview": "Research overview",
    "Dataset": "Dataset explorer",
    "Preprocessing": "Preprocessing",
    "PCA": "PCA analysis",
    "Clusters": "Cluster discovery",
    "Validation": "Cluster validation",
    "Biomarkers": "Biomarker explorer",
    "Pathways": "Pathway enrichment",
    "Explainability": "Explainability",
    "Compare": "Comparative analysis",
    "Reports": "Report generator",
    "Methods": "Research methods",
}


def style():
    font = base64.b64encode(
        (ROOT / "app/assets/BarlowCondensed-Medium.ttf").read_bytes()
    ).decode()
    st.html(
        "<style>"
        + (ROOT / "app/assets/style.css").read_text().replace("__FONT__", font)
        + "</style>"
    )


def select_page(page):
    st.query_params["view"] = page
    st.session_state["_pending_page"] = page


def go(page):
    select_page(page)
    st.rerun()


def menu_changed():
    page = st.session_state.get("page_menu")
    if page in PAGES:
        select_page(page)


def search_gene():
    query = st.session_state.get("global_gene", "")
    st.session_state["gene_query"] = query
    st.session_state["biomarker_query"] = query
    select_page("Biomarkers")


def shell(page, data):
    pending = st.session_state.pop("_pending_page", None)
    if pending is not None or "page_menu" not in st.session_state:
        st.session_state["page_menu"] = pending or page
    st.html(
        '<nav class="workspace-nav"><a class="workspace-brand" href="?view=Home" target="_self">MOLECULAR DISCOVERY<small>CANCER SUBTYPE EXPLORER / G87</small></a><div class="workspace-links"><a href="?view=Home" target="_self">The molecular world ↗</a><a href="?view=Methods" target="_self">Our approach</a><a href="?view=Reports" target="_self">Generate report ↗</a></div></nav>'
    )
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        menu = st.selectbox(
            "Research sections",
            list(PAGES),
            index=None,
            format_func=lambda x: PAGES[x],
            key="page_menu",
            on_change=menu_changed,
        )
    with c2:
        ids = (
            sorted(data.get("clusters").cluster.unique().astype(int).tolist())
            if not data.get("clusters").empty
            else []
        )
        st.selectbox("Focus cluster", ["All clusters", *ids], key="focus_cluster")
    with c3:
        with st.form("global_search", border=False):
            a, b = st.columns([3, 1.2], vertical_alignment="bottom")
            query = a.text_input(
                "Find a gene or probe",
                placeholder="Gene symbol or probe ID",
                key="global_gene",
            )
            b.form_submit_button("Find", use_container_width=True, on_click=search_gene)


def heading(number, title, description):
    st.html(
        f'<div class="page-heading"><div class="eyebrow"><span>RESEARCH EXPLORER / {html.escape(str(number))}</span><span>GOLUB LEUKEMIA · G87</span></div><h1>{html.escape(title)}</h1><div class="page-deck">{html.escape(description)}</div></div>'
    )


def missing(data, name, message=None):
    if not data.get(name).empty:
        return False
    st.warning(
        message or f"{name.replace('_', ' ').title()} results are not available yet."
    )
    if name in data.errors:
        with st.expander("Artifact validation details"):
            st.code(data.errors[name])
    return True


def focus(data):
    ids = (
        sorted(data.get("clusters").cluster.unique())
        if not data.get("clusters").empty
        else []
    )
    selected = st.session_state.get("focus_cluster", "All clusters")
    return ids if selected == "All clusters" else [selected]


def download(frame, name):
    st.download_button(
        "Export table · CSV",
        frame.to_csv(index=False).encode(),
        name + ".csv",
        "text/csv",
        key="csv_" + name,
        on_click="ignore",
    )


def footer():
    st.html(
        '<div class="research-foot"><span>CANCER SUBTYPE DISCOVERY / GROUP G87</span><span>Exploratory research · Independent validation required · Not for clinical diagnosis</span></div>'
    )


def landing(data):
    from app.components.landing import render

    render(data)
