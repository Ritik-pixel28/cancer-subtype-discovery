import json
import pandas as pd
import streamlit as st
from app.components.layout import heading, missing
from app.services.research import fingerprint
from src.reporting.engine import (
    ReportOptions,
    SECTIONS,
    METHODS,
    LIMITATIONS,
    build_report,
)


@st.fragment
def figure_exports(figures):
    if not figures:
        st.info("This report has no figures for the selected sections.")
        return
    st.subheader("Scientific figure exports")
    name = st.selectbox("Figure", list(figures))
    selected = figures[name]
    st.image(
        selected["png"], caption=name + " · generated report figure", width="stretch"
    )
    a, b = st.columns(2)
    a.download_button(
        "Download figure · PNG",
        selected["png"],
        name.lower() + ".png",
        "image/png",
        on_click="ignore",
    )
    b.download_button(
        "Download figure · SVG",
        selected["svg"],
        name.lower() + ".svg",
        "image/svg+xml",
        on_click="ignore",
    )


def reports(d):
    heading(
        "11",
        "Make the evidence shareable.",
        "Generate a real PDF, a browser-readable HTML report, or a complete archive of selected tables, scientific figures and provenance.",
    )
    if missing(d, "clusters"):
        return
    with st.form("report_configuration"):
        a, b = st.columns(2)
        kind = a.selectbox(
            "Report type",
            ["Executive summary", "Full technical report", "Cluster-specific report"],
        )
        cluster = b.selectbox(
            "Cluster (used for cluster-specific reports)",
            sorted(d.get("clusters").cluster.unique()),
        )
        sections = st.multiselect("Include sections", SECTIONS, default=SECTIONS)
        st.caption(
            "Scientific limitations and report scope are always included. Missing Phase 7/8 sections are explicitly marked pending. Executive summaries are concise; full tables are in the ZIP archive."
        )
        generate = st.form_submit_button("Generate research report ↗", type="primary")
    current = fingerprint(d.root)
    if generate:
        options = ReportOptions(
            kind,
            int(cluster) if kind == "Cluster-specific report" else None,
            tuple(sections),
        )
        try:
            with st.spinner(
                "Validating results, drawing scientific figures and typesetting the report…"
            ):
                result = build_report(d, options)
            st.session_state["report_result"] = result
            st.session_state["report_version"] = current
            st.session_state["report_description"] = kind + (
                f" · Cluster {int(cluster)}"
                if options.cluster is not None
                else " · Full cohort"
            )
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Report could not be generated: {exc}")
    if (
        st.session_state.get("report_result")
        and st.session_state.get("report_version") == current
    ):
        result = st.session_state["report_result"]
        st.success("Prepared: " + st.session_state["report_description"])
        a, b, c = st.columns(3)
        a.download_button(
            "Download PDF",
            result["pdf"],
            "cancer_subtype_report.pdf",
            "application/pdf",
            type="primary",
            on_click="ignore",
        )
        b.download_button(
            "Download HTML",
            result["html"],
            "cancer_subtype_report.html",
            "text/html",
            on_click="ignore",
        )
        c.download_button(
            "Report + data + figures · ZIP",
            result["zip"],
            "cancer_subtype_research.zip",
            "application/zip",
            on_click="ignore",
        )
        st.download_button(
            "Provenance · JSON",
            result["metadata"],
            "research_provenance.json",
            "application/json",
            on_click="ignore",
        )
        figure_exports(result["figures"])
    st.subheader("Result availability")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Artifact": name.replace("_", " ").title(),
                    "Status": status,
                    "Rows": len(d.get(name)),
                }
                for name, status in d.status.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def methods(d):
    heading(
        "12",
        "Every result has a history.",
        "Inspect the methods, configuration, software environment and exact analytical artifacts behind this research workspace.",
    )
    stages = {
        "Data and label separation": "Golub leukemia gene-expression measurements and known ALL/AML labels are stored separately. The known labels are used only for external evaluation and interpretation.",
        "Median absolute deviation": "MAD ranks each probe by median absolute deviation from its median. The top configured number of probes is retained. High variance does not imply biological importance.",
        "Standardization and PCA": "Each selected probe is centered and divided by its population standard deviation. PCA reduces the selected representation to the configured number of components. Labels do not enter these steps.",
        "Clustering and consensus": "K-Means, Ward and GMM are compared. Repeated consensus subsampling assesses pairwise co-clustering, and the saved exploratory k=3 partition supplies the downstream labels.",
        "Validation": "ARI measures agreement with known classes; silhouette measures separation. Gap uses the 1-SE rule. PAC measures ambiguous pair frequencies. Bootstrap Jaccard intervals describe resampling stability.",
        "Candidate probe discovery": "Cluster-versus-rest Mann-Whitney U tests are adjusted with BH-FDR. Median differences and rank-biserial effect sizes describe magnitude and direction. GPL80 maps selected strong probes to gene symbols.",
        "Future interpretation": "Phase 7 will provide pathway enrichment with its background gene universe. Phase 8 will provide a surrogate classifier, evaluated fidelity and SHAP results. Neither establishes causal biology.",
    }
    for title, text in stages.items():
        with st.expander(title):
            st.write(text)
    with st.expander("Scientific limitations", expanded=True):
        st.write(LIMITATIONS)
    with st.expander("Molecular animation provenance"):
        st.write(
            "The homepage renders 486 DNA atoms from the published 1BNA B-DNA dodecamer. Atomic coordinates are preserved up to centering, rigid rotation and uniform scene scale. Nearby atoms are connected using a covalent-radius distance heuristic. Colors, lighting, repeated background copies and camera motion are artistic. This is a structural reference, not a cancer-specific gene or patient sequence."
        )
        st.markdown(
            "[RCSB PDB 1BNA](https://www.rcsb.org/structure/1BNA) · Drew et al. (1981), doi:10.1073/pnas.78.4.2179."
        )
    metadata = d.provenance()
    with st.expander("Actual pipeline configuration"):
        st.json(d.config)
    with st.expander("Software and code version"):
        st.write("Application base commit: " + metadata["application_commit"])
        st.write(
            "Working tree contains changes: " + str(metadata["working_tree_modified"])
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Package": metadata["software"].keys(),
                    "Version": metadata["software"].values(),
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Artifact checksums and status"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Artifact": k, "SHA-256": v}
                    for k, v in metadata["artifact_sha256"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.json(d.status)
        if d.errors:
            st.json(d.errors)
    st.download_button(
        "Download reproducibility record",
        json.dumps(metadata, indent=2, default=str),
        "reproducibility.json",
        "application/json",
    )
    st.caption(
        "Checksums identify exact files; they do not validate biological conclusions. The app reads saved results and never reruns clustering in response to navigation or filtering."
    )
