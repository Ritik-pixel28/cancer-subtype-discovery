import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.components.layout import heading, missing, focus, download, go as navigate
from app.charts import show, pca_scatter, heatmap, COLORS, MAP


def overview(d):
    heading(
        "01",
        "Different samples. Shared questions.",
        "An interactive platform for exploratory molecular subgroup analysis. Follow the evidence from the expression matrix to candidate biomarkers.",
    )
    if missing(d, "clusters"):
        return
    s = d.summary
    b = d.get("biomarkers")
    c = d.get("clusters")
    for cols, items in [
        (
            st.columns(4),
            [
                ("Patient samples", len(c)),
                ("Expression probes", s.get("probes", "—")),
                ("Selected probes", s.get("selected_probes", "—")),
                ("PCA components", s.get("pca_components", "—")),
            ],
        ),
        (
            st.columns(4),
            [
                ("Exploratory groups", c.cluster.nunique()),
                ("Strong candidate entries", len(b)),
                (
                    "Annotated entries",
                    b.gene_symbol.notna().sum() if not b.empty else "—",
                ),
                (
                    "PCA variance retained",
                    f"{d.get('variance').explained_variance_ratio.sum():.1%}"
                    if not d.get("variance").empty
                    else "—",
                ),
            ],
        ),
    ]:
        for col, (label, value) in zip(cols, items):
            col.metric(
                label, f"{value:,}" if isinstance(value, (int, np.integer)) else value
            )
    a, z = st.columns(2)
    with a:
        st.subheader("The molecular landscape")
        if not missing(d, "pca"):
            show(pca_scatter(d.get("pca")), "overview_pca")
    with z:
        st.subheader("Three exploratory groups")
        counts = (
            c.cluster.value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="samples")
        )
        show(
            px.bar(
                counts,
                x="cluster",
                y="samples",
                color=counts.cluster.astype(str),
                color_discrete_map=MAP,
                text="samples",
            ),
            "overview_counts",
        )
    st.subheader("Follow the research")
    stages = [
        ("Dataset", "01 / Dataset", "Public leukemia expression measurements."),
        (
            "Preprocessing",
            "02 / Preparation",
            "Select variable probes and standardize.",
        ),
        ("PCA", "03 / Representation", "Inspect variance in fewer dimensions."),
        ("Clusters", "04 / Discovery", "Explore the saved consensus partition."),
        ("Validation", "05 / Evidence", "Assess separation, stability and agreement."),
        ("Biomarkers", "06 / Signals", "Investigate cluster-associated probes."),
    ]
    for offset in range(0, len(stages), 3):
        cols = st.columns(3)
        for col, (route, title, text) in zip(cols, stages[offset : offset + 3]):
            with col:
                with st.container(border=True):
                    st.markdown("**" + title + "**")
                    st.caption(text)
                    if st.button(
                        "Explore " + title.split("/ ")[1] + " ↗",
                        key="stage_" + route,
                        use_container_width=True,
                    ):
                        navigate(route)
    st.info(
        "The selected k=3 partition is exploratory. Mixed validation evidence, a small cohort and same-cohort candidate discovery require independent validation. Phase 7 and Phase 8 will add interpretation when validated outputs become available."
    )
    status = pd.DataFrame(
        {
            "Stage": ["Pathway enrichment", "Explainability"],
            "Status": [d.status.get("pathways"), d.status.get("importance")],
        }
    )
    st.dataframe(status, hide_index=True, width="stretch")


def dataset(d):
    heading(
        "02",
        "Meet the expression landscape.",
        "Inspect the public Golub cohort, verify data quality, and explore individual probes without losing their identities.",
    )
    if missing(d, "expression"):
        return
    expr = d.get("expression")
    s = d.summary
    cols = st.columns(4)
    for col, (label, val) in zip(
        cols,
        [
            ("Samples", len(expr)),
            ("Measured probes", len(expr.columns)),
            ("Known ALL", s.get("known_labels", {}).get("ALL", "—")),
            ("Known AML", s.get("known_labels", {}).get("AML", "—")),
        ],
    ):
        col.metric(label, val)
    assignment = d.get("clusters")
    tab1, tab2, tab3 = st.tabs(
        ["Expression explorer", "Quality & distributions", "Reference labels"],
    )
    with tab1:
        a, b = st.columns([2, 1])
        choices = list(expr.columns)
        default = d.get("biomarkers").gene.drop_duplicates().head(3).tolist()
        probes = a.multiselect(
            "Select probes (up to 12)",
            choices,
            default=default,
            max_selections=12,
            format_func=d.probe_label,
        )
        query = b.text_input("Sample ID contains", placeholder="e.g. 39")
        rows = expr[expr.index.str.contains(query, case=False, regex=False)]
        allowed = assignment[assignment.cluster.isin(focus(d))].sample_id
        rows = rows.loc[rows.index.isin(allowed), probes]
        if not probes or rows.empty:
            st.info("Choose a probe and matching samples to inspect expression.")
        else:
            st.dataframe(rows.reset_index(), hide_index=True, width="stretch")
            download(rows.reset_index(), "expression_selection")
            st.subheader("Descriptive statistics")
            st.dataframe(rows.describe().T, width="stretch")
    with tab2:
        numeric = expr.to_numpy()
        quality = {
            "Missing measurements": int(expr.isna().sum().sum()),
            "Duplicate sample IDs": int(expr.index.duplicated().sum()),
            "Constant probes": int((expr.nunique() <= 1).sum()),
            "Negative measurements": int((expr < 0).sum().sum()),
            "Nonnumeric columns": len(expr.columns)
            - len(expr.select_dtypes(include="number").columns),
            "Aligned to assignments": len(expr) == len(assignment)
            and set(expr.index) == set(assignment.sample_id),
        }
        st.dataframe(
            pd.DataFrame(
                {
                    "Check": quality.keys(),
                    "Observed result": [str(v) for v in quality.values()],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        counts, edges = np.histogram(numeric.ravel(), bins=80)
        show(
            go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts)).update_layout(
                xaxis_title="Original expression value",
                yaxis_title="Measurement count",
            ),
            "expression_histogram",
        )
        ids = st.multiselect(
            "Compare sample distributions",
            expr.index.tolist(),
            default=expr.index[:4].tolist(),
            max_selections=8,
        )
        if ids:
            long = expr.loc[ids].T.melt(var_name="sample_id", value_name="expression")
            show(
                px.box(long, x="sample_id", y="expression", points=False),
                "sample_distribution",
            )
        st.caption(
            "An unusual expression distribution does not establish that a sample is invalid. Negative input values are why a naive log2(x+1) transform is not applied."
        )
    with tab3:
        pca = d.get("pca")
        if "cancer_type" not in pca:
            st.info("Reference labels are unavailable.")
            return
        labels = (
            pca.cancer_type.value_counts()
            .rename_axis("known_label")
            .reset_index(name="samples")
        )
        show(
            px.bar(labels, x="known_label", y="samples", text="samples"),
            "reference_labels",
        )
        st.caption(
            "ALL and AML are known dataset labels. They are kept separate from the discovered clusters and were not used for feature selection, PCA or clustering."
        )


def preprocessing(d):
    heading(
        "03",
        "A considered preparation.",
        "Understand feature variability and the transformation from original expression measurements to a comparable numerical representation.",
    )
    if missing(d, "mad") or missing(d, "expression"):
        return
    mad = d.get("mad")
    expr = d.get("expression")
    selected = mad[mad.selected.astype(bool)]
    a, b, c = st.columns(3)
    a.metric("Input probes", len(mad))
    b.metric("Retained by MAD", len(selected))
    c.metric("Excluded probes", len(mad) - len(selected))
    fig = px.line(mad, x="rank", y="mad")
    fig.add_vline(
        x=len(selected),
        line_dash="dash",
        line_color="#b77550",
        annotation_text=f"Top {len(selected):,} retained",
    )
    show(fig, "mad_ranking")
    st.caption(
        "Higher variability does not imply biological importance or cancer specificity. MAD measures dispersion around each probe’s median."
    )
    include = st.selectbox("Show ranked probes", ["All", "Selected", "Excluded"])
    ranked = mad if include == "All" else mad[mad.selected == (include == "Selected")]
    st.dataframe(ranked, hide_index=True, width="stretch")
    download(ranked, "mad_selection")
    st.subheader("Before and after standardization")
    probe = st.selectbox(
        "Selected probe", selected.probe_id.tolist(), format_func=d.probe_label
    )
    original = expr[probe]
    scaled = (original - original.mean()) / original.std(ddof=0)
    frame = pd.DataFrame(
        {
            "sample_id": expr.index,
            "Original": original.values,
            "Standardized": scaled.values,
        }
    )
    a, b = st.columns(2)
    for column, col in [("Original", a), ("Standardized", b)]:
        with col:
            st.metric(column + " mean", f"{frame[column].mean():.4f}")
            st.caption(
                f"Population standard deviation: {frame[column].std(ddof=0):.4f}"
            )
            show(px.histogram(frame, x=column, nbins=20), "scaling_" + column, 320)
    st.caption(
        "The existing pipeline uses gene-wise centering and population standard deviation (ddof=0). No cancer labels enter this transformation."
    )


def pca(d):
    heading(
        "04",
        "A new perspective on variation.",
        "Rotate, compare and inspect a reduced representation of the same samples. Projection changes the view, never the saved assignments.",
    )
    if missing(d, "pca"):
        return
    frame = d.get("pca")
    frame = frame[frame.cluster.isin(focus(d))]
    a, b, c = st.columns(3)
    dimension = a.selectbox("Projection", ["2D", "3D"])
    axes = [x for x in frame if x.startswith("PC")]
    x = b.selectbox("Horizontal component", axes, index=0)
    y = c.selectbox("Vertical component", [a for a in axes if a != x], index=0)
    components = [x, y]
    if dimension == "3D":
        components.append(
            st.selectbox("Depth component", [a for a in axes if a not in components])
        )
    color = st.radio(
        "Color samples by",
        ["cluster"] + (["cancer_type"] if "cancer_type" in frame else []),
        horizontal=True,
        format_func=lambda x: "Consensus cluster"
        if x == "cluster"
        else "Known ALL / AML label",
    )
    show(pca_scatter(frame, components, color), "pca_explorer", 520)
    download(frame, "pca_selection")
    st.caption(
        "Distances in two or three plotted components do not reproduce every distance in the 30-component clustering space."
    )
    variance = d.get("variance")
    if not variance.empty:
        a, b = st.columns(2)
        with a:
            show(
                px.bar(
                    variance,
                    x="component",
                    y="explained_variance_ratio",
                    labels={"explained_variance_ratio": "Explained variance ratio"},
                ),
                "scree",
            )
        with b:
            show(
                px.line(
                    variance.assign(
                        cumulative=variance.explained_variance_ratio.cumsum()
                    ),
                    x="component",
                    y="cumulative",
                    markers=True,
                ),
                "cumulative_variance",
            )
        st.info(
            f"{len(variance)} configured components retain {variance.explained_variance_ratio.sum():.1%} of the selected, standardized expression variance. This describes the configured choice; it does not establish an optimal component count."
        )


def clusters(d):
    heading(
        "05",
        "Patterns, not predetermined labels.",
        "Investigate the saved consensus groups, compare algorithm evidence, and inspect sample-level expression profiles.",
    )
    if missing(d, "clusters"):
        return
    c = d.get("clusters")
    selected = focus(d)
    subset = c[c.cluster.isin(selected)]
    tab1, tab2, tab3 = st.tabs(
        ["Samples & neighbours", "Expression heatmap", "Algorithm comparison"],
    )
    with tab1:
        counts = (
            subset.cluster.value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="samples")
        )
        counts["percentage"] = counts.samples / len(c) * 100
        show(
            px.bar(
                counts,
                x="cluster",
                y="samples",
                text="samples",
                color=counts.cluster.astype(str),
                color_discrete_map=MAP,
            ),
            "cluster_distribution",
            320,
        )
        st.dataframe(counts, hide_index=True, width="stretch")
        query = st.text_input("Search sample ID")
        filtered = subset[subset.sample_id.str.contains(query, regex=False, case=False)]
        if filtered.empty:
            st.info("No sample matches the current cluster and ID filters.")
            return
        sample = st.selectbox("Inspect sample", filtered.sample_id.tolist())
        row = d.get("pca")
        if not row.empty:
            chosen = row[row.sample_id == sample]
            st.dataframe(
                chosen[
                    ["sample_id", "cluster"]
                    + (["cancer_type"] if "cancer_type" in row else [])
                ],
                hide_index=True,
                width="stretch",
            )
            pcs = [x for x in row if x.startswith("PC")]
            a = row.set_index("sample_id")[pcs]
            dist = (
                np.sqrt(((a - a.loc[sample]) ** 2).sum(axis=1))
                .sort_values()
                .drop(sample)
            )
            neighbours = (
                dist.head(5)
                .rename("euclidean_distance_30pc")
                .rename_axis("sample_id")
                .reset_index()
                .merge(c, on="sample_id")
            )
            st.subheader("Nearest samples in the clustering representation")
            st.dataframe(neighbours, hide_index=True, width="stretch")
            st.caption(
                "Euclidean distance across all saved PCA components. Nearest does not imply the same clinical class."
            )
        st.dataframe(filtered, hide_index=True, width="stretch")
        download(filtered, "cluster_assignments")
    with tab2:
        if missing(d, "expression") or missing(d, "biomarkers"):
            return
        a, b = st.columns(2)
        cluster = a.selectbox("Candidate source cluster", sorted(c.cluster.unique()))
        top = b.selectbox("Top candidate probes", [10, 20, 50])
        marks = d.get("biomarkers")
        probes = (
            marks[marks.cluster == cluster]
            .sort_values(["fdr", "gene"])
            .gene.drop_duplicates()
            .head(top)
            .tolist()
        )
        show(
            heatmap(d, probes, selected),
            "cluster_expression_heatmap",
            max(420, len(probes) * 15),
        )
        st.caption(
            "Probe-wise z-scores across the full cohort; sample columns are ordered by saved cluster. C0/C1/C2 labels annotate the columns."
        )
    with tab3:
        v = d.get("validation")
        boot = d.get("bootstrap")
        if v.empty:
            st.info("Algorithm validation results unavailable.")
            return
        k = st.selectbox(
            "Number of clusters (k)",
            sorted(v.k.unique()),
            index=list(sorted(v.k.unique())).index(3),
        )
        result = (
            v[v.k == k].merge(boot, on=["method", "k"], how="left")
            if not boot.empty
            else v[v.k == k]
        )
        metric = st.selectbox(
            "Comparison metric",
            ["ari", "silhouette"]
            + (["stability_mean"] if "stability_mean" in result else []),
        )
        show(px.bar(result, x="method", y=metric), "algorithm_comparison")
        st.dataframe(result, hide_index=True, width="stretch")
        download(result, "algorithm_comparison")
        st.caption(
            "Bootstrap results exist only for their saved k values. Missing stability cells are not zero. ARI is agreement, not diagnostic accuracy."
        )


def validation(d):
    heading(
        "06",
        "Question every apparent pattern.",
        "Compare internal separation, external agreement, resampling stability and consensus ambiguity. No single metric is a verdict.",
    )
    v = d.get("validation")
    if not v.empty:
        methods = st.multiselect(
            "Methods", v.method.unique(), default=v.method.unique()
        )
        f = v[v.method.isin(methods)]
        a, b = st.columns(2)
        with a:
            show(
                px.line(f, x="k", y="silhouette", color="method", markers=True),
                "silhouette",
            )
        with b:
            show(px.line(f, x="k", y="ari", color="method", markers=True), "ari")
        st.caption(
            "Silhouette describes separation in PCA space. ARI compares with external ALL/AML labels after clustering. These curves represent K-Means, Ward and GMM, not the consensus partition."
        )
    a, b = st.columns(2)
    gap = d.get("gap")
    pac = d.get("pac")
    with a:
        st.subheader("Gap statistic · the 1-SE rule")
        if not gap.empty:
            ordered = gap.sort_values("k")
            chosen = int(ordered.iloc[-1].k)
            for i in range(len(ordered) - 1):
                if (
                    ordered.iloc[i].gap
                    >= ordered.iloc[i + 1].gap - ordered.iloc[i + 1].gap_std
                ):
                    chosen = int(ordered.iloc[i].k)
                    break
            fig = px.line(ordered, x="k", y="gap", error_y="gap_std", markers=True)
            fig.add_vline(
                x=chosen, line_dash="dash", annotation_text=f"1-SE: k={chosen}"
            )
            show(fig, "gap")
            st.caption(
                "Smallest k satisfying Gap(k) ≥ Gap(k+1) − uncertainty(k+1); not simply the largest gap."
            )
    with b:
        st.subheader("PAC · ambiguity across pairs")
        if not pac.empty:
            chosen = int(pac.loc[pac.pac.idxmin(), "k"])
            fig = px.line(pac, x="k", y="pac", markers=True)
            fig.add_vline(
                x=chosen, line_dash="dash", annotation_text=f"Minimum: k={chosen}"
            )
            show(fig, "pac")
            st.caption(
                "Lower PAC means fewer ambiguous sample pairs in the 0.1–0.9 interval; diagonal pairs are excluded."
            )
    boot = d.get("bootstrap")
    if not boot.empty:
        st.subheader("Resampling stability")
        fig = px.scatter(
            boot.assign(
                upper=boot.ci_upper - boot.stability_mean,
                lower=boot.stability_mean - boot.ci_lower,
            ),
            x="stability_mean",
            y="method",
            error_x="upper",
            error_x_minus="lower",
        )
        fig.update_xaxes(range=[0, 1])
        show(fig, "bootstrap", 300)
        st.caption(
            "Best-match Jaccard similarity with 95% empirical resampling intervals. These are not clinical confidence intervals."
        )
        st.dataframe(boot, hide_index=True, width="stretch")
    st.subheader("Pairwise consensus matrix")
    matrix = d.get("consensus")
    if matrix.empty:
        st.info(
            "The original 72 × 72 consensus matrix was not saved in the supplied outputs. Final cluster labels cannot reconstruct it. Export consensus_k3_matrix.csv from Phase 5 to enable this view."
        )
    else:
        ids = d.get("clusters").sort_values(["cluster", "sample_id"]).sample_id.tolist()
        show(
            go.Figure(
                go.Heatmap(
                    z=matrix.loc[ids, ids],
                    x=ids,
                    y=ids,
                    zmin=0,
                    zmax=1,
                    colorscale="Teal",
                )
            ),
            "consensus_matrix",
            600,
        )
    st.warning(
        "The cohort is small, external agreement is modest, and stability intervals vary by algorithm. The exploratory three-group solution needs independent biological validation."
    )
