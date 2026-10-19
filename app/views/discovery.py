import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.components.layout import heading, missing, focus, download, go as navigate
from app.charts import show, heatmap, pca_scatter, MAP, COLORS


def biomarkers(d):
    heading(
        "07",
        "Look closer at the signals.",
        "Search cluster-associated probes, inspect their statistical evidence, and compare actual expression distributions.",
    )
    if missing(d, "all_biomarkers"):
        return
    frame = d.get("all_biomarkers")
    strong = d.get("biomarkers")
    cols = st.columns(3)
    for i, c in enumerate(sorted(frame.cluster.unique())):
        marks = strong[strong.cluster == c] if not strong.empty else strong
        cols[i % 3].metric(f"Cluster {int(c)} · strong candidate entries", len(marks))
    a, b, c = st.columns([2, 1, 1])
    query = a.text_input(
        "Gene symbol or probe ID",
        value=st.session_state.get("gene_query", ""),
        key="biomarker_query",
    )
    source = b.selectbox("Result set", ["Strong candidates", "All tested probes"])
    annotation = c.selectbox("Annotation", ["All", "Mapped", "Unmapped"])
    a, b, c = st.columns(3)
    threshold = a.number_input(
        "Maximum BH-FDR", min_value=0.0, max_value=1.0, value=0.05, format="%.4f"
    )
    effect = b.slider("Minimum absolute rank-biserial effect", 0.0, 1.0, 0.8, step=0.05)
    direction = c.selectbox("Expression direction", ["All", "higher", "lower"])
    chosen = focus(d)
    f = frame[
        frame.cluster.isin(chosen)
        & (frame.fdr <= threshold)
        & (frame.rank_biserial.abs() >= effect)
    ].copy()
    if source == "Strong candidates":
        keys = (
            strong[["gene", "cluster"]]
            if not strong.empty
            else pd.DataFrame(columns=["gene", "cluster"])
        )
        f = f.merge(keys, on=["gene", "cluster"], how="inner", validate="one_to_one")
    if query:
        mask = f.gene.str.contains(
            query, case=False, regex=False
        ) | f.gene_symbol.fillna("").str.contains(query, case=False, regex=False)
        f = f[mask]
    if annotation != "All":
        f = f[f.gene_symbol.notna() if annotation == "Mapped" else f.gene_symbol.isna()]
    if direction != "All":
        f = f[f.direction == direction]
    st.caption(
        f"{len(f):,} matching probe–cluster entries. Annotation is loaded from the Phase 6 strong-candidate mapping; absent symbols do not establish that a probe is unmappable."
    )
    columns = [
        "gene",
        "gene_symbol",
        "cluster",
        "p_value",
        "fdr",
        "rank_biserial",
        "median_difference",
        "direction",
    ]
    tabs = st.tabs(
        [
            "Candidate table",
            "Significance & effect",
            "Expression profiles",
            "Probe & gene overlap",
        ],
    )
    with tabs[0]:
        if f.empty:
            st.info("No results match the selected filters.")
        st.dataframe(
            f[columns].sort_values(["cluster", "fdr"]),
            hide_index=True,
            width="stretch",
        )
        download(f, "candidate_selection")
    with tabs[1]:
        cluster = st.selectbox(
            "Volcano-style plot · cluster", sorted(frame.cluster.unique())
        )
        axis = st.selectbox("Effect measure", ["median_difference", "rank_biserial"])
        minimum = st.number_input(
            "Minimum absolute median difference", min_value=0.0, value=0.0
        )
        plot = frame[frame.cluster == cluster].copy()
        plot["significance"] = -np.log10(plot.fdr.clip(lower=np.finfo(float).tiny))
        plot["evidence"] = np.where(
            (plot.fdr <= threshold)
            & (plot.rank_biserial.abs() >= effect)
            & (plot.median_difference.abs() >= minimum),
            "Meets thresholds",
            "Other tested probes",
        )
        fig = px.scatter(
            plot,
            x=axis,
            y="significance",
            color="evidence",
            hover_name="gene",
            hover_data=["gene_symbol", "fdr", "direction"],
            color_discrete_map={
                "Meets thresholds": "#a86644",
                "Other tested probes": "#8babab",
            },
            labels={
                "significance": "−log₁₀(BH-FDR)",
                "median_difference": "Median expression difference",
                "rank_biserial": "Rank-biserial effect size",
            },
        )
        if threshold > 0:
            fig.add_hline(y=-np.log10(threshold), line_dash="dot")
        show(fig, "biomarker_volcano")
        st.caption(
            "All tested probes for this cluster are plotted. This is not a log-fold-change plot; raw expression includes negative values."
        )
    with tabs[2]:
        if missing(d, "expression"):
            return
        probes = f.gene.drop_duplicates().tolist()
        if not probes:
            st.info(
                "Choose filters that include at least one probe to inspect expression."
            )
        else:
            probe = st.selectbox("Expression probe", probes, format_func=d.probe_label)
            long = d.expression_long([probe])
            long["cluster"] = long.cluster.astype(int).astype(str)
            show(
                px.box(
                    long,
                    x="cluster",
                    y="expression",
                    points="all",
                    color="cluster",
                    color_discrete_map=MAP,
                    hover_data=["sample_id"],
                    labels={"expression": "Original expression measurement"},
                ),
                "probe_boxplot",
            )
            st.dataframe(
                frame[frame.gene == probe][columns],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Annotation source: the existing GPL80 mapping from Phase 6. Select a specific probe when several probes map to the same gene; no silent averaging is performed."
            )
    with tabs[3]:
        level = st.radio("Overlap unit", ["Probe ID", "Gene symbol"], horizontal=True)
        key = "gene" if level == "Probe ID" else "gene_symbol"
        mapped = strong.dropna(subset=[key])
        overlap = (
            mapped.groupby(key)
            .agg(
                clusters=(
                    "cluster",
                    lambda x: ", ".join(str(int(c)) for c in sorted(set(x))),
                ),
                cluster_count=("cluster", "nunique"),
                probe_count=("gene", "nunique"),
            )
            .reset_index()
        )
        st.dataframe(
            overlap.sort_values("cluster_count", ascending=False),
            hide_index=True,
            width="stretch",
        )
        download(overlap, "candidate_overlap_" + key)
        st.caption(
            "Gene-level overlap deduplicates symbols and excludes unmapped entries; probe-level overlap preserves the measurement identifiers."
        )


def compare(d):
    heading(
        "10",
        "Compare, side by side.",
        "Bring cluster summaries and probe expression into the same workspace. Save the selection as a portable JSON configuration.",
    )
    if missing(d, "clusters") or missing(d, "biomarkers"):
        return
    ids = sorted(d.get("clusters").cluster.unique())
    a, b = st.columns(2)
    ca = a.selectbox("Cluster A", ids, index=0)
    cb = b.selectbox("Cluster B", ids, index=min(2, len(ids) - 1))
    marks = d.get("biomarkers")
    rows = []
    for c in [ca, cb]:
        subset = marks[marks.cluster == c]
        rows.append(
            {
                "Cluster": int(c),
                "Patients": int((d.get("clusters").cluster == c).sum()),
                "Strong candidate entries": len(subset),
                "Annotated entries": int(subset.gene_symbol.notna().sum()),
                "Unique mapped symbols": subset.gene_symbol.nunique(),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    sa = set(marks[marks.cluster == ca].gene_symbol.dropna())
    sb = set(marks[marks.cluster == cb].gene_symbol.dropna())
    st.write(
        f"**{len(sa & sb)} shared mapped gene symbols** · {len(sa - sb)} unique to A · {len(sb - sa)} unique to B"
    )
    st.dataframe(
        pd.DataFrame({"Shared gene symbol": sorted(sa & sb)}),
        hide_index=True,
        width="stretch",
    )
    p = d.get("pca")
    if not p.empty:
        show(pca_scatter(p[p.cluster.isin([ca, cb])]), "comparison_pca")
    selected = []
    if not missing(d, "expression"):
        choices = marks.gene.drop_duplicates().tolist()
        selected = st.multiselect(
            "Compare probes (select two)",
            choices,
            default=choices[:2],
            max_selections=2,
            format_func=d.probe_label,
        )
        if selected:
            long = d.expression_long(selected, clusters=[ca, cb])
            long["cluster"] = long.cluster.astype(int).astype(str)
            show(
                px.box(
                    long,
                    x="cluster",
                    y="expression",
                    color="cluster",
                    facet_col="probe_id",
                    points="all",
                    color_discrete_map=MAP,
                ),
                "comparison_expression",
            )
        if len(selected) == 2:
            expr = d.get("expression")
            assignment = d.get("clusters")
            xy = expr[selected].reset_index().merge(assignment, on="sample_id")
            xy = xy[xy.cluster.isin([ca, cb])]
            xy["cluster"] = xy.cluster.astype(int).astype(str)
            show(
                px.scatter(
                    xy,
                    x=selected[0],
                    y=selected[1],
                    color="cluster",
                    hover_name="sample_id",
                    color_discrete_map=MAP,
                ),
                "probe_correlation",
            )
            r = xy[selected].corr(method="spearman").iloc[0, 1]
            st.caption(
                f"Spearman correlation across selected samples: {r:.3f}. Exploratory association does not imply regulation or causation."
            )
    for name, field in [("pathways", "pathway_name"), ("importance", "feature_id")]:
        frame = d.get(name)
        if not frame.empty:
            with st.expander("Compare " + name):
                st.dataframe(
                    frame[frame.cluster.isin([ca, cb])],
                    hide_index=True,
                    width="stretch",
                )
    p8 = d.phase8_characterization()
    if p8:
        with st.expander("Subtype Characterization & Statistical Comparisons"):
            comp = p8.get("comparison", pd.DataFrame())
            if not comp.empty:
                pair_comp = comp[
                    ((comp.cluster_a == ca) & (comp.cluster_b == cb))
                    | ((comp.cluster_a == cb) & (comp.cluster_b == ca))
                ]
                if not pair_comp.empty:
                    st.dataframe(pair_comp, hide_index=True, width="stretch")
            summary = p8.get("summary", pd.DataFrame())
            if not summary.empty:
                st.dataframe(
                    summary[summary.cluster.isin([ca, cb])],
                    hide_index=True,
                    width="stretch",
                )
    configuration = {
        "dataset": d.summary.get("dataset"),
        "cluster_a": int(ca),
        "cluster_b": int(cb),
        "probes": selected,
        "comparison": "descriptive",
        "artifact_sources": d.summary.get("source_sha256", {}),
    }
    st.download_button(
        "Save comparison · JSON",
        json.dumps(configuration, indent=2),
        "comparison.json",
        "application/json",
    )


def pathways(d):
    heading(
        "08",
        "The biological context.",
        "Explore the saved Phase 7 over-representation results. Enrichment does not prove activation or causality.",
    )
    if missing(
        d,
        "pathways",
        "Pathway analysis is awaiting your partner’s validated Phase 7 results.",
    ):
        st.write(
            "The integration is ready for pathway tables, database filters, overlap counts, and method-specific enrichment statistics."
        )
        with st.expander("Required result format"):
            st.code(
                "reports/enrichment/pathway_results.csv\ncluster, database, pathway_id, pathway_name, adjusted_p_value, overlap_count, overlapping_genes\nOptional: raw_p_value, pathway_size, enrichment_score\nMetadata: reports/enrichment/metadata.json"
            )
        return
    frame = d.get("pathways")
    st.caption(
        f"{len(frame):,} saved pathway tests · filters below apply to the original adjusted p-values. "
        "Native Phase 7 exports use pathway term names as lookup keys; accession IDs are not supplied."
    )
    a, b, c = st.columns(3)
    database = a.multiselect(
        "Pathway databases", frame.database.unique(), default=frame.database.unique()
    )
    threshold = b.number_input("Adjusted p-value limit", 0.0, 1.0, 0.05)
    minimum = c.number_input("Minimum overlap", 0, value=1)
    query = st.text_input("Pathway name contains")
    f = frame[
        frame.cluster.isin(focus(d))
        & frame.database.isin(database)
        & (frame.adjusted_p_value <= threshold)
        & (frame.overlap_count >= minimum)
        & frame.pathway_name.str.contains(query, case=False, regex=False)
    ].sort_values("adjusted_p_value")
    if f.empty:
        st.info("Phase 7 results are loaded, but no pathways match these filters. Try another cluster or adjust the filters.")
        return
    axis = "enrichment_score" if "enrichment_score" in f else "significance"
    plot = f.head(25).copy()
    plot["significance"] = -np.log10(
        plot.adjusted_p_value.clip(lower=np.finfo(float).tiny)
    )
    label = (
        d.enrichment_meta.get("enrichment_measure", "Reported enrichment score")
        if axis == "enrichment_score"
        else "−log₁₀(adjusted p-value)"
    )
    show(
        px.scatter(
            plot,
            x=axis,
            y="pathway_name",
            size="overlap_count",
            color="adjusted_p_value",
            hover_data=["database", "cluster"],
            labels={axis: label},
            color_continuous_scale="Teal_r",
        ),
        "pathway_dotplot",
        650,
    )
    st.dataframe(f, hide_index=True, width="stretch")
    download(f, "pathway_selection")
    choices = f.index.tolist()
    chosen = st.selectbox(
        "Pathway details",
        choices,
        format_func=lambda i: f"{f.loc[i, 'database']} / {f.loc[i, 'pathway_name']} / C{int(f.loc[i, 'cluster'])}",
    )
    st.json(f.loc[chosen].to_dict())
    if d.enrichment_meta:
        st.json(d.enrichment_meta)
    else:
        st.warning(
            "Database version, analysis method and background gene universe have not been provided. Interpret these enrichment results with that provenance gap in mind."
        )



def characterization(d):
    heading("SUBTYPE CHARACTERIZATION", "Understand the discovered groups.",
            "Explore the saved subtype characterization: sample composition, candidate probes, pathway evidence and cross-cluster comparisons.")
    if missing(d, "characterization", "Subtype characterization results are unavailable."):
        return
    summary = d.get("characterization")
    selected = summary[summary.cluster.isin(focus(d))]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, [
        ("Groups in view", len(selected)),
        ("Samples in view", int(selected.sample_count.sum())),
        ("Strong candidate entries", int(selected.strong_candidate_count.sum())),
        ("Significant pathway entries", int(selected.significant_pathways_total.sum())),
    ]):
        col.metric(label, value)
    st.caption("Subtype characterization results loaded · Use Focus cluster above to inspect a group. Candidate and pathway totals count entries across groups and libraries, not unique genes or pathways.")
    overview, comparisons, figures = st.tabs(["Group profiles", "Statistical comparisons", "Saved figures"])
    with overview:
        labels = selected[["cluster", "known_ALL_count", "known_AML_count"]].melt(
            id_vars="cluster", var_name="Known label", value_name="Samples")
        labels["Known label"] = labels["Known label"].map({"known_ALL_count": "ALL", "known_AML_count": "AML"})
        labels["cluster"] = labels.cluster.astype(int).astype(str)
        show(px.bar(labels, x="cluster", y="Samples", color="Known label", barmode="stack",
                    color_discrete_map={"ALL": "#498a92", "AML": "#c69a56"}), "phase8_labels", 360)
        st.caption("Known ALL/AML labels are used for interpretation after clustering.")
        st.subheader("Cluster summaries")
        columns = ["cluster", "sample_count", "known_ALL_count", "known_AML_count",
                   "strong_candidate_count", "significant_pathways_total"]
        st.dataframe(selected[columns], hide_index=True, width="stretch")
        download(selected, "phase8_cluster_summary")
        st.info("These are exploratory groups. The same cohort was used for discovery and characterization; independent validation is required. No significant pathway entries at the saved threshold does not establish an absence of biological differences.")
    with comparisons:
        p8 = d.phase8_characterization()
        comp = p8["comparison"] if p8 else pd.DataFrame()
        if comp.empty:
            st.info("Saved cross-cluster comparisons are unavailable.")
        else:
            if len(selected) < len(summary):
                comp = comp[comp.cluster_a.isin(selected.cluster) | comp.cluster_b.isin(selected.cluster)]
            st.caption("Comparisons involving the selected cluster(s). Fisher p-values are the saved pairwise values. Infinite odds ratios indicate a zero cell in the ALL/AML table; PCA centroid distances use the saved comparison representation.")
            st.dataframe(comp, hide_index=True, width="stretch")
            download(comp, "phase8_statistical_comparisons")
    with figures:
        options = {
            "Sample distribution": "cluster_sample_distribution.png",
            "Biomarker counts": "biomarker_counts_by_cluster.png",
            "Pathway enrichment counts": "pathway_enrichment_counts_by_cluster.png",
            "Top biomarkers": "top_biomarkers_effect_size.png",
            "Cluster 0 pathways": "top_pathways_cluster0.png",
            "PCA distribution": "cross_cluster_pca_distribution.png",
        }
        available = {label: d.root / "reports/phase8/figures" / filename
                     for label, filename in options.items()
                     if (d.root / "reports/phase8/figures" / filename).is_file()}
        if available:
            chosen = st.selectbox("Subtype characterization figure", list(available))
            st.caption("Saved analysis figures show their original scope and do not change with Focus cluster.")
            st.image(str(available[chosen]), caption=chosen, width="stretch")
            st.download_button("Download Subtype characterization figure · PNG", available[chosen].read_bytes(),
                               available[chosen].name, "image/png", on_click="ignore")
        else:
            st.info("No saved Subtype characterization figures are available.")


def explainability(d):
    heading(
        "09",
        "Understand the surrogate.",
        "Inspect the model trained to reproduce cluster assignments. SHAP explains that surrogate’s output, not the causal biology of leukemia.",
    )
    if missing(
        d,
        "importance",
        "Optional model explainability outputs have not been supplied. Subtype characterization is available separately.",
    ):
        if not d.get("characterization").empty:
            if st.button("View Subtype Characterization ↗", key="explain_phase8"):
                navigate("Characterization")
        st.write(
            "Prepared views include global importance, sample-level contributions, observed feature values and independently evaluated surrogate fidelity."
        )
        with st.expander("Integration files"):
            st.code(
                "reports/explainability/global_importance.csv\nreports/explainability/sample_explanations.csv\nreports/explainability/surrogate_predictions.csv\nreports/explainability/model_metadata.json"
            )
        return
    frame = d.get("importance")
    a, b = st.columns(2)
    cluster = a.selectbox("Explained target cluster", sorted(frame.cluster.unique()))
    top = b.selectbox("Top features", [10, 20, 50])
    f = (
        frame[frame.cluster == cluster]
        .sort_values("mean_abs_shap", ascending=False)
        .head(top)
    )
    show(
        px.bar(f, x="mean_abs_shap", y="feature_id", orientation="h"),
        "global_shap",
        max(400, top * 18),
    )
    download(f, "shap_global")
    st.subheader("Model provenance and fidelity")
    if d.model:
        st.json(d.model)
    else:
        st.warning(
            "Model metadata and evaluation strategy are missing; fidelity cannot be assessed from feature importance alone."
        )
    pred = d.get("predictions")
    if not pred.empty:
        split = st.selectbox(
            "Prediction evaluation split", pred.evaluation_split.unique()
        )
        prediction = pred[pred.evaluation_split == split].merge(
            d.get("clusters"), on="sample_id", validate="one_to_one"
        )
        fidelity = (prediction.cluster == prediction.predicted_cluster).mean()
        st.metric("Surrogate agreement with saved assignments", f"{fidelity:.1%}")
        matrix = pd.crosstab(prediction.cluster, prediction.predicted_cluster)
        show(
            px.imshow(
                matrix,
                text_auto=True,
                labels={"x": "Surrogate prediction", "y": "Original consensus cluster"},
                color_continuous_scale="Teal",
            ),
            "surrogate_confusion",
        )
        st.caption(
            "Agreement on training samples can be optimistic. Interpret the selected split using the provided evaluation protocol."
        )
    else:
        st.info(
            "Sample-level surrogate predictions have not been supplied; fidelity and confusion matrix are unavailable."
        )
    sample = d.get("explanations")
    sample = sample[sample.target_cluster == cluster] if not sample.empty else sample
    if sample.empty:
        st.info("Sample-level SHAP contributions have not been supplied.")
        return
    st.subheader("Sample-level explanation")
    chosen = st.selectbox("Sample", sample.sample_id.unique())
    sf = sample[sample.sample_id == chosen].copy()
    sf["magnitude"] = sf.shap_value.abs()
    sf = sf.sort_values("magnitude", ascending=False).head(top)
    original = d.get("clusters").set_index("sample_id").loc[chosen, "cluster"]
    st.write(
        f"Original consensus assignment: **Cluster {int(original)}** · Explained target: **Cluster {int(cluster)}**"
    )
    prediction = pred[pred.sample_id == chosen] if not pred.empty else pred
    if not prediction.empty:
        st.dataframe(prediction, hide_index=True, width="stretch")
    else:
        st.caption("Surrogate prediction unavailable for this sample.")
    units = d.model.get("output_units", "unspecified model-output units")
    show(
        px.bar(
            sf,
            x="shap_value",
            y="feature_id",
            color="feature_value",
            orientation="h",
            labels={"shap_value": f"SHAP contribution ({units})"},
            color_continuous_scale="RdBu_r",
        ),
        "sample_shap",
        500,
    )
    st.subheader("Across samples")
    topids = f.feature_id.tolist()
    summary = sample[sample.feature_id.isin(topids)]
    show(
        px.scatter(
            summary,
            x="shap_value",
            y="feature_id",
            color="feature_value",
            hover_data=["sample_id"],
            labels={"shap_value": f"SHAP contribution ({units})"},
            color_continuous_scale="RdBu_r",
        ),
        "shap_summary",
        500,
    )
