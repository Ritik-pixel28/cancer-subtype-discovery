"""Deterministic data selection, scientific figures and portable PDF/HTML reports."""

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from html import escape
import base64
import hashlib
import io
import json
import os
import zipfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.pagesizes import A4

SECTIONS = [
    "Dataset",
    "Preprocessing",
    "PCA",
    "Clusters",
    "Validation",
    "Biomarkers",
    "Pathways",
    "Explainability",
    "Methods",
    "Limitations",
    "Provenance",
]
PALETTE = ["#277c8e", "#c38b31", "#8273a6"]
LIMITATIONS = "This is exploratory research, not a diagnostic system. The cohort contains 72 historical leukemia samples. Cluster-validation signals are mixed, agreement with external ALL/AML labels is modest, and bootstrap lower bounds do not establish uniformly strong stability. Candidate probes were identified in the same cohort used for discovery. Probe annotations may be missing or shared across genes. Independent datasets and biological validation are required before clinical interpretations. Pathway enrichment and surrogate SHAP explanations do not establish causality."
METHODS = "Known ALL/AML labels are withheld from feature selection, PCA and clustering. The existing pipeline selects the top 2,000 variable probes by median absolute deviation, applies gene-wise standardization, and retains 30 principal components with seed 42. K-Means, Ward and GMM are compared alongside a consensus solution. Validation uses silhouette, external ARI, Gap statistic, PAC and resampling Jaccard stability. Cluster-versus-rest biomarker analysis uses Mann-Whitney U, BH-FDR, median differences and rank-biserial effect size. Strong candidates use an absolute rank-biserial threshold of 0.8. Presentation reads saved outputs without rerunning clustering."


@dataclass(frozen=True)
class ReportOptions:
    kind: str = "Executive summary"
    cluster: int | None = None
    sections: tuple = tuple(SECTIONS)


def selected_tables(d, o):
    if o.kind not in [
        "Executive summary",
        "Full technical report",
        "Cluster-specific report",
    ]:
        raise ValueError("Unknown report type")
    if d.get("clusters").empty:
        raise ValueError("Valid cluster assignments are required to generate a report")
    if o.kind == "Cluster-specific report" and o.cluster not in set(
        d.get("clusters").cluster
    ):
        raise ValueError("Select a valid report cluster")
    if set(o.sections) - set(SECTIONS):
        raise ValueError("Unknown report section")
    tables = {}
    mapping = {
        "Dataset": ["clusters"],
        "Preprocessing": ["mad"],
        "PCA": ["pca", "variance"],
        "Clusters": ["clusters"],
        "Validation": ["validation", "bootstrap", "gap", "pac"],
        "Biomarkers": ["biomarkers"],
        "Pathways": ["pathways"],
        "Explainability": ["importance"],
    }
    for section in o.sections:
        for name in mapping.get(section, []):
            if name in d.errors:
                raise ValueError(f"Cannot report invalid {name}: {d.errors[name]}")
            frame = d.get(name).copy()
            if (
                o.kind == "Cluster-specific report"
                and name in ["clusters", "pca", "biomarkers", "pathways", "importance"]
                and not frame.empty
            ):
                frame = frame[frame.cluster == o.cluster]
            tables[name] = frame
    return tables


def figure(d, name, tables):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#314d54",
            "text.color": "#253d42",
            "axes.edgecolor": "#bdc9c5",
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )
    fig, ax = plt.subplots(figsize=(7.1, 3.7), layout="constrained")
    fig.patch.set_facecolor("white")
    if name == "Dataset":
        labels = d.summary.get("known_labels", {})
        ax.bar(labels.keys(), labels.values())
        ax.set(
            ylabel="Samples", title="Known reference labels (not discovered clusters)"
        )
    elif name == "Preprocessing":
        frame = tables.get("mad", pd.DataFrame())
        if frame.empty:
            plt.close(fig)
            return None
        ax.plot(frame["rank"], frame.mad)
        ax.axvline(
            int(frame.selected.sum()),
            color=PALETTE[1],
            ls="--",
            label="Selected cutoff",
        )
        ax.legend()
        ax.set(
            xlabel="Probe rank", ylabel="MAD", title="Unsupervised feature variability"
        )
    elif name == "PCA":
        frame = tables.get("pca", pd.DataFrame())
        if frame.empty:
            plt.close(fig)
            return None
        for cluster, group in frame.groupby("cluster"):
            ax.scatter(
                group.PC1,
                group.PC2,
                s=24,
                label=f"Cluster {int(cluster)}",
                color=PALETTE[int(cluster) % 3],
            )
        ax.legend(frameon=False)
        ax.set(
            xlabel="PC1",
            ylabel="PC2",
            title="PCA projection of saved consensus assignments",
        )
    elif name == "Clusters":
        frame = tables.get("clusters", pd.DataFrame())
        if frame.empty:
            plt.close(fig)
            return None
        counts = frame.cluster.value_counts().sort_index()
        ax.bar(
            [f"C{int(c)}" for c in counts.index],
            counts.values,
            color=[PALETTE[int(c) % 3] for c in counts.index],
        )
        ax.set(ylabel="Samples", title="Exploratory consensus cluster sizes")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.3, str(v), ha="center")
    elif name == "Validation":
        frame = tables.get("pac", pd.DataFrame())
        if frame.empty:
            plt.close(fig)
            return None
        ax.plot(frame.k, frame.pac, "o-")
        best = frame.loc[frame.pac.idxmin()]
        ax.scatter([best.k], [best.pac], color=PALETTE[1], s=80)
        ax.set(
            xlabel="Number of clusters (k)",
            ylabel="PAC",
            title="Consensus ambiguity (lower is less ambiguous)",
        )
    elif name == "Biomarkers":
        frame = tables.get("biomarkers", pd.DataFrame())
        if frame.empty:
            plt.close(fig)
            return None
        for cluster, group in frame.groupby("cluster"):
            ax.scatter(
                group.rank_biserial,
                -np.log10(group.fdr.clip(lower=np.finfo(float).tiny)),
                label=f"C{int(cluster)}",
                s=14,
                alpha=0.8,
                color=PALETTE[int(cluster) % 3],
            )
        ax.legend(frameon=False)
        ax.set(
            xlabel="Rank-biserial effect size",
            ylabel="-log10(BH-FDR)",
            title="Strong candidate probes (preselected set)",
        )
    else:
        plt.close(fig)
        return None
    files = {}
    for ext in ["png", "svg"]:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format=ext,
            dpi=180,
            metadata={"Creator": "Cancer Subtype Discovery Explorer"}
            if ext == "svg"
            else None,
        )
        files[ext] = buf.getvalue()
    plt.close(fig)
    return files


def section_content(d, o, tables):
    scope = (
        f"Cluster {o.cluster}" if o.kind == "Cluster-specific report" else "Full cohort"
    )
    result = []
    for section in o.sections:
        frame = pd.DataFrame()
        text = ""
        chart = None
        if section == "Dataset":
            text = f"Golub leukemia reference cohort. Report scope: {scope}. Expression values are probe measurements, not necessarily unique genes. The reference ALL/AML labels are used only for post-clustering interpretation."
            s = d.summary
            frame = pd.DataFrame(
                {
                    "Property": [
                        "Samples",
                        "Expression probes",
                        "MAD-selected probes",
                        "PCA components",
                        "Seed",
                    ],
                    "Value": [
                        s.get(k, "Unavailable")
                        for k in [
                            "samples",
                            "probes",
                            "selected_probes",
                            "pca_components",
                            "seed",
                        ]
                    ],
                }
            )
        elif section == "Preprocessing":
            text = "Probes are ranked using median absolute deviation. The top 2,000 are standardized to zero mean and unit population standard deviation. High variability is not evidence of biological importance. Negative raw values preclude a naive log2(x+1) transformation."
            if "mad" in tables:
                frame = tables["mad"].head(12)
        elif section == "PCA":
            v = tables.get("variance", pd.DataFrame())
            text = (
                f"The configured 30-component representation retains {v.explained_variance_ratio.sum():.1%} of selected-feature variance."
                if not v.empty
                else "Explained variance is unavailable."
            )
            text += " Saved cluster labels are joined by sample ID. Low-dimensional plots do not preserve every distance used by clustering."
            frame = v.head(10)
        elif section == "Clusters":
            c = tables.get("clusters", pd.DataFrame())
            text = f"Report scope: {scope}. The three-group consensus partition remains exploratory."
            if not c.empty:
                frame = (
                    c.cluster.value_counts()
                    .sort_index()
                    .rename_axis("Cluster")
                    .reset_index(name="Samples")
                )
        elif section == "Validation":
            text = "Global algorithm evidence, not per-cluster diagnostic performance. ARI is external label agreement; silhouette is internal separation. Bootstrap bounds are empirical resampling intervals. Gap is interpreted using the 1-SE rule. PAC measures ambiguous consensus sample pairs. The original pairwise consensus matrix is not available unless separately exported."
            v = tables.get("validation", pd.DataFrame())
            b = tables.get("bootstrap", pd.DataFrame())
            if not v.empty:
                frame = (
                    v[v.k == 3].merge(b, on=["method", "k"], how="left")
                    if not b.empty
                    else v[v.k == 3]
                )
        elif section == "Biomarkers":
            b = tables.get("biomarkers", pd.DataFrame())
            text = f"{len(b)} strong candidate probe-cluster entries in this report scope. Gene symbols can be shared or missing. These candidates are not independently validated clinical biomarkers. Technical tables show the top entries by FDR; complete selected tables are included in the accompanying bundle."
            if not b.empty:
                frame = b.sort_values(["fdr", "gene"])[
                    ["gene", "gene_symbol", "cluster", "fdr", "rank_biserial"]
                ].head(12)
        elif section == "Pathways":
            p = tables.get("pathways", pd.DataFrame())
            text = (
                "Awaiting validated Phase 7 results. No pathway findings were inferred or fabricated."
                if p.empty
                else "Reported enrichment from Phase 7. Enrichment does not establish pathway activation or causality. Metadata: "
                + json.dumps(d.enrichment_meta)
            )
            if not p.empty:
                frame = p[
                    ["cluster", "database", "pathway_name", "adjusted_p_value"]
                ].head(15)
        elif section == "Explainability":
            p = tables.get("importance", pd.DataFrame())
            text = (
                "Awaiting validated Phase 8 results. No SHAP values were inferred or fabricated."
                if p.empty
                else "SHAP explains the surrogate classifier, not causal biology or the unsupervised algorithm directly. Model metadata: "
                + json.dumps(d.model)
            )
            if not p.empty:
                frame = p.head(15)
        elif section == "Methods":
            text = METHODS
        elif section == "Limitations":
            text = LIMITATIONS
        elif section == "Provenance":
            text = (
                "Source configuration, checksums and software versions are included in provenance.json inside the report bundle. Exported analysis base: "
                + str(d.summary.get("source_commit", "Unavailable"))
                + ". Seed: "
                + str(d.summary.get("seed", "Unavailable"))
                + ". Application code may include uncommitted changes; this state is recorded rather than attributed silently to the base commit."
            )
            p = d.provenance()
            frame = pd.DataFrame(
                {
                    "Package": list(p["software"]),
                    "Version": list(p["software"].values()),
                }
            )
        chart = figure(d, section, tables)
        result.append({"name": section, "text": text, "table": frame, "figure": chart})
    return result


def build_report(d, options=ReportOptions()):
    tables = selected_tables(d, options)
    # Scientific scope and limitations are always explicit even when optional sections are omitted.
    sections = section_content(d, options, tables)
    provenance = d.provenance()
    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    provenance["report"] = {
        "kind": options.kind,
        "cluster": options.cluster,
        "sections": list(options.sections),
        "generated_at": timestamp,
    }
    pdf = render_pdf(d, options, sections, provenance)
    env = Environment(
        loader=FileSystemLoader(d.root / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    rendered = []
    for section in sections:
        item = dict(section)
        item["table_html"] = (
            section["table"].to_html(
                index=False, escape=True, float_format=lambda x: f"{x:.5g}"
            )
            if not section["table"].empty
            else ""
        )
        item["image"] = (
            base64.b64encode(section["figure"]["png"]).decode()
            if section["figure"]
            else None
        )
        rendered.append(item)
    html = env.get_template("research_report.html").render(
        kind=options.kind,
        cluster=options.cluster,
        sections=rendered,
        provenance=provenance,
        limitations=LIMITATIONS,
        timestamp=timestamp,
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("research_report.pdf", pdf)
        z.writestr("research_report.html", html)
        z.writestr("provenance.json", json.dumps(provenance, indent=2, default=str))
        for name, table in tables.items():
            if not table.empty:
                z.writestr("tables/" + name + ".csv", table.to_csv(index=False))
        for section in sections:
            if section["figure"]:
                for ext, b in section["figure"].items():
                    z.writestr("figures/" + section["name"].lower() + "." + ext, b)
    return {
        "pdf": pdf,
        "html": html.encode(),
        "zip": archive.getvalue(),
        "metadata": json.dumps(provenance, indent=2, default=str).encode(),
        "figures": {s["name"]: s["figure"] for s in sections if s["figure"]},
    }


def render_pdf(d, o, sections, provenance):
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            fontName="Helvetica",
            fontSize=31,
            leading=35,
            textColor=colors.HexColor("#23464c"),
            spaceAfter=24,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportDeck",
            fontSize=11,
            leading=17,
            textColor=colors.HexColor("#506b70"),
            spaceAfter=12,
        )
    )
    styles.add(ParagraphStyle(name="Cell", fontSize=7, leading=10, wordWrap="CJK"))
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 14
    styles["Heading1"].textColor = colors.HexColor("#285c64")
    story = []
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=46,
        leftMargin=46,
        topMargin=55,
        bottomMargin=45,
        title="Cancer Subtype Discovery | Group G87",
        author="Ritik Arya & Prakhar Joshi",
    )
    story.extend(
        [
            Paragraph("GROUP G87 / RESEARCH REPORT", styles["ReportDeck"]),
            Spacer(1, 28),
            Paragraph(
                "Cancer Subtype Discovery<br/>Using Gene Expression Data",
                styles["ReportTitle"],
            ),
            Paragraph(escape(o.kind), styles["Heading2"]),
            Paragraph(
                "An exploratory molecular subgroup discovery platform",
                styles["ReportDeck"],
            ),
            Spacer(1, 20),
        ]
    )
    count = (
        len(d.get("clusters"))
        if o.cluster is None
        else int((d.get("clusters").cluster == o.cluster).sum())
    )
    story.append(
        Paragraph(
            f"Report scope: {'Full cohort' if o.cluster is None else 'Cluster ' + str(o.cluster)} | {count} samples",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "Dataset: Golub leukemia | Selected solution: exploratory consensus k=3",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 18))
    story.append(Paragraph("Research objective", styles["Heading2"]))
    story.append(
        Paragraph(
            "Investigate whether gene-expression patterns reveal reproducible molecular structure among leukemia samples, assess the reliability of the discovered partition, and identify candidate cluster-associated probes for subsequent biological interpretation.",
            styles["ReportDeck"],
        )
    )
    story.append(
        Paragraph(
            "Independent validation is required. This report does not support clinical diagnosis.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 26))
    story.append(
        Paragraph(
            "Generated: " + escape(provenance["report"]["generated_at"]),
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "Application base commit: " + escape(provenance["application_commit"]),
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "Working tree modified: " + str(provenance["working_tree_modified"]),
            styles["BodyText"],
        )
    )

    def section_block(s, compact=False):
        block = [
            Paragraph(escape(s["name"]), styles["Heading1"]),
            Paragraph(escape(s["text"]), styles["BodyText"]),
            Spacer(1, 10),
        ]
        if s["figure"]:
            block.extend(
                [
                    Image(io.BytesIO(s["figure"]["png"]), width=480, height=250),
                    Spacer(1, 10),
                ]
            )
        frame = s["table"]
        if not frame.empty:
            # Narrow, wrapped tables remain legible on A4; full precision is preserved in CSV exports.
            frame = frame.head(5 if compact else 12)
            cols = list(frame.columns)
            rows = [
                [
                    Paragraph(escape(str(c)).replace("_", " "), styles["Cell"])
                    for c in cols
                ]
            ]
            for _, r in frame.iterrows():
                rows.append(
                    [
                        Paragraph(
                            escape(
                                "—"
                                if pd.isna(v)
                                else f"{v:.4g}"
                                if isinstance(v, (float, np.floating))
                                else str(v)
                            ),
                            styles["Cell"],
                        )
                        for v in r
                    ]
                )
            table = Table(
                rows,
                colWidths=[480 / len(cols)] * len(cols),
                repeatRows=1,
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e1eae4")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8d4cc")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            block.append(table)
        return block

    if o.kind == "Executive summary":
        # Four concise pages, respecting selected sections without expanding into an appendix.
        story.append(PageBreak())
        story.append(Paragraph("Methodology and dataset", styles["Heading1"]))
        for s in sections:
            if s["name"] in ["Dataset", "Preprocessing", "PCA", "Methods"]:
                story.append(Paragraph(escape(s["name"]), styles["Heading2"]))
                story.append(Paragraph(escape(s["text"]), styles["BodyText"]))
        story.append(PageBreak())
        story.append(Paragraph("Exploratory findings", styles["Heading1"]))
        findings = [
            s for s in sections if s["name"] in ["Clusters", "Validation", "Biomarkers"]
        ]
        for s in findings:
            story.append(Paragraph(escape(s["name"]), styles["Heading2"]))
            story.append(Paragraph(escape(s["text"]), styles["BodyText"]))
        validation = next(
            (s["table"] for s in findings if s["name"] == "Validation"), pd.DataFrame()
        )
        if not validation.empty:
            compact = validation[
                [
                    c
                    for c in ["method", "ari", "silhouette", "stability_mean"]
                    if c in validation
                ]
            ]
            rows = [
                [
                    Paragraph(str(c).replace("_", " "), styles["Cell"])
                    for c in compact.columns
                ]
            ]
            for _, row in compact.iterrows():
                rows.append(
                    [
                        Paragraph(
                            escape(
                                f"{v:.4f}"
                                if isinstance(v, (float, np.floating))
                                else str(v)
                            ),
                            styles["Cell"],
                        )
                        for v in row
                    ]
                )
            tab = Table(
                rows, colWidths=[480 / len(compact.columns)] * len(compact.columns)
            )
            tab.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e1eae4")),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d4cc")),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([Spacer(1, 12), tab, Spacer(1, 8)])
        chart = next((s["figure"] for s in findings if s["figure"]), None)
        if chart:
            story.append(Image(io.BytesIO(chart["png"]), width=470, height=245))
        story.append(PageBreak())
        story.append(
            Paragraph("Limits, interpretation and provenance", styles["Heading1"])
        )
        story.append(Paragraph(escape(LIMITATIONS), styles["BodyText"]))
        for s in sections:
            if s["name"] in ["Pathways", "Explainability", "Provenance"]:
                story.extend(
                    [
                        Paragraph(escape(s["name"]), styles["Heading2"]),
                        Paragraph(escape(s["text"]), styles["BodyText"]),
                    ]
                )
    else:
        for s in sections:
            story.append(PageBreak())
            story.extend(section_block(s))
        if "Limitations" not in o.sections:
            story.extend(
                [
                    PageBreak(),
                    Paragraph("Scientific limitations", styles["Heading1"]),
                    Paragraph(escape(LIMITATIONS), styles["BodyText"]),
                ]
            )
    story.extend(
        [
            Spacer(1, 18),
            Paragraph("References", styles["Heading2"]),
            Paragraph(
                "Golub et al. (1999). Molecular classification of cancer. Science 286:531-537. doi:10.1126/science.286.5439.531.<br/>Drew et al. (1981). Structure of a B-DNA dodecamer. PNAS 78:2179-2183. doi:10.1073/pnas.78.4.2179. Structural illustration only.<br/>Original implementation, configuration and result artifacts: Ritik-pixel28/cancer-subtype-discovery.",
                styles["BodyText"],
            ),
        ]
    )

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd6cf"))
        canvas.line(46, 35, A4[0] - 46, 35)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#5b7377"))
        canvas.drawString(
            46, 24, "GROUP G87 / EXPLORATORY RESEARCH / NOT FOR CLINICAL DIAGNOSIS"
        )
        canvas.drawRightString(A4[0] - 46, 24, str(doc.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=page, onLaterPages=page)
    return output.getvalue()
