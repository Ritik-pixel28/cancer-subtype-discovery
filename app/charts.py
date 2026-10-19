import io
from functools import lru_cache
import plotly.io as pio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COLORS = ["#277c8e", "#c38b31", "#8273a6"]
MAP = {str(i): c for i, c in enumerate(COLORS)}


@lru_cache(maxsize=12)
def figure_html(figure_json):
    """Portable bytes work on both Streamlit 1.51 and the pinned app runtime."""
    return pio.from_json(figure_json).to_html(include_plotlyjs=True).encode()


def show(fig, key, height=420):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#43595e", size=12),
        margin=dict(l=25, r=20, t=45, b=40),
        height=height,
        colorway=COLORS,
        legend=dict(orientation="h", y=1.13, x=0),
        hoverlabel=dict(bgcolor="#fff"),
    )
    fig.update_xaxes(gridcolor="#dfe4dc", zerolinecolor="#bdcbc3")
    fig.update_yaxes(gridcolor="#dfe4dc", zerolinecolor="#bdcbc3")
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": key, "scale": 2},
        },
    )
    with st.expander("Export this figure", expanded=False):
        a, b = st.columns(2)
        a.download_button(
            "Interactive figure · HTML",
            figure_html(fig.to_json()),
            key + ".html",
            "text/html",
            key=key + "_html",
            on_click="ignore",
        )
        # Plotly's browser renderer exports vectors without a server-side browser dependency.
        b.caption(
            "Use the chart camera for PNG. Download the vector-ready HTML and choose SVG from the export toolbar below."
        )
        svg_figure = go.Figure(fig).update_layout(height=180)
        st.plotly_chart(
            svg_figure,
            width="stretch",
            key=key + "_svg",
            config={
                "displaylogo": False,
                "toImageButtonOptions": {"format": "svg", "filename": key},
            },
        )
    return fig


def pca_scatter(frame, components=("PC1", "PC2"), color="cluster"):
    frame = frame.copy()
    frame["cluster"] = frame.cluster.astype(int).astype(str)
    kwargs = dict(
        data_frame=frame,
        x=components[0],
        y=components[1],
        color=color,
        color_discrete_map=MAP if color == "cluster" else None,
        hover_name="sample_id",
        hover_data=["cluster"],
        opacity=0.88,
    )
    if len(components) == 3:
        fig = px.scatter_3d(**kwargs, z=components[2])
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(bgcolor="#f3f2ed"))
    else:
        fig = px.scatter(**kwargs)
        fig.update_traces(marker=dict(size=10, line=dict(width=0.6, color="#fff")))
    return fig


def heatmap(data, probes, clusters=None):
    expr = data.get("expression")
    assignment = data.get("clusters").sort_values(["cluster", "sample_id"])
    if clusters is not None:
        assignment = assignment[assignment.cluster.isin(clusters)]
    values = expr.loc[assignment.sample_id, probes]
    values = (values - expr[probes].mean()) / expr[probes].std(ddof=0).replace(
        0, np.nan
    )
    x = [f"C{int(c)} / {s}" for c, s in zip(assignment.cluster, assignment.sample_id)]
    fig = go.Figure(
        go.Heatmap(
            z=values.T,
            x=x,
            y=[data.probe_label(p) for p in probes],
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="z-score"),
        )
    )
    fig.update_layout(
        xaxis_title="Samples ordered by consensus cluster",
        yaxis_title="Expression probe",
        yaxis_autorange="reversed",
    )
    return fig
