import base64, json, math
from html import escape
import streamlit as st
import streamlit.components.v1 as components
from app.components.layout import go
from app.components.layout import ROOT


def image(svg):
    return (
        '<img alt="Scientific data visualization" src="data:image/svg+xml;base64,'
        + base64.b64encode(svg.encode()).decode()
        + '">'
    )


def render(data):
    atoms = json.loads((ROOT / "app/assets/dna.json").read_text())
    fallback = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 750" aria-hidden="true">'
    for atom in sorted(atoms["atoms"], key=lambda a: a["p"][2]):
        x, y, z = atom["p"]
        fallback += f'<circle cx="{650 + x * 12 + y * 4}" cy="{350 - y * 12 + x * 4}" r="{4 + z * 0.04}" fill="{chr(35)}e1e9d9"/>'
    fallback += "</svg>"
    geneart = fallback
    expr = data.get("expression")
    clusters = data.get("pca")
    heat = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 280" aria-hidden="true">'
    if not expr.empty:
        values = expr.iloc[:20, :30].to_numpy()
        v = (values - values.min()) / (values.max() - values.min())
        for i, row in enumerate(v):
            for j, value in enumerate(row):
                heat += f'<rect x="{j * 13}" y="{i * 14}" width="11" height="12" fill="rgb({int(70 + value * 130)},{int(110 + value * 90)},{int(117 + value * 65)})"/>'
    heat += "</svg>"
    points = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 280" aria-hidden="true">'
    if not clusters.empty:
        colors = ["#80bbbd", "#e6be86", "#c6cadd"]
        for _, r in clusters.iterrows():
            x = (
                35
                + (r.PC1 - clusters.PC1.min())
                / (clusters.PC1.max() - clusters.PC1.min())
                * 310
            )
            y = (
                240
                - (r.PC2 - clusters.PC2.min())
                / (clusters.PC2.max() - clusters.PC2.min())
                * 200
            )
            points += f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[int(r.cluster) % 3]}" opacity=".85"/>'
    points += "</svg>"
    s = data.summary
    replacements = {
        "__FALLBACK__": image(fallback),
        "__EXPRESSION_ART__": image(heat),
        "__CLUSTER_ART__": image(points),
        "__GENE_ART__": image(geneart),
        "__SAMPLES__": str(len(data.get("clusters")))
        if not data.get("clusters").empty
        else "—",
        "__PROBES__": f"{s['probes']:,}" if s else "—",
        "__CLUSTERS__": str(data.get("clusters").cluster.nunique())
        if not data.get("clusters").empty
        else "—",
        "__CANDIDATES__": str(len(data.get("biomarkers")))
        if not data.get("biomarkers").empty
        else "—",
    }
    body = (ROOT / "app/assets/landing.html").read_text()
    for key, value in replacements.items():
        body = body.replace(key, value)
    body = body.split('<script type="module">')[0]
    st.html(
        '<style>.block-container{padding:0!important;max-width:none!important}[data-testid="stMainBlockContainer"]>div{gap:0!important}</style>'
    )
    component = components.declare_component(
        "molecular_landing_v2", path=str(ROOT / "app/assets")
    )
    route = component(markup=body, dna=atoms, key="molecular_landing")
    if isinstance(route, str) and route in [
        "Home",
        "Overview",
        "Dataset",
        "Clusters",
        "Biomarkers",
        "Reports",
        "Methods",
    ]:
        go(route)
