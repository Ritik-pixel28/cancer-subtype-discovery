from pathlib import Path

import GEOparse
import pandas as pd


INPUT_PATH = Path(
    "reports/biomarkers/strong_candidates.csv"
)

OUTPUT_PATH = Path(
    "reports/biomarkers/"
    "strong_candidates_annotated.csv"
)

ANNOTATION_DIR = Path(
    "data/annotation"
)


def main():
    print("=" * 70)
    print("PHASE 6 — PROBE ANNOTATION")
    print("=" * 70)

    strong = pd.read_csv(INPUT_PATH)

    print()
    print(
        f"Strong candidate rows: {len(strong)}"
    )

    print(
        f"Unique probes: "
        f"{strong['gene'].nunique()}"
    )

    ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Downloading/loading GPL80 annotation...")

    gpl = GEOparse.get_GEO(
        geo="GPL80",
        destdir=str(ANNOTATION_DIR),
        silent=True,
    )

    annotation = gpl.table.copy()

    print(
        f"GPL80 annotation rows: "
        f"{len(annotation)}"
    )

    print()
    print("Available annotation columns:")
    print(annotation.columns.tolist())

    required_columns = [
        "ID",
        "Gene Symbol",
        "Gene Title",
        "ENTREZ_GENE_ID",
    ]

    available_columns = [
        column
        for column in required_columns
        if column in annotation.columns
    ]

    annotation = annotation[
        available_columns
    ].copy()

    annotation = annotation.rename(
        columns={
            "ID": "gene",
            "Gene Symbol": "gene_symbol",
            "Gene Title": "gene_title",
            "ENTREZ_GENE_ID": "entrez_gene_id",
        }
    )

    annotated = strong.merge(
        annotation,
        on="gene",
        how="left",
    )

    annotated.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    mapped = annotated[
        "gene_symbol"
    ].notna().sum()

    print()
    print(
        f"Mapped rows: "
        f"{mapped}/{len(annotated)}"
    )

    print()
    print("Preview:")

    preview_columns = [
        "gene",
        "gene_symbol",
        "gene_title",
        "cluster",
        "direction",
        "fdr",
        "rank_biserial",
    ]

    preview_columns = [
        c
        for c in preview_columns
        if c in annotated.columns
    ]

    print(
        annotated[
            preview_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()