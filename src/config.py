from typing import List, Literal, Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_expr_path: str
    raw_label_path: str
    processed_dir: str = "data/processed"

    n_top_genes: int = Field(default=2000, ge=100, le=7129)
    variance_metric: Literal["mad", "variance"] = "mad"
    log_transform: Literal["log2p1", "log1p"] = "log2p1"

    batch_check_cols: List[str] = []
    label_col: str = "cancer_type"
    sample_id_col: str = "sample_id"


class DimRedConfig(BaseModel):
    pca_n_components: int = Field(default=30, ge=2, le=100)
    umap_n_neighbors: int = Field(default=15, ge=2, le=100)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    umap_random_state: int = 42


class ClusteringConfig(BaseModel):
    methods: List[
        Literal[
            "hierarchical_ward",
            "kmeans",
            "gmm",
            "consensus",
        ]
    ] = [
        "hierarchical_ward",
        "kmeans",
        "gmm",
        "consensus",
    ]

    k_range: List[int] = [2, 3, 4, 5, 6, 7, 8]

    consensus_n_resamples: int = Field(
        default=100,
        ge=10,
        le=500,
    )

    consensus_p_item: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
    )

    consensus_p_feature: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
    )

    consensus_random_state: int = 42


class ValidationConfig(BaseModel):
    bootstrap_n_iter: int = Field(
        default=100,
        ge=10,
        le=1000,
    )

    bootstrap_sample_frac: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
    )

    ari_threshold_pass: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )

    stability_threshold_pass: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )


class BiomarkerConfig(BaseModel):
    test: Literal["mannwhitneyu", "ttest"] = "mannwhitneyu"
    fdr_method: Literal["fdr_bh", "bonferroni"] = "fdr_bh"

    fdr_threshold: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
    )

    logfc_threshold: float = Field(
        default=1.0,
        ge=0.0,
    )

    top_n_report: int = 20


class EnrichmentConfig(BaseModel):
    gene_sets: List[str] = [
        "KEGG_2021_Human",
        "Reactome_2022",
        "Hallmark_2020",
        "GO_Biological_Process_2021",
    ]

    organism: str = "Human"
    min_size: int = 5
    max_size: int = 500

    fdr_threshold: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
    )


class ExplainabilityConfig(BaseModel):
    surrogate_model: Literal[
        "RandomForestClassifier"
    ] = "RandomForestClassifier"

    n_estimators: int = 300
    max_depth: Optional[int] = None
    random_state: int = 42
    n_jobs: int = -1


class ReportConfig(BaseModel):
    title: str = "Cancer Subtype Discovery Report"
    author: str = "Ritik Arya & Prakhar Joshi"

    include_shap_summary: bool = True
    include_methods_section: bool = True

    output_path: str = "reports/final_report.pdf"


class AppConfig(BaseModel):
    theme: Literal["light", "dark"] = "light"
    cache_ttl_seconds: int = 3600


class PipelineConfig(BaseModel):
    seed: int = 42

    data: DataConfig
    dimensionality_reduction: DimRedConfig
    clustering: ClusteringConfig
    validation: ValidationConfig
    biomarkers: BiomarkerConfig
    enrichment: EnrichmentConfig
    explainability: ExplainabilityConfig
    report: ReportConfig
    app: AppConfig

    @classmethod
    def from_yaml(cls, path: str):
        config = OmegaConf.load(path)
        return cls(**OmegaConf.to_container(config, resolve=True))