"""
New Clustering Package - GPT + Cosine Similarity Theme Extraction

A theme-first clustering approach that:
1. Discovers themes first using GPT on representative samples
2. Generates multiple key phrases per theme to cast a wider net
3. Assigns messages to themes via cosine similarity
4. Validates quality through automated checks

Usage:
    from new_clustering import run_pipeline, ThemeClusteringPipeline
    from src.theme_clustering.theme_config import ThemeClusteringConfig

    # Quick run with defaults
    output = run_pipeline(
        messages=["message 1", "message 2", ...],
        session_ids=["session_1", "session_2", ...],
        seed_themes=["Delivery", "Returns"],  # optional
        client_name="my_client",
    )

    # Access results
    for cluster in output.clusters:
        print(f"{cluster.theme.theme_name}: {cluster.message_count} messages")

    # Or with custom config
    config = ThemeClusteringConfig(...)
    pipeline = ThemeClusteringPipeline(config)
    output = pipeline.run(messages, session_ids)

Modules:
    - theme_config: Centralized configuration (all thresholds in one place)
    - theme_models: Data classes (Theme, MessageAssignment, ClusterResult, etc.)
    - theme_prompts: GPT prompt templates
    - theme_preprocessing: Minimal text preprocessing
    - theme_sampling: Adaptive stratified sampling
    - theme_discovery: GPT-based theme discovery
    - theme_embedding: Key phrase and message embedding
    - theme_assignment: Cosine similarity message assignment
    - theme_quality: Quality assurance (coherence, merging, etc.)
    - theme_clustering: Main pipeline orchestration
"""

from .theme_clustering import (
    ThemeClusteringPipeline,
    run_pipeline,
    main,
)

from .theme_config import (
    ThemeClusteringConfig,
    SamplingConfig,
    DiscoveryConfig,
    EmbeddingConfig,
    AssignmentConfig,
    QualityConfig,
    SecondPassConfig,
    PreprocessingConfig,
    OutputConfig,
    DEFAULT_CONFIG,
)

from .theme_models import (
    Theme,
    MessageAssignment,
    ClusterResult,
    QualityReport,
    PipelineInput,
    PipelineOutput,
    AssignmentStatus,
)

__all__ = [
    # Main pipeline
    "ThemeClusteringPipeline",
    "run_pipeline",
    "main",

    # Configuration
    "ThemeClusteringConfig",
    "SamplingConfig",
    "DiscoveryConfig",
    "EmbeddingConfig",
    "AssignmentConfig",
    "QualityConfig",
    "SecondPassConfig",
    "PreprocessingConfig",
    "OutputConfig",
    "DEFAULT_CONFIG",

    # Data models
    "Theme",
    "MessageAssignment",
    "ClusterResult",
    "QualityReport",
    "PipelineInput",
    "PipelineOutput",
    "AssignmentStatus",
]

__version__ = "1.0.0"
