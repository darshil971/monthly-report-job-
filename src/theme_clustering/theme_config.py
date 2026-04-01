"""
Theme Clustering Configuration - Centralized Thresholds and Parameters

All configurable parameters for the GPT + Cosine Similarity theme extraction pipeline.
Modify values here to tune the pipeline behavior.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SamplingConfig:
    """Phase 1: Adaptive Sampling Configuration"""

    # Sample size thresholds
    small_dataset_max: int = 500           # Use all messages if total <= this
    medium_dataset_max: int = 1000          # Medium dataset threshold
    large_dataset_max: int = 1500          # Large dataset threshold

    # Sample percentages
    medium_sample_percent: float = 0.75     # Sample 60% for medium datasets
    large_sample_percent: float = 0.33      # Sample 20% for large datasets

    # Absolute caps
    medium_sample_cap: int = 500           # Max samples for medium datasets
    large_sample_cap: int = 650            # Max samples for large datasets
    very_large_sample_cap: int = 800       # Fixed cap for very large datasets

    # Sampling method
    use_hdbscan: bool = True              # Use HDBSCAN instead of mini-KMeans for sampling

    # Stratified sampling (KMeans)
    mini_clusters_divisor: int = 5         # k = sample_size // this value
    random_state: int = 42


@dataclass
class DiscoveryConfig:
    """Phase 2: Theme Discovery Configuration"""

    # GPT batching
    messages_per_gpt_call: int = 200       # Max messages per GPT discovery call

    # Theme constraints
    min_themes: int = 5                    # Minimum expected themes per batch
    key_phrases_per_theme: int = 10         # Target key phrases per theme (6-10)
    min_key_phrases: int = 6               # Minimum key phrases per theme

    # GPT parameters
    gpt_temperature: float = 0.2           # Temperature for theme discovery
    gpt_max_tokens: int = 6000             # Max tokens for GPT response (raised from 4000 to prevent truncation on dense batches)
    gpt_timeout: int = 120                 # Timeout in seconds

    # Consolidation
    consolidation_temperature: float = 0.2  # Temperature for theme consolidation
    always_consolidate: bool = True        # Always run consolidation after discovery


@dataclass
class EmbeddingConfig:
    """Phase 3: Embedding Configuration"""

    # Model configuration
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 200
    max_retries: int = 3


@dataclass
class AssignmentConfig:
    """Phase 4: Message Assignment Configuration"""

    # Primary thresholds (from approved plan)
    primary_threshold: float = 0.5        # Min similarity for confident assignment
    confidence_gap: float = 0.04           # Min gap between best and 2nd best
    borderline_threshold: float = 0.45     # Below primary but may reassign in refinement

    # Hybrid fusion weights (sum to 1.0)
    w_cosine: float = 0.55                    # Semantic signal weight
    w_bm25: float = 0.30                      # Lexical keyword signal weight
    w_fuzzy: float = 0.15                     # Surface-form / typo tolerance weight

    # Usecase-aware first-pass boost (Phase 3.6 → Phase 4)
    usecase_boost_enabled: bool = True        # Enable additive usecase boost during Phase 4 assignment
    usecase_additive_boost: float = 0.25      # Additive boost for usecase-matching clusters (capped at 1.0)

    # NOTE: No threshold relaxation based on misc size per user request
    # Misc messages accumulate and pipeline can be rerun on misc bucket if needed

    def __post_init__(self):
        """Validate that fusion weights sum to 1.0 (or very close)"""
        weight_sum = self.w_cosine + self.w_bm25 + self.w_fuzzy
        if not (0.99 <= weight_sum <= 1.01):  # Allow small floating-point error
            raise ValueError(
                f"Fusion weights must sum to 1.0. Current: "
                f"w_cosine={self.w_cosine} + w_bm25={self.w_bm25} + w_fuzzy={self.w_fuzzy} = {weight_sum}"
            )


@dataclass
class SecondPassConfig:
    """Phase 6: Second Pass on Miscellaneous Messages

    Two independent toggles:
    - borderline_reassignment: Force-assign BORDERLINE messages to their best theme (no GPT)
    - enabled: Usecase-aware misc rescue — GPT tags clusters with usecases,
      then boosts fusion scores for misc messages whose usecase matches a cluster.

    If borderline_reassignment=False and enabled=True, the usecase rescue will
    also act on borderline messages (they remain unassigned and eligible for boost).
    """

    # Stage 1: borderline reassignment (no GPT, just force-assign)
    borderline_reassignment: bool = True
    borderline_reassignment_threshold: float = 0.60  # Only reassign if best_similarity > this

    # Usecase-aware rescue toggle
    enabled: bool = False                  # Enable usecase-aware misc rescue

    # Usecase rescue parameters
    boost_factor: float = 1.20             # Multiplicative boost for usecase-matching clusters
    rescue_threshold: float = 0.5         # Min boosted fusion score to reassign from misc

    # False-negative (FN) signal rescue (runs after usecase rescue)
    fn_rescue: bool = True                 # Enable FN signal-based rescue
    fn_fusion_threshold: float = 0.75      # High-confidence FN: fusion alone exceeds this
    fn_bm25_norm_threshold: float = 0.95   # Medium-confidence FN: near-saturated BM25 ...
    fn_fusion_threshold_medium: float = 0.65  # ... combined with minimum fusion score


@dataclass
class QualityConfig:
    """Phase 5: Quality Assurance Configuration"""

    # Theme size thresholds
    min_messages_per_theme: int = 5        # Themes with fewer messages dissolved to misc

    # Redundancy detection (for merging similar themes)
    redundancy_similarity_threshold: float = 0.8  # Themes with centroid sim > this are merged

    # Coherence thresholds (from intent_expansion/config.py)
    cv_coherence_threshold: float = 0.1    # Coefficient of variation threshold
    min_coherence_score: float = 0.2       # Minimum internal coherence score

    # IQR-based outlier detection
    iqr_multiplier: float = 0.5            # Lower values find more outliers

    # Silhouette validation
    silhouette_positive_threshold: float = 0.6  # 60% of points must have positive silhouette
    min_avg_silhouette: float = 0.05       # Minimum average silhouette

    # Merge validation (from uf_option_generator.py approach)
    merge_threshold_buffer: float = 0.05   # 5% above median distance for merge validation
    distance_percentile: float = 25.0      # Use 25th percentile for candidate selection


@dataclass
class PreprocessingConfig:
    """Preprocessing Configuration (matching vector_copy.py)"""

    # Minimum message length
    min_message_length: int = 7            # Filter messages shorter than this

    # Language detection
    default_language: str = "en"           # Fallback if detection fails

    # Indian language codes
    indian_languages: tuple = (
        'hi', 'bn', 'te', 'ta', 'gu', 'kn', 'ml', 'mr', 'pa', 'or'
    )


@dataclass
class OutputConfig:
    """Output Configuration"""

    # Directory for intermediate outputs
    output_dir: str = "new_clustering/outputs"

    # Toggle generation of optional output files
    generate_borderline_diagnostics: bool = False   # <client>_<page>_borderline_diagnostics.txt
    generate_voc_report: bool = True                # voc_reports/clusters_<client>_<page>_voc_report.html (client-facing)
    generate_html_report: bool = True              # debug_report_<client>_<page>_<ts>.html (internal, off by default)
    generate_themes_json: bool = False              # <client>_<page>_themes.json

    # Output file names
    output_files: dict = field(default_factory=lambda: {
        "sampled_messages": "sampled_messages.json",
        "discovered_themes": "discovered_themes.json",
        "theme_embeddings": "theme_embeddings.npy",
        "message_assignments": "message_assignments.json",
        "final_clusters": "final_clusters.json",
        "miscellaneous": "miscellaneous_messages.json",
    })


@dataclass
class ThemeClusteringConfig:
    """
    Master configuration class combining all sub-configurations.

    Usage:
        config = ThemeClusteringConfig()
        # Or customize:
        config = ThemeClusteringConfig(
            sampling=SamplingConfig(very_large_sample_cap=400),
            assignment=AssignmentConfig(primary_threshold=0.6)
        )
    """

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    second_pass: SecondPassConfig = field(default_factory=SecondPassConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Global settings
    random_state: int = 42
    verbose: bool = True

    def __post_init__(self):
        """Ensure random_state is consistent across configs"""
        self.sampling.random_state = self.random_state


# Default configuration instance for easy import
DEFAULT_CONFIG = ThemeClusteringConfig()
