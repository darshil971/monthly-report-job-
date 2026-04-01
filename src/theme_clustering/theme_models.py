"""
Theme Clustering Data Models

Data classes for the GPT + Cosine Similarity theme extraction pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from enum import Enum


class AssignmentStatus(Enum):
    """Status of message assignment to themes"""
    CONFIDENT = "confident"          # Assigned with high confidence
    BORDERLINE = "borderline"        # Low confidence, may be reassigned
    MISCELLANEOUS = "miscellaneous"  # Could not assign to any theme


@dataclass
class Theme:
    """
    Represents a discovered theme with its key phrases and embeddings.

    Attributes:
        theme_id: Unique identifier for the theme
        theme_name: Human-readable name (1-4 words)
        description: Longer description of what the theme covers
        key_phrases: List of 4-6 phrases that represent this theme
        example_messages: Sample messages from the discovery phase
        phrase_embeddings: Embeddings for each key phrase (n_phrases, 1024)
        theme_embedding: Centroid embedding (mean of phrase_embeddings)
        is_seed: Whether this was a user-provided seed theme
        message_count: Number of messages assigned to this theme
        coherence_score: Coherence metric for quality validation
    """
    theme_id: int
    theme_name: str
    description: str
    key_phrases: List[str]
    example_messages: List[str] = field(default_factory=list)
    phrase_embeddings: Optional[np.ndarray] = None
    theme_embedding: Optional[np.ndarray] = None
    is_seed: bool = False
    message_count: int = 0
    coherence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary (excludes numpy arrays)"""
        return {
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "description": self.description,
            "key_phrases": self.key_phrases,
            "example_messages": self.example_messages,
            "is_seed": self.is_seed,
            "message_count": self.message_count,
            "coherence_score": self.coherence_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Theme':
        """Create Theme from dictionary"""
        return cls(
            theme_id=data["theme_id"],
            theme_name=data["theme_name"],
            description=data["description"],
            key_phrases=data["key_phrases"],
            example_messages=data.get("example_messages", []),
            is_seed=data.get("is_seed", False),
            message_count=data.get("message_count", 0),
            coherence_score=data.get("coherence_score"),
        )


@dataclass
class MessageAssignment:
    """
    Represents a single message's assignment to a theme.

    Attributes:
        message_idx: Index in the original message list
        message_text: The actual message text
        session_id: Session ID for the message (optional)
        assigned_theme_id: ID of assigned theme (-1 for miscellaneous)
        best_similarity: Highest similarity score to any theme
        confidence_gap: Gap between best and second-best theme
        status: Assignment status (confident/borderline/miscellaneous)
        all_similarities: Similarity scores to all themes (for analysis)
        best_matching_phrase: The key phrase that had highest similarity
        cluster_title: Title of the cluster containing the best_matching_phrase
    """
    message_idx: int
    message_text: str
    session_id: Optional[str] = None
    assigned_theme_id: int = -1
    best_similarity: float = 0.0
    confidence_gap: float = 0.0
    status: AssignmentStatus = AssignmentStatus.MISCELLANEOUS
    all_similarities: Dict[int, float] = field(default_factory=dict)
    best_matching_phrase: Optional[str] = None
    cluster_title: Optional[str] = None        # Title of cluster containing best_matching_phrase
    was_borderline: bool = False               # True if reassigned from borderline in Stage 1
    was_usecase_rescued: bool = False          # True if reassigned via usecase-aware boost in Phase 6
    was_fn_rescued: bool = False               # True if reassigned via false-negative signal rescue
    was_usecase_boosted: bool = False          # True if Phase 4 additive boost changed cluster assignment
    original_cluster_title: Optional[str] = None  # Pre-boost cluster title (set when was_usecase_boosted=True)
    signal_scores: Dict[str, float] = field(default_factory=dict)  # Per-signal breakdown: cosine, bm25_raw, bm25_norm, fuzzy, fusion
    user_intent: Optional[str] = None         # Secondary usecase label from the source data
    parent_usecases: List[str] = field(default_factory=list)  # Usecases tagged by GPT for the cluster

    @property
    def is_confident(self) -> bool:
        return self.status == AssignmentStatus.CONFIDENT

    @property
    def is_borderline(self) -> bool:
        return self.status == AssignmentStatus.BORDERLINE

    @property
    def is_miscellaneous(self) -> bool:
        return self.status == AssignmentStatus.MISCELLANEOUS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "message_idx": self.message_idx,
            "message_text": self.message_text,
            "session_id": self.session_id,
            "assigned_theme_id": self.assigned_theme_id,
            "best_similarity": float(self.best_similarity),
            "confidence_gap": float(self.confidence_gap),
            "status": self.status.value,
            "all_similarities": {str(k): float(v) for k, v in self.all_similarities.items()},
            "best_matching_phrase": self.best_matching_phrase,
            "cluster_title": self.cluster_title,
            "was_borderline": self.was_borderline,
            "was_usecase_rescued": self.was_usecase_rescued,
            "was_fn_rescued": self.was_fn_rescued,
            "signal_scores": {k: round(v, 4) for k, v in self.signal_scores.items()} if self.signal_scores else {},
        }


@dataclass
class ClusterResult:
    """
    Final cluster result combining theme with assigned messages.

    Attributes:
        theme: The Theme object
        messages: List of MessageAssignment objects
        session_ids: Unique session IDs in this cluster
        quality_metrics: Coherence and other quality metrics
    """
    theme: Theme
    messages: List[MessageAssignment] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    parent_usecases: List[str] = field(default_factory=list)  # Usecases tagged by GPT
    category: str = ""  # "pre-sales", "post-sales", or "miscellaneous"
    metadata: Dict[str, Any] = field(default_factory=dict)   # session-level metrics (A2C, orders, UTM, avg conv length)
    performance: Dict[str, Any] = field(default_factory=dict)  # percentile-based performance classifications

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def session_count(self) -> int:
        return len(set(self.session_ids))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary for export"""
        # Count how many messages matched each key phrase (non-miscellaneous clusters only).
        # Uses the best_matching_phrase already recorded on each MessageAssignment so no
        # re-embedding is needed.  Messages whose matching_phrase is None (unmatched edge
        # cases) are simply skipped.
        from collections import Counter
        phrase_counts: Dict[str, int] = dict(
            Counter(
                m.best_matching_phrase
                for m in self.messages
                if m.best_matching_phrase is not None
            )
        )

        result = {
            "cluster_id": self.theme.theme_id,
            "cluster_title": self.theme.theme_name,
            "description": self.theme.description,
            "key_phrases": self.theme.key_phrases,
            "key_phrase_counts": phrase_counts,
            "message_count": self.message_count,
            "session_count": self.session_count,
            "is_seed": self.theme.is_seed,
            "category": self.category,
            "parent_usecases": self.parent_usecases,
            "quality_metrics": self.quality_metrics,
            "messages": [
                {
                    "text": m.message_text,
                    "session_id": m.session_id,
                    "user_intent": m.user_intent,
                    "parent_usecases": m.parent_usecases,
                    "similarity": float(m.best_similarity),
                    "matching_phrase": m.best_matching_phrase,
                    "cluster_title": m.cluster_title,
                    "borderline": m.was_borderline or m.status == AssignmentStatus.BORDERLINE,
                    "usecase_rescued": m.was_usecase_rescued,
                    "fn_rescued": m.was_fn_rescued,
                    "usecase_boosted": m.was_usecase_boosted,
                    "original_cluster": m.original_cluster_title,
                    "signals": {k: round(v, 4) for k, v in m.signal_scores.items()} if m.signal_scores else {},
                }
                for m in self.messages
            ],
            "session_ids": list(set(self.session_ids)),
            "session_metadata": self.metadata,
            "performance": self.performance,
        }
        return result


@dataclass
class QualityReport:
    """
    Quality assessment report for the clustering results.

    Attributes:
        total_messages: Total input messages
        assigned_messages: Messages assigned to themes
        miscellaneous_messages: Messages in miscellaneous bucket
        coverage_percent: Percentage of messages assigned (not misc)
        num_themes: Number of final themes
        redundant_themes_merged: Number of redundant themes merged
        tiny_themes_dissolved: Number of tiny themes dissolved
        borderline_reassigned: Number of borderline messages reassigned (when misc > 15%)
        coherence_scores: Per-theme coherence scores
        avg_coherence: Average coherence across themes
        silhouette_score: Overall silhouette score (if computed)
    """
    total_messages: int = 0
    assigned_messages: int = 0
    miscellaneous_messages: int = 0
    coverage_percent: float = 0.0
    num_themes: int = 0
    redundant_themes_merged: int = 0
    tiny_themes_dissolved: int = 0
    borderline_reassigned: int = 0
    misc_validated_by_gpt: int = 0
    fn_rescued: int = 0
    secondary_themes_discovered: int = 0
    coherence_scores: Dict[int, float] = field(default_factory=dict)
    avg_coherence: float = 0.0
    silhouette_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "total_messages": self.total_messages,
            "assigned_messages": self.assigned_messages,
            "miscellaneous_messages": self.miscellaneous_messages,
            "coverage_percent": round(self.coverage_percent, 2),
            "num_themes": self.num_themes,
            "redundant_themes_merged": self.redundant_themes_merged,
            "tiny_themes_dissolved": self.tiny_themes_dissolved,
            "borderline_reassigned": self.borderline_reassigned,
            "misc_validated_by_gpt": self.misc_validated_by_gpt,
            "fn_rescued": self.fn_rescued,
            "secondary_themes_discovered": self.secondary_themes_discovered,
            "coherence_scores": {str(k): round(v, 4) for k, v in self.coherence_scores.items()},
            "avg_coherence": round(self.avg_coherence, 4),
            "silhouette_score": round(self.silhouette_score, 4) if self.silhouette_score else None,
        }


@dataclass
class PipelineInput:
    """
    Input container for the theme clustering pipeline.

    Attributes:
        messages: List of message texts to cluster
        session_ids: Optional list of session IDs (parallel to messages)
        seed_themes: Optional list of seed theme names to include
        client_name: Name of the client (for output naming)
        date_range: Date range string (for output naming)
    """
    messages: List[str]
    session_ids: Optional[List[str]] = None
    seed_themes: Optional[List[str]] = None
    client_name: str = "default"
    date_range: str = ""

    def __post_init__(self):
        """Validate inputs"""
        if self.session_ids and len(self.session_ids) != len(self.messages):
            raise ValueError(
                f"session_ids length ({len(self.session_ids)}) must match "
                f"messages length ({len(self.messages)})"
            )


@dataclass
class PipelineOutput:
    """
    Output container from the theme clustering pipeline.

    Attributes:
        clusters: List of ClusterResult objects
        miscellaneous: ClusterResult for miscellaneous bucket
        quality_report: QualityReport with metrics
        themes: List of discovered Theme objects
    """
    clusters: List[ClusterResult] = field(default_factory=list)
    miscellaneous: Optional[ClusterResult] = None
    quality_report: Optional[QualityReport] = None
    themes: List[Theme] = field(default_factory=list)
    global_metadata: Dict[str, Any] = field(default_factory=dict)  # global medians + percentile thresholds (pre-sales)

    def to_export_dict(self) -> Dict[str, Any]:
        """Convert to export format compatible with existing systems"""
        clusters_list = [c.to_dict() for c in self.clusters]

        # Add miscellaneous as a special cluster
        if self.miscellaneous and self.miscellaneous.messages:
            misc_dict = self.miscellaneous.to_dict()
            misc_dict["cluster_title"] = "Miscellaneous"
            misc_dict["is_miscellaneous"] = True
            clusters_list.append(misc_dict)

        return {
            "clusters": clusters_list,
            "quality_report": self.quality_report.to_dict() if self.quality_report else {},
            "num_clusters": len(self.clusters),
            "total_messages": (
                self.quality_report.total_messages if self.quality_report else 0
            ),
            "global_metadata": self.global_metadata,
        }
