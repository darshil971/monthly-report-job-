"""
Theme Clustering Assignment Module (Phase 4)

Hybrid message-to-theme assignment using three weighted signals:
  1. Cosine Similarity  — semantic matching via embeddings
  2. BM25Plus           — lexical/keyword matching
  3. Fuzzy String Match — surface-form / typo tolerance (token_sort_ratio)

Signals are normalised to [0,1] and blended into a fusion score that drives
both theme ranking AND threshold decisions. Weights are configured in
AssignmentConfig (w_cosine, w_bm25, w_fuzzy).
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Plus
from rapidfuzz import fuzz

from .theme_config import AssignmentConfig, DEFAULT_CONFIG
from .theme_models import Theme, MessageAssignment, AssignmentStatus


class ThemeAssigner:
    """
    Assigns messages to themes using hybrid similarity (Cosine + BM25 + Fuzzy)
    with weighted score fusion.

    Assignment strategy:
    1. For each message, compute three signals against ALL key phrases in ALL themes
    2. Per-theme scores: max across phrases for each signal
    3. Normalise BM25 to [0,1] via sigmoid (avoids per-message min-max inflation)
    4. Fusion score = w_cosine*cos + w_bm25*norm_bm25 + w_fuzzy*fuzzy
    5. Fusion score drives both ranking and threshold decisions:
       - CONFIDENT  if fusion >= primary_threshold AND gap >= confidence_gap
       - BORDERLINE if fusion >= borderline_threshold
       - MISCELLANEOUS otherwise
    """

    def __init__(self, config: AssignmentConfig = None):
        self.config = config or DEFAULT_CONFIG.assignment

        # Populated once per pipeline run via build_bm25_index()
        self._bm25_index: Optional[BM25Plus] = None
        self._all_phrases: List[str] = []
        self._phrase_to_theme_idx: List[int] = []
        self._theme_list: List[Theme] = []

    # ------------------------------------------------------------------
    # One-time setup (called once before the message loop)
    # ------------------------------------------------------------------

    def build_bm25_index(self, themes: List[Theme]) -> None:
        """
        Build BM25Plus index from all key phrases across all themes.
        Must be called once after Phase 3.5 (merging) before assignment.

        Args:
            themes: Finalised list of themes with key_phrases populated
        """
        self._theme_list = themes

        self._all_phrases = []
        self._phrase_to_theme_idx = []

        for theme_idx, theme in enumerate(themes):
            for phrase in theme.key_phrases:
                self._all_phrases.append(phrase)
                self._phrase_to_theme_idx.append(theme_idx)

        # Tokenise phrases for BM25
        tokenised_corpus = [p.lower().split() for p in self._all_phrases]
        self._bm25_index = BM25Plus(tokenised_corpus)

        print(f"[ASSIGN] Built BM25Plus index: {len(self._all_phrases)} phrases "
              f"across {len(themes)} themes")

    # ------------------------------------------------------------------
    # Per-message scoring
    # ------------------------------------------------------------------

    def _compute_cosine_scores(
        self,
        message_embedding: np.ndarray,
        themes: List[Theme],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Signal 1: Cosine similarity between message and all theme phrases.

        Returns:
            cos_scores: array of shape (num_themes,) — max cosine sim per theme
            matching_phrases: list of best-matching phrase per theme
        """
        n_themes = len(themes)
        cos_scores = np.zeros(n_themes)
        matching_phrases = [""] * n_themes

        for idx, theme in enumerate(themes):
            if theme.phrase_embeddings is None or len(theme.phrase_embeddings) == 0:
                continue

            sims = cosine_similarity(
                message_embedding.reshape(1, -1),
                theme.phrase_embeddings
            )[0]

            best_phrase_idx = int(np.argmax(sims))
            cos_scores[idx] = float(sims[best_phrase_idx])
            if best_phrase_idx < len(theme.key_phrases):
                matching_phrases[idx] = theme.key_phrases[best_phrase_idx]

        return cos_scores, matching_phrases

    def _compute_bm25_scores(
        self,
        message_text: str,
        themes: List[Theme],
    ) -> np.ndarray:
        """
        Signal 2: BM25Plus scores for the message against all theme phrases.

        Returns:
            bm25_scores: array of shape (num_themes,) — max BM25 score per theme
        """
        n_themes = len(themes)
        bm25_scores = np.full(n_themes, -1e9)

        query_tokens = message_text.lower().split()
        if not query_tokens:
            return np.zeros(n_themes)

        # BM25Plus returns scores for all indexed phrases at once
        phrase_scores = self._bm25_index.get_scores(query_tokens)

        # Aggregate: max BM25 score per theme
        for phrase_idx, score in enumerate(phrase_scores):
            theme_idx = self._phrase_to_theme_idx[phrase_idx]
            if score > bm25_scores[theme_idx]:
                bm25_scores[theme_idx] = score

        # Replace -1e9 sentinel with 0 for themes that had no phrases
        bm25_scores[bm25_scores < -1e8] = 0.0

        return bm25_scores

    def _compute_fuzzy_scores(
        self,
        message_text: str,
        themes: List[Theme],
    ) -> np.ndarray:
        """
        Signal 3: Fuzzy string matching (token_sort_ratio) against all theme phrases.

        Returns:
            fuzzy_scores: array of shape (num_themes,) — max fuzzy score per theme, in [0, 1]
        """
        n_themes = len(themes)
        fuzzy_scores = np.zeros(n_themes)
        msg_lower = message_text.lower()

        for idx, theme in enumerate(themes):
            best_fuzzy = 0.0
            for phrase in theme.key_phrases:
                score = fuzz.token_sort_ratio(msg_lower, phrase.lower()) / 100.0
                if score > best_fuzzy:
                    best_fuzzy = score
            fuzzy_scores[idx] = best_fuzzy

        return fuzzy_scores

    @staticmethod
    def _normalise_bm25(bm25_scores: np.ndarray) -> np.ndarray:
        """
        Sigmoid normalisation of BM25 scores to [0, 1].

        Unlike min-max (where the top theme ALWAYS gets 1.0), sigmoid
        preserves the relative magnitude: a dominant match gets ~0.95
        while a mediocre match gets ~0.6. This prevents BM25 from being
        a binary "best match gets full bonus" signal.

        The shift and scale are tuned for typical BM25Plus score ranges
        where meaningful matches produce scores of 2-8 and noise is 0-1.

        Returns:
            normalised array of same shape, in [0, 1]
        """
        # sigmoid(x) = 1 / (1 + exp(-(x - center) / scale))
        # center=3: scores around 3 map to 0.5 (moderate match)
        # scale=2: controls steepness — scores 0→~0.18, 3→0.5, 6→~0.82, 10→~0.97
        center = 3.0
        scale = 2.0
        return 1.0 / (1.0 + np.exp(-(bm25_scores - center) / scale))

    def _compute_fusion_scores(
        self,
        cos_scores: np.ndarray,
        bm25_scores: np.ndarray,
        fuzzy_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted fusion of three normalised signals.

        Args:
            cos_scores:   shape (num_themes,) — already in [0,1]
            bm25_scores:  shape (num_themes,) — unbounded, will be normalised via sigmoid
            fuzzy_scores: shape (num_themes,) — already in [0,1]

        Returns:
            fusion_scores: shape (num_themes,) — weighted blend in [0,1], higher is better
            norm_bm25:     shape (num_themes,) — normalised BM25 scores (for metadata)
        """
        norm_bm25 = self._normalise_bm25(bm25_scores)

        w_cos = self.config.w_cosine
        w_bm25 = self.config.w_bm25
        w_fuzzy = self.config.w_fuzzy

        fusion_scores = (
            w_cos   * cos_scores +
            w_bm25  * norm_bm25 +
            w_fuzzy * fuzzy_scores
        )
        return fusion_scores, norm_bm25

    # ------------------------------------------------------------------
    # Assignment (per-message)
    # ------------------------------------------------------------------

    def assign_message(
        self,
        message_idx: int,
        message_text: str,
        message_embedding: np.ndarray,
        themes: List[Theme],
        session_id: Optional[str] = None,
        message_usecase: Optional[str] = None,
        theme_id_usecases: Optional[Dict[int, set]] = None,
        boost_amount: float = 0.0,
    ) -> MessageAssignment:
        """
        Assign a single message to the best matching theme using hybrid fusion.

        Args:
            message_idx: Index of message in original list
            message_text: Preprocessed/cleaned message text
            message_embedding: Message embedding (1024D)
            themes: List of Theme objects
            session_id: Optional session ID

        Returns:
            MessageAssignment object
        """
        # Ensure BM25 index is built for the current theme set
        # (handles second-pass calls where themes may have changed)
        if self._theme_list is not themes:
            self.build_bm25_index(themes)

        n_themes = len(themes)

        # Edge case: empty message
        if not message_text.strip():
            return MessageAssignment(
                message_idx=message_idx,
                message_text=message_text,
                session_id=session_id,
                assigned_theme_id=-1,
                best_similarity=0.0,
                confidence_gap=0.0,
                status=AssignmentStatus.MISCELLANEOUS,
                all_similarities={},
                best_matching_phrase=None,
            )

        # Edge case: no themes
        if n_themes == 0:
            return MessageAssignment(
                message_idx=message_idx,
                message_text=message_text,
                session_id=session_id,
                assigned_theme_id=-1,
                best_similarity=0.0,
                confidence_gap=0.0,
                status=AssignmentStatus.MISCELLANEOUS,
                all_similarities={},
                best_matching_phrase=None,
            )

        # --- Compute three signals ---
        cos_scores, matching_phrases = self._compute_cosine_scores(message_embedding, themes)
        bm25_scores = self._compute_bm25_scores(message_text, themes)
        fuzzy_scores = self._compute_fuzzy_scores(message_text, themes)

        # --- Weighted fusion ---
        fusion_scores, norm_bm25 = self._compute_fusion_scores(
            cos_scores, bm25_scores, fuzzy_scores
        )

        # --- Apply additive usecase boost (Phase 3.6 → Phase 4) ---
        boosted_scores = fusion_scores.copy()
        was_usecase_boosted = False
        original_cluster_title = None
        if message_usecase and theme_id_usecases and boost_amount > 0:
            pre_boost_best_idx = int(np.argmax(fusion_scores))
            for i, theme in enumerate(themes):
                if theme.theme_id in theme_id_usecases:
                    if message_usecase in theme_id_usecases[theme.theme_id]:
                        boosted_scores[i] = min(fusion_scores[i] + boost_amount, 1.0)
            post_boost_best_idx = int(np.argmax(boosted_scores))
            if post_boost_best_idx != pre_boost_best_idx:
                was_usecase_boosted = True
                original_cluster_title = themes[pre_boost_best_idx].theme_name

        # --- Rank by (boosted) fusion score ---
        best_idx = int(np.argmax(boosted_scores))

        if n_themes == 1:
            second_fusion = 0.0
        else:
            boosted_masked = boosted_scores.copy()
            boosted_masked[best_idx] = -np.inf
            second_idx = int(np.argmax(boosted_masked))
            second_fusion = float(boosted_scores[second_idx])

        # --- Decision logic uses boosted fusion score ---
        best_fusion = float(boosted_scores[best_idx])
        confidence_gap = best_fusion - second_fusion

        # Cosine score of the best theme (for provenance/metadata)
        best_cos = float(cos_scores[best_idx])
        best_phrase = matching_phrases[best_idx]
        best_theme_id = themes[best_idx].theme_id

        # Single theme edge case: treat gap as 1.0
        if n_themes == 1:
            confidence_gap = 1.0

        # Threshold decisions — fusion score drives both ranking and thresholds
        if best_fusion >= self.config.primary_threshold and confidence_gap >= self.config.confidence_gap:
            status = AssignmentStatus.CONFIDENT
            assigned_theme_id = best_theme_id
        elif best_fusion >= self.config.borderline_threshold:
            status = AssignmentStatus.BORDERLINE
            assigned_theme_id = -1
        else:
            status = AssignmentStatus.MISCELLANEOUS
            assigned_theme_id = -1

        # Build all_similarities dict (boosted fusion scores, keyed by theme_id)
        all_similarities = {
            themes[i].theme_id: float(boosted_scores[i]) for i in range(n_themes)
        }

        # Signal metadata for the best theme (for diagnostics/export)
        signal_detail = {
            "cosine": best_cos,
            "bm25_raw": float(bm25_scores[best_idx]),
            "bm25_norm": float(norm_bm25[best_idx]),
            "fuzzy": float(fuzzy_scores[best_idx]),
            "fusion": best_fusion,
        }

        assignment = MessageAssignment(
            message_idx=message_idx,
            message_text=message_text,
            session_id=session_id,
            assigned_theme_id=assigned_theme_id,
            best_similarity=best_fusion,
            confidence_gap=confidence_gap,
            status=status,
            all_similarities=all_similarities,
            best_matching_phrase=best_phrase,
            signal_scores=signal_detail,
        )
        if was_usecase_boosted:
            assignment.was_usecase_boosted = True
            assignment.original_cluster_title = original_cluster_title
        return assignment

    # ------------------------------------------------------------------
    # Batch assignment
    # ------------------------------------------------------------------

    def assign_messages(
        self,
        messages: List[str],
        message_embeddings: np.ndarray,
        themes: List[Theme],
        session_ids: Optional[List[str]] = None,
        original_indices: Optional[List[int]] = None,
        user_intents_map: Optional[Dict[int, str]] = None,
        theme_id_usecases: Optional[Dict[int, set]] = None,
        boost_amount: float = 0.0,
    ) -> Tuple[List[MessageAssignment], Dict[str, int]]:
        """
        Assign all messages to themes using hybrid weighted fusion.

        Args:
            messages: List of preprocessed message texts
            message_embeddings: Message embedding matrix (n_messages, dim)
            themes: List of Theme objects with embeddings
            session_ids: Optional list of session IDs
            original_indices: Optional list of original indices (before preprocessing filtering/dedup)

        Returns:
            Tuple of (assignments, stats)
            - assignments: List of MessageAssignment objects
            - stats: Dict with assignment statistics
        """
        n = len(messages)
        w_cos = self.config.w_cosine
        w_bm25 = self.config.w_bm25
        w_fuzzy = self.config.w_fuzzy

        print(f"[ASSIGN] Assigning {n} messages to {len(themes)} themes (hybrid weighted fusion)...")
        print(f"[ASSIGN] Signals: Cosine ({w_cos}) + BM25Plus ({w_bm25}) + Fuzzy ({w_fuzzy})")
        print(f"[ASSIGN] Thresholds: primary={self.config.primary_threshold}, "
              f"gap={self.config.confidence_gap}, borderline={self.config.borderline_threshold}")

        # Build BM25 index once before the message loop
        self.build_bm25_index(themes)

        if session_ids is None:
            session_ids = [None] * n

        if original_indices is None:
            original_indices = list(range(n))

        assignments = []
        stats = {
            "confident": 0,
            "borderline": 0,
            "miscellaneous": 0,
            "usecase_boosted": 0,
        }

        for i in range(n):
            # Use original_idx for tracking (maps back to original user_intents list)
            original_idx = original_indices[i] if i < len(original_indices) else i
            msg_usecase = user_intents_map.get(original_idx) if user_intents_map else None
            assignment = self.assign_message(
                message_idx=original_idx,
                message_text=messages[i],
                message_embedding=message_embeddings[i],
                themes=themes,
                session_id=session_ids[i],
                message_usecase=msg_usecase,
                theme_id_usecases=theme_id_usecases,
                boost_amount=boost_amount,
            )

            assignments.append(assignment)

            # Update stats
            if assignment.status == AssignmentStatus.CONFIDENT:
                stats["confident"] += 1
            elif assignment.status == AssignmentStatus.BORDERLINE:
                stats["borderline"] += 1
            else:
                stats["miscellaneous"] += 1
            if assignment.was_usecase_boosted:
                stats["usecase_boosted"] += 1

            # Progress logging
            if (i + 1) % 500 == 0 or i == n - 1:
                print(f"[ASSIGN] Processed {i + 1}/{n} messages")

        # Log stats
        print(f"[ASSIGN] Results:")
        print(f"  - Confident: {stats['confident']} ({100 * stats['confident'] / n:.1f}%)")
        print(f"  - Borderline: {stats['borderline']} ({100 * stats['borderline'] / n:.1f}%)")
        print(f"  - Miscellaneous: {stats['miscellaneous']} ({100 * stats['miscellaneous'] / n:.1f}%)")
        if stats["usecase_boosted"]:
            print(f"  - Usecase-boosted (cluster changed by 🏷️ boost): {stats['usecase_boosted']}")

        return assignments, stats

    def get_assignments_by_theme(
        self,
        assignments: List[MessageAssignment],
        themes: List[Theme],
    ) -> Dict[int, List[MessageAssignment]]:
        """
        Group assignments by theme ID.

        Args:
            assignments: List of MessageAssignment objects
            themes: List of Theme objects

        Returns:
            Dict mapping theme_id -> list of assignments
        """
        by_theme = {theme.theme_id: [] for theme in themes}
        by_theme[-1] = []  # Miscellaneous

        for assignment in assignments:
            tid = assignment.assigned_theme_id
            if tid in by_theme:
                by_theme[tid].append(assignment)
            else:
                by_theme[-1].append(assignment)

        return by_theme


def assign_messages(
    messages: List[str],
    message_embeddings: np.ndarray,
    themes: List[Theme],
    session_ids: Optional[List[str]] = None,
    config: AssignmentConfig = None,
    original_indices: Optional[List[int]] = None,
    user_intents_map: Optional[Dict[int, str]] = None,
    theme_id_usecases: Optional[Dict[int, set]] = None,
    boost_amount: float = 0.0,
) -> Tuple[List[MessageAssignment], Dict[str, int]]:
    """
    Convenience function for message assignment.

    Args:
        messages: List of message texts
        message_embeddings: Message embedding matrix
        themes: List of Theme objects
        session_ids: Optional session IDs
        config: Optional assignment config
        original_indices: Optional original indices before preprocessing
        user_intents_map: Optional dict mapping original_idx -> usecase string
        theme_id_usecases: Optional dict mapping theme_id -> set of usecases
        boost_amount: Additive boost to apply to usecase-matching clusters

    Returns:
        Tuple of (assignments, stats)
    """
    assigner = ThemeAssigner(config)
    return assigner.assign_messages(
        messages, message_embeddings, themes, session_ids, original_indices,
        user_intents_map=user_intents_map,
        theme_id_usecases=theme_id_usecases,
        boost_amount=boost_amount,
    )
