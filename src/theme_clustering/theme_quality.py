"""
Theme Clustering Quality Module (Phase 5)

Quality assurance including:
- Coherence validation (CV, IQR-based outlier detection)
- Redundancy detection and theme merging
- Tiny theme dissolution
- Silhouette analysis

Uses logic from intent_expansion/coherence.py and uf_option_generator.py.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score

from .theme_config import QualityConfig, DEFAULT_CONFIG
from .theme_models import (
    Theme, MessageAssignment, ClusterResult, QualityReport, AssignmentStatus
)


class QualityAssessor:
    """
    Quality assessment and refinement for theme clusters.

    Performs:
    1. Tiny theme removal (< min_messages_per_theme)
    2. Redundancy detection (centroid similarity > threshold)
    3. Coherence validation (CV, IQR, silhouette)
    4. Final quality metrics
    """

    def __init__(self, config: QualityConfig = None):
        self.config = config or DEFAULT_CONFIG.quality

    # =========================================================================
    # COHERENCE METRICS (from intent_expansion/coherence.py)
    # =========================================================================

    def compute_cluster_coherence_score(
        self,
        cluster_embeddings: np.ndarray,
    ) -> float:
        """
        Compute custom coherence score for a cluster.

        Formula:
        - avg_similarity = average pairwise cosine similarity
        - similarity_std = std of upper-triangle similarities
        - coherence = avg_similarity * (1 - similarity_std)

        Args:
            cluster_embeddings: Embeddings for cluster (n_messages, dim)

        Returns:
            Coherence score (0-1, higher is better)
        """
        if len(cluster_embeddings) <= 1:
            return 1.0

        # Compute pairwise similarities
        similarities = cosine_similarity(cluster_embeddings)

        # Get upper triangle (excluding diagonal)
        n = len(cluster_embeddings)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarities[upper_tri_indices]

        if len(pairwise_sims) == 0:
            return 1.0

        # Compute metrics
        avg_similarity = np.mean(pairwise_sims)
        similarity_std = np.std(pairwise_sims)

        # Coherence score
        coherence = avg_similarity * (1 - similarity_std)

        return float(coherence)

    def assess_dispersion_with_iqr(
        self,
        embeddings: np.ndarray,
        centroid: np.ndarray,
    ) -> Dict:
        """
        Assess dispersion using Coefficient of Variation + IQR outlier detection.

        From intent_expansion/coherence.py:
        - cv = std(distances) / mean(distances)
        - q1, q3 = percentiles 25 & 75
        - iqr = q3 - q1
        - outlier_threshold = q3 + iqr_multiplier * iqr
        - orphan_indices = distances > outlier_threshold

        Args:
            embeddings: Message embeddings (n_messages, dim)
            centroid: Centroid vector (dim,)

        Returns:
            Dict with dispersion metrics
        """
        # Compute cosine distances to centroid
        similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        distances = 1 - similarities

        # Coefficient of Variation
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        cv = std_dist / mean_dist if mean_dist > 0 else 0.0

        # IQR-based outlier detection
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        outlier_threshold = q3 + self.config.iqr_multiplier * iqr

        # Identify outliers
        outlier_mask = distances > outlier_threshold
        outlier_indices = np.where(outlier_mask)[0]
        outlier_count = len(outlier_indices)
        outlier_fraction = outlier_count / len(embeddings) if len(embeddings) > 0 else 0

        # Determine if coherent
        is_coherent = cv < self.config.cv_coherence_threshold

        return {
            "is_coherent": bool(is_coherent),
            "cv": float(cv),
            "outlier_indices": outlier_indices.tolist(),
            "outlier_count": int(outlier_count),
            "outlier_fraction": float(outlier_fraction),
            "distance_stats": {
                "mean": float(mean_dist),
                "std": float(std_dist),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "outlier_threshold": float(outlier_threshold),
            }
        }

    def compute_silhouette_analysis(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """
        Perform silhouette analysis for cluster separation validation.

        Args:
            embeddings: Message embeddings
            labels: Cluster labels (-1 for miscellaneous)

        Returns:
            Dict with silhouette metrics
        """
        # Only compute for non-misc points
        valid_mask = labels >= 0
        if np.sum(valid_mask) < 2:
            return {"error": "Insufficient points for silhouette analysis"}

        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]

        # Need at least 2 clusters
        unique_labels = np.unique(valid_labels)
        if len(unique_labels) < 2:
            return {"error": "Need at least 2 clusters for silhouette analysis"}

        try:
            silhouette_vals = silhouette_samples(valid_embeddings, valid_labels, metric='cosine')
            overall_silhouette = silhouette_score(valid_embeddings, valid_labels, metric='cosine')
        except Exception as e:
            return {"error": str(e)}

        # Per-cluster analysis
        results = {}
        valid_clusters = []
        invalid_clusters = []

        for label in unique_labels:
            mask = valid_labels == label
            cluster_silhouettes = silhouette_vals[mask]

            positive_count = np.sum(cluster_silhouettes > 0)
            positive_fraction = positive_count / len(cluster_silhouettes)
            avg_silhouette = np.mean(cluster_silhouettes)

            is_valid = (
                positive_fraction >= self.config.silhouette_positive_threshold and
                avg_silhouette >= self.config.min_avg_silhouette
            )

            results[int(label)] = {
                "avg_silhouette": float(avg_silhouette),
                "positive_fraction": float(positive_fraction),
                "is_valid": bool(is_valid),
            }

            if is_valid:
                valid_clusters.append(int(label))
            else:
                invalid_clusters.append(int(label))

        return {
            "overall_silhouette": float(overall_silhouette),
            "per_cluster": results,
            "valid_clusters": valid_clusters,
            "invalid_clusters": invalid_clusters,
        }

    # =========================================================================
    # REDUNDANCY DETECTION & MERGING (from uf_option_generator.py approach)
    # =========================================================================

    def detect_redundant_themes(
        self,
        themes: List[Theme],
    ) -> List[Tuple[int, int]]:
        """
        Detect redundant themes using GPT-based semantic understanding.

        Process:
        1. Find similar theme pairs using centroid similarity
        2. For each pair, ask GPT if they should merge
        3. Only merge if GPT decides they represent the same intent

        Args:
            themes: List of Theme objects with theme_embedding

        Returns:
            List of (theme_id1, theme_id2) pairs to merge
        """
        if len(themes) < 2:
            return []

        print(f"[QUALITY] Detecting redundant themes among {len(themes)} themes...")

        # Build centroid matrix
        centroids = []
        theme_ids = []
        theme_lookup = {}

        for theme in themes:
            if theme.theme_embedding is not None:
                centroids.append(theme.theme_embedding)
                theme_ids.append(theme.theme_id)
                theme_lookup[theme.theme_id] = theme

        if len(centroids) < 2:
            return []

        centroid_matrix = np.array(centroids)

        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(centroid_matrix)

        # Find candidate pairs above threshold (using lower threshold for GPT evaluation)
        # Use 0.65 instead of 0.85 to catch more potential merges for GPT to evaluate
        candidate_threshold = 0.65
        print(f"[QUALITY] Candidate similarity threshold: {candidate_threshold:.4f}")

        # Find pairs above threshold
        candidate_pairs = []
        for i in range(len(theme_ids)):
            for j in range(i + 1, len(theme_ids)):
                sim = similarity_matrix[i, j]
                if sim >= candidate_threshold:
                    tid1, tid2 = theme_ids[i], theme_ids[j]
                    t1 = theme_lookup[tid1]
                    t2 = theme_lookup[tid2]

                    candidate_pairs.append({
                        'theme_a': {
                            'id': tid1,
                            'name': t1.theme_name,
                            'description': t1.description,
                            'key_phrases': t1.key_phrases,
                        },
                        'theme_b': {
                            'id': tid2,
                            'name': t2.theme_name,
                            'description': t2.description,
                            'key_phrases': t2.key_phrases,
                        },
                        'similarity': float(sim),
                    })

        if not candidate_pairs:
            print(f"[QUALITY] No similar theme pairs found above threshold")
            return []

        print(f"[QUALITY] Found {len(candidate_pairs)} candidate pairs for GPT evaluation")

        # Ask GPT to decide which pairs should merge
        merge_pairs = self._gpt_merge_decision(candidate_pairs, themes)

        return merge_pairs

    def _gpt_merge_decision(
        self,
        candidate_pairs: List[dict],
        themes: List[Theme],
    ) -> List[Tuple[int, int]]:
        """
        Use GPT to decide which theme pairs should be merged.

        Uses get_cluster_merge_decision_prompt which formats each pair with
        full details and returns per-pair merge/separate decisions.

        Args:
            candidate_pairs: List of candidate pair dicts with theme info
            themes: Full list of themes (for context)

        Returns:
            List of (theme_id1, theme_id2) pairs that should be merged
        """
        from .theme_prompts import get_cluster_merge_decision_prompt

        # Import GPT call function
        from src.utils.openai_utils import gpt_5_2_chat, GPT4Input

        print(f"[QUALITY] Calling GPT to evaluate {len(candidate_pairs)} candidate pairs...")

        # Convert candidate_pairs to the format expected by get_cluster_merge_decision_prompt
        theme_pairs = [
            {
                'theme_a': {
                    'name': pair['theme_a']['name'],
                    'description': pair['theme_a']['description'],
                    'key_phrases': pair['theme_a']['key_phrases'],
                },
                'theme_b': {
                    'name': pair['theme_b']['name'],
                    'description': pair['theme_b']['description'],
                    'key_phrases': pair['theme_b']['key_phrases'],
                },
                'similarity': pair['similarity'],
            }
            for pair in candidate_pairs
        ]

        prompt = get_cluster_merge_decision_prompt(theme_pairs=theme_pairs)

        gpt4_inputs = [
            GPT4Input(actor="system", text=prompt),
            GPT4Input(actor="user", text="Analyze these theme pairs and decide which should be merged.")
        ]

        response = gpt_5_2_chat(gpt4_inputs, temperature=0.3, max_tokens=2000, timeout=120)

        # Extract content from LLMResponse
        if response and hasattr(response, 'content'):
            response = response.content

        if not response:
            print(f"[QUALITY] Warning: GPT merge decision failed, using no merges")
            return []

        # Parse GPT response — get_cluster_merge_decision_prompt returns a JSON array:
        # [{"pair_number": 1, "decision": "merge"|"separate", "reason": "..."}]
        merge_pairs = []
        try:
            import json
            import re

            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            parsed = json.loads(cleaned)

            # Expect a JSON array of per-pair decisions
            decisions = parsed if isinstance(parsed, list) else parsed.get("decisions", [])

            for decision in decisions:
                pair_num = decision.get('pair_number', 0)
                action = decision.get('decision', '').lower()
                reason = decision.get('reason', '')

                # pair_number is 1-indexed, candidate_pairs is 0-indexed
                pair_idx = pair_num - 1
                if pair_idx < 0 or pair_idx >= len(candidate_pairs):
                    continue

                pair = candidate_pairs[pair_idx]
                name_a = pair['theme_a']['name']
                name_b = pair['theme_b']['name']

                if action == 'merge':
                    tid1 = pair['theme_a']['id']
                    tid2 = pair['theme_b']['id']
                    merge_pairs.append((tid1, tid2))
                    print(f"[QUALITY] GPT Decision: MERGE '{name_a}' <-> '{name_b}'")
                    print(f"[QUALITY]   Reason: {reason}")
                elif action == 'separate':
                    print(f"[QUALITY] GPT Decision: KEEP SEPARATE '{name_a}' <-> '{name_b}'")
                    print(f"[QUALITY]   Reason: {reason}")

        except Exception as e:
            print(f"[QUALITY] Warning: Failed to parse GPT merge decisions: {e}")
            print(f"[QUALITY] Response preview: {response[:300]}...")

        print(f"[QUALITY] GPT approved {len(merge_pairs)} merges out of {len(candidate_pairs)} candidates")

        return merge_pairs

    def merge_themes(
        self,
        themes: List[Theme],
        merge_pairs: List[Tuple[int, int]],
        assignments: Optional[List[MessageAssignment]] = None,
    ) -> Tuple[List[Theme], Optional[List[MessageAssignment]], int]:
        """
        Merge redundant themes.

        Strategy:
        - If assignments provided: Keep theme with more messages, update assignments
        - If no assignments (pre-assignment phase): Keep theme with lower ID (arbitrary but deterministic)

        Args:
            themes: List of Theme objects
            merge_pairs: List of (theme_id1, theme_id2) pairs
            assignments: Optional list of MessageAssignment objects (None if called before assignment)

        Returns:
            Tuple of (merged_themes, updated_assignments_or_None, n_merged)
        """
        if not merge_pairs:
            return themes, assignments, 0

        print(f"[QUALITY] Merging {len(merge_pairs)} redundant theme pairs...")

        # Build merge map using union-find
        merge_map = {}  # Maps theme_id -> canonical theme_id

        # Get message counts per theme (if assignments available)
        theme_message_counts = defaultdict(int)
        if assignments:
            for assignment in assignments:
                if assignment.assigned_theme_id >= 0:
                    theme_message_counts[assignment.assigned_theme_id] += 1

        # Process pairs
        for id1, id2 in merge_pairs:
            # Find roots
            root1 = id1
            while root1 in merge_map:
                root1 = merge_map[root1]
            root2 = id2
            while root2 in merge_map:
                root2 = merge_map[root2]

            if root1 != root2:
                if assignments:
                    # Merge smaller into larger based on message counts
                    count1 = theme_message_counts.get(root1, 0)
                    count2 = theme_message_counts.get(root2, 0)

                    if count1 >= count2:
                        merge_map[root2] = root1
                    else:
                        merge_map[root1] = root2
                else:
                    # No assignments yet - merge higher ID into lower ID (arbitrary but deterministic)
                    if root1 < root2:
                        merge_map[root2] = root1
                    else:
                        merge_map[root1] = root2

        # Get final merge map
        final_merge_map = {}
        for tid in set(list(merge_map.keys()) + [t.theme_id for t in themes]):
            root = tid
            while root in merge_map:
                root = merge_map[root]
            if root != tid:
                final_merge_map[tid] = root

        # Merge themes
        theme_by_id = {t.theme_id: t for t in themes}
        canonical_themes = {}

        for theme in themes:
            if theme.theme_id in final_merge_map:
                # This theme is being merged
                canonical_id = final_merge_map[theme.theme_id]
                if canonical_id in canonical_themes:
                    # Combine key phrases
                    existing = canonical_themes[canonical_id]
                    existing_phrases = set(existing.key_phrases)
                    for phrase in theme.key_phrases:
                        if phrase not in existing_phrases:
                            existing.key_phrases.append(phrase)
            else:
                canonical_themes[theme.theme_id] = theme

        # Update assignments if provided
        updated_assignments = None
        if assignments:
            updated_assignments = []
            for assignment in assignments:
                if assignment.assigned_theme_id in final_merge_map:
                    assignment.assigned_theme_id = final_merge_map[assignment.assigned_theme_id]
                updated_assignments.append(assignment)

        merged_themes = list(canonical_themes.values())
        n_merged = len(themes) - len(merged_themes)

        print(f"[QUALITY] Merged {n_merged} themes, {len(merged_themes)} remain")

        return merged_themes, updated_assignments, n_merged

    # =========================================================================
    # BORDERLINE MESSAGE REASSIGNMENT
    # =========================================================================

    def reassign_borderline_messages(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
    ) -> Tuple[List[MessageAssignment], int]:
        """
        Reassign BORDERLINE messages when misc is high.

        If miscellaneous rate > 15%, reassign BORDERLINE messages to their best theme.
        BORDERLINE = messages with similarity >= borderline_threshold but didn't meet
        confidence gap requirement during initial assignment.

        MISCELLANEOUS messages are NOT reassigned - they genuinely don't match any theme.

        Args:
            themes: List of Theme objects
            assignments: List of MessageAssignment objects

        Returns:
            Tuple of (updated_assignments, n_reassigned)
        """
        # Count miscellaneous messages
        misc_count = sum(1 for a in assignments if a.assigned_theme_id < 0)
        misc_percent = 100 * misc_count / len(assignments) if assignments else 0

        print(f"[QUALITY] Miscellaneous: {misc_count}/{len(assignments)} ({misc_percent:.1f}%)")

        # Only reassign if misc > 15%
        if misc_percent <= 15:
            print(f"[QUALITY] Miscellaneous rate is acceptable, skipping reassignment")
            return assignments, 0

        print(f"[QUALITY] Miscellaneous rate is HIGH, reassigning BORDERLINE messages...")

        # Find BORDERLINE messages only
        borderline_count = sum(1 for a in assignments if a.status == AssignmentStatus.BORDERLINE)

        print(f"[QUALITY] Found {borderline_count} borderline messages eligible for reassignment")

        if borderline_count == 0:
            print(f"[QUALITY] No borderline messages to reassign")
            return assignments, 0

        # Build theme_id lookup
        theme_id_set = {t.theme_id for t in themes}

        # Reassign BORDERLINE messages only
        n_reassigned = 0
        updated_assignments = []

        for assignment in assignments:
            # Only reassign BORDERLINE status
            if assignment.status == AssignmentStatus.BORDERLINE:
                # Find best theme from all_similarities
                if assignment.all_similarities:
                    # Sort by similarity, pick best valid theme
                    sorted_themes = sorted(
                        assignment.all_similarities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    for theme_id, similarity in sorted_themes:
                        if theme_id in theme_id_set:
                            # Reassign to this theme
                            assignment.assigned_theme_id = theme_id
                            assignment.status = AssignmentStatus.CONFIDENT
                            n_reassigned += 1
                            break

            updated_assignments.append(assignment)

        print(f"[QUALITY] Reassigned {n_reassigned} messages to themes")

        return updated_assignments, n_reassigned

    # =========================================================================
    # TINY THEME DISSOLUTION
    # =========================================================================

    def dissolve_tiny_themes(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
    ) -> Tuple[List[Theme], List[MessageAssignment], int]:
        """
        Dissolve themes with too few messages.

        Messages from tiny themes go to miscellaneous.

        Args:
            themes: List of Theme objects
            assignments: List of MessageAssignment objects

        Returns:
            Tuple of (remaining_themes, updated_assignments, n_dissolved)
        """
        print(f"[QUALITY] Checking for tiny themes (< {self.config.min_messages_per_theme} messages)...")

        # Count messages per theme
        theme_counts = defaultdict(int)
        for assignment in assignments:
            if assignment.assigned_theme_id >= 0:
                theme_counts[assignment.assigned_theme_id] += 1

        # Identify tiny themes
        tiny_theme_ids = set()
        for theme in themes:
            count = theme_counts.get(theme.theme_id, 0)
            if count < self.config.min_messages_per_theme:
                tiny_theme_ids.add(theme.theme_id)
                print(f"[QUALITY] Tiny theme: '{theme.theme_name}' ({count} messages) -> dissolving")

        if not tiny_theme_ids:
            print(f"[QUALITY] No tiny themes found")
            return themes, assignments, 0

        # Update assignments - mark tiny theme messages as BORDERLINE so
        # Stage 1 borderline reassignment can reroute them to other themes
        updated_assignments = []
        for assignment in assignments:
            if assignment.assigned_theme_id in tiny_theme_ids:
                assignment.assigned_theme_id = -1
                assignment.status = AssignmentStatus.BORDERLINE
            updated_assignments.append(assignment)

        # Remove tiny themes
        remaining_themes = [t for t in themes if t.theme_id not in tiny_theme_ids]

        print(f"[QUALITY] Dissolved {len(tiny_theme_ids)} tiny themes")

        return remaining_themes, updated_assignments, len(tiny_theme_ids)

    # =========================================================================
    # MAIN QUALITY ASSESSMENT
    # =========================================================================

    def assess_quality(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
        message_embeddings: np.ndarray,
        n_merged_pre_assignment: int = 0,
    ) -> Tuple[List[Theme], List[MessageAssignment], QualityReport]:
        """
        Run quality assessment and refinement (post-assignment).

        Steps:
        1. Dissolve tiny themes (< min_messages_per_theme)
        2. Reassign borderline messages if misc is high
        3. Compute coherence metrics
        4. Compute silhouette analysis
        5. Generate quality report

        NOTE: Redundant theme merging happens BEFORE assignment in Phase 3.5,
        so it's not included here.

        Args:
            themes: List of Theme objects (already merged in Phase 3.5)
            assignments: List of MessageAssignment objects
            message_embeddings: Message embedding matrix
            n_merged_pre_assignment: Number of themes merged in Phase 3.5 (for reporting)

        Returns:
            Tuple of (refined_themes, updated_assignments, quality_report)
        """
        print(f"\n[QUALITY] Starting quality assessment...")
        print(f"[QUALITY] Input: {len(themes)} themes, {len(assignments)} assignments")

        report = QualityReport(total_messages=len(assignments))

        # Step 1: Dissolve tiny themes
        themes, assignments, n_dissolved = self.dissolve_tiny_themes(themes, assignments)
        report.tiny_themes_dissolved = n_dissolved
        report.redundant_themes_merged = n_merged_pre_assignment  # From Phase 3.5

        # Step 2: Borderline reassignment moved to Phase 6 (SecondPassRunner Stage 1)
        report.borderline_reassigned = 0  # Updated by second pass

        # Step 3: Compute coherence for each theme
        print(f"\n[QUALITY] Computing coherence scores...")

        assignments_by_theme = defaultdict(list)
        for i, assignment in enumerate(assignments):
            assignments_by_theme[assignment.assigned_theme_id].append(i)

        coherence_scores = {}
        for theme in themes:
            indices = assignments_by_theme.get(theme.theme_id, [])
            if len(indices) >= 2:
                cluster_embeddings = message_embeddings[indices]
                coherence = self.compute_cluster_coherence_score(cluster_embeddings)
                coherence_scores[theme.theme_id] = coherence
                theme.coherence_score = coherence
                theme.message_count = len(indices)

                status = "VALID" if coherence >= self.config.min_coherence_score else "LOW"
                print(f"[QUALITY]   {theme.theme_name}: coherence={coherence:.4f} [{status}]")
            else:
                coherence_scores[theme.theme_id] = 1.0  # Single message = perfect coherence
                theme.coherence_score = 1.0
                theme.message_count = len(indices)

        report.coherence_scores = coherence_scores
        report.avg_coherence = np.mean(list(coherence_scores.values())) if coherence_scores else 0.0

        # Step 4: Silhouette analysis
        labels = np.array([a.assigned_theme_id for a in assignments])
        silhouette_result = self.compute_silhouette_analysis(message_embeddings, labels)

        if "overall_silhouette" in silhouette_result:
            report.silhouette_score = silhouette_result["overall_silhouette"]
            print(f"[QUALITY] Overall silhouette score: {report.silhouette_score:.4f}")

        # Step 5: Compute final stats
        confident_count = sum(1 for a in assignments if a.status == AssignmentStatus.CONFIDENT)
        misc_count = sum(1 for a in assignments if a.assigned_theme_id < 0)

        report.assigned_messages = confident_count
        report.miscellaneous_messages = misc_count
        report.coverage_percent = 100 * confident_count / len(assignments) if assignments else 0
        report.num_themes = len(themes)

        print(f"\n[QUALITY] Final Report:")
        print(f"  - Themes: {report.num_themes}")
        print(f"  - Coverage: {report.coverage_percent:.1f}%")
        print(f"  - Miscellaneous: {report.miscellaneous_messages} ({100 * misc_count / len(assignments):.1f}%)")
        print(f"  - Avg coherence: {report.avg_coherence:.4f}")
        if report.silhouette_score:
            print(f"  - Silhouette: {report.silhouette_score:.4f}")

        return themes, assignments, report


def assess_quality(
    themes: List[Theme],
    assignments: List[MessageAssignment],
    message_embeddings: np.ndarray,
    config: QualityConfig = None,
    n_merged_pre_assignment: int = 0,
) -> Tuple[List[Theme], List[MessageAssignment], QualityReport]:
    """
    Convenience function for quality assessment.

    Args:
        themes: List of Theme objects
        assignments: List of MessageAssignment objects
        message_embeddings: Message embedding matrix
        config: Optional quality config
        n_merged_pre_assignment: Number of themes merged before assignment (for reporting)

    Returns:
        Tuple of (refined_themes, updated_assignments, quality_report)
    """
    assessor = QualityAssessor(config)
    return assessor.assess_quality(themes, assignments, message_embeddings, n_merged_pre_assignment)
