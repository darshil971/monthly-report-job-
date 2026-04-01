"""
Theme Clustering Second Pass Module (Phase 6) — 3-Stage Miscellaneous Rescue

Stage 1: Borderline Reassignment (always runs, no GPT)
    - Reassign BORDERLINE messages to their best-matching theme
    - Flag with was_borderline=True for provenance tracking

Stage 2: GPT Validation of Misc Messages (conditional)
    - Batch misc messages and ask GPT to validate their closest phrase match
    - VALID messages get assigned to their best-matching theme
    - INVALID messages pass to Stage 3
    - Logs rescued messages to stage2_validated_messages.txt

Stage 3: Blind Theme Discovery + Merge (conditional)
    - Run the FULL discovery prompt on remaining misc (same quality as 1st pass)
    - Embed new themes, merge duplicates against existing via embedding + GPT
    - Assign remaining misc to surviving new themes
    - Dissolve tiny new themes
"""

import json
import os
import re
import time
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .theme_config import ThemeClusteringConfig, DEFAULT_CONFIG
from .theme_models import Theme, MessageAssignment, AssignmentStatus
from .theme_prompts import get_misc_validation_prompt, get_cluster_merge_decision_prompt

# Import GPT utilities
import sys
# sys.path removed - using package imports
from src.utils.openai_utils import gpt_5_2_chat, gpt_5_nano, GPT4Input


class SecondPassRunner:
    """
    3-Stage second pass on miscellaneous messages.

    Stage 1: Borderline reassignment (free, always runs)
    Stage 2: GPT validation of closest matches (moderate cost, conditional)
    Stage 3: Blind discovery + merge (expensive, conditional)
    """

    def __init__(
        self,
        config: ThemeClusteringConfig,
        discoverer,       # ThemeDiscovery
        embedder,         # ThemeEmbedder
        assigner,         # ThemeAssigner
        sampler=None,     # AdaptiveSampler (for Stage 3 HDBSCAN)
    ):
        self.config = config
        self.discoverer = discoverer
        self.embedder = embedder
        self.assigner = assigner
        self.sampler = sampler
        self.sp_config = config.second_pass

    # =========================================================================
    # STAGE 1: BORDERLINE REASSIGNMENT
    # =========================================================================

    def reassign_borderline(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
    ) -> Tuple[List[MessageAssignment], int]:
        """
        Reassign BORDERLINE messages to their best-matching theme.

        Only reassigns messages whose best fusion score exceeds
        borderline_reassignment_threshold (config-driven). No GPT needed.

        Args:
            themes: Current theme list
            assignments: Current assignments

        Returns:
            Tuple of (updated_assignments, n_reassigned)
        """
        threshold = self.sp_config.borderline_reassignment_threshold
        print(f"\n[PASS2-S1] Stage 1: Borderline Reassignment (threshold={threshold})")

        theme_id_set = {t.theme_id for t in themes}
        n_reassigned = 0
        n_skipped = 0

        for assignment in assignments:
            if assignment.status != AssignmentStatus.BORDERLINE:
                continue

            if not assignment.all_similarities:
                continue

            # Skip if best fusion score is below threshold
            if assignment.best_similarity <= threshold:
                n_skipped += 1
                continue

            # Find best valid theme
            sorted_themes = sorted(
                assignment.all_similarities.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for theme_id, similarity in sorted_themes:
                if theme_id in theme_id_set:
                    assignment.assigned_theme_id = theme_id
                    assignment.status = AssignmentStatus.CONFIDENT
                    assignment.was_borderline = True
                    n_reassigned += 1
                    break

        print(f"[PASS2-S1] Reassigned {n_reassigned} borderline messages "
              f"(skipped {n_skipped} below threshold {threshold})")
        return assignments, n_reassigned

    # =========================================================================
    # STAGE 2: GPT VALIDATION OF MISC MESSAGES
    # =========================================================================

    def validate_misc_messages(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
        output_dir: str = None,
    ) -> Tuple[List[MessageAssignment], int]:
        """
        Use GPT to validate whether misc messages' best phrase matches are correct.

        Batches messages at validation_batch_size, asks GPT to confirm/reject
        each match. Logs rescued messages to a text file for auditing.

        Args:
            themes: Current theme list
            assignments: Current assignments
            output_dir: Directory to save validation log

        Returns:
            Tuple of (updated_assignments, n_validated)
        """
        print(f"\n[PASS2-S2] Stage 2: GPT Validation of Misc Messages")

        # Build theme lookup
        theme_lookup = {t.theme_id: t for t in themes}

        # Collect misc messages with their best match info
        misc_entries = []
        for assignment in assignments:
            if assignment.assigned_theme_id >= 0:
                continue  # Already assigned

            # Find best theme from all_similarities
            if not assignment.all_similarities:
                continue

            best_theme_id = max(
                assignment.all_similarities,
                key=assignment.all_similarities.get,
            )
            best_sim = assignment.all_similarities[best_theme_id]

            if best_theme_id not in theme_lookup:
                continue

            misc_entries.append({
                "assignment": assignment,
                "best_theme_id": best_theme_id,
                "best_sim": best_sim,
                "best_phrase": assignment.best_matching_phrase or "",
                "theme": theme_lookup[best_theme_id],
            })

        if not misc_entries:
            print(f"[PASS2-S2] No misc messages to validate")
            return assignments, 0

        # Split by minimum similarity threshold
        min_sim = self.sp_config.min_validation_similarity
        validatable = [e for e in misc_entries if e["best_sim"] >= min_sim]
        skipped = len(misc_entries) - len(validatable)

        print(f"[PASS2-S2] Misc messages: {len(misc_entries)} total")
        print(f"[PASS2-S2] Validatable (sim >= {min_sim}): {len(validatable)}")
        print(f"[PASS2-S2] Skipped (sim < {min_sim}): {skipped}")

        if not validatable:
            print(f"[PASS2-S2] No messages above similarity threshold, skipping GPT validation")
            return assignments, 0

        # Batch and validate
        batch_size = self.sp_config.validation_batch_size
        n_validated = 0
        rescued_log = []

        for batch_start in range(0, len(validatable), batch_size):
            batch = validatable[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(validatable) + batch_size - 1) // batch_size

            print(f"[PASS2-S2] Validating batch {batch_num}/{total_batches} ({len(batch)} messages)")

            # Build prompt input
            messages_with_matches = []
            for i, entry in enumerate(batch):
                messages_with_matches.append({
                    "idx": i,
                    "message": entry["assignment"].message_text,
                    "theme_name": entry["theme"].theme_name,
                    "theme_description": entry["theme"].description,
                    "best_phrase": entry["best_phrase"],
                    "similarity": entry["best_sim"],
                })

            prompt = get_misc_validation_prompt(messages_with_matches)

            # GPT call — use gpt-5-nano for faster validation
            gpt_inputs = [
                GPT4Input(actor="system", text=prompt),
                GPT4Input(actor="user", text="Validate these message-to-theme matches."),
            ]

            response = None
            _max_retries = 3
            for _attempt in range(1 + _max_retries):
                try:
                    result = gpt_5_nano(gpt_inputs, temperature=0.2, max_tokens=3000, timeout=120)
                    response = result.content if result and hasattr(result, 'content') else None
                    break
                except Exception as _e:
                    _msg = str(_e)
                    if '401' in _msg or '403' in _msg or 'invalid_api_key' in _msg.lower() or 'incorrect api key' in _msg.lower():
                        print(f"[PASS2-S2] Auth error (non-retryable): {_e}")
                        break
                    if _attempt < _max_retries:
                        _delay = 2 ** (_attempt + 1)
                        print(f"[PASS2-S2] LLM attempt {_attempt + 1} failed: {_e!r}. Retrying in {_delay}s…")
                        time.sleep(_delay)
                    else:
                        print(f"[PASS2-S2] LLM failed after {_max_retries} retries: {_e!r}")

            if not response:
                print(f"[PASS2-S2] GPT call failed for batch {batch_num}, treating all as INVALID")
                continue

            # Parse response
            decisions = self._parse_validation_response(response, len(batch))

            # Apply decisions
            batch_validated = 0
            for decision in decisions:
                idx = decision.get("idx")
                verdict = decision.get("decision", "").upper()

                if idx is None or idx < 0 or idx >= len(batch):
                    continue

                if verdict == "VALID":
                    entry = batch[idx]
                    a = entry["assignment"]
                    a.assigned_theme_id = entry["best_theme_id"]
                    a.status = AssignmentStatus.CONFIDENT
                    batch_validated += 1
                    n_validated += 1

                    rescued_log.append(
                        f'[RESCUED] Message: "{a.message_text}"\n'
                        f'  \u2192 Assigned to: "{entry["theme"].theme_name}" '
                        f'| Phrase: "{entry["best_phrase"]}" '
                        f'| Sim: {entry["best_sim"]:.3f}\n'
                    )

            print(f"[PASS2-S2] Batch {batch_num}: {batch_validated}/{len(batch)} validated")

        # Save log file
        if rescued_log and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_path = os.path.join(output_dir, "stage2_validated_messages.txt")
            with open(log_path, 'w') as f:
                f.write(f"Stage 2 GPT Validation -- {n_validated} messages rescued\n")
                f.write("=" * 70 + "\n\n")
                for entry in rescued_log:
                    f.write(entry + "\n")
            print(f"[PASS2-S2] Saved validation log to: {log_path}")

        print(f"[PASS2-S2] Total validated: {n_validated}/{len(validatable)}")
        return assignments, n_validated

    def _parse_validation_response(
        self,
        response: str,
        expected_count: int,
    ) -> List[dict]:
        """Parse GPT validation response. Returns list of {idx, decision, reason}."""
        try:
            cleaned = response.strip()

            if cleaned.startswith("```json"):
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            decisions = json.loads(cleaned)

            if isinstance(decisions, list):
                return decisions
            return []

        except (json.JSONDecodeError, Exception) as e:
            print(f"[PASS2-S2] Failed to parse validation response: {e}")
            print(f"[PASS2-S2] Response preview: {response[:300]}...")
            return []

    # =========================================================================
    # STAGE 3: BLIND DISCOVERY + MERGE
    # =========================================================================

    def discover_and_merge(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
        all_embeddings: np.ndarray,
        cleaned_messages: List[str],
        cleaned_session_ids: List[str],
    ) -> Tuple[List[Theme], List[MessageAssignment], Dict[str, int]]:
        """
        Discover new micro-themes from remaining misc, then merge against existing.

        Workflow:
        1. Collect misc messages + their embeddings
        2. HDBSCAN sampling on misc embeddings (diverse representative subset)
        3. Run FULL discovery prompt on sampled messages (same quality as 1st pass)
        4. Embed new themes, merge duplicates against existing
        5. Assign ALL remaining misc to surviving new themes (not just sampled)
        6. Dissolve tiny themes

        Args:
            themes: Current theme list
            assignments: Current assignments
            all_embeddings: Full embedding matrix
            cleaned_messages: All cleaned message texts
            cleaned_session_ids: All session IDs

        Returns:
            Tuple of (updated_themes, updated_assignments, stage3_stats)
        """
        print(f"\n[PASS2-S3] Stage 3: Blind Discovery + Merge")

        stats = {
            "new_themes_created": 0,
            "new_themes_merged_into_existing": 0,
            "new_themes_dissolved": 0,
            "misc_rescued_by_new_themes": 0,
        }

        # Collect remaining misc messages + their embeddings
        misc_indices = [
            i for i, a in enumerate(assignments) if a.assigned_theme_id < 0
        ]
        misc_messages = [cleaned_messages[i] for i in misc_indices]
        misc_session_ids = [cleaned_session_ids[i] for i in misc_indices]
        misc_embeddings = all_embeddings[misc_indices]

        print(f"[PASS2-S3] Remaining misc messages: {len(misc_messages)}")

        if len(misc_messages) < 20:
            print(f"[PASS2-S3] Too few misc messages for discovery (<20), skipping")
            return themes, assignments, stats

        # Step 1: HDBSCAN sampling on misc embeddings for diverse discovery input
        discovery_messages = misc_messages
        if self.sampler is not None and len(misc_messages) > self.config.sampling.small_dataset_max:
            sample_size = self.sampler.calculate_sample_size(len(misc_messages))
            print(f"[PASS2-S3] HDBSCAN sampling {sample_size} from {len(misc_messages)} misc for discovery...")

            sampled_msgs, sampled_sids, sampled_idx, metadata = self.sampler.hdbscan_sample(
                messages=misc_messages,
                session_ids=misc_session_ids,
                embeddings=misc_embeddings,
                sample_size=sample_size,
            )
            discovery_messages = sampled_msgs
            print(f"[PASS2-S3] Sampled {len(discovery_messages)} messages for discovery "
                  f"(method: {metadata.get('method', 'unknown')})")
        else:
            print(f"[PASS2-S3] Using all {len(misc_messages)} misc messages for discovery "
                  f"(below sampling threshold or no sampler)")

        # Step 2: Discover new themes (blind -- no seeds)
        # Uses the FULL discovery prompt via ThemeDiscovery, which handles batching
        print(f"[PASS2-S3] Running blind theme discovery on {len(discovery_messages)} messages...")
        new_themes = self.discoverer.discover_themes(
            messages=discovery_messages,
            seed_themes=None,  # Blind discovery
        )

        if not new_themes:
            print(f"[PASS2-S3] No new themes discovered")
            return themes, assignments, stats

        print(f"[PASS2-S3] Discovered {len(new_themes)} candidate micro-themes")

        # Step 2: Assign new IDs that don't collide with existing
        max_id = max((t.theme_id for t in themes), default=-1)
        for i, theme in enumerate(new_themes):
            theme.theme_id = max_id + 1 + i

        # Step 3: Embed only new themes
        print(f"[PASS2-S3] Embedding {len(new_themes)} new themes...")
        new_themes = self.embedder.embed_themes(new_themes)

        # Step 4: Merge check against existing themes
        new_themes, n_merged = self._merge_against_existing(
            new_themes, themes, assignments, cleaned_messages, cleaned_session_ids
        )
        stats["new_themes_merged_into_existing"] = n_merged

        if not new_themes:
            print(f"[PASS2-S3] All new themes merged into existing, no novel themes remain")
            return themes, assignments, stats

        # Step 5: Add surviving new themes to the main list
        themes.extend(new_themes)
        stats["new_themes_created"] = len(new_themes)

        # Step 6: Assign ALL remaining misc to all themes (existing + new)
        n_rescued = self._assign_misc_to_themes(
            themes, assignments, all_embeddings,
            cleaned_messages, cleaned_session_ids,
        )
        stats["misc_rescued_by_new_themes"] = n_rescued

        # Step 7: Dissolve tiny new themes
        themes, assignments, n_dissolved = self._dissolve_tiny_new_themes(
            themes, assignments, new_themes
        )
        stats["new_themes_dissolved"] = n_dissolved
        stats["new_themes_created"] -= n_dissolved

        return themes, assignments, stats

    def _merge_against_existing(
        self,
        new_themes: List[Theme],
        existing_themes: List[Theme],
        assignments: List[MessageAssignment],
        cleaned_messages: List[str],
        cleaned_session_ids: List[str],
    ) -> Tuple[List[Theme], int]:
        """
        Check new themes for duplicates against existing themes.

        Uses embedding pre-filtering (centroid similarity > 0.65) to limit
        GPT calls, then asks GPT to confirm merge/keep-separate.

        Merged themes: their key phrases get absorbed into the existing theme.

        Returns:
            Tuple of (surviving_new_themes, n_merged)
        """
        print(f"[PASS2-S3] Checking {len(new_themes)} new themes against "
              f"{len(existing_themes)} existing for duplicates...")

        # Compute centroid similarity between new and existing themes
        existing_with_embeddings = [
            t for t in existing_themes if t.theme_embedding is not None
        ]
        new_with_embeddings = [
            t for t in new_themes if t.theme_embedding is not None
        ]

        if not existing_with_embeddings or not new_with_embeddings:
            print(f"[PASS2-S3] Missing embeddings, skipping merge check")
            return new_themes, 0

        existing_centroids = np.array([t.theme_embedding for t in existing_with_embeddings])
        new_centroids = np.array([t.theme_embedding for t in new_with_embeddings])

        sim_matrix = cosine_similarity(new_centroids, existing_centroids)

        # Find candidate pairs above threshold
        merge_threshold = 0.65
        candidate_pairs = []

        for i, new_t in enumerate(new_with_embeddings):
            for j, ex_t in enumerate(existing_with_embeddings):
                sim = float(sim_matrix[i, j])
                if sim >= merge_threshold:
                    candidate_pairs.append({
                        "theme_a_name": new_t.theme_name,
                        "theme_b_name": ex_t.theme_name,
                        "similarity": sim,
                        "_new_theme": new_t,
                        "_existing_theme": ex_t,
                    })

        if not candidate_pairs:
            print(f"[PASS2-S3] No similar pairs found (threshold={merge_threshold}), all themes are novel")
            return new_themes, 0

        print(f"[PASS2-S3] Found {len(candidate_pairs)} candidate pairs for GPT merge check")

        # Format pairs for get_cluster_merge_decision_prompt
        theme_pairs = [
            {
                'theme_a': {
                    'name': p["_new_theme"].theme_name,
                    'description': p["_new_theme"].description,
                    'key_phrases': p["_new_theme"].key_phrases,
                },
                'theme_b': {
                    'name': p["_existing_theme"].theme_name,
                    'description': p["_existing_theme"].description,
                    'key_phrases': p["_existing_theme"].key_phrases,
                },
                'similarity': p["similarity"],
            }
            for p in candidate_pairs
        ]

        prompt = get_cluster_merge_decision_prompt(theme_pairs=theme_pairs)

        gpt_inputs = [
            GPT4Input(actor="system", text=prompt),
            GPT4Input(actor="user", text="Decide which new micro-themes are duplicates of existing themes."),
        ]

        response = None
        _max_retries = 3
        for _attempt in range(1 + _max_retries):
            try:
                result = gpt_5_2_chat(gpt_inputs, temperature=0.2, max_tokens=2000, timeout=120)
                response = result.content if result and hasattr(result, 'content') else None
                break
            except Exception as _e:
                _msg = str(_e)
                if '401' in _msg or '403' in _msg or 'invalid_api_key' in _msg.lower() or 'incorrect api key' in _msg.lower():
                    print(f"[PASS2-S3] Auth error (non-retryable): {_e}")
                    break
                if _attempt < _max_retries:
                    _delay = 2 ** (_attempt + 1)
                    print(f"[PASS2-S3] LLM attempt {_attempt + 1} failed: {_e!r}. Retrying in {_delay}s…")
                    time.sleep(_delay)
                else:
                    print(f"[PASS2-S3] LLM failed after {_max_retries} retries: {_e!r}")

        if not response:
            print(f"[PASS2-S3] GPT merge check failed, keeping all new themes")
            return new_themes, 0

        # Parse merge decisions — get_cluster_merge_decision_prompt returns:
        # [{"pair_number": 1, "decision": "merge"|"separate", "reason": "..."}]
        merged_new_theme_names = set()
        try:
            cleaned = response.strip()

            # Strip markdown code blocks
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                parts = cleaned.split("```")
                if len(parts) >= 3:
                    cleaned = parts[1].strip()

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
                new_t = pair["_new_theme"]
                ex_t = pair["_existing_theme"]

                if action == 'merge':
                    merged_new_theme_names.add(new_t.theme_name)

                    # Add new theme's key phrases to existing (union, deduplicate)
                    existing_phrases = set(ex_t.key_phrases)
                    for phrase in new_t.key_phrases:
                        if phrase not in existing_phrases:
                            ex_t.key_phrases.append(phrase)

                    print(f"[PASS2-S3] MERGE: '{new_t.theme_name}' -> '{ex_t.theme_name}'")
                    print(f"[PASS2-S3]   Reason: {reason}")
                elif action == 'separate':
                    print(f"[PASS2-S3] KEEP SEPARATE: '{new_t.theme_name}' vs '{ex_t.theme_name}'")
                    print(f"[PASS2-S3]   Reason: {reason}")

        except Exception as e:
            print(f"[PASS2-S3] Failed to parse merge response: {e}")
            print(f"[PASS2-S3] Response preview: {response[:500]}...")

        n_merged = len(merged_new_theme_names)

        # Re-embed existing themes that absorbed new phrases
        if n_merged > 0:
            modified_existing = []
            for pair in candidate_pairs:
                if pair["_new_theme"].theme_name in merged_new_theme_names:
                    modified_existing.append(pair["_existing_theme"])

            if modified_existing:
                print(f"[PASS2-S3] Re-embedding {len(modified_existing)} modified existing themes...")
                self.embedder.embed_themes(modified_existing)

        # Filter out merged new themes
        surviving = [t for t in new_themes if t.theme_name not in merged_new_theme_names]

        print(f"[PASS2-S3] Merged {n_merged} new themes into existing, {len(surviving)} novel themes remain")
        return surviving, n_merged

    def _assign_misc_to_themes(
        self,
        all_themes: List[Theme],
        assignments: List[MessageAssignment],
        all_embeddings: np.ndarray,
        cleaned_messages: List[str],
        cleaned_session_ids: List[str],
    ) -> int:
        """
        Assign ALL remaining misc messages against all themes (existing + new).

        After Stage 3 discovery + merge, existing themes may have absorbed
        new key phrases, so misc messages that were too far before might
        now match. We re-assign against the full theme set.
        """
        misc_indices = [
            i for i, a in enumerate(assignments) if a.assigned_theme_id < 0
        ]

        if not misc_indices or not all_themes:
            return 0

        print(f"[PASS2-S3] Assigning {len(misc_indices)} misc messages against "
              f"{len(all_themes)} themes (existing + new)...")

        n_rescued = 0
        for idx in misc_indices:
            new_assignment = self.assigner.assign_message(
                message_idx=idx,
                message_text=cleaned_messages[idx],
                message_embedding=all_embeddings[idx],
                themes=all_themes,
                session_id=cleaned_session_ids[idx],
            )

            if new_assignment.assigned_theme_id >= 0:
                assignments[idx].assigned_theme_id = new_assignment.assigned_theme_id
                assignments[idx].best_similarity = new_assignment.best_similarity
                assignments[idx].confidence_gap = new_assignment.confidence_gap
                assignments[idx].status = new_assignment.status
                assignments[idx].all_similarities = new_assignment.all_similarities
                assignments[idx].best_matching_phrase = new_assignment.best_matching_phrase
                n_rescued += 1

        print(f"[PASS2-S3] Rescued {n_rescued}/{len(misc_indices)} messages")
        return n_rescued

    def _dissolve_tiny_new_themes(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
        new_themes: List[Theme],
    ) -> Tuple[List[Theme], List[MessageAssignment], int]:
        """Dissolve new themes that didn't reach minimum size."""
        min_size = self.sp_config.min_new_theme_messages
        new_theme_ids = {t.theme_id for t in new_themes}

        # Count messages per new theme
        new_theme_counts = {tid: 0 for tid in new_theme_ids}
        for a in assignments:
            if a.assigned_theme_id in new_theme_ids:
                new_theme_counts[a.assigned_theme_id] += 1

        # Find tiny themes
        tiny_ids = set()
        for tid, count in new_theme_counts.items():
            if count < min_size:
                theme_name = next(
                    (t.theme_name for t in themes if t.theme_id == tid), "?"
                )
                print(f"[PASS2-S3] Dissolving '{theme_name}' ({count} msgs < {min_size} min)")
                tiny_ids.add(tid)

        if not tiny_ids:
            return themes, assignments, 0

        # Move messages back to misc
        for a in assignments:
            if a.assigned_theme_id in tiny_ids:
                a.assigned_theme_id = -1
                a.status = AssignmentStatus.MISCELLANEOUS

        # Remove tiny themes
        themes = [t for t in themes if t.theme_id not in tiny_ids]

        return themes, assignments, len(tiny_ids)

    # =========================================================================
    # ORCHESTRATION
    # =========================================================================

    def run_stages_2_and_3(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
        all_embeddings: np.ndarray,
        cleaned_messages: List[str],
        cleaned_session_ids: List[str],
        output_dir: str = None,
    ) -> Tuple[List[Theme], List[MessageAssignment], Dict[str, int]]:
        """
        Run Stages 2 and 3 of the second pass.

        Called only when enabled and misc thresholds are met.
        Stage 1 (borderline) is called separately and unconditionally.

        Returns:
            Tuple of (updated_themes, updated_assignments, stats)
        """
        stats = {
            "misc_validated": 0,
            "new_themes_created": 0,
            "new_themes_merged_into_existing": 0,
            "new_themes_dissolved": 0,
            "misc_rescued_by_new_themes": 0,
        }

        # Stage 2: GPT Validation
        assignments, n_validated = self.validate_misc_messages(
            themes, assignments, output_dir
        )
        stats["misc_validated"] = n_validated

        # Stage 3: Discovery + Merge
        themes, assignments, s3_stats = self.discover_and_merge(
            themes, assignments, all_embeddings,
            cleaned_messages, cleaned_session_ids,
        )
        stats.update(s3_stats)

        # Final summary
        misc_remaining = sum(1 for a in assignments if a.assigned_theme_id < 0)
        stats["misc_remaining"] = misc_remaining
        total = len(assignments)

        print(f"\n[PASS2] Second Pass Summary:")
        print(f"  Stage 2 -- GPT validated: {stats['misc_validated']}")
        print(f"  Stage 3 -- New themes: {stats['new_themes_created']}, "
              f"Merged into existing: {stats['new_themes_merged_into_existing']}, "
              f"Dissolved: {stats['new_themes_dissolved']}")
        print(f"  Stage 3 -- Rescued by new themes: {stats.get('misc_rescued_by_new_themes', 0)}")
        print(f"  Misc remaining: {misc_remaining}/{total} "
              f"({100 * misc_remaining / total:.1f}%)")

        return themes, assignments, stats
