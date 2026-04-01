"""
Theme Clustering Pipeline - Main Orchestration

GPT + Hybrid Similarity (Cosine + BM25 + Fuzzy weighted fusion) theme extraction pipeline.
This is the main entry point for the theme clustering system.

Usage:
    from src.theme_clustering.theme_clustering import ThemeClusteringPipeline, run_pipeline

    # Quick run
    results = run_pipeline(messages, session_ids, seed_themes=["Delivery", "Returns"])

    # Or with custom config
    pipeline = ThemeClusteringPipeline(config)
    results = pipeline.run(messages, session_ids, seed_themes)
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np

# Handle both direct execution and module import
# Use try/except to avoid circular import through __init__.py
try:
    from .theme_config import ThemeClusteringConfig, DEFAULT_CONFIG
    from .theme_models import (
        Theme, MessageAssignment, ClusterResult, QualityReport,
        PipelineInput, PipelineOutput, AssignmentStatus
    )
    from .theme_preprocessing import TextPreprocessor
    from .theme_sampling import AdaptiveSampler
    from .theme_discovery import ThemeDiscovery
    from .theme_embedding import ThemeEmbedder
    from .theme_assignment import ThemeAssigner
    from .theme_quality import QualityAssessor
    from .theme_second_pass import SecondPassRunner
except ImportError:
    # Fallback for absolute imports
    from src.theme_clustering.theme_config import ThemeClusteringConfig, DEFAULT_CONFIG
    from src.theme_clustering.theme_models import (
        Theme, MessageAssignment, ClusterResult, QualityReport,
        PipelineInput, PipelineOutput, AssignmentStatus
    )
    from src.theme_clustering.theme_preprocessing import TextPreprocessor
    from src.theme_clustering.theme_sampling import AdaptiveSampler
    from src.theme_clustering.theme_discovery import ThemeDiscovery
    from src.theme_clustering.theme_embedding import ThemeEmbedder
    from src.theme_clustering.theme_assignment import ThemeAssigner
    from src.theme_clustering.theme_quality import QualityAssessor
    from src.theme_clustering.theme_second_pass import SecondPassRunner


def _extract_client_slug(client_name: str) -> str:
    """Strip .myshopify.com from a full Shopify domain, keeping just the store slug."""
    return client_name.replace('.myshopify.com', '')


def _extract_page_slug(page_url: str) -> str:
    """
    Derive a safe filesystem slug from a full page URL.

    Mirrors the filename suffix logic in vector.py so that file naming is
    consistent between both pipelines.

    Examples:
        https://store.com/products/blue-widget  →  blue-widget
        https://store.com/collections/summer    →  summer
        /cart                                   →  cart
        already-a-slug                          →  already-a-slug (returned as-is)
    """
    if not page_url:
        return ""
    # If there's no scheme separator and no leading slash it's already a slug
    if '//' not in page_url and not page_url.startswith('/'):
        return page_url
    for marker in ['/products/', '/blogs/', '/collections/', '/pages/']:
        if marker in page_url:
            page_id = page_url.split(marker)[-1].split('?')[0].split('#')[0]
            return re.sub(r'[^\w\-]', '_', page_id)[:50]
    if '/cart' in page_url:
        return 'cart'
    if '/search' in page_url:
        return 'search'
    page_id = page_url.split('//')[-1].split('/')[-1].split('?')[0].split('#')[0] or 'page'
    return re.sub(r'[^\w\-]', '_', page_id)[:50]


def _expand_page_variants(
    page_urls: List[str],
    chat_data_path: str,
    is_report: bool,
) -> List[str]:
    """
    Given a list of page URLs, return an expanded set that also includes all
    URL variants of the same product found in the actual chat data.

    Shopify products are accessible via both a direct path
    (/products/<slug>) and one or more collection paths
    (/collections/<X>/products/<slug>).  The bot_page field in session data
    stores whichever URL the user actually visited, so filtering with only the
    canonical direct URL can miss a significant portion of sessions.

    Only /products/ URLs are expanded; /collections/, /blogs/, /pages/ etc.
    are left as-is (the collection context matters for those).
    """
    import json as _json

    # Extract product slugs that should be expanded (only direct /products/ URLs)
    slugs_to_expand = []
    for url in page_urls:
        if '/products/' in url and '/collections/' not in url:
            slug = url.split('/products/')[-1].split('?')[0].split('#')[0].rstrip('/')
            if slug:
                slugs_to_expand.append(f'/products/{slug}')

    if not slugs_to_expand:
        return page_urls  # Nothing to expand

    try:
        with open(chat_data_path, 'r', encoding='utf-8') as _f:
            _raw = _json.load(_f)

        all_pages = set(page_urls)

        if is_report:
            # Normalise single-dict format (all sessions in one dict)
            if (isinstance(_raw, list) and len(_raw) == 1
                    and isinstance(_raw[0], dict) and len(_raw[0]) > 10):
                _raw = [{sid: sd} for sid, sd in _raw[0].items()]
            for _session_obj in _raw:
                for _sid, _sdata in _session_obj.items():
                    _meta = _sdata.get('metadata_for_session', [])
                    if _meta:
                        _bp = _meta[0].get('bot_page', '')
                        if _bp and any(_s in _bp for _s in slugs_to_expand):
                            all_pages.add(_bp)
        else:
            if isinstance(_raw, dict):
                for _sid, _sdata in _raw.items():
                    _meta = _sdata.get('session_meta', [])
                    if _meta:
                        _bp = _meta[0].get('bot_page', '')
                        if _bp and any(_s in _bp for _s in slugs_to_expand):
                            all_pages.add(_bp)

        added = len(all_pages) - len(page_urls)
        if added > 0:
            print(
                f"[PAGE-EXPAND] Expanded {len(page_urls)} URL(s) → {len(all_pages)} variants "
                f"(+{added} collection-path variants for the same product)"
            )
        return list(all_pages)

    except Exception as _e:
        print(f"[PAGE-EXPAND] Warning: Could not expand page variants ({_e}), using original URLs")
        return page_urls


class ThemeClusteringPipeline:
    """
    Main orchestration class for GPT + Hybrid Weighted Fusion theme clustering.

    6-Phase Pipeline:
    1. Preprocessing & Sampling
       - UMAP (1024D→50D) used internally for HDBSCAN clustering during sampling
       - Full 1024D embeddings used for all assignment operations
    2. Theme Discovery (GPT)
    3. Theme Embedding
    3.5. Redundant Theme Merging (before assignment to avoid wasted computation)
    4. Message Assignment (Hybrid Weighted Fusion)
       - Three signals: Cosine (0.5) + BM25Plus (0.25) + Fuzzy (0.25)
       - Weighted fusion score drives both ranking AND threshold decisions
       - Per-message signal breakdown exported for diagnostics
    5. Quality Assurance (tiny theme dissolution, coherence metrics)
    6. Second Pass — 3-Stage Miscellaneous Rescue:
       6.1 Borderline reassignment (always runs, no GPT)
       6.2 GPT validation of misc closest matches (conditional)
       6.3 Blind discovery + merge on remaining misc (conditional)

    Attributes:
        config: ThemeClusteringConfig with all parameters
        preprocessor: TextPreprocessor instance
        sampler: AdaptiveSampler instance
        discoverer: ThemeDiscovery instance
        embedder: ThemeEmbedder instance
        assigner: ThemeAssigner instance
        assessor: QualityAssessor instance
    """

    def __init__(self, config: ThemeClusteringConfig = None):
        self.config = config or DEFAULT_CONFIG

        # Initialize components
        self.preprocessor = TextPreprocessor(self.config.preprocessing)
        self.sampler = AdaptiveSampler(self.config.sampling)
        self.discoverer = ThemeDiscovery(self.config.discovery)
        self.embedder = ThemeEmbedder(self.config.embedding)
        self.assigner = ThemeAssigner(self.config.assignment)
        self.assessor = QualityAssessor(self.config.quality)

        # State
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def run(
        self,
        messages: List[str],
        session_ids: Optional[List[str]] = None,
        user_intents: Optional[List[str]] = None,
        seed_themes: Optional[List[str]] = None,
        client_name: str = "default",
        page_name: Optional[str] = None,
        chat_data: Optional[Dict[str, Any]] = None,
        save_outputs: bool = True,
        output_dir_override: Optional[str] = None,
        report_month: Optional[str] = None,
    ) -> Optional[PipelineOutput]:
        """
        Run the full theme clustering pipeline.

        Args:
            messages: List of customer message texts
            session_ids: Optional list of session IDs (parallel to messages)
            user_intents: Optional list of secondary usecase labels (parallel to messages).
                          When provided, enables usecase-aware misc rescue after Phase 5.
            seed_themes: Optional list of seed theme names to include
            client_name: Client name slug (e.g. "flicka-cosmetics")
            page_name: Optional page slug (e.g. "flicka-silk-touch-moisturizer").
                       When provided, all output files are named
                       {client_name}_{page_name}_{file_id}.ext to prevent race conditions
                       when the pipeline runs in parallel for different pages.
            chat_data: Optional dict mapping session_id -> session data (daily format).
                       When provided, per-cluster metadata (A2C %, order %, UTM %,
                       avg conversation length) and performance classifications are computed.
            save_outputs: Whether to save intermediate outputs
            output_dir_override: Optional absolute or relative path to override
                                 self.config.output.output_dir for all saved files.
            report_month: Optional month string (e.g., "February 2026") for VoC report header.
                         Defaults to current month if not provided.

        Returns:
            PipelineOutput with clusters, miscellaneous, and quality report
        """
        pipeline_start = time.time()

        # Build file prefix — used for ALL output files to avoid race conditions
        # when multiple instances run in parallel for different client/page combos.
        # Slugs are auto-derived from the full myshopify domain + raw page URL.
        client_slug = _extract_client_slug(client_name)
        page_slug   = _extract_page_slug(page_name) if page_name else None
        file_prefix = f"{client_slug}_{page_slug}" if page_slug else client_slug
        effective_output_dir = output_dir_override or self.config.output.output_dir

        print("=" * 70)
        print("THEME CLUSTERING PIPELINE")
        print("=" * 70)
        print(f"Input: {len(messages)} messages")
        print(f"File prefix: {file_prefix}")
        if seed_themes:
            print(f"Seed themes: {seed_themes}")
        print()

        # Ensure session_ids
        if session_ids is None:
            session_ids = [f"session_{i}" for i in range(len(messages))]

        # =====================================================================
        # PHASE 1: PREPROCESSING & SAMPLING
        # =====================================================================
        print("-" * 70)
        print("PHASE 1: PREPROCESSING & SAMPLING")
        print("-" * 70)

        # Preprocess messages (returns original indices for user_intents mapping)
        cleaned_messages, cleaned_session_ids, message_to_sessions, original_indices = \
            self.preprocessor.preprocess_messages(messages, session_ids)

        print(f"After preprocessing: {len(cleaned_messages)} unique messages")

        if len(cleaned_messages) < 50:
            print(
                f"\n[PIPELINE] Too few messages after preprocessing "
                f"({len(cleaned_messages)} < 50 minimum). "
                f"Not enough data for meaningful theme clustering — skipping."
            )
            return None

        # Cap at 4000 messages post-preprocessing to keep pipeline tractable
        _MAX_MESSAGES = 4000
        if len(cleaned_messages) > _MAX_MESSAGES:
            import random as _random
            print(f"[PREPROCESS] Message count ({len(cleaned_messages)}) exceeds cap of {_MAX_MESSAGES}, randomly sampling {_MAX_MESSAGES}...")
            _sample_idx = sorted(_random.sample(range(len(cleaned_messages)), _MAX_MESSAGES))
            cleaned_messages    = [cleaned_messages[i]    for i in _sample_idx]
            cleaned_session_ids = [cleaned_session_ids[i] for i in _sample_idx]

        # Embed all messages (full 1024D - used for assignment throughout pipeline)
        # Note: UMAP reduction to 50D occurs only internally during HDBSCAN sampling
        print("\nEmbedding all messages for assignment...")

        # Build cache file path based on file_prefix and message count
        cache_dir = os.path.join(effective_output_dir, "embeddings")
        cache_file = os.path.join(
            cache_dir,
            f"{file_prefix}_embeddings_{len(cleaned_messages)}.npy"
        )

        all_embeddings, self._embedding_cache = self.embedder.embed_messages(
            cleaned_messages, self._embedding_cache, cache_file=cache_file
        )

        # Sample for theme discovery
        # Note: If HDBSCAN enabled, UMAP reduces to 50D internally for clustering only
        # The full 1024D embeddings continue to be used for similarity matching
        sampled_messages, sampled_session_ids, sampled_indices = \
            self.sampler.sample_for_discovery(
                cleaned_messages, cleaned_session_ids, all_embeddings,
                output_dir=effective_output_dir,
                file_prefix=file_prefix,
            )

        print(f"Sampled {len(sampled_messages)} messages for theme discovery")

        # =====================================================================
        # PHASE 2: THEME DISCOVERY (GPT)
        # =====================================================================
        print("\n" + "-" * 70)
        print("PHASE 2: THEME DISCOVERY (GPT)")
        print("-" * 70)

        themes = self.discoverer.discover_themes(sampled_messages, seed_themes)

        print(f"\nDiscovered {len(themes)} themes")

        # =====================================================================
        # PHASE 3: THEME EMBEDDING
        # =====================================================================
        print("\n" + "-" * 70)
        print("PHASE 3: THEME EMBEDDING")
        print("-" * 70)

        themes = self.embedder.embed_themes(themes)

        # =====================================================================
        # PHASE 3.5: REDUNDANT THEME MERGING
        # =====================================================================
        print("\n" + "-" * 70)
        print("PHASE 3.5: REDUNDANT THEME MERGING")
        print("-" * 70)
        print("[MERGE] Detecting and merging redundant themes BEFORE assignment")
        print("[MERGE] This prevents wasted computation assigning to themes that will be merged")

        merge_pairs = self.assessor.detect_redundant_themes(themes)
        themes, _, n_merged = self.assessor.merge_themes(themes, merge_pairs, assignments=None)

        print(f"[MERGE] Merged {n_merged} redundant themes")
        print(f"[MERGE] Proceeding to assignment with {len(themes)} final themes")

        # =====================================================================
        # PHASE 3.6: CLUSTER USECASE TAGGING (runs once, reused by Phase 4 + 6)
        # =====================================================================
        cluster_usecases_map: Dict[str, list] = {}
        theme_id_usecases: Dict[int, set] = {}
        msg_usecase_idx: Dict[int, str] = {}   # original_idx -> usecase

        if user_intents is not None and self.config.assignment.usecase_boost_enabled:
            print("\n" + "-" * 70)
            print("PHASE 3.6: CLUSTER USECASE TAGGING")
            print("-" * 70)

            # Build original_idx -> usecase map (via original_indices from preprocessing)
            for i, orig_idx in enumerate(original_indices):
                if orig_idx < len(user_intents) and user_intents[orig_idx]:
                    uc = user_intents[orig_idx]
                    if uc and uc != "unknown":
                        msg_usecase_idx[orig_idx] = uc

            unique_usecases = sorted(set(msg_usecase_idx.values()))
            print(f"[PHASE3.6] {len(unique_usecases)} unique usecases across {len(msg_usecase_idx)} messages")

            if unique_usecases:
                cluster_usecases_map, theme_id_usecases = self._tag_clusters_with_usecases(
                    themes, unique_usecases
                )
        else:
            print("\n[PHASE3.6] Skipped (user_intents=None or usecase_boost_enabled=False)")

        # =====================================================================
        # PHASE 4: MESSAGE ASSIGNMENT
        # =====================================================================
        print("\n" + "-" * 70)
        print("PHASE 4: MESSAGE ASSIGNMENT")
        print("-" * 70)
        print("[ASSIGN] Hybrid Weighted Fusion: Cosine (0.5) + BM25Plus (0.25) + Fuzzy (0.25)")
        print("[ASSIGN] Fusion score drives both ranking and threshold decisions")
        if theme_id_usecases:
            boost = self.config.assignment.usecase_additive_boost
            print(f"[ASSIGN] Usecase boost enabled: additive +{boost} for matching-usecase clusters")

        assignments, assignment_stats = self.assigner.assign_messages(
            cleaned_messages, all_embeddings, themes, cleaned_session_ids,
            original_indices=original_indices,
            user_intents_map=msg_usecase_idx if theme_id_usecases else None,
            theme_id_usecases=theme_id_usecases if theme_id_usecases else None,
            boost_amount=self.config.assignment.usecase_additive_boost if theme_id_usecases else 0.0,
        )

        # =====================================================================
        # PHASE 5: QUALITY ASSURANCE
        # =====================================================================
        print("\n" + "-" * 70)
        print("PHASE 5: QUALITY ASSURANCE")
        print("-" * 70)
        print("[QUALITY] Post-assignment refinement: tiny theme dissolution, coherence metrics")

        themes, assignments, quality_report = self.assessor.assess_quality(
            themes, assignments, all_embeddings, n_merged_pre_assignment=n_merged
        )

        # =====================================================================
        # STAMP USER INTENT ON ALL MESSAGES (if provided)
        # =====================================================================
        # Ensure every message gets its user_intent populated, regardless of Phase 6 config
        if user_intents is not None:
            for assignment in assignments:
                idx = assignment.message_idx
                if idx < len(user_intents):
                    uc = user_intents[idx]
                    assignment.user_intent = uc if uc else "unknown"

        # =====================================================================
        # PHASE 6: SECOND PASS ON MISCELLANEOUS
        # =====================================================================
        print("\n" + "-" * 70)
        print("PHASE 6: SECOND PASS ON MISCELLANEOUS")
        print("-" * 70)

        sp_config = self.config.second_pass
        # cluster_usecases_map and theme_id_usecases were populated in Phase 3.6 (or remain {} if skipped)

        # --- Stage 1: Borderline Reassignment (toggleable, no GPT) ---
        pre_stage1_coverage = quality_report.coverage_percent

        if sp_config.borderline_reassignment:
            second_pass = SecondPassRunner(
                self.config, self.discoverer, self.embedder, self.assigner,
                sampler=self.sampler,
            )
            assignments, n_borderline = second_pass.reassign_borderline(themes, assignments)
            quality_report.borderline_reassigned = n_borderline

            if self.config.output.generate_borderline_diagnostics:
                self._export_borderline_diagnostics(
                    assignments, themes, effective_output_dir, file_prefix
                )
        else:
            n_borderline = 0
            quality_report.borderline_reassigned = 0
            print(f"[PASS2] Stage 1 skipped (borderline_reassignment=False)")

        # Recompute misc stats after Stage 1
        misc_count = sum(1 for a in assignments if a.assigned_theme_id < 0)
        misc_percent = (
            100 * misc_count / len(assignments) if assignments else 0
        )
        assigned_count = len(assignments) - misc_count

        quality_report.assigned_messages = assigned_count
        quality_report.miscellaneous_messages = misc_count
        quality_report.coverage_percent = (
            100 * assigned_count / len(assignments) if assignments else 0
        )

        print(f"[PASS2] After Stage 1: {misc_count} misc ({misc_percent:.1f}%), "
              f"coverage {quality_report.coverage_percent:.1f}% "
              f"(was {pre_stage1_coverage:.1f}%)")

        # --- Usecase-Aware Rescue (config-driven) ---
        if sp_config.enabled:
            if user_intents is not None:
                assignments, n_usecase_rescued, cluster_usecases_map = self._usecase_aware_rescue(
                    themes=themes,
                    assignments=assignments,
                    user_intents=user_intents,
                    cluster_usecases_map=cluster_usecases_map,
                    theme_id_usecases=theme_id_usecases,
                )

                # Recompute final coverage
                final_assigned = sum(1 for a in assignments if a.assigned_theme_id >= 0)
                final_misc = sum(1 for a in assignments if a.assigned_theme_id < 0)
                quality_report.assigned_messages = final_assigned
                quality_report.miscellaneous_messages = final_misc
                quality_report.coverage_percent = (
                    100 * final_assigned / len(assignments) if assignments else 0
                )
                quality_report.misc_validated_by_gpt = n_usecase_rescued

                print(f"[PASS2] After usecase rescue: coverage {quality_report.coverage_percent:.1f}%, "
                      f"{n_usecase_rescued} messages rescued")
            else:
                print(f"[PASS2] Usecase rescue enabled but no user_intents provided — skipping")
        else:
            print(f"[PASS2] Usecase rescue disabled (config.second_pass.enabled=False)")

        # --- False-Negative Signal Rescue (runs after usecase rescue) ---
        if sp_config.fn_rescue:
            assignments, n_fn_rescued = self._fn_rescue(themes, assignments)

            final_assigned = sum(1 for a in assignments if a.assigned_theme_id >= 0)
            final_misc = sum(1 for a in assignments if a.assigned_theme_id < 0)
            quality_report.assigned_messages = final_assigned
            quality_report.miscellaneous_messages = final_misc
            quality_report.coverage_percent = (
                100 * final_assigned / len(assignments) if assignments else 0
            )
            quality_report.fn_rescued = n_fn_rescued

            print(f"[PASS2] After FN rescue: coverage {quality_report.coverage_percent:.1f}%, "
                  f"{n_fn_rescued} false negatives rescued")
        else:
            print(f"[PASS2] FN rescue disabled (config.second_pass.fn_rescue=False)")

        # =====================================================================
        # ANNOTATE MISC MESSAGES WITH CLUSTER TITLES
        # =====================================================================
        print("\n" + "-" * 70)
        print("ANNOTATING MISCELLANEOUS MESSAGES")
        print("-" * 70)

        # Add cluster titles to misc messages based on their matching phrases
        assignments = self._annotate_cluster_titles(themes, assignments)
        print("[ANNOTATE] Added cluster title references to misc messages")

        # =====================================================================
        # BUILD OUTPUT
        # =====================================================================
        print("\n" + "-" * 70)
        print("BUILDING OUTPUT")
        print("-" * 70)

        output = self._build_output(
            themes, assignments, message_to_sessions, quality_report,
            cluster_usecases_map=cluster_usecases_map,
        )

        # =====================================================================
        # CATEGORY TAGGING (pre-sales / post-sales / miscellaneous)
        # =====================================================================
        print("\n" + "-" * 70)
        print("CATEGORY TAGGING")
        print("-" * 70)

        self._tag_categories(output)

        # =====================================================================
        # CLUSTER METADATA (session-level metrics: A2C, orders, UTM, conv length)
        # =====================================================================
        if chat_data is not None:
            print("\n" + "-" * 70)
            print("CLUSTER METADATA")
            print("-" * 70)
            self._add_cluster_metadata(output, chat_data)

        # Guard: nothing meaningful to report if all themes were dissolved
        if not output.clusters:
            elapsed = time.time() - pipeline_start
            print(
                f"\n[PIPELINE] 0 non-miscellaneous clusters after quality assurance "
                f"(all themes dissolved). Skipping output generation."
                f"\n[PIPELINE] Total execution time: {elapsed / 60:.1f} minutes"
            )
            return None

        # Save outputs if requested
        if save_outputs:
            self._save_outputs(output, file_prefix, client_display=client_name, page_url=page_name,
                               output_dir_override=output_dir_override, report_month=report_month)

        elapsed = time.time() - pipeline_start
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Final clusters: {len(output.clusters)}")
        print(f"Miscellaneous: {output.miscellaneous.message_count if output.miscellaneous else 0}")
        print(f"Coverage: {quality_report.coverage_percent:.1f}%")
        print(f"Total execution time: {elapsed / 60:.1f} minutes")

        return output

    def _export_borderline_diagnostics(
        self,
        assignments: List,
        themes: List,
        output_dir: str,
        file_prefix: str,
    ) -> None:
        """
        Export a diagnostics file listing all borderline-reassigned messages.

        For each message that was_borderline=True, shows:
        - The message text
        - Which theme it was assigned to (1st choice) and its fusion score
        - The 2nd closest theme and its fusion score
        - The gap between them
        - Signal breakdown (cosine, bm25, fuzzy, fusion)

        Written to: <output_dir>/<file_prefix>_borderline_diagnostics.txt
        """

        theme_name_map = {t.theme_id: t.theme_name for t in themes}

        borderline_msgs = [a for a in assignments if a.was_borderline]
        if not borderline_msgs:
            print("[DIAG] No borderline messages to export")
            return

        os.makedirs(output_dir, exist_ok=True)
        diag_path = os.path.join(output_dir, f"{file_prefix}_borderline_diagnostics.txt")

        lines = []
        lines.append(f"BORDERLINE REASSIGNMENT DIAGNOSTICS — {len(borderline_msgs)} messages")
        lines.append(f"These messages had fusion >= borderline_threshold but failed the confidence gap check.")
        lines.append(f"They were force-assigned to their best theme in Stage 1 (Second Pass).")
        lines.append("=" * 90)
        lines.append("")

        for i, a in enumerate(borderline_msgs, 1):
            # Sort all_similarities to get 1st and 2nd best themes
            sorted_sims = sorted(
                a.all_similarities.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            best_tid, best_score = sorted_sims[0] if sorted_sims else (-1, 0.0)
            second_tid, second_score = sorted_sims[1] if len(sorted_sims) > 1 else (-1, 0.0)

            best_name = theme_name_map.get(best_tid, "Unknown")
            second_name = theme_name_map.get(second_tid, "Unknown")
            gap = best_score - second_score

            lines.append(f"[{i}] \"{a.message_text}\"")
            lines.append(f"    Assigned to:  {best_name} (id={best_tid})")
            lines.append(f"    Fusion score: {best_score:.4f}")
            lines.append(f"    2nd choice:   {second_name} (id={second_tid})")
            lines.append(f"    2nd score:    {second_score:.4f}")
            lines.append(f"    Gap:          {gap:.4f}")
            if a.signal_scores:
                s = a.signal_scores
                lines.append(f"    Signals:      cos={s.get('cosine', 0):.4f}  "
                             f"bm25_raw={s.get('bm25_raw', 0):.2f}  "
                             f"bm25_norm={s.get('bm25_norm', 0):.4f}  "
                             f"fuzzy={s.get('fuzzy', 0):.4f}")
            lines.append(f"    Phrase:       {a.best_matching_phrase}")
            lines.append("")

        with open(diag_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"[DIAG] Exported borderline diagnostics to: {diag_path}")

    def _annotate_cluster_titles(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
    ) -> List[MessageAssignment]:
        """
        Annotate assignments with cluster titles.

        For assigned messages: use theme_id to look up the cluster title directly.
        For misc messages: fall back to phrase -> theme lookup via best_matching_phrase.

        Args:
            themes: List of themes with key_phrases
            assignments: List of message assignments

        Returns:
            Updated list of assignments with cluster_title field populated
        """
        # Build mappings
        theme_map = {t.theme_id: t.theme_name for t in themes}
        phrase_to_theme = {}
        for theme in themes:
            for phrase in theme.key_phrases:
                phrase_to_theme[phrase] = theme.theme_name

        for assignment in assignments:
            if assignment.assigned_theme_id >= 0:
                # Assigned messages: direct lookup by theme_id
                assignment.cluster_title = theme_map.get(assignment.assigned_theme_id)
            elif assignment.best_matching_phrase:
                # Misc messages: lookup by best matching phrase
                assignment.cluster_title = phrase_to_theme.get(assignment.best_matching_phrase)

        return assignments

    def _tag_clusters_with_usecases(
        self,
        themes: List[Theme],
        unique_usecases: List[str],
    ) -> tuple:
        """
        Call GPT to tag each cluster with 1-2 relevant usecases from unique_usecases.

        Used by Phase 3.6 (before assignment) so the same map can be reused by
        Phase 4 (first-pass boost) and Phase 6 (misc rescue) without a second GPT call.

        Args:
            themes: Final themes after Phase 3.5 (redundant merge)
            unique_usecases: Sorted list of unique usecase strings from user_intents

        Returns:
            (cluster_usecases_map, theme_id_usecases):
                - cluster_usecases_map: {cluster_name -> [usecase, ...]}
                - theme_id_usecases:    {theme_id -> set(usecases)}
            Returns ({}, {}) gracefully on GPT failure.
        """
        import json as _json
        import re as _re
        from src.utils.openai_utils import gpt_5_2_chat, GPT4Input

        try:
            from .theme_prompts import get_usecase_tagging_prompt
        except ImportError:
            from src.theme_clustering.theme_prompts import get_usecase_tagging_prompt

        cluster_info = [
            {"cluster_name": t.theme_name, "description": t.description}
            for t in themes
        ]

        prompt = get_usecase_tagging_prompt(cluster_info, unique_usecases)
        print(f"[USECASE] Asking GPT to tag {len(themes)} clusters with {len(unique_usecases)} usecases...")

        gpt_inputs = [
            GPT4Input(actor="system", text="You are a VoC analyst. Return only valid JSON."),
            GPT4Input(actor="user", text=prompt),
        ]
        raw_response = None
        _max_retries = 3
        for _attempt in range(1 + _max_retries):
            try:
                result = gpt_5_2_chat(gpt_inputs, temperature=0.2, max_tokens=4000, timeout=120)
                raw_response = result.content if result and hasattr(result, 'content') else None
                break
            except Exception as _e:
                _msg = str(_e)
                if '401' in _msg or '403' in _msg or 'invalid_api_key' in _msg.lower() or 'incorrect api key' in _msg.lower():
                    print(f"[USECASE] Auth error (non-retryable): {_e}")
                    break
                if _attempt < _max_retries:
                    _delay = 2 ** (_attempt + 1)
                    print(f"[USECASE] LLM attempt {_attempt + 1} failed: {_e!r}. Retrying in {_delay}s…")
                    time.sleep(_delay)
                else:
                    print(f"[USECASE] LLM failed after {_max_retries} retries: {_e!r}")

        if not raw_response:
            print("[USECASE] WARNING: Empty GPT response — skipping usecase tagging")
            return {}, {}

        try:
            cleaned = _re.sub(r'^```(?:json)?\s*', '', raw_response.strip())
            cleaned = _re.sub(r'\s*```$', '', cleaned)
            tagging_result = _json.loads(cleaned)
        except _json.JSONDecodeError:
            print(f"[USECASE] WARNING: Failed to parse GPT response — skipping usecase tagging")
            print(f"[USECASE] Raw: {raw_response[:200]}...")
            return {}, {}

        # Build cluster_name -> usecases
        cluster_usecases_map: Dict[str, list] = {}
        for entry in tagging_result:
            name = entry.get("cluster_name", "")
            ucs = entry.get("usecases", [])
            if name:
                cluster_usecases_map[name] = ucs if ucs else []

        # Build theme_id -> set of usecases
        theme_id_usecases: Dict[int, set] = {}
        for theme in themes:
            ucs = cluster_usecases_map.get(theme.theme_name, [])
            if ucs:
                theme_id_usecases[theme.theme_id] = set(ucs)

        tagged_count = sum(1 for ucs in cluster_usecases_map.values() if ucs)
        print(f"[USECASE] GPT tagged {tagged_count}/{len(themes)} clusters with usecases")

        return cluster_usecases_map, theme_id_usecases

    def _usecase_aware_rescue(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
        user_intents: List[str],
        cluster_usecases_map: Optional[Dict[str, list]] = None,
        theme_id_usecases: Optional[Dict[int, set]] = None,
    ) -> tuple:
        """
        Usecase-aware misc rescue — the new Phase 6 second pass.

        Steps:
        1. Build message-index -> usecase mapping from user_intents
        2. Call GPT to tag each cluster with 1-2 relevant usecases
        3. For each unassigned message (misc + borderline if Stage 1 was skipped):
           find clusters that share its usecase, boost their fusion scores
           by boost_factor, re-rank, and reassign if boosted score >= rescue_threshold.

        Args:
            themes: Current themes after Phase 5 (+ optional borderline reassignment)
            assignments: Current assignments
            user_intents: List of usecase labels parallel to original messages
            cluster_usecases_map: Pre-computed map from Phase 3.6 (skips GPT call if provided)
            theme_id_usecases: Pre-computed map from Phase 3.6 (skips GPT call if provided)

        Returns:
            (assignments, n_rescued, cluster_usecases_map):
                - Updated assignments
                - Count of rescued messages
                - Dict mapping cluster_name -> list of usecase strings (for output)
        """
        sp_config = self.config.second_pass
        boost_factor = sp_config.boost_factor
        rescue_threshold = sp_config.rescue_threshold

        # --- Step 1: Build message_idx -> usecase mapping ---
        # user_intents is parallel to the ORIGINAL messages (before preprocessing).
        # We use assignment.message_idx to index into user_intents.
        # Note: assignment.user_intent is already stamped after Phase 5 for all messages.
        # Here we just build the msg_usecase mapping for boost/rescue logic.
        msg_usecase = {}
        for assignment in assignments:
            idx = assignment.message_idx
            if idx < len(user_intents):
                uc = user_intents[idx]
                # user_intent already set post-Phase 5, but refresh for safety
                if not assignment.user_intent:
                    assignment.user_intent = uc if uc else "unknown"
                if uc and uc != "unknown":
                    msg_usecase[assignment.message_idx] = uc

        # Discover unique usecases
        unique_usecases = sorted(set(msg_usecase.values()))
        if not unique_usecases:
            print("[USECASE] No usecases found in user_intents — skipping rescue")
            return assignments, 0, cluster_usecases_map or {}

        print(f"[USECASE] Found {len(unique_usecases)} unique usecases: {unique_usecases[:10]}...")

        # --- Step 2: Use pre-computed cluster tagging (from Phase 3.6) or call GPT ---
        if not cluster_usecases_map or not theme_id_usecases:
            # Fallback: GPT was not called in Phase 3.6 (e.g. boost disabled), call now
            cluster_usecases_map, theme_id_usecases = self._tag_clusters_with_usecases(
                themes, unique_usecases
            )
        else:
            tagged_count = sum(1 for ucs in cluster_usecases_map.values() if ucs)
            print(f"[USECASE] Reusing Phase 3.6 tagging: {tagged_count}/{len(themes)} clusters tagged")

        # Build reverse map: usecase -> set of theme_ids
        usecase_to_themes: Dict[str, set] = {}
        for tid, ucs in theme_id_usecases.items():
            for uc in ucs:
                usecase_to_themes.setdefault(uc, set()).add(tid)

        # --- Step 3: Boost + reassign unassigned messages ---
        n_rescued = 0
        for assignment in assignments:
            if assignment.assigned_theme_id >= 0:
                continue  # Already assigned (confident or borderline-reassigned)

            msg_uc = msg_usecase.get(assignment.message_idx)
            if not msg_uc:
                continue  # No usecase for this message

            # Find clusters tagged with this usecase
            matching_theme_ids = usecase_to_themes.get(msg_uc, set())
            if not matching_theme_ids:
                continue

            # Get fusion scores for all clusters, boost matching ones
            best_boosted_tid = -1
            best_boosted_score = 0.0

            for tid, fusion_score in assignment.all_similarities.items():
                if tid in matching_theme_ids:
                    boosted = fusion_score * boost_factor
                else:
                    boosted = fusion_score

                if boosted > best_boosted_score:
                    best_boosted_score = boosted
                    best_boosted_tid = tid

            # Reassign if boosted score exceeds threshold
            if best_boosted_score >= rescue_threshold and best_boosted_tid >= 0:
                assignment.assigned_theme_id = best_boosted_tid
                assignment.best_similarity = best_boosted_score
                assignment.status = AssignmentStatus.CONFIDENT
                assignment.was_usecase_rescued = True
                n_rescued += 1

        print(f"[USECASE] Rescued {n_rescued} messages from misc via usecase boost "
              f"(boost={boost_factor}, threshold={rescue_threshold})")
        return assignments, n_rescued, cluster_usecases_map

    def _fn_rescue(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
    ) -> tuple:
        """
        False-negative signal rescue — runs after usecase rescue.

        For each still-unassigned message, applies the following ruleset
        using pre-computed signal_scores (cosine, bm25_raw, bm25_norm, fusion):

          if bm25_raw == 0          → True misc (no lexical overlap) — skip
          elif fusion > fn_fusion_threshold            → Rescue (high confidence)
          elif bm25_norm > fn_bm25_norm_threshold
               AND fusion > fn_fusion_threshold_medium → Rescue (medium confidence)
          else                      → Keep as misc

        The best theme is taken from all_similarities (max fusion score).

        Args:
            themes: Final themes list (used to validate theme_ids)
            assignments: Current assignments (after usecase rescue)

        Returns:
            (assignments, n_fn_rescued)
        """
        sp_config = self.config.second_pass
        fn_fusion_threshold = sp_config.fn_fusion_threshold
        fn_bm25_norm_threshold = sp_config.fn_bm25_norm_threshold
        fn_fusion_threshold_medium = sp_config.fn_fusion_threshold_medium

        valid_theme_ids = {t.theme_id for t in themes}
        n_fn_rescued = 0

        for assignment in assignments:
            if assignment.assigned_theme_id >= 0:
                continue  # Already assigned

            signals = assignment.signal_scores
            if not signals:
                continue  # No signal data (empty message edge case)

            bm25_raw = signals.get("bm25_raw", 0.0)

            # Rule 1: No lexical overlap → true misc, never rescue
            if bm25_raw == 0:
                continue

            bm25_norm = signals.get("bm25_norm", 0.0)
            fusion = signals.get("fusion", 0.0)

            # Rules 2 & 3: Check if FN
            is_fn = (
                fusion > fn_fusion_threshold
                or (bm25_norm > fn_bm25_norm_threshold and fusion > fn_fusion_threshold_medium)
            )

            if not is_fn:
                continue

            # Find the best theme from all_similarities
            if not assignment.all_similarities:
                continue

            best_tid = max(
                (tid for tid in assignment.all_similarities if tid in valid_theme_ids),
                key=lambda tid: assignment.all_similarities[tid],
                default=None,
            )
            if best_tid is None:
                continue

            assignment.assigned_theme_id = best_tid
            assignment.best_similarity = assignment.all_similarities[best_tid]
            assignment.status = AssignmentStatus.CONFIDENT
            assignment.was_fn_rescued = True
            n_fn_rescued += 1

        print(f"[FN] Rescued {n_fn_rescued} false-negative messages "
              f"(fusion>{fn_fusion_threshold} OR bm25_norm>{fn_bm25_norm_threshold}+fusion>{fn_fusion_threshold_medium})")
        return assignments, n_fn_rescued

    def _build_output(
        self,
        themes: List[Theme],
        assignments: List[MessageAssignment],
        message_to_sessions: Dict[str, List[str]],
        quality_report: QualityReport,
        cluster_usecases_map: Optional[Dict[str, list]] = None,
    ) -> PipelineOutput:
        """
        Build PipelineOutput from themes and assignments.

        Args:
            themes: Final list of themes
            assignments: Final list of assignments
            message_to_sessions: Dict mapping message -> all session_ids
            quality_report: Quality report
            cluster_usecases_map: Optional mapping of cluster_name -> list of usecases

        Returns:
            PipelineOutput object
        """
        if cluster_usecases_map is None:
            cluster_usecases_map = {}

        # Pre-populate parent_usecases on all assignments from cluster_usecases_map
        theme_map = {t.theme_name: cluster_usecases_map.get(t.theme_name, []) for t in themes}
        cluster_title_map = {t.theme_name: cluster_usecases_map.get(t.theme_name, []) for t in themes}

        for assignment in assignments:
            if assignment.assigned_theme_id >= 0:
                # For assigned messages, look up parent_usecases by theme_id
                theme = next((t for t in themes if t.theme_id == assignment.assigned_theme_id), None)
                if theme:
                    assignment.parent_usecases = cluster_usecases_map.get(theme.theme_name, [])
            elif assignment.cluster_title:
                # For misc messages, look up by cluster_title (best matching theme)
                assignment.parent_usecases = cluster_usecases_map.get(assignment.cluster_title, [])

        # Group assignments by theme
        theme_assignments = {t.theme_id: [] for t in themes}
        misc_assignments = []

        for assignment in assignments:
            if assignment.assigned_theme_id >= 0:
                if assignment.assigned_theme_id in theme_assignments:
                    theme_assignments[assignment.assigned_theme_id].append(assignment)
                else:
                    misc_assignments.append(assignment)
            else:
                misc_assignments.append(assignment)

        # Build ClusterResult for each theme
        clusters = []
        for theme in themes:
            theme_msgs = theme_assignments.get(theme.theme_id, [])

            # Sort messages by similarity (descending) - most similar first
            theme_msgs.sort(key=lambda msg: msg.best_similarity, reverse=True)

            # Expand session_ids using message_to_sessions
            all_session_ids = []
            for assignment in theme_msgs:
                sessions = message_to_sessions.get(assignment.message_text, [])
                if sessions:
                    all_session_ids.extend(sessions)
                elif assignment.session_id:
                    all_session_ids.append(assignment.session_id)

            cluster = ClusterResult(
                theme=theme,
                messages=theme_msgs,
                session_ids=list(set(all_session_ids)),
                quality_metrics={
                    "coherence": theme.coherence_score or 0.0,
                    "message_count": len(theme_msgs),
                    "session_count": len(set(all_session_ids)),
                },
                parent_usecases=cluster_usecases_map.get(theme.theme_name, []),
            )
            clusters.append(cluster)

        # Sort by message count
        clusters.sort(key=lambda c: c.message_count, reverse=True)

        # Build miscellaneous cluster
        # Sort misc messages by similarity (descending) - closest to threshold first
        misc_assignments.sort(key=lambda msg: msg.best_similarity, reverse=True)

        misc_session_ids = []
        for assignment in misc_assignments:
            sessions = message_to_sessions.get(assignment.message_text, [])
            if sessions:
                misc_session_ids.extend(sessions)
            elif assignment.session_id:
                misc_session_ids.append(assignment.session_id)

        misc_theme = Theme(
            theme_id=-1,
            theme_name="Miscellaneous",
            description="Messages that could not be confidently assigned to any theme",
            key_phrases=[],
        )
        miscellaneous = ClusterResult(
            theme=misc_theme,
            messages=misc_assignments,
            session_ids=list(set(misc_session_ids)),
        )

        return PipelineOutput(
            clusters=clusters,
            miscellaneous=miscellaneous,
            quality_report=quality_report,
            themes=themes,
        )

    # Keywords that strongly indicate a post-purchase (post-sales) context.
    # Any cluster whose title+description contains one of these (word-level) will
    # be classified as post-sales by the keyword fallback when LLM tagging fails.
    # Keywords that strongly indicate a POST-purchase (post-sales) context.
    # Deliberately excludes: 'order', 'checkout', 'payment', 'cod', 'emi' — those
    # indicate the customer is TRYING to buy (pre-sales), not that an order is placed.
    _POST_SALES_KEYWORDS = frozenset({
        'delivery', 'delivered', 'track', 'tracking', 'cancel',
        'cancellation', 'return', 'refund', 'exchange', 'shipped',
        'received', 'dispatch', 'dispatched',
        'logistics', 'shipment', 'courier', 'transit',
        'replacement', 'damaged', 'wrong product', 'not received',
    })

    @staticmethod
    def _keyword_category(theme_name: str, description: str) -> str:
        """
        Fallback heuristic: classify a cluster as pre-sales or post-sales
        based on keyword presence in its name and description.

        Returns 'post-sales' if any strong post-purchase keyword is found;
        otherwise defaults to 'pre-sales'.
        """
        combined = (theme_name + ' ' + description).lower()
        if any(kw in combined for kw in ThemeClusteringPipeline._POST_SALES_KEYWORDS):
            return 'post-sales'
        return 'pre-sales'

    def _tag_categories(self, output: PipelineOutput) -> None:
        """
        Tag each cluster as pre-sales, post-sales, or miscellaneous using GPT 5 nano.

        Sends all cluster names + descriptions in one LLM call.
        A keyword-based fallback always fires for any cluster left untagged —
        so cluster.category is NEVER left as "" after this method returns.
        """
        try:
            from .theme_prompts import get_category_tagging_prompt
        except ImportError:
            from src.theme_clustering.theme_prompts import get_category_tagging_prompt

        # Always tag the Miscellaneous bucket directly
        if output.miscellaneous:
            output.miscellaneous.category = "miscellaneous"

        if not output.clusters:
            return

        # Build input for prompt
        cluster_inputs = [
            {
                "cluster_name": c.theme.theme_name,
                "description": c.theme.description,
            }
            for c in output.clusters
        ]

        prompt = get_category_tagging_prompt(cluster_inputs)

        try:
            from src.utils.openai_utils import GPT4Input, gpt_5_2_chat

            inputs = [
                GPT4Input(actor="system", text=prompt),
                GPT4Input(actor="user", text="Tag each cluster. Return ONLY the JSON array."),
            ]
            max_retries = 3
            response = None
            for attempt in range(1 + max_retries):
                try:
                    response = gpt_5_2_chat(inputs, temperature=0.2, max_tokens=1000, timeout=60)
                    break
                except Exception as _e:
                    _msg = str(_e)
                    if '401' in _msg or '403' in _msg or 'invalid_api_key' in _msg.lower() or 'incorrect api key' in _msg.lower():
                        print(f"[CATEGORY] Auth error (non-retryable): {_e}")
                        break
                    if attempt < max_retries:
                        _delay = 2 ** (attempt + 1)
                        print(f"[CATEGORY] LLM attempt {attempt + 1} failed: {_e!r}. Retrying in {_delay}s…")
                        time.sleep(_delay)
                    else:
                        print(f"[CATEGORY] LLM failed after {max_retries} retries: {_e!r}")

            if response and response.content:
                # Parse response
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```json")[-1] if "```json" in raw else raw.split("```")[1]
                    raw = raw.split("```")[0]
                raw = raw.strip()

                import json as _json
                tags = _json.loads(raw)

                # Build lookup: cluster_name -> category
                category_map = {}
                for item in tags:
                    name = item.get("cluster_name", "")
                    cat = item.get("category", "").lower().strip()
                    # Only accept pre-sales or post-sales from LLM
                    if cat in ("pre-sales", "post-sales"):
                        category_map[name] = cat

                # Apply to clusters
                tagged = 0
                for cluster in output.clusters:
                    cat = category_map.get(cluster.theme.theme_name)
                    if cat:
                        cluster.category = cat
                        tagged += 1

                print(f"[CATEGORY] LLM tagged {tagged}/{len(output.clusters)} clusters")
            else:
                print("[CATEGORY] Warning: Empty LLM response — will use keyword fallback for all clusters")

        except Exception as e:
            print(f"[CATEGORY] Warning: LLM category tagging failed ({e!r}) — keyword fallback will run")

        # ----------------------------------------------------------------
        # Keyword-based fallback for any cluster still without a category.
        # This runs regardless of whether LLM succeeded, ensuring every
        # cluster has a valid category before _add_performance_classifications().
        # ----------------------------------------------------------------
        fallback_count = 0
        for cluster in output.clusters:
            if not cluster.category:
                cluster.category = self._keyword_category(
                    cluster.theme.theme_name, cluster.theme.description
                )
                fallback_count += 1

        if fallback_count:
            print(f"[CATEGORY] Keyword fallback applied to {fallback_count} untagged clusters")

        print(
            f"[CATEGORY] Final: "
            f"pre-sales={sum(1 for c in output.clusters if c.category == 'pre-sales')}, "
            f"post-sales={sum(1 for c in output.clusters if c.category == 'post-sales')}"
        )

    def _add_cluster_metadata(
        self,
        output: PipelineOutput,
        chat_data: Dict[str, Any],
    ) -> None:
        """
        Compute per-cluster session metrics and store in cluster.metadata.

        Follows the exact logic of cluster_metadata.py:
          - unique_session_count
          - avg_human_messages_per_session
          - a2c_sessions_count / a2c_sessions_percentage
          - order_sessions_count / order_sessions_percentage
          - utm_sessions_count / utm_sessions_percentage

        A2C:   any customer message starting with "add to cart - " (case-insensitive)
        Order: session has a 'shopify_order_details' user_field key
        UTM:   order session where shopify_order_details.has_verifast_utm == True

        After computing per-cluster metrics, calls _add_performance_classifications()
        to derive percentile-based show flags.

        Args:
            output: PipelineOutput with clusters and miscellaneous
            chat_data: Dict mapping session_id -> {"messages": [...], "user_fields": [...], ...}
        """
        import json as _json

        all_clusters = list(output.clusters)
        if output.miscellaneous:
            all_clusters.append(output.miscellaneous)

        for cluster in all_clusters:
            unique_sessions = list(set(cluster.session_ids))

            if not unique_sessions:
                cluster.metadata = {
                    'unique_session_count': 0,
                    'avg_human_messages_per_session': 0.0,
                    'a2c_sessions_count': 0,
                    'a2c_sessions_percentage': 0.0,
                    'order_sessions_count': 0,
                    'order_sessions_percentage': 0.0,
                    'utm_sessions_count': 0,
                    'utm_sessions_percentage': 0.0,
                }
                continue

            total_human_messages = 0
            a2c_sessions: set = set()
            order_sessions: set = set()
            utm_sessions: set = set()
            missing_sessions = 0

            for session_id in unique_sessions:
                session_data = chat_data.get(session_id, {})
                if not session_data:
                    missing_sessions += 1
                    continue

                messages = session_data.get('messages', [])
                user_fields = session_data.get('user_fields', [])

                # Count human (customer) messages.
                # Some data sources use 'human' instead of 'customer' as the actor label.
                total_human_messages += sum(
                    1 for msg in messages
                    if msg.get('actor') in ('customer', 'human')
                )

                # A2C: first customer/human message starting with "add to cart - "
                for msg in messages:
                    if msg.get('actor') in ('customer', 'human'):
                        if msg.get('text', '').strip().lower().startswith('add to cart - '):
                            a2c_sessions.add(session_id)
                            break

                # Order + UTM: look for shopify_order_details user_field
                for uf in user_fields:
                    if uf.get('key') == 'shopify_order_details':
                        order_sessions.add(session_id)
                        val = uf.get('value', {})
                        if isinstance(val, str):
                            try:
                                val = _json.loads(val)
                            except _json.JSONDecodeError:
                                val = {}
                        if isinstance(val, dict) and val.get('has_verifast_utm') is True:
                            utm_sessions.add(session_id)

            # Warn if a large fraction of sessions are missing from chat_data
            if missing_sessions > 0:
                miss_pct = missing_sessions / len(unique_sessions) * 100
                if miss_pct > 20:
                    print(
                        f"[METADATA] Warning: {cluster.theme.theme_name} — "
                        f"{missing_sessions}/{len(unique_sessions)} sessions "
                        f"({miss_pct:.0f}%) not found in chat_data. "
                        f"Check IS_REPORT_FORMAT setting."
                    )

            n = len(unique_sessions)
            a2c_count = len(a2c_sessions)
            order_count = len(order_sessions)
            utm_count = len(utm_sessions)

            cluster.metadata = {
                'unique_session_count': n,
                'avg_human_messages_per_session': round(total_human_messages / n, 2) if n > 0 else 0.0,
                'a2c_sessions_count': a2c_count,
                'a2c_sessions_percentage': round(a2c_count / n * 100, 2) if n > 0 else 0.0,
                'order_sessions_count': order_count,
                'order_sessions_percentage': round(order_count / n * 100, 2) if n > 0 else 0.0,
                'utm_sessions_count': utm_count,
                'utm_sessions_percentage': round(utm_count / n * 100, 2) if n > 0 else 0.0,
            }

            print(
                f"[METADATA] {cluster.theme.theme_name}: "
                f"sessions={n}, "
                f"A2C={a2c_count}({cluster.metadata['a2c_sessions_percentage']}%), "
                f"orders={order_count}({cluster.metadata['order_sessions_percentage']}%), "
                f"UTM={utm_count}({cluster.metadata['utm_sessions_percentage']}%)"
            )

        # Now compute percentile-based performance classifications
        self._add_performance_classifications(output)

    def _add_performance_classifications(
        self,
        output: PipelineOutput,
    ) -> None:
        """
        Derive percentile-based performance classifications for each cluster.

        Percentile thresholds (25th / 75th) are computed from pre-sales clusters ONLY.
        Show flags (show_a2c, show_order, show_utm) are set only for clusters that are
        BOTH significant (>=2.5% session volume AND >=25 sessions) AND pre-sales.

        Follows the exact logic of enhance_cluster_metadata.py.
        Sets cluster.performance for every cluster and output.global_metadata.

        Args:
            output: PipelineOutput (cluster.metadata must already be populated)
        """
        # Collect metric values from pre-sales clusters only
        a2c_pcts: List[float] = []
        order_pcts: List[float] = []
        utm_pcts: List[float] = []

        for cluster in output.clusters:
            if cluster.category != 'pre-sales':
                print(
                    f"[METADATA] Skipping '{cluster.theme.theme_name}' "
                    f"(category={cluster.category!r}) from percentile calculation"
                )
                continue
            meta = cluster.metadata
            if not meta:
                continue
            a2c_pcts.append(meta.get('a2c_sessions_percentage', 0.0))
            order_pcts.append(meta.get('order_sessions_percentage', 0.0))
            utm_pcts.append(meta.get('utm_sessions_percentage', 0.0))

        # If no pre-sales clusters qualified (can happen if all clusters are post-sales or
        # categories were not set), fall back to ALL clusters so medians are not 0.
        if not a2c_pcts:
            print(
                "[METADATA] Warning: no pre-sales clusters for percentile calc — "
                "falling back to all clusters"
            )
            for cluster in output.clusters:
                meta = cluster.metadata
                if not meta:
                    continue
                a2c_pcts.append(meta.get('a2c_sessions_percentage', 0.0))
                order_pcts.append(meta.get('order_sessions_percentage', 0.0))
                utm_pcts.append(meta.get('utm_sessions_percentage', 0.0))

        # Compute medians + percentile thresholds
        global_metrics: Dict[str, Any] = {}
        percentile_thresholds: Dict[str, float] = {}

        if a2c_pcts:
            global_metrics['median_a2c_percentage'] = round(float(np.median(a2c_pcts)), 2)
            global_metrics['median_order_percentage'] = round(float(np.median(order_pcts)), 2)
            global_metrics['median_utm_percentage'] = round(float(np.median(utm_pcts)), 2)
            percentile_thresholds['a2c_25th'] = round(float(np.percentile(a2c_pcts, 25)), 2)
            percentile_thresholds['a2c_75th'] = round(float(np.percentile(a2c_pcts, 75)), 2)
            percentile_thresholds['order_25th'] = round(float(np.percentile(order_pcts, 25)), 2)
            percentile_thresholds['order_75th'] = round(float(np.percentile(order_pcts, 75)), 2)
            percentile_thresholds['utm_25th'] = round(float(np.percentile(utm_pcts, 25)), 2)
            percentile_thresholds['utm_75th'] = round(float(np.percentile(utm_pcts, 75)), 2)
        else:
            global_metrics['median_a2c_percentage'] = 0.0
            global_metrics['median_order_percentage'] = 0.0
            global_metrics['median_utm_percentage'] = 0.0
            for metric in ['a2c', 'order', 'utm']:
                percentile_thresholds[f'{metric}_25th'] = 0.0
                percentile_thresholds[f'{metric}_75th'] = 0.0

        global_metrics['percentile_thresholds'] = percentile_thresholds

        print(
            f"[METADATA] Global medians (pre-sales only — {len(a2c_pcts)} clusters): "
            f"A2C={global_metrics['median_a2c_percentage']}%, "
            f"Order={global_metrics['median_order_percentage']}%, "
            f"UTM={global_metrics['median_utm_percentage']}%"
        )
        print(
            f"[METADATA] Percentile thresholds: "
            f"A2C 25th/75th={percentile_thresholds['a2c_25th']}/{percentile_thresholds['a2c_75th']}%  "
            f"Order 25th/75th={percentile_thresholds['order_25th']}/{percentile_thresholds['order_75th']}%  "
            f"UTM 25th/75th={percentile_thresholds['utm_25th']}/{percentile_thresholds['utm_75th']}%"
        )

        # Total sessions = sum of unique_session_count per cluster (same as enhance_cluster_metadata.py)
        all_clusters = list(output.clusters)
        if output.miscellaneous:
            all_clusters.append(output.miscellaneous)

        total_sessions = sum(
            c.metadata.get('unique_session_count', 0) for c in all_clusters if c.metadata
        )
        print(f"[METADATA] Total sessions (sum across all clusters): {total_sessions}")

        # Classify every cluster (including miscellaneous)
        for cluster in all_clusters:
            meta = cluster.metadata or {}
            n = meta.get('unique_session_count', 0)
            session_volume_pct = (n / total_sessions * 100) if total_sessions > 0 else 0.0
            is_significant = (session_volume_pct >= 2.5) and (n >= 25)
            is_pre_sales = (cluster.category == 'pre-sales')

            a2c_pct = meta.get('a2c_sessions_percentage', 0.0)
            order_pct = meta.get('order_sessions_percentage', 0.0)
            utm_pct = meta.get('utm_sessions_percentage', 0.0)

            performance: Dict[str, Any] = {
                'session_volume_percentage': round(session_volume_pct, 2),
                'session_count': n,
                'is_significant': is_significant,
                'sales_stage': cluster.category,
                'a2c_performance': 'average',
                'order_performance': 'average',
                'utm_performance': 'average',
                'show_a2c': False,
                'show_order': False,
                'show_utm': False,
            }

            # A2C performance
            if a2c_pct <= percentile_thresholds.get('a2c_25th', 0):
                performance['a2c_performance'] = 'poor'
                performance['show_a2c'] = is_significant and is_pre_sales
            elif a2c_pct >= percentile_thresholds.get('a2c_75th', 0):
                performance['a2c_performance'] = 'good'
                performance['show_a2c'] = is_significant and is_pre_sales

            # Order performance
            if order_pct <= percentile_thresholds.get('order_25th', 0):
                performance['order_performance'] = 'poor'
                performance['show_order'] = is_significant and is_pre_sales
            elif order_pct >= percentile_thresholds.get('order_75th', 0):
                performance['order_performance'] = 'good'
                performance['show_order'] = is_significant and is_pre_sales

            # UTM performance
            if utm_pct <= percentile_thresholds.get('utm_25th', 0):
                performance['utm_performance'] = 'poor'
                performance['show_utm'] = is_significant and is_pre_sales
            elif utm_pct >= percentile_thresholds.get('utm_75th', 0):
                performance['utm_performance'] = 'good'
                performance['show_utm'] = is_significant and is_pre_sales

            cluster.performance = performance

            print(
                f"[METADATA] Performance — {cluster.theme.theme_name}: "
                f"vol={session_volume_pct:.1f}%, significant={is_significant}, "
                f"A2C={performance['a2c_performance']}, "
                f"order={performance['order_performance']}, "
                f"UTM={performance['utm_performance']}"
            )

        output.global_metadata = global_metrics

    def _save_outputs(
        self,
        output: PipelineOutput,
        file_prefix: str,
        client_display: str = "",
        page_url: Optional[str] = None,
        output_dir_override: Optional[str] = None,
        report_month: Optional[str] = None,
    ):
        """
        Save outputs to files.

        All filenames follow the convention: {file_prefix}_{file_id}.ext
        where file_prefix = {client_slug}_{page_slug} (or just client_slug if
        no page_name was provided). This prevents race conditions when running
        multiple pipeline instances in parallel for different client/page combos.

        Args:
            output: PipelineOutput to save
            file_prefix: Combined client+page slug for file naming
            client_display: Human-readable client name for report headers
            page_url: Raw page URL passed to the VoC report for display
            output_dir_override: Optional path to override self.config.output.output_dir
            report_month: Optional month string (e.g., "February 2026") for VoC report header
        """
        output_dir = output_dir_override or self.config.output.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save final clusters (always generated)
        clusters_file = os.path.join(output_dir, f"{file_prefix}_clusters.json")
        with open(clusters_file, 'w', encoding='utf-8') as f:
            json.dump(output.to_export_dict(), f, ensure_ascii=False, indent=2)
        print(f"[OUTPUT] Saved clusters to: {clusters_file}")

        # Save themes separately (toggleable)
        if self.config.output.generate_themes_json:
            themes_file = os.path.join(output_dir, f"{file_prefix}_themes.json")
            themes_data = [t.to_dict() for t in output.themes]
            with open(themes_file, 'w', encoding='utf-8') as f:
                json.dump(themes_data, f, ensure_ascii=False, indent=2)
            print(f"[OUTPUT] Saved themes to: {themes_file}")

        # Generate client-facing VoC report (enabled by default)
        if self.config.output.generate_voc_report:
            voc_dir = os.path.join(output_dir, '..', 'voc_reports')
            os.makedirs(voc_dir, exist_ok=True)
            voc_path = os.path.join(voc_dir, f"clusters_{file_prefix}_voc_report.html")
            try:
                try:
                    from .theme_voc_report import generate_voc_report
                except ImportError:
                    from src.theme_clustering.theme_voc_report import generate_voc_report

                generate_voc_report(
                    clusters_export=output.to_export_dict(),
                    output_path=voc_path,
                    client_display=client_display or file_prefix,
                    page_url=page_url,
                    report_month=report_month,
                )
                print(f"[OUTPUT] Saved VoC report to: {voc_path}")
            except Exception as e:
                print(f"[OUTPUT] Warning: Failed to generate VoC report: {e}")

        # Generate internal debug HTML report (disabled by default)
        if self.config.output.generate_html_report:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = os.path.join(output_dir, f"debug_report_{file_prefix}_{ts}.html")
            try:
                try:
                    from .theme_html_report import generate_html_report
                except ImportError:
                    from src.theme_clustering.theme_html_report import generate_html_report

                generate_html_report(
                    clusters_data=output.to_export_dict(),
                    quality_report=output.quality_report.to_dict(),
                    output_path=html_file,
                    client_name=file_prefix,
                )
                print(f"[OUTPUT] Saved debug HTML report to: {html_file}")
            except Exception as e:
                print(f"[OUTPUT] Warning: Failed to generate debug HTML report: {e}")


def run_pipeline(
    messages: List[str],
    session_ids: Optional[List[str]] = None,
    user_intents: Optional[List[str]] = None,
    seed_themes: Optional[List[str]] = None,
    client_name: str = "default",
    page_name: Optional[str] = None,
    chat_data: Optional[Dict[str, Any]] = None,
    config: ThemeClusteringConfig = None,
    save_outputs: bool = True,
    output_dir_override: Optional[str] = None,
    report_month: Optional[str] = None,
) -> Optional[PipelineOutput]:
    """
    Convenience function to run the theme clustering pipeline.

    Args:
        messages: List of customer message texts
        session_ids: Optional list of session IDs
        user_intents: Optional list of usecase labels (parallel to messages).
                      Enables usecase-aware misc rescue when provided.
        seed_themes: Optional list of seed theme names
        client_name: Client name slug (e.g. "flicka-cosmetics")
        page_name: Optional page slug for parallel-safe file naming
        chat_data: Optional raw session data dict for cluster metadata computation
        config: Optional custom configuration
        save_outputs: Whether to save outputs to files
        output_dir_override: Optional path to override the default output directory
        report_month: Optional month string (e.g., "February 2026") for VoC report header

    Returns:
        PipelineOutput with clusters, miscellaneous, and quality report
    """
    pipeline = ThemeClusteringPipeline(config)
    return pipeline.run(
        messages=messages,
        session_ids=session_ids,
        user_intents=user_intents,
        seed_themes=seed_themes,
        client_name=client_name,
        page_name=page_name,
        chat_data=chat_data,
        save_outputs=save_outputs,
        output_dir_override=output_dir_override,
        report_month=report_month,
    )


# =============================================================================
# MAIN FUNCTION - Configure inputs here
# =============================================================================

def main():
    """
    Main entry point for running the theme clustering pipeline.

    Configure your inputs directly in this function - no command line args needed.
    """

    # =========================================================================
    # CONFIGURATION - MODIFY THESE VALUES
    # =========================================================================

    # Path to your chat data JSON file
    CHAT_DATA_PATH = "./output/february_all_report/typsy-beauty.myshopify.com/raw/typsy-beauty.myshopify.com.json"

    # Full Shopify domain — slugs for all output files are auto-derived from this.
    # Example: "flicka-cosmetics-india.myshopify.com" → prefix "flicka-cosmetics-india"
    CLIENT_NAME = "typsy-beauty.myshopify.com"

    # Seed themes (optional) - themes you want to ensure are included
    # Set to None if you want fully automatic discovery
    SEED_THEMES = None  # Or: ["Delivery & Shipping", "Returns & Refunds", "Product Usage"]

    # Whether the data is in "report" format vs "daily" format
    IS_REPORT_FORMAT = True

    # =========================================================================
    # FILTERING OPTIONS (from vector.py)
    # =========================================================================

    # Filter by specific bot pages (full URLs). None = include all pages.
    # When GET_TOP_PDT_PAGE is True this list is overridden automatically.
    FILTER_PAGES = None # Or: None

    # Automatically discover and use the top product page(s) from the data.
    # When True, FILTER_PAGES is replaced by the top TOP_K_PAGES pages.
    GET_TOP_PDT_PAGE = False
    TOP_K_PAGES = None  # How many top pages to pick
    # "together" — cluster all selected pages in one combined run (single output).
    # "separate" — run the pipeline independently for each page (one output per page).
    PAGE_METHOD = "separate"

    # Filter by keyword in bot_page (None = no keyword filter)
    FILTER_KEYWORD = None  # Or: "saree" to include only pages containing "saree"

    # Filter by specific secondary usecases (None = include all)
    FILTER_USECASES = None  # Or: ["usecase1", "usecase2"]

    # Filter to messages preceding AI responses containing these phrases (None = include all)
    # Can be a single string or list of strings
    FILTER_PHRASE = ["I don't have"]  # Or: "I recommend" or ["phrase1", "phrase2"]

    # Remove bubble-click messages (frequent, repeated messages)
    REMOVE_BUBBLES = True

    # Maximum messages to process (sampling will be applied if exceeded during loading)
    # Set to None for no limit, or a number like 3000
    MAX_MESSAGES_LOAD = None

    # =========================================================================
    # PIPELINE CONFIGURATION
    # =========================================================================

    try:
        from .theme_config import ThemeClusteringConfig, SamplingConfig
    except ImportError:
        from theme_config import ThemeClusteringConfig, SamplingConfig

    config = ThemeClusteringConfig(
        sampling=SamplingConfig(use_hdbscan=True),
    )

    # =========================================================================
    # SETUP
    # =========================================================================

    from src.data.vector_helpers import load_messages_from_report, get_top_frequent_pages
    from src.data.cluster_metadata_helpers import load_chat_data

    # ── Auto top-page detection ────────────────────────────────────────────
    if GET_TOP_PDT_PAGE:
        top_pages = get_top_frequent_pages(CHAT_DATA_PATH, report=IS_REPORT_FORMAT, top_k=TOP_K_PAGES)
        if top_pages:
            FILTER_PAGES = top_pages
            print(f"[Auto-page] Top {TOP_K_PAGES} page(s): {FILTER_PAGES}")
        else:
            print("[Auto-page] Warning: no pages found — running on ALL pages (no filter)")
            FILTER_PAGES = None

    # ── Load full chat data (shared across all page runs) ──────────────────
    chat_data = None
    try:
        chat_data = load_chat_data(CHAT_DATA_PATH, report=IS_REPORT_FORMAT)
    except Exception as _e:
        print(f"[WARN] Failed to load chat data for metadata phase ({_e}). "
              f"Cluster metadata (A2C/orders/UTM) will be skipped.")

    # =========================================================================
    # SEPARATE-PAGE MODE: run the pipeline once per page
    # =========================================================================

    def _load_df(filter_pages):
        if IS_REPORT_FORMAT:
            return load_messages_from_report(
                CHAT_DATA_PATH,
                remove_bubbles=REMOVE_BUBBLES,
                filter_pages=filter_pages,
                filter_secondary_usecases=FILTER_USECASES,
                phrase=FILTER_PHRASE,
                keyword=FILTER_KEYWORD,
                max_messages=MAX_MESSAGES_LOAD or 10000,
            )
        else:
            return load_messages(
                CHAT_DATA_PATH,
                remove_bubbles=REMOVE_BUBBLES,
                filter_pages=filter_pages,
                filter_secondary_usecases=FILTER_USECASES,
                phrase=FILTER_PHRASE,
                keyword=FILTER_KEYWORD,
                max_messages=MAX_MESSAGES_LOAD or 10000,
            )

    if FILTER_PAGES and len(FILTER_PAGES) > 1 and PAGE_METHOD == "separate":
        outputs = []
        for i, page_url in enumerate(FILTER_PAGES, 1):
            print(f"\n[PAGE {i}/{len(FILTER_PAGES)}] {page_url}")
            df_p = _load_df([page_url])
            if df_p is None or df_p.empty:
                print(f"  No messages found for page — skipping.")
                continue

            messages_p    = df_p['text'].tolist()
            session_ids_p = df_p['session_id'].tolist()
            user_intents_p = df_p['user_intent'].tolist() if 'user_intent' in df_p.columns else None

            print(f"  Loaded {len(messages_p)} messages, {len(set(session_ids_p))} sessions")

            out = run_pipeline(
                messages=messages_p,
                session_ids=session_ids_p,
                user_intents=user_intents_p,
                seed_themes=SEED_THEMES,
                client_name=CLIENT_NAME,
                page_name=page_url,
                chat_data=chat_data,
                config=config,
                save_outputs=True,
            )
            if out is not None:
                outputs.append(out)
        return outputs

    # =========================================================================
    # TOGETHER / SINGLE-PAGE MODE
    # =========================================================================

    print("Loading chat data with filters...")
    df = _load_df(FILTER_PAGES)

    if df is None or df.empty:
        print("[ERROR] No messages loaded after filtering. Check FILTER_PAGES and CHAT_DATA_PATH.")
        return None

    messages    = df['text'].tolist()
    session_ids = df['session_id'].tolist()
    user_intents = df['user_intent'].tolist() if 'user_intent' in df.columns else None

    if user_intents:
        n_intents = len(set(u for u in user_intents if u and u != "unknown"))
        print(f"  - User intents: {n_intents} unique secondary usecases")

    print(f"Loaded {len(messages)} customer messages after filtering")
    print(f"  - Unique sessions: {len(set(session_ids))}")

    # For naming: use page URL only when there is exactly one page in the filter.
    # When FILTER_PAGES was expanded (multi-variant), the canonical URL is FILTER_PAGES[0]
    # as returned by get_top_frequent_pages (or the user-supplied first URL).
    page_for_naming = FILTER_PAGES[0] if FILTER_PAGES else None

    output = run_pipeline(
        messages=messages,
        session_ids=session_ids,
        user_intents=user_intents,
        seed_themes=SEED_THEMES,
        client_name=CLIENT_NAME,
        page_name=page_for_naming,
        chat_data=chat_data,
        config=config,
        save_outputs=True,
    )

    if output is None:
        return None

    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for i, cluster in enumerate(output.clusters):
        print(f"\n{i+1}. {cluster.theme.theme_name}")
        print(f"   Messages: {cluster.message_count}, Sessions: {cluster.session_count}")
        print(f"   Key phrases: {', '.join(cluster.theme.key_phrases[:3])}...")
        print(f"   Sample: \"{cluster.messages[0].message_text[:80]}...\"" if cluster.messages else "")

    if output.miscellaneous:
        print(f"\nMiscellaneous: {output.miscellaneous.message_count} messages")

    return output


if __name__ == "__main__":
    main()
