"""
Theme Clustering Discovery Module (Phase 2)

GPT-based theme discovery with batch processing and consolidation.
"""

import json
import re
import time
from typing import List, Optional, Dict, Any

from .theme_config import DiscoveryConfig, DEFAULT_CONFIG
from .theme_models import Theme
from .theme_prompts import (
    get_theme_discovery_prompt,
    get_cross_batch_merge_pairs_prompt,
)

# Import GPT utilities from parent directory
import sys
# sys.path removed - using package imports
from src.utils.openai_utils import gpt_5_2_chat, GPT4Input


class ThemeDiscovery:
    """
    GPT-based theme discovery from customer messages.

    Workflow:
    1. Batch messages (100 per GPT call)
    2. Extract themes with 4-6 key phrases each
    3. Consolidate if multiple batches produce overlapping themes
    4. Output: 10-25 themes with name, description, key_phrases
    """

    def __init__(self, config: DiscoveryConfig = None):
        self.config = config or DEFAULT_CONFIG.discovery

    def _gpt_call(
        self,
        prompt: str,
        user_text: str,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None,
    ) -> Optional[str]:
        """
        Make a GPT API call with automatic retry on transient failures.

        Retries up to 3 times with exponential back-off (2s, 4s, 8s).
        Does NOT retry on 401/403 (auth errors) — returns None immediately.

        Args:
            prompt: System prompt
            user_text: User message
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            timeout: Request timeout

        Returns:
            Response content or None on failure
        """
        temperature = temperature or self.config.gpt_temperature
        max_tokens = max_tokens or self.config.gpt_max_tokens
        timeout = timeout or self.config.gpt_timeout

        gpt_inputs = [
            GPT4Input(actor="system", text=prompt),
            GPT4Input(actor="user", text=user_text),
        ]

        max_retries = 3
        for attempt in range(1 + max_retries):
            try:
                result = gpt_5_2_chat(
                    gpt_inputs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                if result and hasattr(result, 'content'):
                    return result.content
                return None
            except Exception as e:
                msg = str(e)
                if '401' in msg or '403' in msg or 'invalid_api_key' in msg.lower() or 'incorrect api key' in msg.lower():
                    print(f"[DISCOVERY] Auth error (non-retryable): {e}")
                    return None
                if attempt < max_retries:
                    delay = 2 ** (attempt + 1)  # 2, 4, 8
                    print(f"[DISCOVERY] LLM call attempt {attempt + 1} failed: {e!r}. Retrying in {delay}s…")
                    time.sleep(delay)
                else:
                    print(f"[DISCOVERY] LLM call failed after {max_retries} retries: {e!r}")
                    return None
        return None

    def _parse_themes_json(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse themes from GPT JSON response.

        Args:
            response: GPT response string

        Returns:
            List of theme dictionaries
        """
        if not response:
            return []

        try:
            # Clean response
            cleaned = response.strip()

            # Remove markdown code blocks if present
            if cleaned.startswith("```json"):
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            # Find JSON array
            json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            themes = json.loads(cleaned)

            if isinstance(themes, list):
                return themes
            else:
                print(f"[DISCOVERY] Warning: Expected list, got {type(themes)}")
                return []

        except json.JSONDecodeError as e:
            print(f"[DISCOVERY] JSON parse error: {e}")
            print(f"[DISCOVERY] Response preview: {response[:500]}...")
            return []

    def _validate_theme(self, theme_dict: Dict[str, Any]) -> bool:
        """
        Validate a theme dictionary has required fields.

        Args:
            theme_dict: Theme dictionary from GPT

        Returns:
            True if valid
        """
        required_fields = ["theme_name", "key_phrases"]

        for field in required_fields:
            if field not in theme_dict:
                return False

        # Check key_phrases is a non-empty list
        if not isinstance(theme_dict["key_phrases"], list):
            return False
        if len(theme_dict["key_phrases"]) < self.config.min_key_phrases:
            return False

        return True

    def discover_themes_batch(
        self,
        messages: List[str],
        seed_themes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Discover themes from a single batch of messages.

        Args:
            messages: List of message texts (should be <= messages_per_gpt_call)
            seed_themes: Optional seed themes to include

        Returns:
            List of theme dictionaries
        """
        prompt = get_theme_discovery_prompt(
            messages=messages,
            seed_themes=seed_themes,
            min_themes=self.config.min_themes,
            max_themes=100,  # No artificial cap — let GPT discover as many as needed
            key_phrases_per_theme=self.config.key_phrases_per_theme,
        )

        response = self._gpt_call(
            prompt=prompt,
            user_text="Analyze these customer messages and extract themes.",
        )

        if not response:
            print("[DISCOVERY] GPT call failed, no response")
            return []

        themes = self._parse_themes_json(response)

        # Validate and filter
        valid_themes = []
        for theme in themes:
            if self._validate_theme(theme):
                valid_themes.append(theme)
            else:
                print(f"[DISCOVERY] Skipping invalid theme: {theme.get('theme_name', 'Unknown')}")

        print(f"[DISCOVERY] Batch discovered {len(valid_themes)} valid themes")

        # No refinement needed - discovery prompt already generates high-quality phrases
        return valid_themes

    def consolidate_themes(
        self,
        themes_batch_1: List[Dict[str, Any]],
        themes_batch_2: List[Dict[str, Any]],
        seed_themes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate themes from two batches using a pairs-only approach.

        Instead of asking the LLM to re-output ALL themes (which causes token overflow
        and catastrophic theme loss), we ask only for the pairs of cross-batch duplicates,
        then merge them programmatically.

        Safe by default: if the LLM returns [], all themes are kept unchanged.

        Args:
            themes_batch_1: Themes from first batch
            themes_batch_2: Themes from second batch
            seed_themes: Seed themes to preserve (never merged away)

        Returns:
            Deduplicated list of themes (all input themes preserved unless a pair is merged)
        """
        print(f"[DISCOVERY] Deduplicating {len(themes_batch_1)} + {len(themes_batch_2)} themes "
              f"(pairs-only approach)")

        prompt = get_cross_batch_merge_pairs_prompt(themes_batch_1, themes_batch_2)

        # Output is tiny (just pairs), so a small token budget is fine
        response = self._gpt_call(
            prompt=prompt,
            user_text="Identify cross-batch duplicate pairs only. Return [] if none.",
            temperature=0.1,
            max_tokens=1500,
        )

        if not response:
            print("[DISCOVERY] Deduplication LLM call failed — keeping all themes unchanged")
            return themes_batch_1 + themes_batch_2

        merge_pairs = self._parse_merge_pairs_response(response)
        print(f"[DISCOVERY] LLM identified {len(merge_pairs)} duplicate pair(s) to merge")

        # Apply pairs programmatically
        result = self._apply_merge_pairs(themes_batch_1, themes_batch_2, merge_pairs, seed_themes)
        print(f"[DISCOVERY] Deduplicated to {len(result)} themes")
        return result

    def _parse_merge_pairs_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the pairs-only merge response.

        Expected format: [{"batch1": "Theme A", "batch2": "Theme B"}, ...]
        Returns list of dicts with "batch1" and "batch2" keys.
        Falls back to [] on any parse error (safe — no merging).
        """
        if not response:
            return []

        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            # Handle case where model returns just "[]"
            cleaned = cleaned.strip()
            if cleaned == "[]":
                return []

            parsed = json.loads(cleaned)

            if not isinstance(parsed, list):
                print(f"[DISCOVERY] Pairs response was not a list: {type(parsed)}")
                return []

            # Validate each pair
            valid_pairs = []
            for item in parsed:
                if isinstance(item, dict) and "batch1" in item and "batch2" in item:
                    valid_pairs.append({"batch1": item["batch1"], "batch2": item["batch2"]})
            return valid_pairs

        except json.JSONDecodeError as e:
            print(f"[DISCOVERY] Pairs parse error: {e} — skipping all merges (safe fallback)")
            return []

    def _apply_merge_pairs(
        self,
        themes_batch_1: List[Dict[str, Any]],
        themes_batch_2: List[Dict[str, Any]],
        merge_pairs: List[Dict[str, str]],
        seed_themes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Programmatically merge batch 2 themes into batch 1 based on identified pairs.

        For each valid pair:
          - Find the matching themes in each batch by name
          - Combine key_phrases (deduplicated, capped at 12)
          - Keep batch 1's theme name, description, and example_messages
          - Mark the batch 2 theme as merged (exclude from final list)

        All unmatched themes from both batches are kept unchanged.
        Seed themes are never merged away.
        """
        b1_by_name = {t.get("theme_name", ""): t for t in themes_batch_1}
        b2_by_name = {t.get("theme_name", ""): t for t in themes_batch_2}

        seed_set = set(s.lower() for s in (seed_themes or []))
        b2_merged_names: set = set()

        result = list(themes_batch_1)  # Start with all batch 1 themes

        for pair in merge_pairs:
            b1_name = pair["batch1"]
            b2_name = pair["batch2"]

            # Skip if either theme doesn't exist in the respective batch
            if b1_name not in b1_by_name or b2_name not in b2_by_name:
                print(f"[DISCOVERY] Pair skipped (theme not found): '{b1_name}' ↔ '{b2_name}'")
                continue

            # Never merge away a seed theme
            if b1_name.lower() in seed_set or b2_name.lower() in seed_set:
                print(f"[DISCOVERY] Pair skipped (seed theme protected): '{b1_name}' ↔ '{b2_name}'")
                continue

            b1_theme = b1_by_name[b1_name]
            b2_theme = b2_by_name[b2_name]

            # Merge phrases: union, deduplicated, capped at 12
            b1_phrases = b1_theme.get("key_phrases", [])
            b2_phrases = b2_theme.get("key_phrases", [])
            seen_phrases: set = set()
            merged_phrases = []
            for p in b1_phrases + b2_phrases:
                p_lower = p.lower().strip()
                if p_lower not in seen_phrases:
                    seen_phrases.add(p_lower)
                    merged_phrases.append(p)
            b1_theme["key_phrases"] = merged_phrases[:12]

            # Merge example_messages (keep unique)
            b1_examples = b1_theme.get("example_messages", [])
            b2_examples = b2_theme.get("example_messages", [])
            all_examples = list(dict.fromkeys(b1_examples + b2_examples))
            b1_theme["example_messages"] = all_examples[:4]

            b2_merged_names.add(b2_name)
            print(f"[DISCOVERY] Merged: '{b2_name}' → '{b1_name}' "
                  f"({len(merged_phrases)} combined phrases)")

        # Add all batch 2 themes not consumed by a merge
        for theme in themes_batch_2:
            if theme.get("theme_name", "") not in b2_merged_names:
                result.append(theme)

        return result

    def discover_themes(
        self,
        messages: List[str],
        seed_themes: Optional[List[str]] = None,
    ) -> List[Theme]:
        """
        Main theme discovery method.

        Handles batching for large datasets and consolidation.

        Args:
            messages: All messages to analyze (can be pre-sampled)
            seed_themes: Optional seed themes to include

        Returns:
            List of Theme objects
        """
        n = len(messages)
        batch_size = self.config.messages_per_gpt_call

        print(f"[DISCOVERY] Starting theme discovery on {n} messages")
        print(f"[DISCOVERY] Batch size: {batch_size}")

        # Calculate number of batches
        n_batches = (n + batch_size - 1) // batch_size
        print(f"[DISCOVERY] Will process {n_batches} batch(es)")

        # Track themes per discovery batch (not a half-split — actual batches)
        themes_by_batch: List[List[Dict[str, Any]]] = []

        # Process batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n)
            batch_messages = messages[start_idx:end_idx]

            print(f"[DISCOVERY] Processing batch {i+1}/{n_batches} ({len(batch_messages)} messages)")

            batch_themes = self.discover_themes_batch(
                messages=batch_messages,
                seed_themes=seed_themes if i == 0 else None,  # Seed only in first batch
            )

            if batch_themes:
                themes_by_batch.append(batch_themes)

        # Consolidate: only run cross-batch deduplication when we have 2+ distinct batches.
        # With a single batch the discovery prompt already ensures internal distinctness —
        # running consolidation on arbitrary halves of the same batch confuses the model.
        if len(themes_by_batch) == 0:
            all_themes = []
        elif len(themes_by_batch) == 1:
            all_themes = themes_by_batch[0]
            print(f"[DISCOVERY] Single batch — skipping deduplication (all themes already distinct)")
        else:
            # Progressive consolidation: fold batches in from left to right
            all_themes = themes_by_batch[0]
            for batch_idx, next_batch in enumerate(themes_by_batch[1:], start=2):
                print(f"[DISCOVERY] Deduplicating batch 1..{batch_idx-1} ({len(all_themes)} themes) "
                      f"vs batch {batch_idx} ({len(next_batch)} themes)")
                all_themes = self.consolidate_themes(all_themes, next_batch, seed_themes)

            print(f"[DISCOVERY] Post-consolidation: {len(all_themes)} themes")

        # Convert to Theme objects
        themes = []
        for idx, theme_dict in enumerate(all_themes):
            theme = Theme(
                theme_id=idx,
                theme_name=theme_dict.get("theme_name", f"Theme {idx}"),
                description=theme_dict.get("description", ""),
                key_phrases=theme_dict.get("key_phrases", []),
                example_messages=theme_dict.get("example_messages", []),
                is_seed=self._is_seed_theme(theme_dict.get("theme_name", ""), seed_themes),
            )
            themes.append(theme)

        print(f"[DISCOVERY] Final: {len(themes)} themes discovered")

        # Log themes
        for theme in themes:
            seed_marker = " [SEED]" if theme.is_seed else ""
            print(f"  - {theme.theme_name}{seed_marker}: {len(theme.key_phrases)} phrases")

        return themes

    def _is_seed_theme(
        self,
        theme_name: str,
        seed_themes: Optional[List[str]],
    ) -> bool:
        """Check if theme name matches any seed theme."""
        if not seed_themes:
            return False

        theme_lower = theme_name.lower()
        for seed in seed_themes:
            if seed.lower() in theme_lower or theme_lower in seed.lower():
                return True

        return False


def discover_themes(
    messages: List[str],
    seed_themes: Optional[List[str]] = None,
    config: DiscoveryConfig = None,
) -> List[Theme]:
    """
    Convenience function for theme discovery.

    Args:
        messages: Messages to analyze
        seed_themes: Optional seed themes
        config: Optional discovery config

    Returns:
        List of Theme objects
    """
    discovery = ThemeDiscovery(config)
    return discovery.discover_themes(messages, seed_themes)
