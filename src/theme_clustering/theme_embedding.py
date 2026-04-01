"""
Theme Clustering Embedding Module (Phase 3)

Key phrase embedding and centroid calculation for themes.
Also handles message embedding.
"""

import os
from typing import List, Optional, Dict, Tuple
import numpy as np

from .theme_config import EmbeddingConfig, DEFAULT_CONFIG
from .theme_models import Theme
from .theme_preprocessing import TextPreprocessor

# Import embedding utilities from parent directory
import sys
# sys.path removed - using package imports
from src.utils.openai_utils import get_embedding_model


class ThemeEmbedder:
    """
    Embedder for themes and messages.

    Handles:
    1. Embedding all key phrases for each theme
    2. Computing theme centroids (mean of phrase embeddings)
    3. Embedding messages for assignment
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or DEFAULT_CONFIG.embedding
        self._model = None

    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            print(f"[EMBEDDING] Loading embedding model: {self.config.embedding_model}")
            self._model = get_embedding_model()
        return self._model

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress

        Returns:
            Embedding matrix (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        n = len(texts)
        batch_size = self.config.embedding_batch_size
        all_embeddings = []

        for i in range(0, n, batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (n + batch_size - 1) // batch_size

            if show_progress:
                print(f"[EMBEDDING] Batch {batch_num}/{total_batches} ({len(batch)} texts)")

            # Retry logic for robustness
            for attempt in range(self.config.max_retries):
                try:
                    embeddings = self.model.embed_documents(batch)
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        print(f"[EMBEDDING] Retry {attempt + 1} after error: {e}")
                    else:
                        print(f"[EMBEDDING] Failed after {self.config.max_retries} attempts: {e}")
                        # Fill with zeros for failed batch
                        all_embeddings.extend([[0.0] * self.config.embedding_dimensions] * len(batch))

        return np.array(all_embeddings)

    def embed_themes(self, themes: List[Theme]) -> List[Theme]:
        """
        Embed all key phrases for each theme and compute centroids.

        Phrases are preprocessed with the same TextPreprocessor used for
        messages so that both live in the same embedding space.

        Args:
            themes: List of Theme objects

        Returns:
            Updated Theme objects with phrase_embeddings and theme_embedding
        """
        print(f"[EMBEDDING] Embedding {len(themes)} themes...")

        # Preprocess phrases the same way messages are preprocessed
        # so they occupy the same embedding space
        preprocessor = TextPreprocessor()

        # Collect all unique key phrases (preprocessed for embedding)
        all_phrases = []           # preprocessed versions (for embedding)
        all_phrases_raw = []       # original versions (for lookup)
        phrase_to_theme_idx = {}   # Maps raw phrase -> (theme_idx, phrase_idx)

        for theme_idx, theme in enumerate(themes):
            for phrase_idx, phrase in enumerate(theme.key_phrases):
                if phrase not in phrase_to_theme_idx:
                    phrase_to_theme_idx[phrase] = (theme_idx, phrase_idx)
                    cleaned = preprocessor.clean_text(phrase)
                    # Fall back to original if cleaning empties it
                    all_phrases.append(cleaned if cleaned else phrase)
                    all_phrases_raw.append(phrase)

        print(f"[EMBEDDING] Total unique key phrases: {len(all_phrases)}")

        # Embed all phrases at once (preprocessed versions)
        if all_phrases:
            phrase_embeddings = self.embed_texts(all_phrases, show_progress=True)
        else:
            phrase_embeddings = np.array([])

        # Build raw_phrase -> embedding lookup
        phrase_to_embedding = {
            raw_phrase: phrase_embeddings[i]
            for i, raw_phrase in enumerate(all_phrases_raw)
        }

        # Assign embeddings to themes
        for theme in themes:
            # Get embeddings for this theme's phrases
            theme_phrase_embs = []
            for phrase in theme.key_phrases:
                if phrase in phrase_to_embedding:
                    theme_phrase_embs.append(phrase_to_embedding[phrase])

            if theme_phrase_embs:
                theme.phrase_embeddings = np.array(theme_phrase_embs)
                # Centroid = mean of phrase embeddings
                theme.theme_embedding = np.mean(theme.phrase_embeddings, axis=0)
            else:
                # Empty theme - use zero vectors
                theme.phrase_embeddings = np.zeros((1, self.config.embedding_dimensions))
                theme.theme_embedding = np.zeros(self.config.embedding_dimensions)

        print(f"[EMBEDDING] Theme embedding complete")
        for theme in themes:
            print(f"  - {theme.theme_name}: {theme.phrase_embeddings.shape[0]} phrase embeddings")

        return themes

    def embed_messages(
        self,
        messages: List[str],
        cache: Optional[Dict[str, np.ndarray]] = None,
        cache_file: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Embed messages, using cache when available.

        Supports both in-memory cache (dict) and file-based cache (.npy).
        If cache_file is provided, will save/load embeddings to/from disk.

        Args:
            messages: List of message texts
            cache: Optional dict mapping message -> embedding (in-memory cache)
            cache_file: Optional path to .npy file for persistent caching

        Returns:
            Tuple of (embedding_matrix, updated_cache)
        """
        print(f"[EMBEDDING] Embedding {len(messages)} messages...")

        # Try to load from file cache first
        if cache_file and os.path.exists(cache_file):
            try:
                print(f"[EMBEDDING] Loading cached embeddings from: {cache_file}")
                cached_embeddings = np.load(cache_file)

                # Validate cache — check both row count AND embedding dimension
                expected_dim = self.config.embedding_dimensions
                row_ok  = cached_embeddings.shape[0] == len(messages)
                dim_ok  = len(cached_embeddings.shape) == 2 and cached_embeddings.shape[1] == expected_dim
                if row_ok and dim_ok:
                    print(f"[EMBEDDING] Using cached embeddings (shape: {cached_embeddings.shape})")

                    # Build in-memory cache from file
                    cache = cache or {}
                    for i, msg in enumerate(messages):
                        cache[msg] = cached_embeddings[i]

                    return cached_embeddings, cache
                else:
                    if not row_ok:
                        print(f"[EMBEDDING] Cache row mismatch ({cached_embeddings.shape[0]} != {len(messages)}), regenerating...")
                    else:
                        print(f"[EMBEDDING] Cache dimension mismatch "
                              f"({cached_embeddings.shape[1]} != {expected_dim}), regenerating...")
            except Exception as e:
                print(f"[EMBEDDING] Failed to load cached embeddings: {e}, regenerating...")

        cache = cache or {}

        # Find messages needing embedding
        to_embed = []
        to_embed_indices = []

        for i, msg in enumerate(messages):
            if msg not in cache:
                to_embed.append(msg)
                to_embed_indices.append(i)

        print(f"[EMBEDDING] {len(to_embed)} new messages to embed, {len(messages) - len(to_embed)} cached")

        # Embed new messages
        if to_embed:
            new_embeddings = self.embed_texts(to_embed, show_progress=True)

            # Update cache
            for msg, emb in zip(to_embed, new_embeddings):
                cache[msg] = emb

        # Build result matrix
        embeddings = np.array([cache[msg] for msg in messages])

        # Save to file cache if specified
        if cache_file:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                np.save(cache_file, embeddings)
                print(f"[EMBEDDING] Saved embeddings to cache: {cache_file}")
            except Exception as e:
                print(f"[EMBEDDING] Warning: Failed to save embeddings cache: {e}")

        return embeddings, cache


def embed_themes(
    themes: List[Theme],
    config: EmbeddingConfig = None,
) -> List[Theme]:
    """
    Convenience function for theme embedding.

    Args:
        themes: List of Theme objects
        config: Optional embedding config

    Returns:
        Updated Theme objects with embeddings
    """
    embedder = ThemeEmbedder(config)
    return embedder.embed_themes(themes)


def embed_messages(
    messages: List[str],
    cache: Optional[Dict[str, np.ndarray]] = None,
    config: EmbeddingConfig = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convenience function for message embedding.

    Args:
        messages: List of message texts
        cache: Optional embedding cache
        config: Optional embedding config

    Returns:
        Tuple of (embedding_matrix, updated_cache)
    """
    embedder = ThemeEmbedder(config)
    return embedder.embed_messages(messages, cache)
