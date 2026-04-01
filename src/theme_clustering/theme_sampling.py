"""
Theme Clustering Sampling Module (Phase 1)

Adaptive sampling with stratified selection for theme discovery.
Ensures diverse message coverage across the embedding space.
"""

import random
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import LocalOutlierFactor

try:
    import umap
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("[SAMPLING] Warning: umap/hdbscan not available, falling back to KMeans")

from .theme_config import SamplingConfig, DEFAULT_CONFIG


class AdaptiveSampler:
    """
    Adaptive message sampler for theme discovery.

    Sampling strategy based on dataset size:
    - Small (≤100): Use all messages
    - Medium (101-500): Sample 60% with diversity
    - Large (501-1500): Sample 20% (max 300) stratified
    - Very Large (1500+): Sample 350 with embedding-space diversity
    """

    def __init__(self, config: SamplingConfig = None):
        self.config = config or DEFAULT_CONFIG.sampling

    def calculate_sample_size(self, total: int) -> int:
        """
        Calculate optimal sample size based on dataset size.

        Args:
            total: Total number of messages

        Returns:
            Number of messages to sample
        """
        if total <= self.config.small_dataset_max:
            return total  # Use all
        elif total <= self.config.medium_dataset_max:
            return min(
                self.config.medium_sample_cap,
                int(total * self.config.medium_sample_percent)
            )
        elif total <= self.config.large_dataset_max:
            return min(
                self.config.large_sample_cap,
                int(total * self.config.large_sample_percent)
            )
        else:
            return self.config.very_large_sample_cap  # Fixed cap

    def random_sample(
        self,
        messages: List[str],
        session_ids: List[str],
        sample_size: int,
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Simple random sampling without embeddings.

        Args:
            messages: List of message texts
            session_ids: List of session IDs
            sample_size: Number of messages to sample

        Returns:
            Tuple of (sampled_messages, sampled_session_ids, sampled_indices)
        """
        n = len(messages)
        if sample_size >= n:
            return messages, session_ids, list(range(n))

        random.seed(self.config.random_state)
        indices = random.sample(range(n), sample_size)
        indices.sort()  # Keep original order

        sampled_messages = [messages[i] for i in indices]
        sampled_session_ids = [session_ids[i] for i in indices]

        return sampled_messages, sampled_session_ids, indices

    def stratified_sample(
        self,
        messages: List[str],
        session_ids: List[str],
        embeddings: np.ndarray,
        sample_size: int,
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Stratified sampling using mini-KMeans on embeddings.

        Strategy:
        1. Run mini-KMeans on embeddings (k = sample_size // divisor)
        2. Select centroid-nearest message from each mini-cluster
        3. Add random samples from each mini-cluster
        4. Fill remaining quota with random selection

        Args:
            messages: List of message texts
            session_ids: List of session IDs
            embeddings: Message embeddings (n_messages, dim)
            sample_size: Number of messages to sample

        Returns:
            Tuple of (sampled_messages, sampled_session_ids, sampled_indices)
        """
        n = len(messages)
        if sample_size >= n:
            return messages, session_ids, list(range(n))

        print(f"[SAMPLING] Stratified sampling: {sample_size} from {n} messages")

        # Determine number of mini-clusters
        k = max(5, sample_size // self.config.mini_clusters_divisor)
        k = min(k, n - 1)  # Can't have more clusters than samples

        print(f"[SAMPLING] Using {k} mini-clusters for stratification")

        # Run mini-KMeans
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.config.random_state,
            batch_size=min(256, n),
            n_init=3,
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # Group indices by cluster
        cluster_indices = {i: [] for i in range(k)}
        for idx, label in enumerate(cluster_labels):
            cluster_indices[label].append(idx)

        # Phase 1: Select centroid-nearest message from each cluster
        selected_indices = set()

        for cluster_id in range(k):
            if not cluster_indices[cluster_id]:
                continue

            # Find message nearest to centroid
            cluster_embs = embeddings[cluster_indices[cluster_id]]
            centroid = centroids[cluster_id]

            # Compute distances to centroid
            distances = np.linalg.norm(cluster_embs - centroid, axis=1)
            nearest_local_idx = np.argmin(distances)
            nearest_global_idx = cluster_indices[cluster_id][nearest_local_idx]

            selected_indices.add(nearest_global_idx)

        print(f"[SAMPLING] Phase 1: Selected {len(selected_indices)} centroid-nearest messages")

        # Phase 2: Random samples from each cluster (proportional to cluster size)
        remaining = sample_size - len(selected_indices)
        if remaining > 0:
            # Calculate samples per cluster proportionally
            cluster_sizes = {c: len(idxs) for c, idxs in cluster_indices.items()}
            total_non_selected = n - len(selected_indices)

            for cluster_id in range(k):
                available = [i for i in cluster_indices[cluster_id] if i not in selected_indices]
                if not available:
                    continue

                # Proportional allocation
                proportion = len(available) / total_non_selected
                n_to_sample = max(1, int(remaining * proportion))
                n_to_sample = min(n_to_sample, len(available))

                random.seed(self.config.random_state + cluster_id)
                sampled = random.sample(available, n_to_sample)
                selected_indices.update(sampled)

        print(f"[SAMPLING] Phase 2: Total {len(selected_indices)} after proportional sampling")

        # Phase 3: Fill remaining quota with random selection
        if len(selected_indices) < sample_size:
            remaining_indices = [i for i in range(n) if i not in selected_indices]
            random.seed(self.config.random_state)
            additional = random.sample(
                remaining_indices,
                min(sample_size - len(selected_indices), len(remaining_indices))
            )
            selected_indices.update(additional)

        # Convert to sorted list
        final_indices = sorted(list(selected_indices))[:sample_size]

        sampled_messages = [messages[i] for i in final_indices]
        sampled_session_ids = [session_ids[i] for i in final_indices]

        print(f"[SAMPLING] Final: {len(final_indices)} messages sampled")

        return sampled_messages, sampled_session_ids, final_indices

    def hdbscan_sample(
        self,
        messages: List[str],
        session_ids: List[str],
        embeddings: np.ndarray,
        sample_size: int,
        output_dir: Optional[str] = None,
        file_prefix: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[int], Dict[str, Any]]:
        """
        HDBSCAN-based sampling using density-aware clustering.

        Strategy (from vector.py):
        1. UMAP dimensionality reduction (1024D → 50D)
        2. LOF outlier detection (~10% removal)
        3. HDBSCAN clustering (min_cluster_size adaptive)
        4. Proportional sampling from each cluster + outliers

        Args:
            messages: List of message texts
            session_ids: List of session IDs
            embeddings: Message embeddings (n_messages, dim)
            sample_size: Target number of messages to sample
            output_dir: Optional directory to save sampling metadata JSON

        Returns:
            Tuple of (sampled_messages, sampled_session_ids, sampled_indices, metadata)
        """
        if not HDBSCAN_AVAILABLE:
            print("[SAMPLING] HDBSCAN not available, falling back to stratified sampling")
            msgs, sids, indices = self.stratified_sample(messages, session_ids, embeddings, sample_size)
            return msgs, sids, indices, {}

        n = len(messages)
        if sample_size >= n:
            metadata = {"method": "all_messages", "total": n}
            return messages, session_ids, list(range(n)), metadata

        print(f"[SAMPLING] HDBSCAN-based sampling: {sample_size} from {n} messages")

        # Step 1: UMAP dimensionality reduction
        print(f"[SAMPLING] Step 1: UMAP reduction {embeddings.shape[1]}D → 50D...")
        umap_reducer = umap.UMAP(
            n_components=50,
            n_neighbors=25,
            min_dist=0.0,
            metric='cosine',
            random_state=self.config.random_state,
            verbose=False
        )
        reduced_embeddings = umap_reducer.fit_transform(embeddings)
        print(f"[SAMPLING] UMAP complete: {reduced_embeddings.shape}")

        # Step 2: Outlier detection with LOF
        print(f"[SAMPLING] Step 2: LOF outlier detection (contamination=0.1)...")
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outlier_labels = lof.fit_predict(reduced_embeddings)

        inlier_mask = outlier_labels == 1
        outlier_mask = outlier_labels == -1
        n_outliers = np.sum(outlier_mask)

        print(f"[SAMPLING] Outliers detected: {n_outliers} ({100*n_outliers/n:.1f}%)")

        # Step 3: HDBSCAN clustering on inliers
        clean_embeddings = reduced_embeddings[inlier_mask]
        inlier_indices = np.where(inlier_mask)[0]

        dataset_size = len(clean_embeddings)
        min_cluster_size = max(5, min(50, dataset_size // 100))
        min_samples = max(3, min_cluster_size - 2)

        print(f"[SAMPLING] Step 3: HDBSCAN clustering (min_cluster_size={min_cluster_size}, "
              f"min_samples={min_samples})...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.0,
        )

        cluster_labels_clean = clusterer.fit_predict(clean_embeddings)

        # Map back to full dataset
        cluster_labels = np.full(n, -2, dtype=int)  # -2 = outlier (from LOF)
        cluster_labels[inlier_indices] = cluster_labels_clean

        # Count clusters
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_hdbscan_noise = np.sum(cluster_labels == -1)
        n_lof_outliers = np.sum(cluster_labels == -2)

        print(f"[SAMPLING] HDBSCAN found {n_clusters} clusters")
        print(f"[SAMPLING] Noise: {n_hdbscan_noise} HDBSCAN noise, {n_lof_outliers} LOF outliers")

        # Step 4: Proportional sampling from each cluster + noise
        cluster_sizes = defaultdict(int)
        cluster_indices = defaultdict(list)

        for idx, label in enumerate(cluster_labels):
            cluster_sizes[label] += 1
            cluster_indices[label].append(idx)

        print(f"\n[SAMPLING] Step 4: Proportional sampling from {len(cluster_sizes)} groups...")

        # Calculate samples per cluster (proportional, with minimum guarantees)
        samples_per_cluster = {}
        total_assigned = 0

        # First pass: proportional allocation
        for label in sorted(cluster_sizes.keys()):
            size = cluster_sizes[label]
            proportion = size / n
            n_samples = int(sample_size * proportion)

            # Minimum guarantees
            if label >= 0:  # Real clusters
                n_samples = max(3, n_samples)  # At least 3 from each cluster
            else:  # Noise/outliers
                n_samples = max(2, n_samples)  # At least 2 from noise

            samples_per_cluster[label] = min(n_samples, size)  # Can't sample more than exists
            total_assigned += samples_per_cluster[label]

        # Adjust if we're over/under budget
        if total_assigned > sample_size:
            # Scale down proportionally
            scale = sample_size / total_assigned
            for label in samples_per_cluster:
                samples_per_cluster[label] = max(1, int(samples_per_cluster[label] * scale))

        # Sample from each cluster
        selected_indices = []
        cluster_metadata = []

        for label in sorted(cluster_sizes.keys()):
            indices = cluster_indices[label]
            n_to_sample = min(samples_per_cluster[label], len(indices))

            random.seed(self.config.random_state + int(label) + 1000)
            sampled = random.sample(indices, n_to_sample)
            selected_indices.extend(sampled)

            label_name = f"cluster_{label}" if label >= 0 else ("hdbscan_noise" if label == -1 else "lof_outlier")
            cluster_metadata.append({
                "label": int(label),
                "label_name": label_name,
                "total_size": int(cluster_sizes[label]),
                "sampled": int(n_to_sample),
                "sample_rate": float(n_to_sample / cluster_sizes[label])
            })

            print(f"[SAMPLING]   {label_name}: {n_to_sample}/{cluster_sizes[label]} sampled "
                  f"({100*n_to_sample/cluster_sizes[label]:.1f}%)")

        # If we're still under budget, randomly sample remaining
        if len(selected_indices) < sample_size:
            remaining = sample_size - len(selected_indices)
            available = [i for i in range(n) if i not in selected_indices]
            random.seed(self.config.random_state)
            additional = random.sample(available, min(remaining, len(available)))
            selected_indices.extend(additional)
            print(f"[SAMPLING]   Additional random: {len(additional)} messages")

        # Sort indices
        selected_indices = sorted(selected_indices)[:sample_size]

        # Build output
        sampled_messages = [messages[i] for i in selected_indices]
        sampled_session_ids = [session_ids[i] for i in selected_indices]

        # Metadata for JSON output
        metadata = {
            "method": "hdbscan",
            "total_messages": int(n),
            "sampled_messages": len(selected_indices),
            "sample_rate": float(len(selected_indices) / n),
            "umap_params": {
                "n_components": 50,
                "n_neighbors": 25,
                "min_dist": 0.0,
                "metric": "cosine"
            },
            "lof_params": {
                "n_neighbors": 20,
                "contamination": 0.1,
                "outliers_detected": int(n_lof_outliers)
            },
            "hdbscan_params": {
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "metric": "euclidean",
                "n_clusters_found": int(n_clusters),
                "noise_points": int(n_hdbscan_noise)
            },
            "cluster_sampling": cluster_metadata
        }

        print(f"\n[SAMPLING] Final: {len(selected_indices)} messages sampled from {n_clusters} clusters + noise")

        # Save text file if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save text file with just messages (one per line)
            fname = f"{file_prefix}_hdbscan_sampled_messages.txt" if file_prefix else "hdbscan_sampled_messages.txt"
            text_path = os.path.join(output_dir, fname)
            with open(text_path, 'w') as f:
                for msg in sampled_messages:
                    f.write(msg + '\n')
            print(f"[SAMPLING] Saved sampled messages to: {text_path}")

        return sampled_messages, sampled_session_ids, selected_indices, metadata

    def sample_for_discovery(
        self,
        messages: List[str],
        session_ids: List[str],
        embeddings: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None,
        file_prefix: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Main sampling method for theme discovery.

        Uses HDBSCAN (if configured), stratified sampling, or random sampling.

        Args:
            messages: List of message texts
            session_ids: List of session IDs
            embeddings: Optional message embeddings
            output_dir: Optional directory to save sampling metadata JSON

        Returns:
            Tuple of (sampled_messages, sampled_session_ids, sampled_indices)
        """
        n = len(messages)
        sample_size = self.calculate_sample_size(n)

        print(f"[SAMPLING] Dataset size: {n}, Sample size: {sample_size}")

        if sample_size >= n:
            print(f"[SAMPLING] Using all {n} messages (small dataset)")
            return messages, session_ids, list(range(n))

        # Choose sampling method
        if embeddings is not None and len(embeddings) == n:
            if self.config.use_hdbscan and HDBSCAN_AVAILABLE:
                print(f"[SAMPLING] Using HDBSCAN-based sampling")
                msgs, sids, indices, metadata = self.hdbscan_sample(
                    messages, session_ids, embeddings, sample_size, output_dir, file_prefix
                )
                return msgs, sids, indices
            else:
                if self.config.use_hdbscan and not HDBSCAN_AVAILABLE:
                    print(f"[SAMPLING] Warning: HDBSCAN requested but not available, using KMeans")
                return self.stratified_sample(messages, session_ids, embeddings, sample_size)
        else:
            print(f"[SAMPLING] No embeddings provided, using random sampling")
            return self.random_sample(messages, session_ids, sample_size)


def sample_messages(
    messages: List[str],
    session_ids: List[str],
    embeddings: Optional[np.ndarray] = None,
    config: SamplingConfig = None,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Convenience function for sampling messages.

    Args:
        messages: List of message texts
        session_ids: List of session IDs
        embeddings: Optional message embeddings
        config: Optional sampling config

    Returns:
        Tuple of (sampled_messages, sampled_session_ids, sampled_indices)
    """
    sampler = AdaptiveSampler(config)
    return sampler.sample_for_discovery(messages, session_ids, embeddings)
