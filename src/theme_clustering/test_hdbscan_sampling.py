"""
Test script for HDBSCAN sampling.

This will run HDBSCAN sampling on the sheopals dataset and output the results to JSON.
"""

import sys
sys.path.insert(0, '/home/anirudh/admin-backend')

import json
import numpy as np
from new_clustering.theme_sampling import AdaptiveSampler
from new_clustering.theme_config import SamplingConfig
from new_clustering.theme_preprocessing import TextPreprocessor
from vector import load_messages

def main():
    # Load sheopals data
    print("Loading sheopals data...")
    CHAT_DATA_PATH = "/home/anirudh/admin-backend/chat_data/972808_0101_1501.json"

    df = load_messages(
        CHAT_DATA_PATH,
        remove_bubbles=True,
        max_messages=10000
    )

    messages = df['text'].tolist()
    session_ids = df['session_id'].tolist()

    print(f"Loaded {len(messages)} messages")

    # Apply theme preprocessing (filters short messages, cleans, deduplicates)
    print("\nApplying theme preprocessing...")
    preprocessor = TextPreprocessor()
    cleaned_messages, cleaned_session_ids, message_to_sessions = \
        preprocessor.preprocess_messages(messages, session_ids)

    print(f"After preprocessing: {len(cleaned_messages)} messages")

    # Create embeddings (load from cache if available)
    print("\nLoading/creating embeddings...")
    from openai_utils import get_embedding_model

    embeddings_path = f"new_clustering/outputs/embeddings/sheopals_embeddings_{len(cleaned_messages)}.npy"
    embeddings = None

    # Try to load cached embeddings
    try:
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] == len(cleaned_messages):
            print(f"Loaded cached embeddings: {embeddings.shape}")
        else:
            print(f"Cache size mismatch ({embeddings.shape[0]} != {len(cleaned_messages)}), regenerating...")
            embeddings = None
    except:
        print("No cached embeddings found, creating new ones...")

    # Generate embeddings if needed
    if embeddings is None:
        model = get_embedding_model()
        embeddings = []
        batch_size = 200
        for i in range(0, len(cleaned_messages), batch_size):
            batch = cleaned_messages[i:i+batch_size]
            print(f"  Embedding batch {i//batch_size + 1}/{(len(cleaned_messages)+batch_size-1)//batch_size}")
            batch_embs = model.embed_documents(batch)
            embeddings.extend(batch_embs)
        embeddings = np.array(embeddings)

        import os
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")

    # Run HDBSCAN sampling
    print("\n" + "="*70)
    print("RUNNING HDBSCAN SAMPLING")
    print("="*70 + "\n")

    config = SamplingConfig(use_hdbscan=True)
    sampler = AdaptiveSampler(config)

    output_dir = "new_clustering/outputs"

    sampled_messages, sampled_session_ids, sampled_indices, metadata = \
        sampler.hdbscan_sample(
            messages=cleaned_messages,
            session_ids=cleaned_session_ids,
            embeddings=embeddings,
            sample_size=sampler.calculate_sample_size(len(cleaned_messages)),
            output_dir=output_dir
        )

    print("\n" + "="*70)
    print("SAMPLING COMPLETE")
    print("="*70)
    print(f"\nTotal messages: {len(cleaned_messages)}")
    print(f"Sampled messages: {len(sampled_messages)}")
    print(f"Sample rate: {100*len(sampled_messages)/len(cleaned_messages):.1f}%")
    print(f"\nMetadata saved to: {output_dir}/hdbscan_sampling_metadata.json")

    # Show summary of clusters
    if 'cluster_sampling' in metadata:
        print("\nCluster sampling summary:")
        for cluster in metadata['cluster_sampling'][:10]:  # Show first 10
            print(f"  {cluster['label_name']}: {cluster['sampled']}/{cluster['total_size']} "
                  f"({100*cluster['sample_rate']:.1f}%)")
        if len(metadata['cluster_sampling']) > 10:
            print(f"  ... and {len(metadata['cluster_sampling'])-10} more clusters")

    print("\n✓ HDBSCAN sampling test completed successfully!")
    print(f"✓ JSON output available at: {output_dir}/hdbscan_sampling_metadata.json")

if __name__ == "__main__":
    main()
