"""
Concern-Based Clustering Module with Label Merging

This module performs clustering based on the "Concern/Requirement" field in user_fields,
with advanced label merging capabilities using embeddings and GPT.

Workflow:
1. Load chat data JSON file
2. Extract concerns from user_fields and map them to messages
3. Create initial clusters where each unique concern is a cluster
4. **NEW**: Merge semantically similar concern labels using:
   - OpenAI text-embedding-3-large for semantic similarity
   - GPT-4 for intelligent grouping suggestions
   - Embedding distance validation to prevent over-aggressive merging
5. Categorize clusters into: regular, multi-tag, and low-frequency
6. Export clusters to JSON file with session mapping and metadata

Key Features:
- Handles multi-tag concerns (e.g., ["liver health", "fatty liver"])
- Merges similar labels (e.g., "fatty liver grade 2" + "fatty liver grade two")
- Identifies low-frequency concerns (< threshold messages)
- Preserves all data while categorizing for easy review
- Updates multi-tag clusters when component labels are merged
"""

import json
import os
import ast
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, Counter
from src.utils.openai_utils import gpt4_1_azure_call, GPT4Input, get_embedding_model
from sklearn.metrics.pairwise import cosine_similarity
from src.concern.concern_report import gen_concern_report
from sklearn.neighbors import NearestNeighbors
import umap
import hdbscan


# ==================== GPT-4.1 HELPER FUNCTIONS ====================

def _gpt4_1_call(prompt: str, user_text: str, temperature: float = 0.3, max_tokens: int = 4000, timeout: int = 120) -> str:
    """
    Helper function to make a GPT-4.1 API call with system prompt and user text.

    Args:
        prompt: System prompt
        user_text: User message text
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds

    Returns:
        Response content string or None on failure
    """
    gpt4_inputs = [
        GPT4Input(actor="system", text=prompt),
        GPT4Input(actor="user", text=user_text)
    ]

    result = gpt4_1_azure_call(gpt4_inputs, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return result


def batched_categorize_sales_stages(concern_labels: List[str], batch_size: int = 20) -> Dict[str, str]:
    """
    Categorize multiple concern labels as pre-sales or post-sales in batched API calls.

    Args:
        concern_labels: List of concern label strings to categorize
        batch_size: Maximum labels per API call (default: 20)

    Returns:
        Dictionary mapping concern_label -> sales_stage ('pre-sales' or 'post-sales')
    """
    if not concern_labels:
        return {}

    print(f"[BATCHED] Categorizing {len(concern_labels)} concern labels...")
    results = {}

    # Process in batches
    for batch_start in range(0, len(concern_labels), batch_size):
        batch = concern_labels[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(concern_labels) - 1) // batch_size + 1

        print(f"[BATCHED] Processing sales stage batch {batch_num}/{total_batches} ({len(batch)} labels)")

        # Build labels list for prompt
        labels_list = "\n".join([f'{i}. "{label}"' for i, label in enumerate(batch)])

        prompt = f'''You are an expert customer service analyst specializing in sales funnel classification.

TASK: Classify each of the following customer concern labels as either "pre-sales" or "post-sales".

CONCERN LABELS:
{labels_list}

DEFINITIONS:
- PRE-SALES: Concerns from customers who are researching, considering, or in the process of making their first purchase. Includes product inquiries, pricing questions, feature comparisons, availability checks, best offers, product recommendations, coupons, membership questions, nutrition/ingredient inquiries, "why should I use it", product benefits, etc.

- POST-SALES: Concerns from customers who have already purchased or are dealing with existing orders. Includes order status, delivery tracking, order not received, missing items, order confirmation, pending pickup, delivery issues, change of address, order cancellation, returns, refunds, product complaints (taste, quality), damaged products, account problems, COD issues, etc.

KEYWORD INDICATORS:
- POST-SALES: "order", "delivery", "tracking", "not received", "missing", "damaged", "refund", "cancel", "status", "shipment", "courier", "pending", "dispatch", "COD", "complaint", "return", "confirmation"
- PRE-SALES: "price", "offer", "discount", "coupon", "combo", "product", "recommend", "ingredients", "suitable", "membership", "benefit", "inquiry", "content", "free", "compare", "availability", "why use"

OUTPUT FORMAT: Return a JSON object mapping the label index (as string) to the classification:
{{
  "0": "pre-sales",
  "1": "post-sales",
  "2": "pre-sales"
}}

Return ONLY the JSON object, no other text.'''

        result = _gpt4_1_call(
            prompt=prompt,
            user_text="Classify concern labels as pre-sales or post-sales.",
            temperature=0.1,
            max_tokens=1000,
            timeout=60
        )

        if result:
            try:
                cleaned = result.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)

                for idx_str, stage in parsed.items():
                    idx = int(idx_str)
                    if 0 <= idx < len(batch):
                        label = batch[idx]
                        stage_clean = stage.strip().lower()
                        if "post" in stage_clean:
                            results[label] = "post-sales"
                        else:
                            results[label] = "pre-sales"

            except (json.JSONDecodeError, ValueError) as e:
                print(f"[BATCHED] Failed to parse batch response: {e}, using fallback")
                # Fallback to default
                for label in batch:
                    results[label] = "pre-sales"
        else:
            print(f"[BATCHED] Batch failed, using default pre-sales")
            for label in batch:
                results[label] = "pre-sales"

    print(f"[BATCHED] Completed categorizing {len(results)} labels")
    return results


def load_chat_data(chat_data_file_path: str, report: bool = False) -> Dict[str, Dict]:
    """
    Load chat data from JSON file with error handling for invalid control characters.
    Supports both daily format and report format.

    Args:
        chat_data_file_path: Path to the chat data JSON file
        report: If True, use report format (array with 'chat' field), else use daily format (dict with 'messages' field)

    Returns:
        Dictionary mapping session_id to session data
    """
    print(f"[CONCERN_CLUSTER] Loading chat data from: {chat_data_file_path}")
    print(f"[CONCERN_CLUSTER] Format: {'Report' if report else 'Daily'}")

    try:
        with open(chat_data_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARNING] JSON decode error: {e}")
        print(f"[INFO] Attempting to load with strict=False to handle control characters...")

        with open(chat_data_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace common control characters
        import re
        # Remove null bytes and other control characters except newline, tab, and carriage return
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)

        raw_data = json.loads(content)

    if report:
        # Report format has two variants:
        # Variant 1: Array of separate session objects [{"session_id1": {...}}, {"session_id2": {...}}]
        # Variant 2: Array with single dict containing all sessions [{"session_id1": {...}, "session_id2": {...}, ...}]
        chat_data = {}

        # Check if it's a single-dict format (all sessions in one dict)
        if isinstance(raw_data, list) and len(raw_data) == 1 and isinstance(raw_data[0], dict):
            # Check if this looks like it contains multiple sessions (has many keys with similar structure)
            first_dict = raw_data[0]
            if len(first_dict) > 10:  # Heuristic: if >10 keys, likely all sessions in one dict
                print(f"[CONCERN_CLUSTER] Detected single-dict report format")
                raw_data = [{session_id: session_data} for session_id, session_data in first_dict.items()]

        # Process sessions (works for both variants now)
        for session_obj in raw_data:
            for session_id, session_data in session_obj.items():
                # Convert 'chat' to 'messages' for consistency
                if 'chat' in session_data:
                    session_data['messages'] = session_data.pop('chat')

                # Convert 'user_field' (singular) to 'user_fields' (plural) for consistency
                if 'user_field' in session_data:
                    user_field = session_data.pop('user_field')
                    # Also normalize key_field -> key and val_field -> value
                    if user_field:
                        normalized_user_fields = []
                        for field in user_field:
                            normalized_field = {}
                            if 'key_field' in field:
                                normalized_field['key'] = field['key_field']
                            if 'val_field' in field:
                                normalized_field['value'] = field['val_field']
                            # Copy other fields
                            for k, v in field.items():
                                if k not in ['key_field', 'val_field']:
                                    normalized_field[k] = v
                            normalized_user_fields.append(normalized_field)
                        session_data['user_fields'] = normalized_user_fields
                    else:
                        session_data['user_fields'] = []

                # Convert 'use_cases' to 'usecases' for consistency
                if 'use_cases' in session_data:
                    session_data['usecases'] = session_data.pop('use_cases')

                chat_data[session_id] = session_data
        print(f"[CONCERN_CLUSTER] Loaded {len(chat_data)} sessions from report format")
    else:
        # Daily format: dictionary with session_id as keys
        # {"session_id": {"messages": [...], "user_fields": [...], ...}}
        chat_data = raw_data
        print(f"[CONCERN_CLUSTER] Loaded {len(chat_data)} sessions from daily format")

    return chat_data

def extract_concerns_and_cluster(
    chat_data: Dict[str, Dict],
    filter_pages: Optional[List[str]] = None,
    filter_secondary_usecases: Optional[List[str]] = None,
    phrase: Optional[str] = None,
    keyword: Optional[str] = None,
    remove_bubbles: bool = True
) -> Dict:
    """
    Extract concerns from user_fields and create clusters.

    For each session:
    - Find all "Concern/Requirement" fields in user_fields
    - Map each concern occurrence to the corresponding message
    - If a message has multiple concerns, use the first one
    - Group messages by concern to create clusters
    - Apply filters (page, usecase, phrase, keyword) and remove bubble clicks

    Args:
        chat_data: Dictionary mapping session_id to session data
        filter_pages: List of bot_page values to filter by
        filter_secondary_usecases: List of secondary usecases to filter by
        phrase: Filter for customer messages preceding AI messages with this phrase
        keyword: Filter for bot_page containing this keyword
        remove_bubbles: Whether to remove bubble-click messages (default: True)

    Returns:
        Dictionary with cluster data structure
    """
    print(f"[CONCERN_CLUSTER] Extracting concerns and creating clusters...")

    # Log filter info
    if phrase is not None:
        phrase_list = [phrase] if isinstance(phrase, str) else phrase
        if len(phrase_list) > 1:
            print(f"[INFO] Filtering with {len(phrase_list)} phrases: {phrase_list[:3]}{'...' if len(phrase_list) > 3 else ''}")
        else:
            print(f"[INFO] Filtering with phrase: '{phrase_list[0]}'")

    if filter_pages:
        print(f"[INFO] Filtering by bot_page: {filter_pages}")

    if keyword:
        print(f"[INFO] Filtering by keyword in bot_page: '{keyword}'")

    if filter_secondary_usecases:
        print(f"[INFO] Filtering by secondary_usecases: {filter_secondary_usecases}")

    # Dictionary to hold cluster data: concern (lowercase) -> list of (session_id, message_data)
    concern_clusters = defaultdict(list)
    # Track original casings for each normalized concern to pick the most common one
    concern_original_casings = defaultdict(list)

    # Track statistics
    total_concerns_found = 0
    total_messages_clustered = 0
    sessions_with_concerns = 0

    for session_id, session_data in chat_data.items():
        messages = session_data.get('messages', [])
        user_fields = session_data.get('user_fields', [])
        usecases = session_data.get('usecases', [])
        # Handle case where usecases is explicitly None (e.g., WhatsApp clients)
        if usecases is None:
            usecases = []
        metadata = session_data.get('session_meta', [])

        bot_page = metadata[0].get("bot_page") if metadata else None

        # Filter by bot_page if specified
        if filter_pages is not None and bot_page not in filter_pages:
            continue

        # Filter by keyword in bot_page if specified
        if keyword is not None and (bot_page is None or keyword.lower() not in bot_page.lower()):
            continue

        # Extract all Concern/Requirement fields in order
        concerns_list = []
        for uf in user_fields:
            if uf.get('key') in ['Concern/Requirement', 'Requirement/Concern', 'Health Concern/Requirement', 'invalid_concern/requirement', 'invalid_requirement/concern', 'Common Keywords']:
                value = uf.get('value', [])

                # Handle both list and string values (including double-encoded strings)
                if isinstance(value, str):
                    # Try JSON parsing first
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # Try Python literal eval for strings like "['item1', 'item2']"
                        try:
                            value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            # If all parsing fails, treat as single-item list
                            value = [value]

                concerns_list.append(value)

        if not concerns_list:
            continue

        sessions_with_concerns += 1

        # Get customer messages from this session with their usecases
        customer_messages = []
        customer_msg_count = 0
        for idx, msg in enumerate(messages):
            if msg.get('actor') == 'customer':
                # Get usecase for this customer message
                usecase = usecases[customer_msg_count].get('secondary', 'unknown') if customer_msg_count < len(usecases) else 'unknown'
                customer_messages.append((idx, msg, usecase))
                customer_msg_count += 1

        # Handle phrase filtering if specified
        valid_message_indices = set()
        if phrase is not None:
            phrase_list = [phrase] if isinstance(phrase, str) else phrase
            # Find customer messages that precede AI messages with the phrase
            for i, msg in enumerate(messages):
                ai_text = msg.get('text', '').lower()
                if msg.get('actor') != 'customer' and any(p.lower() in ai_text for p in phrase_list):
                    # Look backwards for the preceding customer message
                    for j in range(i-1, -1, -1):
                        prev_msg = messages[j]
                        if prev_msg.get('actor') == 'customer':
                            valid_message_indices.add(j)
                            break

        # Map concerns to messages
        # Each concern occurrence corresponds to a customer message in order
        for msg_idx, (original_idx, msg, usecase) in enumerate(customer_messages):
            if msg_idx >= len(concerns_list):
                break

            # If phrase filtering is active, skip messages not in valid set
            if phrase is not None and original_idx not in valid_message_indices:
                continue

            # Filter by secondary_usecase if specified
            if filter_secondary_usecases is not None and usecase not in filter_secondary_usecases:
                continue

            concern_value = concerns_list[msg_idx]

            # If concern_value is a list, normalize it by sorting
            if isinstance(concern_value, list) and len(concern_value) > 0:
                # Sort the list to ensure consistent ordering
                sorted_concern = sorted(concern_value)
                # Convert to JSON string representation for consistent cluster key
                primary_concern = json.dumps(sorted_concern, ensure_ascii=False, sort_keys=True)
            elif isinstance(concern_value, str):
                # The value might be a double-encoded string like "['Hair Care', 'Skin Care']"
                # Try multiple parsing strategies
                parsed_list = None

                # Strategy 1: Try JSON parsing (handles ["item1", "item2"])
                try:
                    parsed = json.loads(concern_value)
                    if isinstance(parsed, list):
                        parsed_list = parsed
                except (json.JSONDecodeError, TypeError):
                    pass

                # Strategy 2: Try Python literal eval (handles ['item1', 'item2'])
                if parsed_list is None:
                    try:
                        parsed = ast.literal_eval(concern_value)
                        if isinstance(parsed, list):
                            parsed_list = parsed
                    except (ValueError, SyntaxError, TypeError):
                        pass

                # If we successfully parsed a list, normalize it
                if parsed_list is not None and len(parsed_list) > 0:
                    # Sort and convert to consistent JSON string format
                    sorted_concern = sorted(parsed_list)
                    primary_concern = json.dumps(sorted_concern, ensure_ascii=False, sort_keys=True)
                else:
                    # Not a list format, treat as plain string
                    primary_concern = concern_value.strip()
            else:
                continue

            if not primary_concern:
                continue

            # Store the original casing for display purposes
            original_concern = primary_concern

            # Normalize to lowercase for clustering (case-insensitive grouping)
            normalized_concern = primary_concern.lower()

            # Add message to the cluster using normalized key
            message_data = {
                'session_id': session_id,
                'text': msg.get('text', '').strip(),
                'timestamp': msg.get('created_at', ''),
                'message_index': original_idx,
                'bot_page': bot_page,
                'user_intent': usecase
            }

            concern_clusters[normalized_concern].append(message_data)
            concern_original_casings[normalized_concern].append(original_concern)
            total_messages_clustered += 1

        total_concerns_found += len(concerns_list)

    print(f"[CONCERN_CLUSTER] Statistics:")
    print(f"  - Sessions with concerns: {sessions_with_concerns}")
    print(f"  - Total concern occurrences: {total_concerns_found}")
    print(f"  - Total messages clustered: {total_messages_clustered}")
    print(f"  - Unique concerns (clusters): {len(concern_clusters)}")

    # # BUBBLE-CLICK REMOVAL DISABLED
    # # Reason: Removes valid sessions from different users who clicked the same button
    # # Impact: Session-level metrics become inaccurate
    #
    # # Remove bubble-click messages if enabled
    # if remove_bubbles and total_messages_clustered > 0:
    #     print(f"[CONCERN_CLUSTER] Removing bubble-click messages...")
    #
    #     # Collect all message texts
    #     all_messages = []
    #     for messages in concern_clusters.values():
    #         all_messages.extend([msg['text'] for msg in messages])
    #
    #     # Count frequency of each message
    #     message_counts = Counter(all_messages)
    #
    #     # Identify bubble-click candidates (>=5 times AND >=3 words)
    #     bubble_clicks = {
    #         msg: count for msg, count in message_counts.items()
    #         if count >= 5 and len(msg.split()) >= 3
    #     }
    #
    #     # Additional predefined bubble starters
    #     bubble_starters = [
    #         "add to cart",
    #         "show more variants"
    #     ]
    #
    #     # Remove bubble messages from each cluster
    #     total_removed = 0
    #     for concern, messages in concern_clusters.items():
    #         original_count = len(messages)
    #
    #         # Filter out bubble clicks
    #         filtered_messages = []
    #         for msg in messages:
    #             text = msg['text']
    #             # Check if it's a bubble click pattern
    #             is_bubble = any(text.startswith(bubble_pattern) for bubble_pattern in bubble_clicks.keys())
    #             # Check if it starts with predefined bubble starters
    #             is_starter = any(text.lower().strip().startswith(starter) for starter in bubble_starters)
    #
    #             if not is_bubble and not is_starter:
    #                 filtered_messages.append(msg)
    #
    #         concern_clusters[concern] = filtered_messages
    #         removed = original_count - len(filtered_messages)
    #         total_removed += removed
    #
    #     print(f"[INFO] Removed {total_removed} bubble-click messages "
    #           f"({len(bubble_clicks)} unique patterns)")
    #
    #     # Remove empty clusters after bubble removal
    #     concern_clusters = {k: v for k, v in concern_clusters.items() if len(v) > 0}
    #
    #     # Also clean up the casing tracking for removed clusters
    #     concern_original_casings = {k: v for k, v in concern_original_casings.items() if k in concern_clusters}
    #
    #     print(f"[CONCERN_CLUSTER] Statistics (after bubble removal):")
    #     print(f"  - Total messages: {sum(len(msgs) for msgs in concern_clusters.values())}")
    #     print(f"  - Unique concerns (clusters): {len(concern_clusters)}")

    # Show clusters with multiple casing variants
    print(f"\n[CONCERN_CLUSTER] Analyzing case variants...")
    clusters_with_variants = 0
    for normalized_concern, original_casings in concern_original_casings.items():
        unique_casings = set(original_casings)
        if len(unique_casings) > 1:
            clusters_with_variants += 1
            casing_counts = Counter(original_casings)
            most_common = casing_counts.most_common(1)[0][0]
            variants_str = ", ".join([f"'{c}' ({casing_counts[c]})" for c in unique_casings])
            print(f"[INFO] Merged casings → '{most_common}': {variants_str}")

    if clusters_with_variants == 0:
        print(f"[INFO] No case variant merging needed - all clusters had consistent casing")
    else:
        print(f"[INFO] Merged {clusters_with_variants} clusters with different casings")

    # Convert to structured format
    clusters_data = []
    for cluster_id, (normalized_concern, messages) in enumerate(concern_clusters.items()):
        # Pick the most common original casing for display
        original_casings = concern_original_casings[normalized_concern]
        if original_casings:
            # Count occurrences of each casing variant
            casing_counts = Counter(original_casings)
            # Use the most common casing as the cluster title
            most_common_casing = casing_counts.most_common(1)[0][0]
        else:
            # Fallback to normalized version
            most_common_casing = normalized_concern

        cluster = {
            'cluster_id': cluster_id,
            'cluster_title': most_common_casing,
            'message_count': len(messages),
            'messages': messages
        }
        clusters_data.append(cluster)

    # Sort clusters by message count (descending)
    clusters_data.sort(key=lambda x: x['message_count'], reverse=True)

    # Re-assign cluster IDs after sorting
    for idx, cluster in enumerate(clusters_data):
        cluster['cluster_id'] = idx

    return clusters_data


def create_session_mapping(clusters_data: List[Dict]) -> List[Dict]:
    """
    Create session-based mapping for clusters.

    Instead of individual messages, group by session_id for each cluster.

    Args:
        clusters_data: List of cluster dictionaries with messages

    Returns:
        List of cluster dictionaries with session_ids instead of messages
    """
    print(f"[CONCERN_CLUSTER] Creating session mapping...")

    sessions_data = []
    total_messages_across_clusters = 0
    total_sessions_across_clusters = 0

    for cluster in clusters_data:
        # Group messages by session_id
        session_groups = defaultdict(list)

        for msg in cluster['messages']:
            session_id = msg['session_id']
            session_groups[session_id].append(msg)

        # Create session-based structure
        cluster_sessions = {
            'cluster_id': cluster['cluster_id'],
            'cluster_title': cluster['cluster_title'],
            'session_count': len(session_groups),
            'message_count': cluster['message_count'],
            'session_ids': list(session_groups.keys())
        }

        # Include sales_stage if present
        if 'sales_stage' in cluster:
            cluster_sessions['sales_stage'] = cluster['sales_stage']

        sessions_data.append(cluster_sessions)

        total_messages_across_clusters += cluster['message_count']
        total_sessions_across_clusters += len(session_groups)

        # Debug: Show clusters where message count != session count
        if cluster['message_count'] != len(session_groups):
            avg_msgs_per_session = cluster['message_count'] / len(session_groups)
            print(f"[DEBUG] Cluster '{cluster['cluster_title'][:60]}': "
                  f"{cluster['message_count']} msgs across {len(session_groups)} sessions "
                  f"(avg {avg_msgs_per_session:.1f} msgs/session)")

    print(f"[CONCERN_CLUSTER] Session mapping complete:")
    print(f"  - Total messages: {total_messages_across_clusters}")
    print(f"  - Total unique sessions: {total_sessions_across_clusters}")
    print(f"  - Ratio: {total_messages_across_clusters / total_sessions_across_clusters:.2f} messages per session on average")

    return sessions_data


def extract_atomic_labels_from_concerns(concern_clusters: Dict) -> Tuple[Set[str], Dict[str, int], Dict[str, List]]:
    """
    Extract atomic labels from all concerns (including multi-tag concerns).

    Returns:
        - Set of unique atomic labels
        - Dict mapping atomic label -> total occurrence count
        - Dict mapping atomic label -> list of concern keys where it appears
    """
    print("[LABEL_MERGE] Extracting atomic labels from concerns...")

    atomic_labels = set()
    label_counts = defaultdict(int)
    label_to_concerns = defaultdict(list)

    for concern_key, messages in concern_clusters.items():
        message_count = len(messages)

        # Try to parse as list (multi-tag)
        try:
            if concern_key.startswith('['):
                parsed = json.loads(concern_key)
                if isinstance(parsed, list):
                    # Multi-tag concern
                    for label in parsed:
                        atomic_labels.add(label)
                        label_counts[label] += message_count
                        label_to_concerns[label].append(concern_key)
                else:
                    # Single label (JSON-encoded string)
                    atomic_labels.add(concern_key)
                    label_counts[concern_key] += message_count
                    label_to_concerns[concern_key].append(concern_key)
            else:
                # Plain string label
                atomic_labels.add(concern_key)
                label_counts[concern_key] += message_count
                label_to_concerns[concern_key].append(concern_key)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, treat as plain string
            atomic_labels.add(concern_key)
            label_counts[concern_key] += message_count
            label_to_concerns[concern_key].append(concern_key)

    print(f"[LABEL_MERGE] Found {len(atomic_labels)} unique atomic labels")
    print(f"[LABEL_MERGE] Total concern keys (including multi-tag): {len(concern_clusters)}")

    return atomic_labels, dict(label_counts), dict(label_to_concerns)


def generate_label_embeddings(atomic_labels: Set[str], output_dir: str = "vec_outs", client_name: str = None, date_range: str = None) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for each unique atomic label.

    Args:
        atomic_labels: Set of unique concern label strings
        output_dir: Directory to cache embeddings
        client_name: Client name for the .npy filename (e.g., "alpino-in")
        date_range: Date range for the .npy filename (e.g., "2909_1310")

    Returns:
        Dictionary mapping label -> embedding vector
    """
    print(f"[LABEL_MERGE] Generating embeddings for {len(atomic_labels)} atomic labels...")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Include client_name in cache filename to avoid overwriting between clients
    cache_filename = f"concern_label_embeddings_cache_{client_name}.json" if client_name else "concern_label_embeddings_cache.json"
    cache_path = os.path.join(output_dir, cache_filename)

    # Try to load cached embeddings
    cached_embeddings = {}
    if os.path.exists(cache_path):
        try:
            print(f"[LABEL_MERGE] Loading cached embeddings from {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                # Convert lists back to numpy arrays
                cached_embeddings = {k: np.array(v) for k, v in cached_data.items()}
            print(f"[LABEL_MERGE] Loaded {len(cached_embeddings)} cached embeddings")
        except Exception as e:
            print(f"[LABEL_MERGE] Failed to load cache: {e}")

    # Find labels that need embedding
    labels_to_embed = [label for label in atomic_labels if label not in cached_embeddings]

    if not labels_to_embed:
        print("[LABEL_MERGE] All embeddings found in cache")
        return cached_embeddings

    print(f"[LABEL_MERGE] Need to generate embeddings for {len(labels_to_embed)} new labels")

    # Generate embeddings
    try:
        embeddings_model = get_embedding_model()
        batch_size = 200
        new_embeddings = {}

        for i in range(0, len(labels_to_embed), batch_size):
            batch = labels_to_embed[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(labels_to_embed) - 1) // batch_size + 1
            print(f"  ↳ Embedding batch {batch_num}/{total_batches}")

            response = embeddings_model.embed_documents(batch)

            for label, embedding in zip(batch, response):
                new_embeddings[label] = np.array(embedding)

        # Combine with cached embeddings
        all_embeddings = {**cached_embeddings, **new_embeddings}

        # Save updated cache
        cache_data = {k: v.tolist() for k, v in all_embeddings.items()}
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[LABEL_MERGE] Saved {len(all_embeddings)} embeddings to cache")

        # Save embeddings as .npy file if client_name and date_range are provided
        if client_name and date_range:
            npy_filename = f"concern_embeddings_{client_name}_{date_range}.npy"
            npy_path = os.path.join(output_dir, npy_filename)

            # Create a structured array with labels and embeddings
            # Save as dictionary with labels as keys and embeddings as values
            embedding_matrix = np.array([all_embeddings[label] for label in sorted(all_embeddings.keys())])
            label_list = sorted(all_embeddings.keys())

            # Save both the embedding matrix and labels
            np.save(npy_path, {
                'embeddings': embedding_matrix,
                'labels': label_list,
                'metadata': {
                    'client_name': client_name,
                    'date_range': date_range,
                    'num_labels': len(label_list),
                    'embedding_dim': embedding_matrix.shape[1] if len(embedding_matrix) > 0 else 0
                }
            }, allow_pickle=True)
            print(f"[LABEL_MERGE] Saved embeddings as .npy file: {npy_path}")
            print(f"  ↳ Shape: {embedding_matrix.shape}, Labels: {len(label_list)}")

        return all_embeddings

    except Exception as e:
        print(f"[LABEL_MERGE] Error generating embeddings: {e}")
        return cached_embeddings if cached_embeddings else {}


def cluster_labels_with_hdbscan(
    atomic_labels: List[str],
    label_counts: Dict[str, int],
    label_embeddings: Dict[str, np.ndarray],
    min_cluster_size: int = 5,
    min_samples: int = 3
) -> List[List[str]]:
    """
    Use HDBscan to cluster semantically similar concern labels.

    This replaces the GPT-based approach that was limited to 500 labels.
    Now we can handle all labels without truncation.

    Args:
        atomic_labels: List of unique atomic label strings
        label_counts: Dictionary mapping label -> message count
        label_embeddings: Dictionary mapping label -> embedding vector
        min_cluster_size: Minimum size for HDBSCAN clusters
        min_samples: Minimum samples parameter for HDBSCAN

    Returns:
        List of merge groups, where each group is a list of label strings to merge
    """
    print(f"[LABEL_MERGE] Using HDBSCAN to cluster similar labels...")
    print(f"[LABEL_MERGE] Processing {len(atomic_labels)} labels (no truncation needed)")

    # Filter labels that have embeddings
    valid_labels = [lbl for lbl in atomic_labels if lbl in label_embeddings]
    if len(valid_labels) < len(atomic_labels):
        print(f"[LABEL_MERGE] Warning: {len(atomic_labels) - len(valid_labels)} labels missing embeddings")

    if len(valid_labels) < 2:
        print("[LABEL_MERGE] Not enough labels with embeddings for clustering")
        return []

    # Create embedding matrix
    embedding_matrix = np.array([label_embeddings[lbl] for lbl in valid_labels])

    # Apply UMAP for dimensionality reduction
    # IMPROVED: Increased n_components (10→35) and n_neighbors (15→25) per CHANGES_IN_VECTOR_COPY.md
    # This preserves more semantic information (96.6% compression vs 99%)
    print(f"[LABEL_MERGE] Reducing dimensionality from {embedding_matrix.shape[1]}D to 35D with UMAP...")
    import umap
    umap_reducer = umap.UMAP(
        n_components=35,  # Changed from 10 to preserve more information
        n_neighbors=min(25, len(valid_labels) - 1),  # Changed from 15 for better global structure
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = umap_reducer.fit_transform(embedding_matrix)

    # Apply HDBSCAN clustering
    # IMPROVED: Changed selection method (leaf→eom) and epsilon (0.25→0.0) per CHANGES_IN_VECTOR_COPY.md
    # eom is more stable, and epsilon=0.0 lets HDBSCAN decide boundaries naturally
    print(f"[LABEL_MERGE] Running HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # Correct - operates on UMAP-transformed space
        cluster_selection_method='eom',  # Changed from 'leaf' for more stable clusters
        cluster_selection_epsilon=0.0,  # Changed from 0.25 for natural boundaries
        prediction_data=True
    )

    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    # Group labels by cluster (excluding noise: -1)
    label_clusters = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id >= 0:  # Skip noise
            label_clusters[cluster_id].append(valid_labels[idx])

    # Convert to list of merge groups (only groups with 2+ labels)
    merge_groups = [labels for labels in label_clusters.values() if len(labels) >= 2]

    print(f"[LABEL_MERGE] HDBSCAN identified {len(merge_groups)} potential merge groups:")
    for i, group in enumerate(merge_groups[:10]):  # Show first 10
        counts_str = ", ".join([f"'{lbl}' ({label_counts.get(lbl, 0)})" for lbl in group[:5]])
        if len(group) > 5:
            counts_str += f", ... ({len(group)-5} more)"
        print(f"[LABEL_MERGE]   Group {i+1}: [{counts_str}]")
 
    if len(merge_groups) > 10:
        print(f"[LABEL_MERGE]   ... and {len(merge_groups)-10} more groups")

    # Count noise points
    n_noise = np.sum(cluster_labels == -1)
    print(f"[LABEL_MERGE] Noise points (unclustered labels): {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")

    return merge_groups


def identify_similar_labels(atomic_labels: List[str], label_counts: Dict[str, int]) -> List[List[str]]:
    """
    DEPRECATED: Use GPT to identify semantically similar concern labels that should be merged.

    This function is kept for backward compatibility but should not be used.
    Use cluster_labels_with_hdbscan() instead as it can handle unlimited labels.

    Args:
        atomic_labels: List of unique atomic label strings
        label_counts: Dictionary mapping label -> message count

    Returns:
        List of merge groups, where each group is a list of label strings to merge
    """
    print(f"[LABEL_MERGE] Using GPT to identify similar labels (DEPRECATED)...")

    # Sort labels by count (descending) for better GPT analysis
    sorted_labels = sorted(atomic_labels, key=lambda x: label_counts.get(x, 0), reverse=True)

    # IMPORTANT: Limit to top labels to avoid truncation
    # With 2374 labels, GPT response gets truncated
    MAX_LABELS_FOR_GPT = 500
    if len(sorted_labels) > MAX_LABELS_FOR_GPT:
        print(f"[LABEL_MERGE] Too many labels ({len(sorted_labels)}), limiting to top {MAX_LABELS_FOR_GPT} by message count")
        sorted_labels = sorted_labels[:MAX_LABELS_FOR_GPT]

    # Create label information for GPT
    label_info = []
    for idx, label in enumerate(sorted_labels):
        count = label_counts.get(label, 0)
        label_info.append(f"{idx}. '{label}' ({count} messages)")

    label_text = "\n".join(label_info)

    prompt = f'''You are an expert data analyst specializing in semantic clustering of customer concern labels.

TASK: Identify groups of concern labels that are semantically similar and should be merged together.

CONCERN LABELS TO ANALYZE (showing top {len(sorted_labels)} by message count):
{label_text}

MERGING CRITERIA:
- Labels that refer to the same concept but with different wording (e.g., "fatty liver grade 2" and "fatty liver grade two")
- Labels that are clear duplicates with minor variations (typos, capitalization, word order)
- Labels where one is clearly a more specific version of another and should be consolidated
- Consider message counts: prefer merging low-count labels into high-count labels

DO NOT MERGE:
- Labels that represent genuinely different concerns (e.g., "liver health" vs "kidney health")
- Labels that differ in specificity in a meaningful way (e.g., "fatty liver" vs "liver cirrhosis")
- Pre-sales vs post-sales concerns
- Different medical conditions even if related to same body part

IMPORTANT INSTRUCTIONS:
1. Analyze each label's semantic meaning and context
2. Group labels that refer to the same underlying concern
3. Each group should contain 2-10 labels maximum
4. Focus on HIGH-COUNT labels first (they matter most)
5. Not every label needs to be merged - only suggest clear semantic matches
6. Return label indices (numbers) not the label text

OUTPUT FORMAT: Return groups as a JSON list of lists containing label INDICES (numbers) only.
Example: [[0, 4, 7], [2, 9], [15, 23, 45]]

If no labels should be merged, return: []

CRITICAL: Return ONLY the JSON list, no explanatory text before or after. Start with [ and end with ].'''

    result = _gpt4_1_call(
        prompt=prompt,
        user_text="Identify similar concern labels for merging.",
        temperature=0.2,
        max_tokens=2000,
        timeout=120
    )

    # Robust fallback parsing with multiple strategies
    try:
        if not result:
            print("[LABEL_MERGE] No result from GPT")
            return []

        # Strategy 1: Direct string parsing
        if isinstance(result, str):
            result_str = result.strip()

            # Extract JSON array if embedded in text
            # Look for content between first [ and last ]
            import re
            json_match = re.search(r'\[.*\]', result_str, re.DOTALL)
            if json_match:
                result_str = json_match.group(0)

            # Try to parse as JSON
            try:
                parsed_result = json.loads(result_str)
            except json.JSONDecodeError as e:
                print(f"[LABEL_MERGE] JSON decode error: {e}")
                # Try eval as fallback
                try:
                    parsed_result = eval(result_str)
                except Exception as e2:
                    print(f"[LABEL_MERGE] Eval failed: {e2}")
                    # Last resort: try to fix incomplete JSON
                    if not result_str.endswith(']'):
                        print("[LABEL_MERGE] Attempting to fix truncated JSON by adding ]")
                        result_str += ']'
                        try:
                            parsed_result = json.loads(result_str)
                        except:
                            print("[LABEL_MERGE] Could not fix truncated JSON")
                            return []
                    else:
                        return []
        else:
            parsed_result = result

        # Strategy 2: Handle different response formats
        merge_groups_indices = None

        if isinstance(parsed_result, list):
            merge_groups_indices = parsed_result
        elif isinstance(parsed_result, dict):
            # Check for common keys
            for key in ['result', 'groups', 'merge_groups', 'merges']:
                if key in parsed_result and isinstance(parsed_result[key], list):
                    merge_groups_indices = parsed_result[key]
                    break

            # If still not found, take first list value
            if merge_groups_indices is None:
                for value in parsed_result.values():
                    if isinstance(value, list):
                        merge_groups_indices = value
                        break

        if merge_groups_indices is None:
            print(f"[LABEL_MERGE] Could not extract merge groups from response type: {type(parsed_result)}")
            return []

        # Strategy 3: Validate and convert indices to labels
        merge_groups = []
        for group_indices in merge_groups_indices:
            if not isinstance(group_indices, list):
                print(f"[LABEL_MERGE] Skipping invalid group: {group_indices}")
                continue

            group_labels = []
            for idx in group_indices:
                if isinstance(idx, int) and 0 <= idx < len(sorted_labels):
                    group_labels.append(sorted_labels[idx])
                else:
                    print(f"[LABEL_MERGE] Invalid index {idx} (max: {len(sorted_labels)-1})")

            if len(group_labels) >= 2:  # Only keep groups with at least 2 labels
                merge_groups.append(group_labels)

        print(f"[LABEL_MERGE] GPT identified {len(merge_groups)} potential merge groups:")
        for i, group in enumerate(merge_groups[:10]):  # Show first 10
            counts_str = ", ".join([f"'{lbl}' ({label_counts.get(lbl, 0)})" for lbl in group[:5]])
            if len(group) > 5:
                counts_str += f", ... ({len(group)-5} more)"
            print(f"[LABEL_MERGE]   Group {i+1}: [{counts_str}]")

        if len(merge_groups) > 10:
            print(f"[LABEL_MERGE]   ... and {len(merge_groups)-10} more groups")

        return merge_groups

    except Exception as e:
        print(f"[LABEL_MERGE] Failed to parse GPT merge suggestions: {e}")
        import traceback
        traceback.print_exc()

    return []


def calculate_label_cluster_centroids(
    label_clusters: List[List[str]],
    label_embeddings: Dict[str, np.ndarray]
) -> Dict[int, np.ndarray]:
    """
    Calculate centroid embeddings for each label cluster (from HDBSCAN).

    Args:
        label_clusters: List of label groups from HDBSCAN
        label_embeddings: Dictionary mapping label -> embedding

    Returns:
        Dictionary mapping cluster_id -> centroid embedding
    """
    centroids = {}

    for cluster_id, label_group in enumerate(label_clusters):
        if not label_group:
            continue

        # Get embeddings for all labels in this cluster
        cluster_embeddings = []
        for label in label_group:
            if label in label_embeddings:
                cluster_embeddings.append(label_embeddings[label])

        if cluster_embeddings:
            # Calculate centroid as mean of all label embeddings
            centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)

    return centroids


def generate_label_cluster_summaries(
    label_clusters: List[List[str]],
    label_counts: Dict[str, int]
) -> Dict[int, str]:
    """
    Generate human-readable summaries for each label cluster.

    Args:
        label_clusters: List of label groups from HDBSCAN
        label_counts: Message counts per label

    Returns:
        Dictionary mapping cluster_id -> summary string
    """
    summaries = {}

    for cluster_id, label_group in enumerate(label_clusters):
        if not label_group:
            summaries[cluster_id] = "Empty cluster"
            continue

        # Sort labels by frequency (highest first)
        sorted_labels = sorted(
            label_group,
            key=lambda x: label_counts.get(x, 0),
            reverse=True
        )

        # Take top 10 for summary
        top_labels = sorted_labels[:10]
        total_messages = sum(label_counts.get(lbl, 0) for lbl in label_group)

        # Create summary
        label_examples = ", ".join([f"'{lbl}'" for lbl in top_labels[:5]])
        more_text = f" + {len(label_group) - 5} more" if len(label_group) > 5 else ""

        summaries[cluster_id] = (
            f"{len(label_group)} labels ({total_messages} messages): "
            f"{label_examples}{more_text}"
        )

    return summaries


def identify_similar_label_clusters(
    label_clusters: List[List[str]],
    label_counts: Dict[str, int],
    batch_size: int = 20
) -> List[List[int]]:
    """
    Use GPT to identify which label-clusters should be merged together in batched API calls.

    This is meta-clustering: clustering the clusters themselves.
    Adapted from vector.py's batched approach.

    Args:
        label_clusters: List of label groups from HDBSCAN
        label_counts: Message counts per label
        batch_size: Maximum clusters per API call (default: 20)

    Returns:
        List of cluster ID groups to merge, e.g. [[0, 3], [1, 5, 7]]
    """
    print(f"[META_CLUSTER] Using GPT to identify similar label-clusters (batched)...")

    # Generate summaries for GPT
    summaries = generate_label_cluster_summaries(label_clusters, label_counts)

    if len(summaries) < 2:
        print("[META_CLUSTER] Less than 2 clusters, no merging possible")
        return []

    # Prepare cluster data list (sorted by ID for consistency)
    cluster_ids = sorted(summaries.keys())

    all_merge_groups = []

    # Process in batches if there are many clusters
    if len(cluster_ids) <= batch_size:
        # Single batch - process all at once
        print(f"[META_CLUSTER] Processing {len(cluster_ids)} clusters in a single batch...")
        batch_merge_groups = _process_meta_cluster_batch(cluster_ids, summaries, label_clusters)
        all_merge_groups.extend(batch_merge_groups)
    else:
        # Multiple batches
        print(f"[META_CLUSTER] Processing {len(cluster_ids)} clusters in batches of {batch_size}...")
        for batch_start in range(0, len(cluster_ids), batch_size):
            batch = cluster_ids[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(cluster_ids) - 1) // batch_size + 1

            print(f"[META_CLUSTER] Processing batch {batch_num}/{total_batches} ({len(batch)} clusters)")

            batch_merge_groups = _process_meta_cluster_batch(batch, summaries, label_clusters)
            all_merge_groups.extend(batch_merge_groups)

    return all_merge_groups


def _process_meta_cluster_batch(
    cluster_ids: List[int],
    summaries: Dict[int, str],
    label_clusters: List[List[str]]
) -> List[List[int]]:
    """
    Process a single batch of label-clusters for meta-clustering.

    Args:
        cluster_ids: List of cluster IDs to process in this batch
        summaries: Dict mapping cluster_id -> summary string
        label_clusters: Full list of label groups (for validation)

    Returns:
        List of cluster ID groups to merge for this batch
    """
    # Build cluster info for GPT
    cluster_info = []
    for cluster_id in cluster_ids:
        cluster_info.append(f"Cluster {cluster_id}: {summaries[cluster_id]}")

    cluster_text = "\n".join(cluster_info)

    prompt = f'''You are an expert data analyst specializing in semantic clustering analysis.

TASK: Identify groups of label-clusters that are semantically similar and should be merged together.

LABEL-CLUSTERS TO ANALYZE:
{cluster_text}

MERGING CRITERIA:
- Clusters should address the same core customer concern or topic
- Similar intent or meaning, even if expressed differently
- Would benefit from being combined for clearer insights
- Avoid merging clusters with different contexts

INSTRUCTIONS:
1. Analyze each cluster's labels for semantic similarity
2. Group clusters that address the same underlying concern
3. Only suggest merges that make semantic sense
4. Each group should contain 2-5 clusters maximum
5. Not every cluster needs to be merged - only suggest clear matches

OUTPUT FORMAT: Return groups as a JSON list of lists containing cluster IDs (numbers) only.
Example: [[0, 4], [2, 7, 9], [5, 12]]

If no clusters should be merged, return: []

Return ONLY the JSON list, no other text or explanation.'''

    result = _gpt4_1_call(
        prompt=prompt,
        user_text="Identify similar label-clusters for merging.",
        temperature=0.2,
        max_tokens=1000,
        timeout=90
    )

    # Parse GPT response
    try:
        if result:
            # Clean up any markdown artifacts
            result = result.replace("```json", "").replace("```", "").strip()
            if isinstance(result, str):
                parsed_result = json.loads(result) if result.strip().startswith('[') else eval(result)
            else:
                parsed_result = result

            # Handle different response formats
            if isinstance(parsed_result, dict):
                if 'result' in parsed_result:
                    merge_groups = parsed_result['result']
                else:
                    # Find any key with list value
                    merge_groups = None
                    for key, value in parsed_result.items():
                        if isinstance(value, list):
                            merge_groups = value
                            break
                    if merge_groups is None:
                        print("[META_CLUSTER] No valid merge groups in batch")
                        return []
            elif isinstance(parsed_result, list):
                merge_groups = parsed_result
            else:
                print(f"[META_CLUSTER] Unexpected response type: {type(parsed_result)}")
                return []

            # Validate cluster IDs
            valid_merge_groups = []
            for group in merge_groups:
                if isinstance(group, list) and len(group) >= 2:
                    # Filter to valid cluster IDs
                    valid_ids = [cid for cid in group if 0 <= cid < len(label_clusters)]
                    if len(valid_ids) >= 2:
                        valid_merge_groups.append(valid_ids)

            print(f"[META_CLUSTER] Batch identified {len(valid_merge_groups)} potential cluster merges:")
            for i, group in enumerate(valid_merge_groups[:10]):
                group_str = ", ".join([f"Cluster {cid}" for cid in group])
                print(f"[META_CLUSTER]   Group {i+1}: [{group_str}]")

            if len(valid_merge_groups) > 10:
                print(f"[META_CLUSTER]   ... and {len(valid_merge_groups) - 10} more groups")

            return valid_merge_groups

    except Exception as e:
        print(f"[META_CLUSTER] Failed to parse batch response: {e}")
        import traceback
        traceback.print_exc()

    return []


def validate_cluster_merges_with_centroids(
    meta_merge_groups: List[List[int]],
    cluster_centroids: Dict[int, np.ndarray],
    label_clusters: List[List[str]]
) -> Tuple[List[List[int]], List[Dict]]:
    """
    Validate meta-merge groups using cluster centroid distances.

    This is the vector.py approach: calculate pairwise distances between
    CLUSTER centroids (not individual labels).

    Args:
        meta_merge_groups: Groups of cluster IDs to merge from GPT
        cluster_centroids: Centroid embedding for each cluster
        label_clusters: Original label groups (for logging)

    Returns:
        Tuple of (validated groups, rejected groups with reasons)
    """
    print(f"[META_CLUSTER] Validating {len(meta_merge_groups)} meta-merge groups with centroid distances...")

    # Calculate all pairwise CENTROID distances for threshold
    cluster_ids = list(cluster_centroids.keys())
    all_centroid_distances = []

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            id1, id2 = cluster_ids[i], cluster_ids[j]
            distance = np.linalg.norm(cluster_centroids[id1] - cluster_centroids[id2])
            all_centroid_distances.append(distance)

    median_distance = np.median(all_centroid_distances)
    merge_threshold = median_distance * 1.02  # UPDATED: 2% window above median (stricter than before)

    print(f"[META_CLUSTER] Median centroid distance: {median_distance:.4f}")
    print(f"[META_CLUSTER] Merge threshold (median × 1.02): {merge_threshold:.4f}")
    print(f"[META_CLUSTER] Calculated from {len(cluster_ids)} cluster centroids (not individual labels)")

    # Validate each meta-merge group
    potential_merges = []  # List of (group, avg_distance) tuples
    rejected_groups = []

    for group in meta_merge_groups:
        if len(group) < 2:
            continue

        # Check all pairwise centroid distances in this group
        group_valid = True
        group_distances = []

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                id1, id2 = group[i], group[j]
                if id1 in cluster_centroids and id2 in cluster_centroids:
                    distance = np.linalg.norm(cluster_centroids[id1] - cluster_centroids[id2])
                    group_distances.append(distance)
                    if distance > merge_threshold:
                        group_valid = False
                        break
            if not group_valid:
                break

        if group_valid and group_distances:
            avg_distance = np.mean(group_distances)
            group_str = ", ".join([f"Cluster {cid}" for cid in group])
            print(f"[META_CLUSTER] ✓ Group [{group_str}] PASSED (avg: {avg_distance:.4f} <= {merge_threshold:.4f})")
            potential_merges.append((group, avg_distance))
        else:
            avg_distance = np.mean(group_distances) if group_distances else 0
            group_str = ", ".join([f"Cluster {cid}" for cid in group])
            print(f"[META_CLUSTER] ✗ Group [{group_str}] FAILED (avg: {avg_distance:.4f} > {merge_threshold:.4f})")
            rejected_groups.append({
                'group': group,
                'reason': 'centroid_distance_exceeded',
                'avg_distance': avg_distance,
                'threshold': merge_threshold
            })

    # Resolve conflicts: if a cluster appears in multiple groups, choose smallest distance
    claimed_clusters = set()
    validated_groups = []

    # Sort by average distance (closest first)
    potential_merges.sort(key=lambda x: x[1])

    for group, avg_distance in potential_merges:
        # Check if any cluster already claimed
        if any(cid in claimed_clusters for cid in group):
            group_str = ", ".join([f"Cluster {cid}" for cid in group])
            print(f"[META_CLUSTER] ✗ Group [{group_str}] SKIPPED - contains already claimed cluster(s)")
            rejected_groups.append({
                'group': group,
                'reason': 'conflict_with_other_merge',
                'avg_distance': avg_distance
            })
            continue

        # Accept this merge
        validated_groups.append(group)
        claimed_clusters.update(group)
        group_str = ", ".join([f"Cluster {cid}" for cid in group])
        print(f"[META_CLUSTER] ✓ Group [{group_str}] ACCEPTED (avg: {avg_distance:.4f})")

    print(f"[META_CLUSTER] Validated {len(validated_groups)}/{len(meta_merge_groups)} meta-merge groups")
    print(f"[META_CLUSTER] Rejected {len(rejected_groups)} groups")

    return validated_groups, rejected_groups


def apply_label_merges(
    concern_clusters: Dict,
    validated_merge_groups: List[List[str]],
    label_counts: Dict[str, int]
) -> Tuple[Dict, Dict[str, str]]:
    """
    Apply validated merges to concern clusters.

    For each merge group, pick the label with the highest count as canonical.
    Update all concern keys (including multi-tag concerns) with merged labels.

    Args:
        concern_clusters: Dict mapping concern_key -> list of messages
        validated_merge_groups: List of validated merge groups
        label_counts: Dict mapping label -> count

    Returns:
        - Updated concern_clusters dict
        - Merge mapping dict (old_label -> canonical_label)
    """
    print(f"[LABEL_MERGE] Applying {len(validated_merge_groups)} validated merges...")

    # Build merge mapping: old_label -> canonical_label
    merge_map = {}
    for group in validated_merge_groups:
        # Pick label with highest count as canonical
        canonical = max(group, key=lambda x: label_counts.get(x, 0))
        for label in group:
            if label != canonical:
                merge_map[label] = canonical

        merged_labels = ", ".join([f"'{lbl}'" for lbl in group if lbl != canonical])
        print(f"[LABEL_MERGE] Merging [{merged_labels}] → '{canonical}' ({label_counts.get(canonical, 0)} messages)")

    # Apply merges to concern clusters
    new_concern_clusters = defaultdict(list)

    for concern_key, messages in concern_clusters.items():
        # Check if this is a multi-tag concern
        try:
            if concern_key.startswith('['):
                parsed = json.loads(concern_key)
                if isinstance(parsed, list):
                    # Multi-tag: update each component label
                    updated_tags = [merge_map.get(tag, tag) for tag in parsed]
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_updated_tags = []
                    for tag in updated_tags:
                        if tag not in seen:
                            seen.add(tag)
                            unique_updated_tags.append(tag)

                    # Create new key
                    if len(unique_updated_tags) == 1:
                        # Multi-tag collapsed to single tag
                        new_key = unique_updated_tags[0]
                    else:
                        # Still multi-tag, recreate JSON key
                        new_key = json.dumps(sorted(unique_updated_tags), ensure_ascii=False)

                    new_concern_clusters[new_key].extend(messages)
                else:
                    # Single label (JSON string)
                    new_key = merge_map.get(concern_key, concern_key)
                    new_concern_clusters[new_key].extend(messages)
            else:
                # Plain string label
                new_key = merge_map.get(concern_key, concern_key)
                new_concern_clusters[new_key].extend(messages)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, treat as plain string
            new_key = merge_map.get(concern_key, concern_key)
            new_concern_clusters[new_key].extend(messages)

    print(f"[LABEL_MERGE] Clusters before merging: {len(concern_clusters)}")
    print(f"[LABEL_MERGE] Clusters after merging: {len(new_concern_clusters)}")

    return dict(new_concern_clusters), merge_map


# DISABLED: Splitting functionality commented out to evaluate base clustering quality
# This entire function is disabled to see pure HDBSCAN + meta-clustering output
pass  # Placeholder to maintain valid Python syntax


# def generate_cluster_titles_with_gpt(
#     clusters_data: List[Dict],
#     sample_size: int = 15
# ) -> List[Dict]:
#     """
#     Generate proper 1-3 word concern label titles for clusters using GPT.

#     ALL clusters marked with 'needs_gpt_title' get new titles.
#     These are semantic concern labels, not descriptive sentences.

#     Args:
#         clusters_data: List of cluster dicts
#         sample_size: Number of sample messages to send to GPT

#     Returns:
#         Updated clusters_data with proper concern label titles
#     """
#     print(f"\n[TITLE_GEN] Generating 1-3 word concern labels for clusters needing titles...")

#     clusters_needing_titles = [c for c in clusters_data if c.get('needs_gpt_title', False)]
#     print(f"[TITLE_GEN] Found {len(clusters_needing_titles)} clusters needing titles")

#     # Track used titles to avoid duplicates
#     used_titles = {}  # title -> count

#     for idx, cluster in enumerate(clusters_needing_titles):
#         messages = cluster['messages']
#         original_concern = cluster.get('split_from', cluster['concern_key'])

#         # Sample messages for GPT
#         if len(messages) <= sample_size:
#             sample_messages = messages
#         else:
#             # Sample evenly across the cluster
#             indices = np.linspace(0, len(messages) - 1, sample_size, dtype=int)
#             sample_messages = [messages[i] for i in indices]

#         message_texts = [msg['text'][:200] for msg in sample_messages]  # Truncate for GPT
#         message_text = "\n".join([f"- {text}" for text in message_texts])

#         prompt = f'''You are analyzing customer concern messages. Create a SHORT, specific concern label (1-3 words MAXIMUM).

# These messages are a subset from the broader concern: "{original_concern}"

# Sample messages from this specific subset:
# {message_text}

# REQUIREMENTS:
# - Label must be 1-3 words ONLY (e.g., "Order Tracking", "Payment Issues", "Product Quality")
# - Must be a NOUN PHRASE describing the concern type
# - Must be specific to what makes THIS subset different from the original concern
# - Do NOT use descriptive sentences
# - Do NOT use "split", "subset", or similar meta-terms

# Examples of GOOD labels:
# - "Order Tracking"
# - "Delivery Delay"
# - "Refund Request"
# - "Product Quality"
# - "Payment Failed"

# Examples of BAD labels:
# - "Order status split 1/4" (too meta)
# - "Customers asking about when their order will arrive" (too long)
# - "Various delivery concerns" (too vague)

# Return ONLY the 1-3 word concern label, nothing else.'''

#         try:
#             new_title = gpt4o_call(
#                 text="Generate concern label",
#                 prompt=prompt,
#                 temperature=0.3,
#                 json_needed=False
#             )

#             if new_title and len(new_title.strip()) > 0:
#                 # Clean up the title
#                 new_title = new_title.strip().strip('"').strip("'")

#                 # Validate length (should be 1-3 words)
#                 word_count = len(new_title.split())
#                 if word_count > 3:
#                     print(f"[TITLE_GEN] Warning: Title too long ({word_count} words), truncating: '{new_title}'")
#                     new_title = " ".join(new_title.split()[:3])

#                 # Check for duplicates and append number if needed
#                 final_title = new_title
#                 if new_title in used_titles:
#                     # Duplicate found, append number
#                     used_titles[new_title] += 1
#                     final_title = f"{new_title} {used_titles[new_title]}"
#                     print(f"[TITLE_GEN] [{idx+1}/{len(clusters_needing_titles)}] Duplicate detected, using: '{final_title}'")
#                 else:
#                     # First occurrence
#                     used_titles[new_title] = 1
#                     print(f"[TITLE_GEN] [{idx+1}/{len(clusters_needing_titles)}] Generated: '{final_title}'")

#                 # Update cluster with final title
#                 cluster['concern_key'] = final_title
#                 cluster['original_temp_key'] = cluster.get('concern_key')
#                 cluster['split_from_original'] = original_concern
#             else:
#                 # Fallback: use original concern key
#                 cluster['concern_key'] = original_concern
#                 print(f"[TITLE_GEN] [{idx+1}/{len(clusters_needing_titles)}] Failed, using original: '{original_concern}'")

#         except Exception as e:
#             print(f"[TITLE_GEN] [{idx+1}/{len(clusters_needing_titles)}] Error: {e}, using original")
#             cluster['concern_key'] = original_concern

#         # Remove the flag
#         cluster.pop('needs_gpt_title', None)

#     print(f"[TITLE_GEN] Completed title generation for {len(clusters_needing_titles)} clusters")
#     return clusters_data


def reassign_miscellaneous_outliers(
    clusters_data: List[Dict],
    chat_data: Dict[str, Dict],
    label_embeddings: Dict[str, np.ndarray],
    miscellaneous_threshold: int = 100,
    distance_threshold: float = 0.3
) -> List[Dict]:
    """
    Reassign messages from overly large "miscellaneous" or "other" clusters to more specific clusters.

    Uses nearest neighbor approach based on embeddings similar to vector.py's reassign_outliers.

    Args:
        clusters_data: List of cluster dicts
        chat_data: Full chat data
        label_embeddings: Embeddings for labels
        miscellaneous_threshold: Clusters larger than this are considered catch-alls
        distance_threshold: Maximum distance for reassignment

    Returns:
        Updated clusters_data with reassigned messages
    """
    print(f"\n[REASSIGN] Reassigning messages from large miscellaneous clusters...")

    # Identify miscellaneous clusters (very large, generic names)
    misc_keywords = ['other', 'miscellaneous', 'general', 'unknown', 'misc']
    misc_clusters = []
    regular_clusters = []

    for cluster in clusters_data:
        concern_key = cluster['concern_key'].lower()
        is_misc = any(keyword in concern_key for keyword in misc_keywords)
        session_count = len(set(msg['session_id'] for msg in cluster['messages']))

        if is_misc and session_count > miscellaneous_threshold:
            misc_clusters.append(cluster)
            print(f"[REASSIGN] Found miscellaneous cluster: '{cluster['concern_key']}' ({session_count} sessions)")
        else:
            regular_clusters.append(cluster)

    if not misc_clusters:
        print("[REASSIGN] No large miscellaneous clusters found")
        return clusters_data

    # Build embeddings for regular clusters
    cluster_embeddings = {}
    for cluster in regular_clusters:
        concern_key = cluster['concern_key']
        # Get base concern key (remove split suffix)
        base_key = cluster.get('split_from', concern_key)
        if base_key in label_embeddings:
            cluster_embeddings[concern_key] = label_embeddings[base_key]

    if not cluster_embeddings:
        print("[REASSIGN] No embeddings available for reassignment")
        return clusters_data

    # For each misc cluster, try to reassign its messages
    reassigned_count = 0
    for misc_cluster in misc_clusters:
        messages = misc_cluster['messages']

        # Sample messages to get representative embedding
        # In practice, we'd want embeddings for each message, but for now use label matching
        print(f"[REASSIGN] Processing {len(messages)} messages from '{misc_cluster['concern_key']}'")

        # For now, keep misc clusters as-is since we need message-level embeddings
        # This is a placeholder for future enhancement
        regular_clusters.append(misc_cluster)

    print(f"[REASSIGN] Reassigned {reassigned_count} messages")
    return regular_clusters + misc_clusters


def remove_multi_tag_concerns(concern_clusters: Dict) -> Tuple[Dict, List[Dict]]:
    """
    Remove all multi-tag concerns from clusters.

    Args:
        concern_clusters: Dict mapping concern_key -> list of messages

    Returns:
        - Updated concern_clusters without multi-tag concerns
        - List of removed multi-tag clusters (for logging)
    """
    print(f"[CLEANUP] Removing multi-tag concerns...")

    single_tag_clusters = {}
    removed_multi_tags = []

    for concern_key, messages in concern_clusters.items():
        # Check if multi-tag
        is_multi_tag = False
        try:
            if concern_key.startswith('['):
                parsed = json.loads(concern_key)
                if isinstance(parsed, list) and len(parsed) > 1:
                    is_multi_tag = True
        except (json.JSONDecodeError, TypeError):
            pass

        if is_multi_tag:
            removed_multi_tags.append({
                'concern_key': concern_key,
                'message_count': len(messages)
            })
        else:
            single_tag_clusters[concern_key] = messages

    total_removed_messages = sum(r['message_count'] for r in removed_multi_tags)
    print(f"[CLEANUP] Removed {len(removed_multi_tags)} multi-tag concerns ({total_removed_messages} messages)")

    return single_tag_clusters, removed_multi_tags


def second_round_clustering(
    low_frequency_clusters: List[Dict],
    regular_clusters: List[Dict],
    label_embeddings: Dict[str, np.ndarray],
    min_message_threshold: int = 5,
    excluded_targets: Optional[List[str]] = None
) -> Tuple[Dict, List[str], Dict[str, str]]:
    """
    Attempt to merge low-frequency concerns (<5 messages) into existing regular clusters.

    Uses GPT recommendations with a stricter distance threshold (median * 1.2).

    Args:
        low_frequency_clusters: List of low-frequency cluster dicts
        regular_clusters: List of regular cluster dicts
        label_embeddings: Dictionary mapping label -> embedding
        min_message_threshold: Minimum messages for regular cluster
        excluded_targets: List of cluster labels that should NOT receive Round 2 merges

    Returns:
        - Updated concern_clusters dict with merged low-frequency concerns
        - List of deleted concern labels that couldn't be merged
        - Dict mapping low_freq_label -> regular_label (Round 2 merge map)
    """
    if not low_frequency_clusters:
        print("[ROUND2] No low-frequency clusters to process")
        return {}, [], {}

    if excluded_targets is None:
        excluded_targets = []

    print(f"\n[ROUND2] Starting second-round clustering for {len(low_frequency_clusters)} low-frequency concerns...")
    if excluded_targets:
        print(f"[ROUND2] Excluding {len(excluded_targets)} targets from Round 2 merges: {excluded_targets}")

    # Extract labels
    low_freq_labels = [c['concern_key'] for c in low_frequency_clusters]
    regular_labels = [c['concern_key'] for c in regular_clusters]

    # Filter out excluded targets from regular labels
    regular_labels_filtered = [lbl for lbl in regular_labels if lbl not in excluded_targets]

    if not regular_labels_filtered:
        print("[ROUND2] All regular clusters are excluded, deleting all low-frequency clusters")
        deleted_labels = low_freq_labels
        regular_map = {c['concern_key']: c['messages'] for c in regular_clusters}
        return regular_map, deleted_labels, {}

    # Build mapping for quick lookup
    low_freq_map = {c['concern_key']: c['messages'] for c in low_frequency_clusters}
    regular_map = {c['concern_key']: c['messages'] for c in regular_clusters}

    # Calculate pairwise distances between low-freq and FILTERED regular clusters
    all_distances = []
    for lf_label in low_freq_labels:
        if lf_label not in label_embeddings:
            continue
        for reg_label in regular_labels_filtered:  # Use filtered list
            if reg_label not in label_embeddings:
                continue

            emb_lf = label_embeddings[lf_label]
            emb_reg = label_embeddings[reg_label]
            similarity = cosine_similarity([emb_lf], [emb_reg])[0][0]
            distance = 1 - similarity
            all_distances.append(distance)

    if not all_distances:
        print("[ROUND2] No valid embeddings for distance calculation")
        return regular_map, low_freq_labels, {}

    median_distance = np.median(all_distances)
    round2_threshold = median_distance * 1.0  # UPDATED: No tolerance - exact median (strictest)

    print(f"[ROUND2] Median distance (low-freq to regular): {median_distance:.4f}")
    print(f"[ROUND2] Round 2 threshold (median × 1.0 - no tolerance): {round2_threshold:.4f}")

    # Ask GPT to suggest merges for low-frequency concerns (use filtered list)
    merge_suggestions = identify_low_frequency_merges(
        low_freq_labels,
        regular_labels_filtered,  # Use filtered list
        {**low_freq_map, **regular_map}
    )

    # Validate suggestions with distance threshold AND excluded targets
    validated_merges = {}  # low_freq_label -> regular_label
    for lf_label, reg_label in merge_suggestions.items():
        # Check if target is excluded
        if reg_label in excluded_targets:
            print(f"[ROUND2] ✗ Rejected '{lf_label}' → '{reg_label}' (target is excluded)")
            continue

        if lf_label not in label_embeddings or reg_label not in label_embeddings:
            continue

        emb_lf = label_embeddings[lf_label]
        emb_reg = label_embeddings[reg_label]
        similarity = cosine_similarity([emb_lf], [emb_reg])[0][0]
        distance = 1 - similarity

        if distance <= round2_threshold:
            validated_merges[lf_label] = reg_label
            lf_count = len(low_freq_map[lf_label])
            print(f"[ROUND2] ✓ Merging '{lf_label}' ({lf_count} msgs) → '{reg_label}' (distance: {distance:.4f})")
        else:
            print(f"[ROUND2] ✗ Rejected '{lf_label}' → '{reg_label}' (distance: {distance:.4f} > {round2_threshold:.4f})")

    # Apply validated merges
    updated_clusters = dict(regular_map)  # Start with regular clusters
    deleted_labels = []
    round2_merge_map = {}  # Track Round 2 merges: low_freq_label -> regular_label

    for lf_label, messages in low_freq_map.items():
        if lf_label in validated_merges:
            # Merge into regular cluster
            target_label = validated_merges[lf_label]
            updated_clusters[target_label].extend(messages)
            round2_merge_map[lf_label] = target_label  # TRACK THIS
        else:
            # Delete this low-frequency concern
            deleted_labels.append(lf_label)
            print(f"[ROUND2] 🗑️  Deleted '{lf_label}' ({len(messages)} messages)")

    print(f"\n[ROUND2] Summary:")
    print(f"  - Low-frequency concerns merged: {len(validated_merges)}")
    print(f"  - Low-frequency concerns deleted: {len(deleted_labels)}")

    return updated_clusters, deleted_labels, round2_merge_map


def identify_low_frequency_merges(
    low_freq_labels: List[str],
    regular_labels: List[str],
    label_counts: Dict[str, List],
    batch_size: int = 30
) -> Dict[str, str]:
    """
    Use GPT to suggest which low-frequency labels should merge into regular clusters in batched API calls.

    Args:
        low_freq_labels: List of low-frequency labels
        regular_labels: List of regular cluster labels
        label_counts: Dict mapping label -> messages (for counting)
        batch_size: Maximum low-freq labels per API call (default: 30)

    Returns:
        Dict mapping low_freq_label -> regular_label
    """
    print(f"[ROUND2] Using GPT to suggest merges for {len(low_freq_labels)} low-frequency concerns (batched)...")

    all_valid_merges = {}

    # Build regular clusters info string once (shared across all batches)
    reg_info = "\n".join([f"  - '{lbl}' ({len(label_counts.get(lbl, []))} messages)" for lbl in regular_labels])

    # Process low-frequency labels in batches
    if len(low_freq_labels) <= batch_size:
        # Single batch
        print(f"[ROUND2] Processing {len(low_freq_labels)} low-freq labels in a single batch...")
        batch_merges = _process_low_freq_batch(low_freq_labels, regular_labels, label_counts, reg_info)
        all_valid_merges.update(batch_merges)
    else:
        # Multiple batches
        print(f"[ROUND2] Processing {len(low_freq_labels)} low-freq labels in batches of {batch_size}...")
        for batch_start in range(0, len(low_freq_labels), batch_size):
            batch = low_freq_labels[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(low_freq_labels) - 1) // batch_size + 1

            print(f"[ROUND2] Processing batch {batch_num}/{total_batches} ({len(batch)} low-freq labels)")

            batch_merges = _process_low_freq_batch(batch, regular_labels, label_counts, reg_info)
            all_valid_merges.update(batch_merges)

    print(f"[ROUND2] GPT suggested {len(all_valid_merges)} total potential merges across all batches")
    return all_valid_merges


def _process_low_freq_batch(
    low_freq_batch: List[str],
    regular_labels: List[str],
    label_counts: Dict[str, List],
    reg_info: str
) -> Dict[str, str]:
    """
    Process a single batch of low-frequency labels for merge suggestions.

    Args:
        low_freq_batch: Batch of low-frequency labels to process
        regular_labels: List of all regular cluster labels
        label_counts: Dict mapping label -> messages (for counting)
        reg_info: Pre-formatted string of regular cluster info

    Returns:
        Dict mapping low_freq_label -> regular_label for this batch
    """
    # Build low-freq info string for this batch
    lf_info = "\n".join([f"  - '{lbl}' ({len(label_counts.get(lbl, []))} messages)" for lbl in low_freq_batch])

    prompt = f'''You are an expert data analyst specializing in semantic clustering.

TASK: Suggest which low-frequency concern labels (< 5 messages) should be merged into existing regular clusters.

LOW-FREQUENCY CONCERNS (candidates for merging):
{lf_info}

REGULAR CLUSTERS (merge targets):
{reg_info}

MERGING CRITERIA:
- Low-frequency label should be semantically very similar to a regular cluster
- Prefer merging if the low-frequency label is clearly a variant or subset of a regular label
- Only suggest merges where meaning is nearly identical
- If no good match exists, don't suggest a merge (label will be deleted)

OUTPUT FORMAT: Return a JSON object mapping low-frequency labels to their target regular labels.
Example: {{"sex timing issue": "sexual health", "fatty liver grade 2": "fatty liver"}}

If a low-frequency label has no good match, simply omit it from the output.

Return ONLY the JSON object, no other text or explanation.'''

    result = _gpt4_1_call(
        prompt=prompt,
        user_text="Suggest merges for low-frequency concerns.",
        temperature=0.2,
        max_tokens=1500,
        timeout=90
    )

    try:
        if result:
            # Clean up any markdown artifacts
            result = result.replace("```json", "").replace("```", "").strip()
            if isinstance(result, str):
                parsed_result = json.loads(result) if result.strip().startswith('{') else eval(result)
            else:
                parsed_result = result

            if isinstance(parsed_result, dict):
                # Filter to only include valid mappings
                valid_merges = {}
                for lf_label, reg_label in parsed_result.items():
                    if lf_label in low_freq_batch and reg_label in regular_labels:
                        valid_merges[lf_label] = reg_label

                print(f"[ROUND2] Batch suggested {len(valid_merges)} potential merges")
                return valid_merges

    except Exception as e:
        print(f"[ROUND2] Failed to parse batch suggestions: {e}")

    return {}


def categorize_concern_sales_stage(concern_label: str) -> str:
    """
    Classify a concern label as pre-sales or post-sales based on the label text.

    Args:
        concern_label: The concern/requirement label text (e.g., "Order Status", "Best combo offers")

    Returns:
        String: 'pre-sales' or 'post-sales'
    """
    prompt = f'''You are an expert customer service analyst specializing in sales funnel classification.

TASK: Classify this customer concern label as either "pre-sales" or "post-sales".

CONCERN LABEL: "{concern_label}"

DEFINITIONS:
- PRE-SALES: Concerns from customers who are researching, considering, or in the process of making their first purchase. Includes product inquiries, pricing questions, feature comparisons, availability checks, best offers, product recommendations, coupons, membership questions, nutrition/ingredient inquiries, "why should I use it", product benefits, etc.

- POST-SALES: Concerns from customers who have already purchased or are dealing with existing orders. Includes order status, delivery tracking, order not received, missing items, order confirmation, pending pickup, delivery issues, change of address, order cancellation, returns, refunds, product complaints (taste, quality), damaged products, account problems, COD issues, etc.

CLASSIFICATION RULES:
1. READ THE LABEL CAREFULLY: Focus on what the customer is primarily trying to do
2. BE DECISIVE: Every concern must be classified as either pre-sales OR post-sales
3. KEYWORD INDICATORS:
   - POST-SALES keywords: "order", "delivery", "tracking", "not received", "missing", "damaged", "refund", "cancel", "status", "shipment", "courier", "pending", "dispatch", "COD", "complaint", "return", "confirmation"
   - PRE-SALES keywords: "price", "offer", "discount", "coupon", "combo", "product", "recommend", "ingredients", "suitable", "membership", "benefit", "inquiry", "content", "free", "compare", "availability", "why use"

EXAMPLES:
- "Order Status" -> post-sales (customer has already ordered)
- "Best combo offers" -> pre-sales (customer exploring purchase options)
- "Delivery Inquiry" -> post-sales (customer asking about existing order)
- "Coupon Code Issue" -> pre-sales (customer preparing to purchase)
- "Order not received" -> post-sales (customer has ordered but not received)
- "Why should I use it?" -> pre-sales (customer considering purchase)
- "Delivery issue" -> post-sales (problem with existing order)
- "Protein Content Inquiry" -> pre-sales (researching product before purchase)

OUTPUT FORMAT: Return exactly one word - either "pre-sales" or "post-sales" - nothing else.'''

    result = _gpt4_1_call(
        prompt=prompt,
        user_text="Classify concern sales stage.",
        temperature=0.1,
        max_tokens=20,
        timeout=30
    )

    # Clean and validate result
    result = result.strip().lower() if result else ""
    if "pre-sales" in result:
        return "pre-sales"
    elif "post-sales" in result:
        return "post-sales"
    else:
        return "pre-sales"  # Default fallback


def categorize_clusters(
    concern_clusters: Dict,
    min_message_threshold: int = 5
) -> Dict[str, List]:
    """
    Categorize clusters into regular and low-frequency groups.
    Multi-tag concerns should already be removed before calling this.

    Args:
        concern_clusters: Dict mapping concern_key -> list of messages
        min_message_threshold: Minimum messages for regular cluster

    Returns:
        Dict with keys: 'regular', 'low_frequency'
    """
    print(f"[CATEGORIZE] Categorizing {len(concern_clusters)} clusters...")

    regular = []
    low_frequency = []

    for concern_key, messages in concern_clusters.items():
        message_count = len(messages)

        cluster_data = {
            'concern_key': concern_key,
            'message_count': message_count,
            'messages': messages
        }

        if message_count < min_message_threshold:
            low_frequency.append(cluster_data)
        else:
            regular.append(cluster_data)

    print(f"[CATEGORIZE] Regular clusters: {len(regular)}")
    print(f"[CATEGORIZE] Low-frequency clusters (< {min_message_threshold} messages): {len(low_frequency)}")

    # Sort each category by message count (descending)
    for category in [regular, low_frequency]:
        category.sort(key=lambda x: x['message_count'], reverse=True)

    # Add sales stage categorization for regular clusters only (batched)
    print(f"\n[CATEGORIZE] Adding sales stage categorization for {len(regular)} regular clusters...")

    # Collect all concern labels for batched processing
    concern_labels = [cluster['concern_key'] for cluster in regular]

    # Use batched categorization for efficiency
    sales_stage_results = batched_categorize_sales_stages(concern_labels)

    # Apply results to clusters
    for cluster in regular:
        concern_label = cluster['concern_key']
        cluster['sales_stage'] = sales_stage_results.get(concern_label, "pre-sales")

    # Print summary
    pre_sales_count = sum(1 for c in regular if c['sales_stage'] == 'pre-sales')
    post_sales_count = sum(1 for c in regular if c['sales_stage'] == 'post-sales')
    print(f"[CATEGORIZE] Sales stage breakdown: {pre_sales_count} pre-sales, {post_sales_count} post-sales")

    return {
        'regular': regular,
        'low_frequency': low_frequency
    }


def merge_concern_labels(
    concern_clusters: Dict,
    chat_data: Dict[str, Dict],
    output_dir: str = "vec_outs",
    min_message_threshold: int = 5,
    merge_enabled: bool = True,
    round2_excluded_targets: Optional[List[str]] = None,
    enable_splitting: bool = True,
    enable_title_generation: bool = True,
    enable_reassignment: bool = True,
    max_session_threshold: int = 250,
    client_name: str = None,
    date_range: str = None
) -> Dict:
    """
    Main function to merge similar concern labels using HDBSCAN clustering workflow.

    NEW HDBSCAN-based workflow (no 500-label limit):
    1. Round 1: Cluster semantically similar labels using HDBSCAN (unlimited labels!)
    2. Remove all multi-tag concerns
    3. Round 2: Merge low-frequency concerns (<5 msgs) into regular clusters
    4. Split large clusters by session volume using KNN
    5. Generate better titles for split clusters with GPT
    6. Reassign miscellaneous outliers to specific clusters
    7. Delete remaining low-frequency concerns

    Args:
        concern_clusters: Dict mapping normalized_concern -> list of messages
        chat_data: Full chat data dict (session_id -> session data)
        output_dir: Directory for caching embeddings
        min_message_threshold: Minimum messages for regular cluster
        merge_enabled: Whether to enable label merging
        round2_excluded_targets: List of cluster labels to exclude from Round 2 merging
        enable_splitting: Whether to split large clusters by session volume
        enable_title_generation: Whether to generate better titles for split clusters
        enable_reassignment: Whether to reassign miscellaneous outliers
        max_session_threshold: Maximum sessions per cluster before splitting

    Returns:
        Dictionary with categorized clusters and merge metadata
    """
    print("\n" + "="*80)
    print("[LABEL_MERGE] Starting concern label merging pipeline...")
    print("="*80 + "\n")

    # Initialize comprehensive history tracking
    cluster_history = {
        'operations': [],  # List of all operations performed
        'label_tracking': {},  # Track each label's journey: original -> merged/deleted/retained
        'timestamp': pd.Timestamp.now().isoformat()
    }

    # ========== STEP 0: REMOVE MULTI-TAG CONCERNS FIRST ==========
    print("\n[STEP 0] Removing multi-tag concerns before clustering...")
    original_cluster_count = len(concern_clusters)
    concern_clusters, removed_multi_tags = remove_multi_tag_concerns(concern_clusters)

    # Track multi-tag removal
    for multi_tag in removed_multi_tags:
        cluster_history['operations'].append({
            'step': 'multi_tag_removal',
            'action': 'removed',
            'label': multi_tag['concern_key'],
            'message_count': multi_tag['message_count'],
            'reason': 'multi_tag_concern'
        })
        cluster_history['label_tracking'][multi_tag['concern_key']] = {
            'status': 'removed',
            'reason': 'multi_tag_concern',
            'message_count': multi_tag['message_count']
        }

    print(f"[STEP 0] Removed {len(removed_multi_tags)} multi-tag concerns")
    print(f"[STEP 0] Proceeding with {len(concern_clusters)} single-tag clusters")

    if not merge_enabled:
        print("[LABEL_MERGE] Merging disabled, categorizing only...")
        categorized = categorize_clusters(concern_clusters, min_message_threshold)
        categorized['removed_multi_tags'] = removed_multi_tags
        categorized['deleted_labels'] = []
        categorized['cluster_history'] = cluster_history
        return categorized

    # ========== ROUND 1: SEMANTIC LABEL MERGING WITH HDBSCAN ==========
    print("\n[ROUND1] Starting semantic label merging with HDBSCAN...")

    # Step 1: Extract atomic labels (now all single-tag)
    atomic_labels, label_counts, label_to_concerns = extract_atomic_labels_from_concerns(concern_clusters)

    if len(atomic_labels) < 2:
        print("[ROUND1] Less than 2 unique labels, skipping round 1 merge")
        round1_merge_map = {}
        round1_validated_groups = []
        merged_clusters = concern_clusters
        label_embeddings = {}  # Initialize empty dict for Round 2 reference
    else:
        # Step 2: Generate embeddings
        label_embeddings = generate_label_embeddings(atomic_labels, output_dir, client_name, date_range)

        if not label_embeddings:
            print("[ROUND1] Failed to generate embeddings, skipping round 1 merge")
            round1_merge_map = {}
            round1_validated_groups = []
            merged_clusters = concern_clusters
            label_embeddings = {}  # Ensure it's defined even if empty
        else:
            # Step 3: Cluster similar labels with HDBSCAN (NO 500 LABEL LIMIT!)
            # Use adaptive min_cluster_size based on dataset size
            dataset_size = len(atomic_labels)
            adaptive_min_cluster_size = max(2, min(10, dataset_size // 200))  # Scale with dataset

            label_clusters = cluster_labels_with_hdbscan(
                list(atomic_labels),
                label_counts,
                label_embeddings,
                min_cluster_size=adaptive_min_cluster_size,
                min_samples=max(2, adaptive_min_cluster_size - 1)
            )

            print(f"[ROUND1] Using adaptive min_cluster_size={adaptive_min_cluster_size} for {dataset_size} labels")

            if not label_clusters:
                print("[ROUND1] No label clusters identified by HDBSCAN")
                round1_merge_map = {}
                round1_validated_groups = []
                round1_rejected_groups = []
                merged_clusters = concern_clusters
            else:
                print(f"[ROUND1] HDBSCAN created {len(label_clusters)} label-clusters")

                # Step 4: META-CLUSTERING - Calculate centroids for each label-cluster
                print("[ROUND1] Calculating centroids for each label-cluster...")
                cluster_centroids = calculate_label_cluster_centroids(label_clusters, label_embeddings)
                print(f"[ROUND1] Calculated {len(cluster_centroids)} cluster centroids")

                # Step 5: Ask GPT which label-CLUSTERS should merge (meta-clustering)
                print("[ROUND1] Asking GPT which label-clusters should merge...")
                meta_merge_groups = identify_similar_label_clusters(label_clusters, label_counts)

                if not meta_merge_groups:
                    print("[ROUND1] GPT found no label-clusters to merge")
                    # Keep label_clusters as final groups
                    validated_groups = label_clusters
                    round1_rejected_groups = []
                else:
                    # Step 6: Validate meta-merges with cluster centroid distances
                    validated_meta_merges, rejected_groups = validate_cluster_merges_with_centroids(
                        meta_merge_groups,
                        cluster_centroids,
                        label_clusters
                    )
                    round1_rejected_groups = rejected_groups

                    # Track rejected meta-merges in history
                    for rejection in rejected_groups:
                        cluster_history['operations'].append({
                            'step': 'round1_meta_validation',
                            'action': 'rejected_cluster_merge',
                            'cluster_ids': rejection['group'],
                            'reason': rejection['reason'],
                            'details': f"Avg distance {rejection.get('avg_distance', 'N/A'):.4f}"
                        })

                    # Step 7: Apply meta-merges by combining label-clusters
                    final_label_groups = []
                    merged_cluster_ids = set()

                    # Merge validated groups
                    for meta_group in validated_meta_merges:
                        # Combine all labels from these clusters
                        combined_labels = []
                        for cluster_id in meta_group:
                            combined_labels.extend(label_clusters[cluster_id])
                            merged_cluster_ids.add(cluster_id)
                        final_label_groups.append(combined_labels)

                    # Add unmerged clusters
                    for cluster_id, label_group in enumerate(label_clusters):
                        if cluster_id not in merged_cluster_ids:
                            final_label_groups.append(label_group)

                    validated_groups = final_label_groups
                    print(f"[ROUND1] Meta-clustering reduced {len(label_clusters)} → {len(validated_groups)} final groups")

                # Step 8: Apply label merges within each final group
                if not validated_groups:
                    print("[ROUND1] No validated groups to merge")
                    round1_merge_map = {}
                    round1_validated_groups = []
                    merged_clusters = concern_clusters
                else:
                    merged_clusters, round1_merge_map = apply_label_merges(
                        concern_clusters, validated_groups, label_counts
                    )
                    round1_validated_groups = validated_groups

                    # Track successful merges in history
                    for old_label, canonical_label in round1_merge_map.items():
                        cluster_history['operations'].append({
                            'step': 'round1_merge',
                            'action': 'merged',
                            'from_label': old_label,
                            'to_label': canonical_label,
                            'message_count': label_counts.get(old_label, 0)
                        })
                        cluster_history['label_tracking'][old_label] = {
                            'status': 'merged',
                            'merged_into': canonical_label,
                            'step': 'round1',
                            'message_count': label_counts.get(old_label, 0)
                        }

    print(f"\n[ROUND1] Complete: {len(concern_clusters)} → {len(merged_clusters)} clusters")

    # ========== CATEGORIZE AFTER ROUND 1 ==========
    categorized_after_round1 = categorize_clusters(merged_clusters, min_message_threshold)

    # ========== ROUND 2: LOW-FREQUENCY CLUSTERING ==========
    if categorized_after_round1['low_frequency']:
        # Re-generate embeddings for any new labels (from round 1 merges)
        all_labels_after_round1 = set(merged_clusters.keys())
        label_embeddings_round2 = generate_label_embeddings(all_labels_after_round1, output_dir, client_name, date_range)

        updated_clusters, deleted_labels, round2_merge_map = second_round_clustering(
            categorized_after_round1['low_frequency'],
            categorized_after_round1['regular'],
            label_embeddings_round2,
            min_message_threshold,
            excluded_targets=round2_excluded_targets
        )

        # Track Round 2 merges in history
        for lf_label, target_label in round2_merge_map.items():
            cluster_history['operations'].append({
                'step': 'round2_merge',
                'action': 'merged_low_frequency',
                'from_label': lf_label,
                'to_label': target_label,
                'reason': 'low_frequency_merge'
            })
            cluster_history['label_tracking'][lf_label] = {
                'status': 'merged',
                'merged_into': target_label,
                'step': 'round2',
                'reason': 'low_frequency'
            }

        # Track Round 2 deletions in history
        for deleted_label in deleted_labels:
            cluster_history['operations'].append({
                'step': 'round2_deletion',
                'action': 'deleted',
                'label': deleted_label,
                'reason': 'low_frequency_no_match'
            })
            cluster_history['label_tracking'][deleted_label] = {
                'status': 'deleted',
                'step': 'round2',
                'reason': 'low_frequency_no_match'
            }
    else:
        print("\n[ROUND2] No low-frequency clusters to process")
        # Just keep regular clusters
        updated_clusters = {c['concern_key']: c['messages'] for c in categorized_after_round1['regular']}
        deleted_labels = []
        round2_merge_map = {}
        label_embeddings_round2 = label_embeddings

    # ========== POST-PROCESSING: SPLIT, TITLE, REASSIGN ==========
    print("\n[POST-PROCESS] Starting cluster splitting and enhancement...")

    # Convert updated_clusters dict to list format for processing
    clusters_list = [
        {
            'concern_key': key,
            'messages': msgs,
            'message_count': len(msgs)
        }
        for key, msgs in updated_clusters.items()
    ]

    # Step 1: Split large clusters by session volume (using KNN-based approach)
    # if enable_splitting:
    #     clusters_list = split_large_clusters_by_session_volume(
    #         clusters_list,
    #         chat_data=chat_data,
    #         max_session_threshold=max_session_threshold,
    #         label_embeddings=label_embeddings_round2
    #     )
    # else:
    #     print("[POST-PROCESS] Cluster splitting disabled")

    # Step 2: Generate better titles for split clusters
    # if enable_title_generation:
    #     clusters_list = generate_cluster_titles_with_gpt(clusters_list, sample_size=10)
    # else:
    #     print("[POST-PROCESS] Title generation disabled")

    # Step 3: Reassign miscellaneous outliers to specific clusters
    if enable_reassignment:
        clusters_list = reassign_miscellaneous_outliers(
            clusters_list,
            chat_data=chat_data,
            label_embeddings=label_embeddings_round2,
            miscellaneous_threshold=100
        )
    else:
        print("[POST-PROCESS] Miscellaneous reassignment disabled")

    # Convert back to dict format
    updated_clusters_final = {c['concern_key']: c['messages'] for c in clusters_list}

    # ========== FINAL CATEGORIZATION ==========
    final_categorized = categorize_clusters(updated_clusters_final, min_message_threshold)

    # Track retained labels (labels that made it to the final output)
    for label in updated_clusters_final.keys():
        if label not in cluster_history['label_tracking']:
            # This is a label that was never merged or deleted
            cluster_history['label_tracking'][label] = {
                'status': 'retained',
                'reason': 'passed_all_filters',
                'message_count': len(updated_clusters_final[label])
            }

    # Count label statuses for summary
    status_counts = {}
    for label_info in cluster_history['label_tracking'].values():
        status = label_info['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    cluster_history['summary'] = {
        'total_original_labels': original_cluster_count,
        'multi_tags_removed': len(removed_multi_tags),
        'labels_merged_round1': len([l for l in cluster_history['label_tracking'].values() if l.get('step') == 'round1' and l['status'] == 'merged']),
        'labels_merged_round2': len([l for l in cluster_history['label_tracking'].values() if l.get('step') == 'round2' and l['status'] == 'merged']),
        'labels_deleted': status_counts.get('deleted', 0),
        'labels_retained': status_counts.get('retained', 0),
        'final_cluster_count': len(updated_clusters_final),
        'status_breakdown': status_counts
    }

    # Add comprehensive metadata
    final_categorized['merge_metadata'] = {
        'round1_merge_map': round1_merge_map,
        'round1_validated_groups': round1_validated_groups,
        'round1_rejected_groups': round1_rejected_groups if 'round1_rejected_groups' in locals() else [],
        'round2_merge_map': round2_merge_map,
        'original_cluster_count': original_cluster_count,
        'after_multi_tag_removal': len(concern_clusters),
        'after_multi_tag_removal_count': len(concern_clusters),  # Add for backward compatibility
        'after_round1_count': len(merged_clusters),
        'after_round2_count': len(updated_clusters),
        'final_cluster_count': len(updated_clusters_final),
        'multi_tags_removed': len(removed_multi_tags),
        'low_frequency_deleted': len(deleted_labels),
        'low_frequency_merged_in_round2': len(round2_merge_map),
        'splitting_enabled': enable_splitting,
        'title_generation_enabled': enable_title_generation,
        'reassignment_enabled': enable_reassignment
    }
    final_categorized['removed_multi_tags'] = removed_multi_tags
    final_categorized['deleted_labels'] = deleted_labels
    final_categorized['cluster_history'] = cluster_history

    print("\n" + "="*80)
    print("[LABEL_MERGE] Concern label merging complete!")
    print("="*80 + "\n")

    # Print comprehensive summary
    summary = cluster_history['summary']
    print("[SUMMARY] Label Journey Report:")
    print(f"  Original labels: {summary['total_original_labels']}")
    print(f"  Multi-tags removed: {summary['multi_tags_removed']}")
    print(f"  Labels merged (Round 1): {summary['labels_merged_round1']}")
    print(f"  Labels merged (Round 2): {summary['labels_merged_round2']}")
    print(f"  Labels deleted: {summary['labels_deleted']}")
    print(f"  Labels retained: {summary['labels_retained']}")
    print(f"  Final clusters: {summary['final_cluster_count']}")
    print(f"\n[SUMMARY] Status Breakdown: {summary['status_breakdown']}")
    print(f"[SUMMARY] Total operations tracked: {len(cluster_history['operations'])}\n")

    return final_categorized


def add_cluster_metadata(
    clusters_sessions: List[Dict],
    chat_data: Dict[str, Dict]
) -> List[Dict]:
    """
    Add metadata to clusters using the session mapping methodology.

    Calculates:
    - Unique session count
    - Average human messages per session
    - A2C (Add-to-cart) sessions count and percentage
    - Order sessions count and percentage
    - UTM sessions count and percentage

    Args:
        clusters_sessions: List of cluster dictionaries with session IDs
        chat_data: Dictionary mapping session_id to session data

    Returns:
        List of cluster dictionaries with added metadata
    """
    print(f"[CONCERN_CLUSTER] Adding cluster metadata...")

    for cluster in clusters_sessions:
        cluster_id = cluster['cluster_id']
        session_ids = cluster.get('session_ids', [])

        if not session_ids:
            cluster['metadata'] = {
                'unique_session_count': 0,
                'avg_human_messages_per_session': 0,
                'a2c_sessions_count': 0,
                'a2c_sessions_percentage': 0,
                'order_sessions_count': 0,
                'order_sessions_percentage': 0,
                'utm_sessions_count': 0,
                'utm_sessions_percentage': 0
            }
            continue

        # Initialize counters
        total_human_messages = 0
        a2c_sessions = set()
        order_sessions = set()
        utm_sessions = set()

        # Process each session
        for session_id in session_ids:
            session_data = chat_data.get(session_id, {})

            if not session_data:
                print(f"[CONCERN_CLUSTER] Warning: Session {session_id} not found in chat data")
                continue

            messages = session_data.get('messages', [])
            user_fields = session_data.get('user_fields', [])

            # Count human messages (actor=customer)
            human_message_count = sum(1 for msg in messages if msg.get('actor') == 'customer')
            total_human_messages += human_message_count

            # Check for A2C sessions
            for msg in messages:
                if msg.get('actor') == 'customer':
                    text = msg.get('text', '').strip()
                    if text.lower().startswith('add to cart - '):
                        a2c_sessions.add(session_id)
                        break

            # Check for shopify_order_details and UTM attribution
            has_order = False
            has_utm = False

            for uf in user_fields:
                if uf.get('key') == 'shopify_order_details':
                    has_order = True
                    val = uf.get('value', {})

                    # Handle string value (parse JSON)
                    if isinstance(val, str):
                        try:
                            val = json.loads(val)
                        except json.JSONDecodeError:
                            val = {}

                    # Check for has_verifast_utm
                    if isinstance(val, dict) and val.get('has_verifast_utm') == True:
                        has_utm = True

                    break

            # Classify sessions
            if has_order:
                order_sessions.add(session_id)
            if has_utm:
                utm_sessions.add(session_id)

        # Calculate metrics
        unique_session_count = len(session_ids)
        avg_human_messages = total_human_messages / unique_session_count if unique_session_count > 0 else 0
        a2c_count = len(a2c_sessions)
        a2c_percentage = (a2c_count / unique_session_count * 100) if unique_session_count > 0 else 0
        order_count = len(order_sessions)
        order_percentage = (order_count / unique_session_count * 100) if unique_session_count > 0 else 0
        utm_count = len(utm_sessions)
        utm_percentage = (utm_count / unique_session_count * 100) if unique_session_count > 0 else 0

        # Add metadata
        cluster['metadata'] = {
            'unique_session_count': unique_session_count,
            'avg_human_messages_per_session': round(avg_human_messages, 2),
            'a2c_sessions_count': a2c_count,
            'a2c_sessions_percentage': round(a2c_percentage, 2),
            'order_sessions_count': order_count,
            'order_sessions_percentage': round(order_percentage, 2),
            'utm_sessions_count': utm_count,
            'utm_sessions_percentage': round(utm_percentage, 2)
        }

        print(f"[CONCERN_CLUSTER] Cluster {cluster_id} ({cluster['cluster_title'][:50]}...): "
              f"Sessions={unique_session_count}, "
              f"Avg_Human_Msgs={avg_human_messages:.2f}, "
              f"A2C={a2c_count}({a2c_percentage:.1f}%), "
              f"Orders={order_count}({order_percentage:.1f}%), "
              f"UTM={utm_count}({utm_percentage:.1f}%)")

    return clusters_sessions


def save_clusters_json(
    clusters_data: List[Dict],
    output_file_path: str,
    chat_data_file_path: str
) -> None:
    """
    Save clusters with messages to JSON file.

    Args:
        clusters_data: List of cluster dictionaries
        output_file_path: Path to save the output JSON file
        chat_data_file_path: Original chat data file path (for metadata)
    """
    output_data = {
        'metadata': {
            'source_file': os.path.basename(chat_data_file_path),
            'clustering_method': 'concern_based',
            'total_clusters': len(clusters_data),
            'total_messages': sum(c['message_count'] for c in clusters_data)
        },
        'clusters': clusters_data
    }

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[CONCERN_CLUSTER] Saved clusters to: {output_file_path}")


def save_sessions_json(
    sessions_data: List[Dict],
    output_file_path: str,
    chat_data_file_path: str
) -> None:
    """
    Save clusters with session mapping to JSON file.

    Args:
        sessions_data: List of cluster dictionaries with session IDs
        output_file_path: Path to save the output JSON file
        chat_data_file_path: Original chat data file path (for metadata)
    """
    # Calculate total unique sessions across all clusters
    all_session_ids = set()
    for cluster in sessions_data:
        all_session_ids.update(cluster.get('session_ids', []))

    output_data = {
        'metadata': {
            'source_file': os.path.basename(chat_data_file_path),
            'clustering_method': 'concern_based',
            'total_clusters': len(sessions_data),
            'total_sessions': len(all_session_ids),
            'total_messages': sum(c['message_count'] for c in sessions_data)
        },
        'clusters': sessions_data
    }

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[CONCERN_CLUSTER] Saved session-mapped clusters to: {output_file_path}")


def convert_categorized_to_clusters_format(categorized: Dict, merge_enabled: bool) -> List[Dict]:
    """
    Convert categorized clusters back to the standard clusters_data format.

    Args:
        categorized: Dictionary with 'regular' and optionally 'low_frequency' keys
        merge_enabled: Whether merging was enabled

    Returns:
        List of cluster dictionaries in standard format
    """
    clusters_data = []
    cluster_id = 0

    # Add regular clusters only (low-frequency should be deleted by now)
    for cluster in categorized.get('regular', []):
        cluster_dict = {
            'cluster_id': cluster_id,
            'cluster_title': cluster['concern_key'],
            'message_count': cluster['message_count'],
            'messages': cluster['messages']
        }

        # Include sales_stage if present
        if 'sales_stage' in cluster:
            cluster_dict['sales_stage'] = cluster['sales_stage']

        clusters_data.append(cluster_dict)
        cluster_id += 1

    return clusters_data


def generate_cluster_names_with_gpt(
    categorized: Dict,
    batch_size: int = 15,
    sample_size: int = 10
) -> Dict[str, str]:
    """
    Generate improved cluster names using GPT-4.1 for final clusters.

    Analyzes the concern_key and sample messages from each cluster to generate:
    - Short (1-3 words preferred, max 5 words)
    - Succinct and clear
    - Representative of the whole cluster
    - In English
    - Suitable as predefined options for future categorization

    Args:
        categorized: Dict with 'regular' key containing list of clusters
        batch_size: Maximum clusters per API call (default: 15)
        sample_size: Number of sample messages to show GPT per cluster (default: 10)

    Returns:
        Dictionary mapping old concern_key -> new generated name
    """
    regular_clusters = categorized.get('regular', [])

    if not regular_clusters:
        print("[NAME_GEN] No regular clusters to rename")
        return {}

    print(f"\n[NAME_GEN] Generating improved names for {len(regular_clusters)} clusters...")
    print(f"[NAME_GEN] Using batch_size={batch_size}, sample_size={sample_size}")

    name_mapping = {}

    # Process in batches
    for batch_start in range(0, len(regular_clusters), batch_size):
        batch = regular_clusters[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(regular_clusters) - 1) // batch_size + 1

        print(f"[NAME_GEN] Processing batch {batch_num}/{total_batches} ({len(batch)} clusters)")

        # Build cluster summaries for this batch
        cluster_summaries = []
        for idx, cluster in enumerate(batch):
            concern_key = cluster['concern_key']
            messages = cluster.get('messages', [])

            # Get sample messages (up to sample_size)
            sample_messages = messages[:sample_size]
            message_texts = [msg.get('text', '').strip() for msg in sample_messages if msg.get('text', '').strip()]

            # Limit message text length to avoid token overflow
            message_texts = [msg[:150] for msg in message_texts]

            summary = f'''Cluster {idx + 1}:
Current Name: "{concern_key}"
Sample Messages ({len(message_texts)} of {len(messages)} total):
{chr(10).join(f'  - "{msg}"' for msg in message_texts)}
'''
            cluster_summaries.append(summary)

        # Build GPT prompt
        all_summaries = "\n\n".join(cluster_summaries)

        prompt = f'''You are an expert at naming customer concern categories for an e-commerce chatbot system. Your task is to generate improved, standardized names for customer concern clusters.

REQUIREMENTS for each name:
1. **Length**: 1-3 words preferred, maximum 5 words
2. **Clarity**: Clear and unambiguous
3. **Specificity**: Representative of the cluster's main theme
4. **Language**: English only
5. **Format**: Title Case (e.g., "Delivery Status", "Product Quality")
6. **Consistency**: Use consistent terminology across related clusters
7. **Purpose**: These names will become predefined options for categorizing future customer concerns

AVOID:
- Generic names like "Other", "Miscellaneous", "Various"
- Overly long descriptive phrases
- Ambiguous or vague terms
- Technical jargon unless necessary

ANALYZE CAREFULLY:
- Look at the current name AND sample messages
- Identify the core concern theme
- If the current name is already good, you may keep it or slightly improve it
- If messages show a different theme than the current name, generate a better name

You will receive {len(batch)} clusters. For each cluster, provide ONLY the new name on a single line.

FORMAT YOUR RESPONSE EXACTLY AS:
Cluster 1: [New Name]
Cluster 2: [New Name]
...
Cluster {len(batch)}: [New Name]

Do NOT include explanations, quotes, or extra text. Just the names.

CLUSTERS TO RENAME:

{all_summaries}'''

        try:
            # Make GPT API call
            response = _gpt4_1_call(
                prompt=prompt,
                user_text="Generate improved cluster names following the requirements above.",
                temperature=0.2,  # Low temperature for consistency
                max_tokens=2000,
                timeout=180
            )

            if not response or not response.strip():
                print(f"[NAME_GEN] Batch {batch_num}: No response from GPT, keeping original names")
                continue

            # Parse response
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # Extract cluster names from formatted response
            for idx, cluster in enumerate(batch):
                concern_key = cluster['concern_key']

                # Look for "Cluster X: Name" pattern
                cluster_label = f"Cluster {idx + 1}:"
                matching_lines = [line for line in lines if line.startswith(cluster_label)]

                if matching_lines:
                    # Extract name after "Cluster X: "
                    new_name = matching_lines[0][len(cluster_label):].strip()

                    # Clean up the name
                    new_name = new_name.strip().strip('"').strip("'").strip()

                    # Validate length (should be <= 5 words)
                    word_count = len(new_name.split())
                    if word_count > 5:
                        print(f"[NAME_GEN]   Warning: Generated name too long ({word_count} words), keeping original: '{concern_key}'")
                        name_mapping[concern_key] = concern_key
                    elif len(new_name) < 3:
                        print(f"[NAME_GEN]   Warning: Generated name too short, keeping original: '{concern_key}'")
                        name_mapping[concern_key] = concern_key
                    else:
                        # Valid name
                        if new_name != concern_key:
                            print(f"[NAME_GEN]   ✓ '{concern_key}' → '{new_name}'")
                        else:
                            print(f"[NAME_GEN]   ✓ '{concern_key}' (unchanged)")
                        name_mapping[concern_key] = new_name
                else:
                    # Fallback: keep original
                    print(f"[NAME_GEN]   Warning: No match for cluster {idx + 1}, keeping original: '{concern_key}'")
                    name_mapping[concern_key] = concern_key

        except Exception as e:
            print(f"[NAME_GEN] Batch {batch_num}: Error calling GPT: {e}")
            print(f"[NAME_GEN] Keeping original names for this batch")
            # Fallback: keep original names
            for cluster in batch:
                concern_key = cluster['concern_key']
                name_mapping[concern_key] = concern_key

    # CRITICAL: Detect and resolve name collisions
    print(f"\n[NAME_GEN] Checking for name collisions...")

    # Build lookup dict for message counts
    concern_key_to_msg_count = {cluster['concern_key']: cluster['message_count'] for cluster in regular_clusters}

    # Build reverse mapping: new_name -> list of old_names
    new_name_to_old = defaultdict(list)
    for old_name, new_name in name_mapping.items():
        new_name_to_old[new_name].append(old_name)

    # Find collisions (new names that map from multiple old names)
    collisions = {new_name: old_names for new_name, old_names in new_name_to_old.items() if len(old_names) > 1}

    if collisions:
        print(f"[NAME_GEN] ⚠️  Found {len(collisions)} name collisions! Resolving...")

        for new_name, old_names in collisions.items():
            print(f"[NAME_GEN]   Collision: '{new_name}' ← {old_names}")

            # Strategy: Keep the cluster with most messages unchanged, append numbers to the rest
            # Sort by message count (descending) to keep the most important cluster unchanged
            old_names_sorted = sorted(old_names, key=lambda x: concern_key_to_msg_count.get(x, 0), reverse=True)

            # Keep first one (highest message count) unchanged
            print(f"[NAME_GEN]     → Keeping '{old_names_sorted[0]}' ({concern_key_to_msg_count.get(old_names_sorted[0], 0)} msgs) as '{new_name}'")

            # Append numbers to the rest
            for idx, old_name in enumerate(old_names_sorted[1:], start=2):
                disambiguated_name = f"{new_name} {idx}"
                name_mapping[old_name] = disambiguated_name
                msg_count = concern_key_to_msg_count.get(old_name, 0)
                print(f"[NAME_GEN]     → Renaming '{old_name}' ({msg_count} msgs) to '{disambiguated_name}' (disambiguation)")
    else:
        print(f"[NAME_GEN] ✓ No name collisions detected")

    # Print summary
    changed_count = sum(1 for old, new in name_mapping.items() if old != new)
    collision_count = sum(1 for new_name, old_names in new_name_to_old.items() if len(old_names) > 1)

    print(f"\n[NAME_GEN] ✓ Name generation complete!")
    print(f"[NAME_GEN]   - Total clusters: {len(name_mapping)}")
    print(f"[NAME_GEN]   - Names changed: {changed_count}")
    print(f"[NAME_GEN]   - Names unchanged: {len(name_mapping) - changed_count}")
    if collision_count > 0:
        print(f"[NAME_GEN]   - Collisions resolved: {collision_count}")

    return name_mapping


def run_concern_clustering(
    chat_data_file_path: str,
    output_dir: str = "vec_outs",
    filter_pages: Optional[List[str]] = None,
    filter_secondary_usecases: Optional[List[str]] = None,
    phrase: Optional[str] = None,
    keyword: Optional[str] = None,
    remove_bubbles: bool = False,
    merge_enabled: bool = True,
    min_message_threshold: int = 5,
    round2_excluded_targets: Optional[List[str]] = None,
    report: bool = False,
    client_name_override: Optional[str] = None,
    date_range_override: Optional[str] = None,
) -> None:
    """
    Main function to run concern-based clustering with optional label merging.

    Args:
        chat_data_file_path: Path to the chat data JSON file
        output_dir: Directory to save output files
        filter_pages: List of bot_page values to filter by
        filter_secondary_usecases: List of secondary usecases to filter by
        phrase: Filter for customer messages preceding AI messages with this phrase
        keyword: Filter for bot_page containing this keyword
        remove_bubbles: Whether to remove bubble-click messages (default: True)
        merge_enabled: Whether to enable concern label merging (default: True)
        min_message_threshold: Minimum messages for regular cluster (default: 5)
        round2_excluded_targets: List of cluster labels to exclude from Round 2 merging
        report: If True, use report format for chat data (array with 'chat' field), else use daily format (dict with 'messages' field)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract client_name and date_range — prefer explicit overrides over filename parsing
    base_name = os.path.basename(chat_data_file_path).replace('.json', '')
    if client_name_override and date_range_override:
        client_name = client_name_override
        date_range = date_range_override
        base_name = f"{client_name_override}_{date_range_override}"
    else:
        # Fallback: parse from filename (expected: "clientname_DDMM_DDMM")
        parts = base_name.rsplit('_', 2)
        if len(parts) == 3:
            client_name = parts[0]
            date_range = f"{parts[1]}_{parts[2]}"
        else:
            client_name = base_name
            date_range = "unknown"

    # Generate output file names using base_name
    clusters_file = os.path.join(output_dir, f"concern_clusters_{base_name}.json")
    sessions_file = os.path.join(output_dir, f"concern_clusters_{base_name}_sessions.json")
    categorized_file = os.path.join(output_dir, f"concern_clusters_{base_name}_categorized.json")

    # Step 1: Load chat data
    try:
        chat_data = load_chat_data(chat_data_file_path, report=report)
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error loading chat data: {e}")
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    if not chat_data:
        print(f"[CONCERN_CLUSTER] ⚠️  No chat data loaded, skipping concern clustering")
        return

    # Step 1.5: Early validation - check if any concerns exist
    print(f"[CONCERN_CLUSTER] Validating concern fields...")
    concern_field_count = 0
    concern_field_names = set()

    for session_id, session_data in chat_data.items():
        user_fields = session_data.get('user_fields', [])
        for field in user_fields:
            key = field.get('key', '').lower()
            # Check for various concern/requirement field patterns
            if any(pattern in key for pattern in ['concern', 'requirement', 'req/con', 'con/req']):
                concern_field_count += 1
                concern_field_names.add(key)

    print(f"[CONCERN_CLUSTER] Found {concern_field_count} concern/requirement fields across all sessions")
    if concern_field_names:
        print(f"[CONCERN_CLUSTER] Field names detected: {concern_field_names}")

    if concern_field_count == 0:
        print(f"[CONCERN_CLUSTER] ⚠️  No concern/requirement fields found in data")
        print(f"[CONCERN_CLUSTER] ℹ️  This client may not have concern tracking enabled")
        print(f"[CONCERN_CLUSTER] Gracefully skipping concern clustering for: {client_name}")
        return

    # Step 2: Extract concerns and create clusters with filtering
    try:
        clusters_data_raw = extract_concerns_and_cluster(
            chat_data,
            filter_pages=filter_pages,
            filter_secondary_usecases=filter_secondary_usecases,
            phrase=phrase,
            keyword=keyword,
            remove_bubbles=remove_bubbles
        )
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error extracting concerns: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    # Convert clusters_data to concern_clusters dict format for merging
    try:
        concern_clusters = {}
        for cluster in clusters_data_raw:
            concern_key = cluster['cluster_title']
            concern_clusters[concern_key] = cluster['messages']

        if not concern_clusters:
            print(f"[CONCERN_CLUSTER] ⚠️  No concern clusters created after extraction")
            print(f"[CONCERN_CLUSTER] ℹ️  This may indicate concerns exist but no valid messages")
            print(f"[CONCERN_CLUSTER] Gracefully skipping concern clustering for: {client_name}")
            return

    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error converting clusters: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    # Step 3: Merge similar concern labels with NEW HDBSCAN workflow!
    try:
        categorized = merge_concern_labels(
            concern_clusters,
            chat_data=chat_data,  # Pass chat_data for session-level operations
            output_dir=output_dir,
            min_message_threshold=min_message_threshold,
            merge_enabled=merge_enabled,
            round2_excluded_targets=round2_excluded_targets,
            enable_splitting=False,  # Enable cluster splitting by session volume
            enable_title_generation=False,  # Enable GPT title generation for split clusters
            enable_reassignment=False,  # Disable for now (needs more work)
            max_session_threshold=250,  # Split clusters with >250 sessions
            client_name=client_name,
            date_range=date_range
        )
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error merging concern labels: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    # Step 4: Convert categorized clusters back to standard format
    try:
        clusters_data = convert_categorized_to_clusters_format(categorized, merge_enabled)
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error converting to clusters format: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    # Step 4.5: Generate improved cluster names with GPT (NEW!)
    try:
        print("\n" + "="*70)
        print("STEP 4.5: GENERATING IMPROVED CLUSTER NAMES WITH GPT-4.1")
        print("="*70)

        name_mapping = generate_cluster_names_with_gpt(
            categorized,
            batch_size=15,  # Process 15 clusters per API call
            sample_size=10   # Show 10 sample messages per cluster
        )

        # Apply name mapping to categorized clusters
        if name_mapping:
            print(f"\n[NAME_GEN] Applying name changes to categorized clusters...")
            for cluster in categorized.get('regular', []):
                old_name = cluster['concern_key']
                if old_name in name_mapping:
                    new_name = name_mapping[old_name]
                    cluster['concern_key'] = new_name

            # Apply name mapping to clusters_data
            print(f"[NAME_GEN] Applying name changes to clusters data...")
            for cluster in clusters_data:
                old_title = cluster['cluster_title']
                if old_title in name_mapping:
                    new_title = name_mapping[old_title]
                    cluster['cluster_title'] = new_title

            # Apply name mapping to merge maps (CRITICAL FIX!)
            print(f"[NAME_GEN] Updating merge maps with new names...")
            merge_metadata = categorized.get('merge_metadata', {})

            # Update round1_merge_map: old_label -> canonical_label
            # Both keys and values may need updating
            if 'round1_merge_map' in merge_metadata:
                old_round1_map = merge_metadata['round1_merge_map'].copy()
                new_round1_map = {}

                for old_label, canonical_label in old_round1_map.items():
                    # Update the old_label key if it was renamed
                    new_old_label = name_mapping.get(old_label, old_label)
                    # Update the canonical_label value if it was renamed
                    new_canonical_label = name_mapping.get(canonical_label, canonical_label)
                    new_round1_map[new_old_label] = new_canonical_label

                merge_metadata['round1_merge_map'] = new_round1_map
                print(f"[NAME_GEN]   - Updated {len(new_round1_map)} Round 1 merge mappings")

            # Update round2_merge_map: low_freq_label -> target_label
            if 'round2_merge_map' in merge_metadata:
                old_round2_map = merge_metadata['round2_merge_map'].copy()
                new_round2_map = {}

                for lf_label, target_label in old_round2_map.items():
                    # Update the lf_label key if it was renamed
                    new_lf_label = name_mapping.get(lf_label, lf_label)
                    # Update the target_label value if it was renamed
                    new_target_label = name_mapping.get(target_label, target_label)
                    new_round2_map[new_lf_label] = new_target_label

                merge_metadata['round2_merge_map'] = new_round2_map
                print(f"[NAME_GEN]   - Updated {len(new_round2_map)} Round 2 merge mappings")

            # Write updated merge_metadata back to categorized (explicit assignment for safety)
            categorized['merge_metadata'] = merge_metadata

            # Update deleted_labels list if any were renamed
            if 'deleted_labels' in categorized:
                old_deleted = categorized['deleted_labels'].copy()
                new_deleted = [name_mapping.get(label, label) for label in old_deleted]
                categorized['deleted_labels'] = new_deleted

            print(f"[NAME_GEN] ✓ Name mapping applied to all data structures (including merge maps)")
        else:
            print(f"[NAME_GEN] No name mapping generated, keeping original names")

    except Exception as e:
        print(f"[CONCERN_CLUSTER] ⚠️  Warning: Error generating cluster names: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Continuing with original names...")
        # Continue processing with original names

    # Step 5: Save categorized output (NEW!)
    try:
        categorized_output = {
            'metadata': {
                'source_file': os.path.basename(chat_data_file_path),
                'clustering_method': 'concern_based_with_label_merging' if merge_enabled else 'concern_based',
                'merge_enabled': merge_enabled,
                'min_message_threshold': min_message_threshold,
                'total_clusters': len(clusters_data),
                'regular_clusters': len(categorized.get('regular', [])),
                'multi_tags_removed': len(categorized.get('removed_multi_tags', [])),
                'labels_deleted': len(categorized.get('deleted_labels', [])),
                'total_messages': sum(c['message_count'] for c in clusters_data)
            },
            'categorized_clusters': categorized
        }

        with open(categorized_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays and other non-serializable objects
            def json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj

            # Make a deep copy and clean it
            clean_output = json.loads(json.dumps(categorized_output, default=json_serializable))
            json.dump(clean_output, f, indent=2, ensure_ascii=False)

        print(f"[CONCERN_CLUSTER] Saved categorized clusters to: {categorized_file}")
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ⚠️  Warning: Failed to save categorized output: {e}")
        # Continue processing even if this fails

    # Step 6: Save clusters with messages (standard format)
    try:
        save_clusters_json(clusters_data, clusters_file, chat_data_file_path)
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error saving clusters JSON: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    # Step 7: Create session mapping
    try:
        sessions_data = create_session_mapping(clusters_data)
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error creating session mapping: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    # Step 8: Add cluster metadata
    try:
        sessions_data_with_metadata = add_cluster_metadata(sessions_data, chat_data)
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error adding cluster metadata: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    # Step 9: Save session-mapped clusters with metadata
    try:
        save_sessions_json(sessions_data_with_metadata, sessions_file, chat_data_file_path)
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ❌ Error saving sessions JSON: {e}")
        import traceback
        traceback.print_exc()
        print(f"[CONCERN_CLUSTER] Skipping concern clustering for this client")
        return

    # Step 10: Export to Excel (NEW!)
    excel_file = os.path.join(output_dir, f"concern_clusters_{base_name}_export.xlsx")
    try:
        export_to_excel(sessions_file, categorized_file, excel_file)
    except Exception as e:
        print(f"[CONCERN_CLUSTER] ⚠️  Warning: Excel export failed: {e}")
        # Don't fail the entire process if Excel export fails
        import traceback
        traceback.print_exc()

    print(f"\n[CONCERN_CLUSTER] ✓ Clustering complete!")
    print(f"[CONCERN_CLUSTER] Output files:")
    print(f"  - Categorized: {categorized_file}")
    print(f"  - Messages: {clusters_file}")
    print(f"  - Sessions: {sessions_file}")
    print(f"  - Excel Export: {excel_file}")

    # Print summary statistics
    if merge_enabled and 'merge_metadata' in categorized:
        merge_meta = categorized['merge_metadata']
        print(f"\n[SUMMARY] Merge Statistics:")
        print(f"  - Original clusters: {merge_meta['original_cluster_count']}")
        print(f"  - After Round 1 merges: {merge_meta['after_round1_count']}")
        print(f"  - After multi-tag removal: {merge_meta['after_multi_tag_removal_count']}")
        print(f"  - Final clusters: {merge_meta['final_cluster_count']}")
        print(f"  - Multi-tag concerns removed: {merge_meta['multi_tags_removed']}")
        print(f"  - Low-frequency merged (Round 2): {merge_meta.get('low_frequency_merged_in_round2', 0)}")
        print(f"  - Low-frequency deleted (Round 2): {merge_meta['low_frequency_deleted']}")
        print(f"  - Round 1 merge groups: {len(merge_meta['round1_validated_groups'])}")
        print(f"  - Round 2 merge mappings: {len(merge_meta.get('round2_merge_map', {}))}")

    print(f"\n[SUMMARY] Final Cluster Breakdown:")
    print(f"  - Regular clusters: {len(categorized.get('regular', []))}")

    # Show deleted and removed items summary
    if 'deleted_labels' in categorized and categorized['deleted_labels']:
        print(f"  - Deleted labels: {len(categorized['deleted_labels'])}")
    if 'removed_multi_tags' in categorized and categorized['removed_multi_tags']:
        total_multi_tag_messages = sum(r['message_count'] for r in categorized['removed_multi_tags'])
        print(f"  - Removed multi-tags: {len(categorized['removed_multi_tags'])} ({total_multi_tag_messages} messages)")
    
    print("\n === GENERATING CONCERN REPORT... ===\n")
    gen_concern_report(sessions_file)


def export_to_excel(
    sessions_file_path: str,
    categorized_file_path: str,
    output_excel_path: str
) -> None:
    """
    Export cluster metadata and merge history to Excel with two sheets.

    Sheet 1: Cluster Metadata (from sessions file)
    Sheet 2: Merge History (from categorized file)

    Args:
        sessions_file_path: Path to the clusters sessions JSON file
        categorized_file_path: Path to the categorized JSON file
        output_excel_path: Path for output Excel file
    """
    print(f"\n[EXCEL] Exporting to Excel: {output_excel_path}")

    # ========== SHEET 1: CLUSTER METADATA ==========
    print("[EXCEL] Loading cluster metadata from sessions file...")

    with open(sessions_file_path, 'r', encoding='utf-8') as f:
        sessions_data = json.load(f)

    file_metadata = sessions_data.get('metadata', {})
    file_name = os.path.basename(sessions_file_path)

    metadata_rows = []
    for cluster in sessions_data.get('clusters', []):
        cluster_id = cluster.get('cluster_id')
        cluster_title = cluster.get('cluster_title', 'N/A')
        cluster_meta = cluster.get('metadata', {})

        num_messages = cluster.get('message_count', 0)

        row = {
            'Source File': file_name,
            'Cluster ID': cluster_id,
            'Cluster Name': cluster_title,
            'Number of Messages': num_messages,
            'Unique Sessions': cluster_meta.get('unique_session_count', 0),
            'Avg Human Messages per Session': cluster_meta.get('avg_human_messages_per_session', 0),
            'A2C Sessions Count': cluster_meta.get('a2c_sessions_count', 0),
            'A2C Sessions %': cluster_meta.get('a2c_sessions_percentage', 0),
            'Order Sessions Count': cluster_meta.get('order_sessions_count', 0),
            'Order Sessions %': cluster_meta.get('order_sessions_percentage', 0),
            'UTM Sessions Count': cluster_meta.get('utm_sessions_count', 0),
            'UTM Sessions %': cluster_meta.get('utm_sessions_percentage', 0),
        }

        metadata_rows.append(row)

    df_metadata = pd.DataFrame(metadata_rows)
    df_metadata = df_metadata.sort_values(['Cluster ID'])

    print(f"[EXCEL] Extracted {len(df_metadata)} clusters for metadata sheet")

    # ========== SHEET 2: MERGE HISTORY ==========
    print("[EXCEL] Loading merge history from categorized file...")

    with open(categorized_file_path, 'r', encoding='utf-8') as f:
        categorized_data = json.load(f)

    merge_history_rows = []
    categorized_clusters = categorized_data.get('categorized_clusters', {})
    merge_metadata = categorized_clusters.get('merge_metadata', {})
    round1_merge_map = merge_metadata.get('round1_merge_map', {})
    round2_merge_map = merge_metadata.get('round2_merge_map', {})

    # Build reverse mapping for Round 1: canonical_label -> list of merged labels
    round1_canonical_to_merged = defaultdict(list)
    for old_label, canonical_label in round1_merge_map.items():
        round1_canonical_to_merged[canonical_label].append(old_label)

    print(f"[EXCEL] Round 1 merge map: {len(round1_merge_map)} entries")
    print(f"[EXCEL] Round 1 canonical clusters: {len(round1_canonical_to_merged)}")

    # Build reverse mapping for Round 2: target_label -> list of low-freq labels
    round2_target_to_merged = defaultdict(list)
    for lf_label, target_label in round2_merge_map.items():
        round2_target_to_merged[target_label].append(lf_label)

    print(f"[EXCEL] Round 2 merge map: {len(round2_merge_map)} entries")
    print(f"[EXCEL] Round 2 target clusters: {len(round2_target_to_merged)}")

    # Add all final clusters
    for cluster in categorized_clusters.get('regular', []):
        cluster_name = cluster['concern_key']

        # Check Round 1 merges
        round1_merged_from = round1_canonical_to_merged.get(cluster_name, [])

        # Check Round 2 merges
        round2_merged_from = round2_target_to_merged.get(cluster_name, [])

        # Debug logging for first 3 clusters
        if len(merge_history_rows) < 3:
            print(f"[EXCEL] DEBUG: Cluster '{cluster_name}'")
            print(f"[EXCEL]   Round 1 merges: {len(round1_merged_from)}")
            print(f"[EXCEL]   Round 2 merges: {len(round2_merged_from)}")

        # Determine merge type
        if round1_merged_from and round2_merged_from:
            merge_type = 'Round 1 + Round 2 Merge'
            all_merged = [cluster_name] + round1_merged_from + round2_merged_from
            num_merged = len(all_merged)
        elif round1_merged_from:
            merge_type = 'Round 1 Merge'
            all_merged = [cluster_name] + round1_merged_from
            num_merged = len(all_merged)
        elif round2_merged_from:
            merge_type = 'Round 2 Merge'
            all_merged = [cluster_name] + round2_merged_from
            num_merged = len(all_merged)
        else:
            merge_type = 'No Merge'
            all_merged = [cluster_name]
            num_merged = 1

        row = {
            'Final Cluster Name': cluster_name,
            'Number of Labels Merged': num_merged,
            'Original Labels': ', '.join(all_merged),
            'Merge Type': merge_type
        }

        merge_history_rows.append(row)

    # Add deleted labels
    deleted_labels = categorized_clusters.get('deleted_labels', [])
    for deleted_label in deleted_labels:
        row = {
            'Final Cluster Name': '[DELETED]',
            'Number of Labels Merged': 0,
            'Original Labels': deleted_label,
            'Merge Type': 'Round 2 Deleted'
        }
        merge_history_rows.append(row)

    # Add removed multi-tags
    removed_multi_tags = categorized_clusters.get('removed_multi_tags', [])
    for multi_tag in removed_multi_tags:
        row = {
            'Final Cluster Name': '[REMOVED]',
            'Number of Labels Merged': 0,
            'Original Labels': multi_tag['concern_key'],
            'Merge Type': 'Multi-tag Removed'
        }
        merge_history_rows.append(row)

    df_merge_history = pd.DataFrame(merge_history_rows)
    df_merge_history = df_merge_history.sort_values(['Merge Type', 'Number of Labels Merged'], ascending=[True, False])

    print(f"[EXCEL] Extracted {len(df_merge_history)} rows for merge history sheet")

    # ========== EXPORT TO EXCEL ==========
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        df_metadata.to_excel(writer, sheet_name='Cluster Metadata', index=False)
        df_merge_history.to_excel(writer, sheet_name='Merge History', index=False)

    print(f"[EXCEL] ✓ Exported to: {output_excel_path}")
    print(f"[EXCEL]   - Sheet 1 'Cluster Metadata': {len(df_metadata)} rows")
    print(f"[EXCEL]   - Sheet 2 'Merge History': {len(df_merge_history)} rows")


if __name__=="__main__":
    # ========== INPUT CONFIGURATION ==========
    # Set your parameters here

    chat_data_file = "./chat_data/cellbells_0101_2901.json"

    # =========================================

    # Check if input file exists
    if not os.path.exists(chat_data_file):
        print(f"[ERROR] File not found: {chat_data_file}")
        exit(1)

    # Run clustering
    run_concern_clustering(
        chat_data_file_path=chat_data_file,
        output_dir="vec_outs",
        filter_pages=None,
        filter_secondary_usecases=None,
        phrase=None,
        keyword=None,
        remove_bubbles=False,
        merge_enabled=True,
        min_message_threshold=5,
        round2_excluded_targets=["other concern"],
        report=False
    )
    
