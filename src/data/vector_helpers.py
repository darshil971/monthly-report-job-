"""
Helper functions extracted from admin-backend/vector.py.
Only the functions needed by the theme clustering pipeline and og_report integration.
"""

import json
import pandas as pd
from collections import Counter
from typing import List, Optional


def load_messages_from_report(
    json_path: str,
    remove_bubbles: bool = True,
    filter_pages: Optional[List[str]] = None,
    filter_secondary_usecases: Optional[List[str]] = None,
    phrase=None,
    keyword: Optional[str] = None,
    max_messages: int = 3000,
) -> pd.DataFrame:
    """
    Load messages from report JSON format (array of objects with 'chat' arrays).
    Returns a DataFrame with columns: session_id, text, timestamp, user_intent, bot_page.
    """
    print("[STEP 1] Loading messages from report format...")

    if phrase is not None:
        phrase_list = [phrase] if isinstance(phrase, str) else phrase
        if len(phrase_list) > 1:
            print(f"[INFO] Filtering with {len(phrase_list)} phrases: {phrase_list[:3]}{'...' if len(phrase_list) > 3 else ''}")
        else:
            print(f"[INFO] Filtering with phrase: '{phrase_list[0]}'")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = []

    for session_obj in data:
        for session_id, session_data in session_obj.items():
            chat_messages = session_data.get("chat", [])

            bot_page = None
            metadata_for_session = session_data.get("metadata_for_session", [])
            if metadata_for_session and len(metadata_for_session) > 0:
                bot_page = metadata_for_session[0].get("bot_page")

            if phrase is not None:
                phrase_list = [phrase] if isinstance(phrase, str) else phrase
                for i, msg in enumerate(chat_messages):
                    ai_text = msg.get("text", "").lower()
                    if (msg.get("actor") != "customer" and
                            any(p.lower() in ai_text for p in phrase_list)):
                        for j in range(i - 1, -1, -1):
                            prev_msg = chat_messages[j]
                            if prev_msg.get("actor") == "customer":
                                customer_msg_idx = sum(
                                    1 for m in chat_messages[:j + 1]
                                    if m.get("actor") == "customer"
                                ) - 1
                                session_usecases = session_data.get("use_cases", []) or []
                                if customer_msg_idx < len(session_usecases):
                                    matched_usecase = session_usecases[customer_msg_idx].get(
                                        "secondary_usecase", "unknown"
                                    )
                                else:
                                    matched_usecase = "unknown"
                                messages.append({
                                    "session_id": session_id,
                                    "text": prev_msg.get("text", "").strip(),
                                    "timestamp": prev_msg.get("created_at", ""),
                                    "user_intent": matched_usecase,
                                    "bot_page": bot_page,
                                })
                                break
            else:
                customer_messages = [m for m in chat_messages if m.get("actor") == "customer"]
                session_usecases = session_data.get("use_cases", []) or []
                for idx, msg in enumerate(customer_messages):
                    if idx < len(session_usecases):
                        matched_usecase = session_usecases[idx].get(
                            "secondary_usecase", "unknown"
                        )
                    else:
                        matched_usecase = "unknown"
                    messages.append({
                        "session_id": session_id,
                        "text": msg.get("text", "").strip(),
                        "timestamp": msg.get("created_at", ""),
                        "user_intent": matched_usecase,
                        "bot_page": bot_page,
                    })

    df = pd.DataFrame(messages)

    # Filter by bot_page
    if filter_pages is not None and not df.empty:
        before = len(df)
        df = df[df["bot_page"].isin(filter_pages)].reset_index(drop=True)
        print(f"[INFO] Filtered bot_page: kept {len(df)}/{before} messages")

    # Filter by keyword in bot_page
    if keyword is not None and not df.empty:
        before = len(df)
        df = df[df["bot_page"].str.contains(keyword, case=False, na=False)].reset_index(drop=True)
        print(f"[INFO] Filtered by keyword '{keyword}': kept {len(df)}/{before} messages")

    # Filter by secondary_usecase
    if filter_secondary_usecases is not None and len(filter_secondary_usecases) > 0 and not df.empty:
        before = len(df)
        df = df[df["user_intent"].isin(filter_secondary_usecases)].reset_index(drop=True)
        print(f"[INFO] Filtered by secondary_usecase: kept {len(df)}/{before} messages")

    # Remove bubble-click messages
    if remove_bubbles and not df.empty:
        message_counts = Counter(df["text"].tolist())
        bubble_clicks = {
            msg: count for msg, count in message_counts.items()
            if count >= 5 and len(msg.split()) >= 3
        }
        before = len(df)
        df = df[~df["text"].isin(bubble_clicks.keys())].reset_index(drop=True)
        print(f"[INFO] Removed {before - len(df)} bubble-click messages")

        bubble_starters = ["add to cart", "show more variants"]
        before = len(df)
        df = df[~df["text"].str.lower().str.strip().str.startswith(tuple(bubble_starters))].reset_index(drop=True)
        print(f"[INFO] Removed {before - len(df)} predefined bubble-start messages")

    # Sample if above threshold
    if len(df) > max_messages:
        print(f"[SAMPLING] Dataset has {len(df)} messages, sampling {max_messages}...")
        df = df.sample(n=max_messages, random_state=42).reset_index(drop=True)

    return df


def get_top_frequent_pages(
    json_path: str,
    report: bool = False,
    top_k: int = 1,
    page_types: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Find the top K most frequent pages from bot_page data.
    Returns list of top page URLs or None if no matches.
    """
    if page_types is None:
        page_types = ["/products/", "/blogs/", "/collections/", "/pages/", "/cart/", "/search/"]

    print(f"[AUTO-PAGE] Finding top {top_k} most frequent pages...")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    page_counts = {}

    if report:
        for session_obj in data:
            for session_id, session_data in session_obj.items():
                metadata_for_session = session_data.get("metadata_for_session", [])
                if metadata_for_session and len(metadata_for_session) > 0:
                    bot_page = metadata_for_session[0].get("bot_page")
                    if bot_page and any(pt in bot_page for pt in page_types):
                        page_counts[bot_page] = page_counts.get(bot_page, 0) + 1
    else:
        if isinstance(data, dict):
            for session_id, session_data in data.items():
                metadata = session_data.get("session_meta", [])
                if metadata:
                    bot_page = metadata[0].get("bot_page") if metadata else None
                    if bot_page and any(pt in bot_page for pt in page_types):
                        page_counts[bot_page] = page_counts.get(bot_page, 0) + 1
        elif isinstance(data, list):
            for session_obj in data:
                for session_id, session_data in session_obj.items():
                    metadata_for_session = session_data.get("metadata_for_session", [])
                    if metadata_for_session and len(metadata_for_session) > 0:
                        bot_page = metadata_for_session[0].get("bot_page")
                        if bot_page and any(pt in bot_page for pt in page_types):
                            page_counts[bot_page] = page_counts.get(bot_page, 0) + 1

    if not page_counts:
        print("[AUTO-PAGE] No matching pages found in the data")
        return None

    sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)
    top_pages = sorted_pages[:top_k]
    total_sessions = sum(page_counts.values())

    print(f"[AUTO-PAGE] Found {len(page_counts)} unique pages")
    for i, (page, count) in enumerate(top_pages, 1):
        pct = (count / total_sessions) * 100
        page_name = page.split("//")[-1] if "//" in page else page
        print(f"[AUTO-PAGE]   {i}. {page_name} - {count} sessions ({pct:.1f}%)")

    return [page for page, _ in top_pages]
