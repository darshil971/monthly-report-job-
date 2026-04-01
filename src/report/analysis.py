"""
GPT-based analysis functions for the monthly report.
Extracted from admin-backend/report.py — get_top_queries, analyze_chat_patterns,
geo_cohort, utm_cohort.
"""

import os
import random
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.openai_utils import GPT4Input, gpt4_1_azure_call


def get_top_queries(message_list: List[str]) -> str:
    """
    Sends a stratified sample of messages to GPT-4.1 and returns
    a comma-separated string of the top 3 customer patterns.
    """
    total_messages = len(message_list)
    max_sample_size = 250

    if total_messages > max_sample_size:
        random.seed(42)
        sampled_messages = random.sample(message_list, max_sample_size)
        sample_size = max_sample_size
    else:
        sampled_messages = message_list
        sample_size = total_messages

    min_threshold = max(10, int(sample_size * 0.1))
    prompt = f"""
Generate a concise, data-driven summary of the three most prevalent customer patterns or concerns
for product, content, and marketing teams, based on chat messages from an e-commerce site cohort.

==== ANALYSIS SCOPE ====
- Total Messages in Subset: {total_messages:,}
- Sample Size Analyzed: {sample_size:,} ({(sample_size/total_messages*100):.1f}%)
- Minimum Frequency Threshold: {min_threshold} messages (10% significance)
- Objective: Surface top unmet needs, content gaps, and messaging opportunities

==== CUSTOMER MESSAGES ====
{chr(10).join(f"- {m}" for m in sampled_messages)}

==== OUTPUT REQUIREMENTS ====
Only return a comma-separated list of three succinct pattern descriptors:
"[pattern1], [pattern2], [pattern3]"

Criteria for patterns:
- Must each appear in at least {min_threshold} messages
- Highly relevant to improving product features, marketing content, or chatbot prompts
- Use professional business language, no filler text or numbering
"""
    system_msg = """
You are a strategic VoC analyst for e-commerce. Identify the top three customer patterns
that are both statistically significant and actionable for product, content, and marketing teams.
Respond only with the comma-separated list of patterns."""

    gpt4_inputs = [
        GPT4Input(actor="system", text=system_msg),
        GPT4Input(actor="user", text=prompt)
    ]
    return gpt4_1_azure_call(gpt4_inputs, temperature=0.2, max_tokens=4000, timeout=120)


def analyze_chat_patterns(
    clean_df: pd.DataFrame,
    output_file: str,
    sample_frac: float = 0.1,
    max_per_usecase: int = 200
) -> str:
    """
    Generate pattern-analysis table by secondary use-case.
    Only includes usecases with >= 5% of total messages.
    """
    # Stratified sampling
    stratified_df = (
        clean_df.groupby('secondary_usecase', group_keys=False)
        .apply(lambda x: x.sample(frac=sample_frac, random_state=42)
               if len(x) > max_per_usecase else x)
        .reset_index(drop=True)
    )
    stratified_df = (
        stratified_df.groupby('secondary_usecase', group_keys=False)
        .apply(lambda x: x.head(max_per_usecase))
        .reset_index(drop=True)
    )

    total_msgs_full = len(clean_df)
    usecase_counts_full = clean_df['secondary_usecase'].fillna('Unspecified').value_counts()
    threshold_count_full = total_msgs_full * 0.05
    filtered_usecases = {
        uc: cnt for uc, cnt in usecase_counts_full.items()
        if cnt > threshold_count_full
    }

    table_rows = []
    for usecase, count in filtered_usecases.items():
        msgs_sampled = stratified_df.loc[
            stratified_df['secondary_usecase'].fillna('Unspecified') == usecase,
            'message_text'
        ].astype(str).tolist()

        top_patterns = get_top_queries(msgs_sampled)
        pct_full = round(count / total_msgs_full * 100, 1)
        table_rows.append((usecase, count, pct_full, top_patterns))

    table_rows.sort(key=lambda x: x[1], reverse=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Customer Chat Analysis\n\n")
        f.write("## Pattern Analysis\n\n")
        f.write(f"_(Only secondary use-cases over 5% of total messages: {threshold_count_full:.0f}+ messages)_\n\n")
        f.write("| User Intent | Message Count | Percentage | Top patterns |\n")
        f.write("|----------|-------|------------|--------------|\n")
        for usecase, count, pct, patterns in table_rows:
            f.write(f"| {usecase} | {count} | {pct}% | {patterns} |\n")
        f.write("\n")

    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def geo_cohort(
    df: pd.DataFrame,
    df_clean: pd.DataFrame,
    sample_frac: float = 0.1,
    max_msgs: int = 200,
    md_path: str = None
) -> Optional[str]:
    """Analyze top 5 locations and generate markdown report."""
    if df["location"].dropna().empty:
        return None

    location_tagged_sessions = df.dropna(subset=["location"])["session_id"].unique()
    total_location_msgs = len(df[df["session_id"].isin(location_tagged_sessions)])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"| Location | % of Messages | Top patterns |\n")
        f.write("|----------|---------------|-----------------|\n")

    location_msg_counts = df.dropna(subset=["location"]).groupby("location").size()
    top_locations = location_msg_counts.nlargest(5)

    for loc, count in top_locations.items():
        percentage = round((count / total_location_msgs) * 100, 1)
        sessions = df_clean[df_clean["location"] == loc]["session_id"].unique()
        loc_msgs = df_clean[df_clean["session_id"].isin(sessions)]["message_text"].astype(str)
        if len(loc_msgs) > max_msgs:
            loc_msgs = loc_msgs.sample(frac=sample_frac, random_state=42).head(max_msgs)
        patterns = get_top_queries(loc_msgs.tolist())
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(f"| {loc} | {percentage}% | {patterns} |\n")

    return md_path


def utm_cohort(
    df: pd.DataFrame,
    df_clean: pd.DataFrame,
    sample_frac: float = 0.1,
    max_msgs: int = 200,
    md_path: str = None
) -> Tuple[Optional[str], List[str]]:
    """Analyze top 3 UTM sources and generate markdown report.
    Returns (md_path, list_of_top_3_utm_sources)."""
    df = df.copy()
    df_clean = df_clean.copy()

    def make_norm(s):
        if pd.isna(s):
            return None
        s2 = str(s).replace(u'\xa0', ' ').strip().lower()
        if s2 in ['', 'none', 'nan', 'null']:
            return None
        return s2

    df['utm_norm'] = df['utm_source'].apply(make_norm)
    df_clean['utm_norm'] = df_clean['utm_source'].apply(make_norm)

    if df['utm_norm'].dropna().empty:
        print(f"[utm_cohort] No UTM data found")
        return None, []

    total_utm_msgs = len(df.dropna(subset=['utm_norm']))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"| UTM Source | % of Messages | Top patterns |\n")
        f.write("|------------|---------------|-----------------|\n")

    counts = df.dropna(subset=['utm_norm']).groupby('utm_norm').size()
    counts = counts.sort_values(ascending=False)
    top_utms = counts.head(3)
    top_utm_sources = top_utms.index.tolist()

    for src, count in top_utms.items():
        percentage = round((count / total_utm_msgs) * 100, 1)
        sessions = df_clean.loc[df_clean['utm_norm'] == src, 'session_id'].unique()
        src_msgs = df_clean.loc[
            df_clean['session_id'].isin(sessions), 'message_text'
        ].astype(str)
        if len(src_msgs) > max_msgs:
            src_msgs = src_msgs.sample(frac=sample_frac, random_state=42).head(max_msgs)
        patterns = get_top_queries(src_msgs.tolist())
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(f"| {src} | {percentage}% | {patterns} |\n")

    return md_path, top_utm_sources


def has_insufficient_data_themes(themes_list: List[str]) -> bool:
    """Check if theme list contains patterns indicating insufficient data."""
    if not themes_list:
        return True

    insufficient_patterns = [
        'no statistically significant patterns',
        'insufficient data for theme extraction',
        'insufficient message volume',
        'not enough data',
        'no patterns detected'
    ]

    for theme in themes_list:
        theme_lower = theme.lower().strip()
        for pattern in insufficient_patterns:
            if pattern in theme_lower:
                return True
    return False
