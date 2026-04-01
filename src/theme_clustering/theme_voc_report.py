"""
Theme Clustering 2.0 — Voice of Customer Report Generator

Client-facing HTML report.  Ported from generate_html_report.py with a thin
adapter layer that maps pipeline 2.0 schema fields to the report format:

  cluster.description       → cluster_summary
  cluster.key_phrases       → keywords  (labelled "Key Phrases" in UI)
  cluster.category          → categories.sales_stage
  cluster.messages[].text   → plain message strings
  global_metadata.median_order_percentage → median_order_percentage
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------

def _categorize_cluster(cluster: Dict[str, Any]) -> str:
    """Return 'pre-sales' or 'post-sales' for a pipeline 2.0 cluster dict."""
    cat = cluster.get('category', '').lower()
    if cat == 'pre-sales':
        return 'pre-sales'
    if cat == 'post-sales':
        return 'post-sales'
    # Fallback: guess from title
    title = cluster.get('cluster_title', '').lower()
    desc  = cluster.get('description', '').lower()
    if any(w in title + desc for w in ['order', 'delivery', 'track', 'cancel', 'return', 'shipped', 'received']):
        return 'post-sales'
    return 'pre-sales'


def _is_greeting_or_random_cluster(cluster: Dict[str, Any]) -> bool:
    """
    Detect greeting/random clusters that should be suppressed from the client report.

    Two-stage check (mirrors generate_html_report.py):
    1. Regex patterns against title + description.
    2. Message-ratio check: if the title is greeting-like AND >60% of the
       cluster's messages are very short or known greetings, suppress it.
    """
    title = cluster.get('cluster_title', '').lower()
    # pipeline 2.0 uses 'description' instead of 'cluster_summary'
    desc  = cluster.get('description', '').lower()

    greeting_patterns = [
        r'\bgreetings?\b', r'\breplies?\b', r'\backnowledg\w+\b',
        r'\brandom\b', r'\bmiscellaneous\b', r'\boutliers?\b',
        r'\bgeneral\b.*\breplies?\b', r'\bgeneral\b.*\bgreetings?\b',
        r'\bcasual\b.*\bmessages?\b', r'\bbrief\b.*\bmessages?\b',
    ]
    content = title + ' ' + desc
    for pattern in greeting_patterns:
        if re.search(pattern, content):
            return True

    if any(w in title for w in ['greeting', 'acknowledgment', 'general', 'casual', 'brief']):
        raw_messages = cluster.get('messages', [])
        if raw_messages:
            # Extended list matching vector.py exactly (includes 'shut up', 'hindi')
            short_greetings = [
                'hi', 'hey', 'hello', 'ok', 'yes', 'no', 'thanks', 'bye', 'yeah',
                'shut up', 'hindi',
            ]
            greeting_count = 0
            for m in raw_messages:
                text = (m.get('text', '') if isinstance(m, dict) else str(m)).lower().strip()
                if len(text) <= 3 or any(g in text for g in short_greetings):
                    greeting_count += 1
            if greeting_count / len(raw_messages) > 0.6:
                return True
    return False


def _extract_plain_messages(cluster: Dict[str, Any]):
    """Return a list of plain message strings from a pipeline 2.0 cluster dict."""
    raw = cluster.get('messages', [])
    result = []
    for m in raw:
        if isinstance(m, dict):
            result.append(m.get('text', ''))
        else:
            result.append(str(m))
    return result


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_voc_report(
    clusters_export: Dict[str, Any],
    output_path: str,
    client_display: str = "Customer",
    page_url: Optional[str] = None,
    report_month: Optional[str] = None,
) -> str:
    """
    Generate the client-facing VoC HTML report from pipeline 2.0 output.

    Args:
        clusters_export: Dict from PipelineOutput.to_export_dict()
        output_path:     Path to write the HTML file
        client_display:  Client name to show in the report header
        page_url:        Optional page URL shown as clickable link in header
        report_month:    Optional month string to show in header (e.g., "February 2026").
                         Defaults to current month if not provided.

    Returns:
        output_path
    """
    print(f"[VOC-REPORT] Generating client VoC report...")

    # Use provided report_month or default to current month
    _report_month = report_month or datetime.now().strftime('%B %Y')

    global_metadata  = clusters_export.get('global_metadata', {})
    quality_report   = clusters_export.get('quality_report', {})
    raw_clusters     = clusters_export.get('clusters', [])

    total_messages          = quality_report.get('total_messages', 0) or clusters_export.get('total_messages', 0)
    median_order_percentage = global_metadata.get('median_order_percentage', 0)

    # ------------------------------------------------------------------
    # Build processed cluster list (adapter layer)
    # ------------------------------------------------------------------
    processed_clusters = []
    for cluster in raw_clusters:
        if cluster.get('cluster_id') == -1:          # skip miscellaneous
            continue
        if _is_greeting_or_random_cluster(cluster):
            continue

        primary_category = _categorize_cluster(cluster)

        session_metadata = cluster.get('session_metadata', {})
        performance      = cluster.get('performance', {})

        msg_count   = cluster.get('message_count', 0)
        percentage  = (msg_count / total_messages * 100) if total_messages > 0 else 0

        order_pct          = session_metadata.get('order_sessions_percentage', 0)
        absolute_diff      = order_pct - median_order_percentage

        processed_clusters.append({
            'id':               cluster.get('cluster_id', 0),
            'title':            cluster.get('cluster_title', 'Untitled').replace('"', ''),
            'category':         primary_category,
            'allCategories':    [primary_category],
            'categoryDisplay':  {'pre-sales': 'Pre-Sales', 'post-sales': 'Post-Sales'}.get(primary_category, 'General'),
            'percentage':       f"{percentage:.2f}",
            'messageCount':     msg_count,
            # 'description' in pipeline 2.0 == 'cluster_summary' in vector.py
            'summary':          cluster.get('description', 'No summary available').replace('"', ''),
            # 'key_phrases' in pipeline 2.0 == 'keywords' in vector.py (shown as "Key Phrases")
            'keywords':         cluster.get('key_phrases', []),
            'keyPhraseCounts':  cluster.get('key_phrase_counts', {}),
            'messages':         _extract_plain_messages(cluster),
            'avgSessionLength': session_metadata.get('avg_human_messages_per_session', 0),
            'metrics':          {'order': order_pct},
            'showOrder':        performance.get('show_order', False),
            'orderPerformance': performance.get('order_performance', 'average'),
            'sessionPercentage': f"{performance.get('session_volume_percentage', 0):.1f}",
            'orderSessionsCount': session_metadata.get('order_sessions_count', 0),
            'absoluteDiffVsMedian': absolute_diff,
            'sessionCount':     session_metadata.get('unique_session_count', 0),
            'isSignificant':    performance.get('is_significant', False),
        })

    # Guard: all clusters were filtered (greeting/misc/etc.) — skip writing empty report
    if not processed_clusters:
        print(f"[VOC-REPORT] No displayable clusters after filtering — skipping report generation.")
        return output_path

    # Sort by message count descending
    processed_clusters.sort(key=lambda x: x['messageCount'], reverse=True)

    total_sessions_display = sum(c['sessionCount'] for c in processed_clusters)
    total_patterns = len(processed_clusters)
    # Strip myshopify.com from display name
    client_name_display = client_display.replace('.myshopify.com', '')

    # For the header link, show just the product handle portion (truncate long collection paths)
    filtered_pages = page_url or ''
    page_display_text = ''
    if filtered_pages:
        # Extract the last meaningful path segment (product handle) for display
        _clean = filtered_pages.split('?')[0].split('#')[0].rstrip('/')
        for _marker in ['/products/', '/blogs/', '/collections/', '/pages/']:
            if _marker in _clean:
                page_display_text = _clean.split(_marker)[-1]
                break
        if not page_display_text:
            page_display_text = _clean.split('/')[-1] or filtered_pages

    # ------------------------------------------------------------------
    # HTML template (verbatim port of generate_html_report.py)
    # ------------------------------------------------------------------
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice of Customer Analysis - {client_name_display}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}

        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        body {{
            font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #D2D9E2;
            background-attachment: fixed;
            color: #3C3C3E;
            line-height: 1.5;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 24px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }}

        .header:hover {{
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
            transform: translateY(-1px);
        }}

        .header-content {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .header-left {{
            flex: 1;
            min-width: 300px;
        }}

        .header-title {{
            font-size: 28px;
            font-weight: 700;
            color: #3C3C3E;
            margin: 0 0 4px 0;
            line-height: 1.2;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}

        .header-subtitle {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            margin: 0;
        }}

        .header-byline {{
            font-size: 16px;
            font-weight: 400;
            color: #3C3C3E;
            font-style: italic;
            margin: 0;
        }}

        .header-date {{
            font-size: 14px;
            font-weight: 600;
            color: #3C3C3E;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 6px 12px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.3);
            margin: 0;
            transition: all 0.3s ease;
        }}

        .header-date:hover {{
            background: rgba(255,255,255,0.25);
            transform: scale(1.02);
        }}

        .header-meta {{
            display: grid;
            grid-template-columns: repeat(2, 140px);
            gap: 12px;
        }}

        .meta-item {{
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 12px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            width: 140px;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .meta-item:hover {{
            background: rgba(255, 255, 255, 0.25);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }}

        .meta-label {{
            font-size: 10px;
            color: #3C3C3E;
            font-weight: 600;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .meta-value {{
            font-size: 20px;
            font-weight: 800;
            color: #3C3C3E;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .meta-sublabel {{
            font-size: 9px;
            color: #3C3C3E;
            font-weight: 500;
            margin-top: 2px;
            opacity: 0.7;
        }}

        .header-meta-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 8px;
        }}

        .page-link {{
            font-size: 14px;
            font-weight: 600;
            color: #3C3C3E;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 6px 12px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.3);
            text-decoration: none;
            transition: all 0.3s ease;
        }}

        .page-link:hover {{
            background: rgba(255,255,255,0.25);
            transform: scale(1.02);
        }}

        .filter-buttons {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}

        .filter-separator {{
            width: 2px;
            height: 24px;
            background: rgba(255, 255, 255, 0.3);
            margin: 0 4px;
        }}

        .filter-btn {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 12px 20px;
            border-radius: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            color: #3C3C3E;
        }}

        .filter-btn:hover {{
            background: rgba(255, 255, 255, 0.25);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }}

        .filter-btn.active {{
            background: rgba(255, 255, 255, 0.5);
            border-color: rgba(255, 255, 255, 0.8);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            transform: translateY(-1px);
            font-weight: 700;
        }}

        .filter-btn[data-filter="pre-sales"].active {{
            background: rgba(143, 173, 225, 0.6);
            border-color: #8FADE1;
            box-shadow: 0 6px 20px rgba(143, 173, 225, 0.4);
        }}

        .filter-btn[data-filter="post-sales"].active {{
            background: rgba(248, 132, 77, 0.6);
            border-color: #F8844D;
            box-shadow: 0 6px 20px rgba(248, 132, 77, 0.4);
        }}

        .clusters-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }}

        .cluster-card {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            min-height: 200px;
        }}

        .cluster-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: rgba(255, 255, 255, 0.3);
        }}

        .cluster-card[data-category="pre-sales"]::before {{
            background: linear-gradient(90deg, #8FADE1 0%, #7A9DD1 100%);
        }}

        .cluster-card[data-category="post-sales"]::before {{
            background: linear-gradient(90deg, #F8844D 0%, #E8743D 100%);
        }}

        .cluster-card:hover {{
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            transform: translateY(-1px);
            background: rgba(255, 255, 255, 0.2);
        }}

        .cluster-header {{
            margin-bottom: 12px;
        }}

        .cluster-title {{
            font-size: 18px;
            font-weight: 700;
            color: #3C3C3E;
            margin-bottom: 16px;
            line-height: 1.3;
            padding-right: 80px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .cluster-percentage {{
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 20px;
            font-weight: 800;
            transition: all 0.3s ease;
        }}

        .cluster-card:hover .cluster-percentage {{
            font-size: 24px;
        }}

        .cluster-card[data-category="pre-sales"] .cluster-percentage {{
            color: #8FADE1;
        }}

        .cluster-card[data-category="post-sales"] .cluster-percentage {{
            color: #F8844D;
        }}

        .cluster-summary {{
            font-size: 14px;
            color: #3C3C3E;
            line-height: 1.5;
            margin-bottom: 16px;
            padding-bottom: 60px;
            overflow: visible;
        }}

        .cluster-footer {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .metric-badge-simple {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
            padding: 6px 12px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 700;
            border: 1px solid rgba(255, 255, 255, 0.4);
            transition: all 0.3s ease;
        }}

        .metric-badge-simple.good {{
            background: rgba(159, 198, 59, 0.4);
            border-color: rgba(159, 198, 59, 0.6);
        }}

        .metric-badge-simple.poor {{
            background: rgba(240, 18, 99, 0.4);
            border-color: rgba(240, 18, 99, 0.6);
        }}

        .click-btn {{
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }}

        .cluster-card[data-category="pre-sales"] .click-btn {{
            background: rgba(143, 173, 225, 0.5);
            color: #3C3C3E;
            border: 1px solid rgba(143, 173, 225, 0.7);
        }}

        .cluster-card[data-category="pre-sales"] .click-btn:hover {{
            background: rgba(143, 173, 225, 0.7);
            transform: scale(1.02);
            box-shadow: 0 6px 24px rgba(143, 173, 225, 0.4);
        }}

        .cluster-card[data-category="post-sales"] .click-btn {{
            background: rgba(248, 132, 77, 0.5);
            color: #3C3C3E;
            border: 1px solid rgba(248, 132, 77, 0.7);
        }}

        .cluster-card[data-category="post-sales"] .click-btn:hover {{
            background: rgba(248, 132, 77, 0.7);
            transform: scale(1.02);
            box-shadow: 0 6px 24px rgba(248, 132, 77, 0.4);
        }}

        /* Deep Dive Page Styles */
        .deep-dive-page {{
            display: none;
        }}

        .nav-bar {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            padding: 16px 20px;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }}

        .nav-bar:hover {{
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }}

        .back-btn {{
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            color: #3C3C3E;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 10px 20px;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }}

        .back-btn:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }}

        .breadcrumb {{
            font-size: 14px;
            color: #3C3C3E;
            font-weight: 500;
        }}

        .detail-header {{
            font-size: 32px;
            font-weight: 700;
            color: #3C3C3E;
            margin: 20px 0;
            text-align: center;
            padding: 24px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }}

        .detail-header:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }}

        .detail-content {{
            display: grid;
            grid-template-columns: 35% 1fr;
            gap: 20px;
            align-items: start;
        }}

        .key-info-panel {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            overflow: hidden;
        }}

        .key-info-panel:hover {{
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }}

        .key-info-panel::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #8FADE1 0%, #7A9DD1 100%);
            border-radius: 20px 20px 0 0;
        }}

        .deep-dive-page.post-sales .key-info-panel::before {{
            background: linear-gradient(90deg, #F8844D 0%, #E8743D 100%);
        }}

        .key-info-title {{
            font-size: 16px;
            font-weight: 700;
            color: #3C3C3E;
            margin: 0 0 16px 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .info-section {{
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .info-section:last-child {{
            border-bottom: none;
            margin-bottom: 0;
        }}

        .info-label {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #3C3C3E;
            opacity: 0.7;
            margin-bottom: 6px;
        }}

        .info-value {{
            font-size: 14px;
            color: #3C3C3E;
            font-weight: 500;
            line-height: 1.5;
        }}

        .info-value.large {{
            font-size: 18px;
            font-weight: 700;
        }}

        .category-badge {{
            display: inline-block;
            background: rgba(143, 173, 225, 0.3);
            color: #3C3C3E;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 14px;
        }}

        .deep-dive-page.post-sales .category-badge {{
            background: rgba(248, 132, 77, 0.3);
        }}

        .keyword-tags-detail {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}

        .keyword-tag-detail {{
            background: rgba(143, 173, 225, 0.3);
            color: #3C3C3E;
            padding: 6px 12px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid rgba(143, 173, 225, 0.5);
            transition: all 0.3s ease;
        }}

        .keyword-tag-detail:hover {{
            background: rgba(143, 173, 225, 0.4);
            transform: scale(1.02);
        }}

        .category-share-badge {{
            font-size: 12px;
            font-weight: 700;
            opacity: 0.85;
            margin-left: 4px;
        }}

        .deep-dive-page.post-sales .keyword-tag-detail {{
            background: rgba(248, 132, 77, 0.3);
            border-color: rgba(248, 132, 77, 0.5);
        }}

        .deep-dive-page.post-sales .keyword-tag-detail:hover {{
            background: rgba(248, 132, 77, 0.4);
        }}

        .metrics-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            transition: all 0.3s ease;
        }}

        .metric-row:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}

        .metric-name {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            color: #3C3C3E;
            opacity: 0.8;
        }}

        .metric-main-value {{
            font-size: 18px;
            font-weight: 800;
            color: #3C3C3E;
        }}

        .metric-comparison {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
            font-weight: 700;
        }}

        .metric-comparison.positive {{
            color: #9FC63B;
        }}

        .metric-comparison.negative {{
            color: #F01263;
        }}

        .metric-baseline {{
            font-size: 10px;
            color: #3C3C3E;
            opacity: 0.6;
            margin-top: 2px;
            text-align: right;
        }}

        .messages-section {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            overflow: hidden;
            transition: all 0.3s ease;
            position: relative;
            display: flex;
            flex-direction: column;
        }}

        .messages-section:hover {{
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }}

        .messages-section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #8FADE1 0%, #7A9DD1 100%);
            border-radius: 20px 20px 0 0;
        }}

        .deep-dive-page.post-sales .messages-section::before {{
            background: linear-gradient(90deg, #F8844D 0%, #E8743D 100%);
        }}

        .messages-header {{
            padding: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            flex-shrink: 0;
        }}

        .messages-title {{
            font-size: 18px;
            font-weight: 700;
            color: #3C3C3E;
            margin-bottom: 0;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .messages-list {{
            flex: 1;
            overflow-y: auto;
            padding: 0;
            min-height: 0;
        }}

        .message-item {{
            padding: 16px 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 16px;
            color: #3C3C3E;
            line-height: 1.6;
            font-weight: 500;
            transition: all 0.2s ease;
        }}

        .message-item:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}

        .message-item:last-child {{
            border-bottom: none;
        }}

        .message-item:nth-child(even) {{
            background: rgba(255, 255, 255, 0.05);
        }}

        .message-item:nth-child(even):hover {{
            background: rgba(255, 255, 255, 0.15);
        }}

        .hidden {{
            display: none;
        }}

        /* Conversion Performance View Styles */
        .conversion-performance-view {{
            margin-bottom: 24px;
        }}

        .info-card-container {{
            margin-bottom: 20px;
        }}

        .info-card {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 16px 20px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }}

        .info-card:hover {{
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
        }}

        .info-icon {{
            font-size: 20px;
            flex-shrink: 0;
            line-height: 1;
        }}

        .info-text {{
            font-size: 14px;
            color: #3C3C3E;
            line-height: 1.6;
            font-weight: 500;
        }}

        .performance-table-container {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            overflow: hidden;
            transition: all 0.3s ease;
        }}

        .performance-table-container:hover {{
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }}

        .performance-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}

        .performance-table thead {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        .performance-table th {{
            padding: 16px 20px;
            text-align: center;
            font-weight: 700;
            color: #3C3C3E;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 12px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            cursor: pointer;
            user-select: none;
            transition: all 0.2s ease;
            position: relative;
        }}

        .performance-table th:hover {{
            background: rgba(255, 255, 255, 0.15);
        }}

        .performance-table th.sortable::after {{
            content: '⇅';
            position: absolute;
            right: 8px;
            opacity: 0.3;
            font-size: 14px;
        }}

        .performance-table th.sort-asc::after {{
            content: '▲';
            opacity: 1;
            color: #8FADE1;
        }}

        .performance-table th.sort-desc::after {{
            content: '▼';
            opacity: 1;
            color: #8FADE1;
        }}

        .performance-table th:first-child {{ width: 40%; }}
        .performance-table th:nth-child(2) {{ width: 20%; }}
        .performance-table th:nth-child(3) {{ width: 20%; }}
        .performance-table th:nth-child(4) {{ width: 20%; }}

        .performance-table tbody tr {{
            transition: all 0.2s ease;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .performance-table tbody tr:nth-child(even) {{
            background: rgba(255, 255, 255, 0.05);
        }}

        .performance-table tbody tr:hover {{
            background: rgba(255, 255, 255, 0.15);
        }}

        .performance-table td {{
            padding: 14px 20px;
            color: #3C3C3E;
            font-weight: 500;
            text-align: center;
        }}

        .performance-table td:first-child {{ text-align: left; }}

        .cluster-title-link {{
            color: #8FADE1;
            text-decoration: underline;
            text-underline-offset: 2px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: block;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.4;
            font-weight: 600;
        }}

        .cluster-title-link:hover {{
            color: #7A9DD1;
        }}

        .metric-value {{
            font-weight: 700;
            color: #3C3C3E;
        }}

        .metric-diff {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            font-weight: 700;
        }}

        .metric-diff.good {{ color: #9FC63B; }}
        .metric-diff.poor {{ color: #F01263; }}

        /* Mobile Responsive */
        @media (max-width: 1024px) {{
            .detail-content {{ grid-template-columns: 1fr; }}
            .key-info-panel {{ margin-bottom: 20px; }}
        }}

        @media (max-width: 768px) {{
            body {{ padding: 16px; }}
            .header-content {{ flex-direction: column; text-align: center; }}
            .header-left {{ min-width: auto; }}
            .header-subtitle {{ justify-content: center; }}
            .header-meta {{ justify-content: center; grid-template-columns: repeat(2, 140px); }}
            .cluster-title {{ padding-right: 70px; }}
            .cluster-percentage {{ font-size: 16px; padding: 6px 8px; }}
            .filter-buttons {{ flex-direction: column; gap: 8px; }}
            .filter-separator {{ width: 100%; height: 2px; margin: 0; }}
            .filter-btn {{ text-align: center; }}
            .clusters-grid {{ grid-template-columns: 1fr; }}
            .nav-bar {{ flex-direction: column; align-items: flex-start; gap: 8px; }}
            .breadcrumb {{ margin-left: 0; }}
            .detail-header {{ margin-left: 0; font-size: 20px; }}
        }}

        /* Scrollbar styling */
        .messages-list::-webkit-scrollbar {{ width: 8px; }}
        .messages-list::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.1); border-radius: 10px; }}
        .messages-list::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.3); border-radius: 10px; }}
        .messages-list::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.5); }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Overview Page -->
        <div id="overview-page">
            <div class="header">
                <div class="header-content">
                    <div class="header-left">
                        <h1 class="header-title">VoC Analysis</h1>
                        <div class="header-subtitle">
                            <div class="header-byline">Discover what your customers are really saying</div>
                            <div class="header-meta-row">
                                <div class="header-date">{ _report_month }</div>
                                { f'<a href="{filtered_pages}" target="_blank" class="page-link">🔗 {page_display_text}</a>' if filtered_pages else '' }
                            </div>
                        </div>
                    </div>
                    <div class="header-meta">
                        <div class="meta-item">
                            <div class="meta-label">Messages</div>
                            <div class="meta-value" id="header-messages-count">{ total_messages if total_messages < 4000 else "4000+" }</div>
                            <div class="meta-sublabel" id="header-sessions-count">{ total_sessions_display } users</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">Conversion %</div>
                            <div class="meta-value" id="header-conversion-value">{ f"{median_order_percentage:.1f}%" }</div>
                            <div class="meta-sublabel">MEDIAN</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="filter-buttons">
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="pre-sales">Pre-Sales</button>
                <button class="filter-btn" data-filter="post-sales">Post-Sales</button>
                <div class="filter-separator"></div>
                <button class="filter-btn" id="conversion-performance-btn" onclick="toggleConversionPerformanceView()">Conversion Performance</button>
            </div>

            <!-- Conversion Performance View (inline) -->
            <div id="conversion-performance-view" class="conversion-performance-view" style="display: none;">
                <div class="info-card-container">
                    <div class="info-card">
                        <div class="info-icon">ℹ️</div>
                        <div class="info-text">
                            Shows significant pre-sales clusters by order conversion.
                        </div>
                    </div>
                </div>

                <div class="performance-table-container">
                    <table class="performance-table" id="performance-table">
                        <thead>
                            <tr>
                                <th class="sortable" data-sort="title">Cluster Title</th>
                                <th class="sortable" data-sort="sessionPercentage">% of Total Sessions</th>
                                <th class="sortable" data-sort="conversion">Conversion %</th>
                                <th class="sortable sort-desc" data-sort="comparison">Comparison to Overall</th>
                            </tr>
                        </thead>
                        <tbody id="performance-table-body">
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="clusters-grid" id="clusters-container">
            </div>
        </div>

        <!-- Deep Dive Page -->
        <div id="deep-dive-page" class="deep-dive-page">
            <div class="nav-bar">
                <button class="back-btn" onclick="goBackToOverview()">← Back</button>
                <div class="breadcrumb">Overview > <span id="current-cluster-name">Cluster Name</span></div>
            </div>

            <div class="detail-header" id="detail-header-title">
                Cluster Title Deep Dive
            </div>

            <div class="detail-content">
                <div class="key-info-panel">
                    <h3 class="key-info-title">KEY INFO</h3>

                    <div class="info-section">
                        <div class="info-label">Cluster Summary</div>
                        <div class="info-value" id="detail-summary">...</div>
                    </div>

                    <div class="info-section keywords">
                        <div class="info-label">Key Phrases</div>
                        <div class="keyword-tags-detail" id="detail-keywords">
                        </div>
                    </div>

                    <div class="info-section">
                        <div class="info-label">Category</div>
                        <div class="info-value">
                            <span class="category-badge" id="detail-category">Category</span>
                        </div>
                    </div>

                    <div class="info-section">
                        <div class="info-label">% of Total Messages</div>
                        <div class="info-value large" id="detail-percentage">0%</div>
                    </div>

                    <div class="info-section">
                        <div class="info-label">Avg Messages per Session</div>
                        <div class="info-value large" id="detail-avg-session">0 msgs</div>
                    </div>

                    <div class="info-section" id="performance-metrics-section">
                        <div class="info-label">Performance Metrics</div>
                        <div class="metrics-list">
                            <div class="metric-row" id="metric-order-row">
                                <div>
                                    <div class="metric-name">conversion %</div>
                                    <div class="metric-main-value" id="metric-order-value">0%</div>
                                </div>
                                <div>
                                    <div class="metric-comparison" id="metric-order-diff">▲ +0%</div>
                                    <div class="metric-baseline" id="metric-order-baseline">vs 0%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="messages-section">
                    <div class="messages-header">
                        <div class="messages-title">Messages</div>
                    </div>
                    <div class="messages-list" id="messages-list">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Cluster data from Python
        const clusterData = {json.dumps(processed_clusters, indent=8)};

        // Global medians
        const globalMedians = {{
            order: {median_order_percentage}
        }};

        // Category mapping
        const categoryMapping = {{
            'pre-sales':  ['pre-sales'],
            'post-sales': ['post-sales'],
            'all':        ['pre-sales', 'post-sales']
        }};

        let currentCategoryFilter = 'all';

        function initializePage() {{
            renderClusters();
            setupFilterButtons();
        }}

        function truncateSummary(text, element) {{
            const lineHeight = 1.5;
            const fontSize = 14;
            const cardPadding = 24;
            const bottomSpace = 60;

            const card = element.closest('.cluster-card');
            if (!card) return text;

            const cardHeight = card.offsetHeight;
            const titleHeight = card.querySelector('.cluster-title').offsetHeight;
            const headerMargin = 28;

            const availableHeight = cardHeight - titleHeight - headerMargin - bottomSpace - (cardPadding * 2);
            const lineHeightPx = fontSize * lineHeight;
            const maxLines = Math.floor(availableHeight / lineHeightPx);

            const temp = document.createElement('div');
            temp.style.cssText = `
                position: absolute;
                visibility: hidden;
                width: ${{element.offsetWidth}}px;
                font-size: ${{fontSize}}px;
                line-height: ${{lineHeight}};
                font-family: 'Poppins', sans-serif;
            `;
            temp.textContent = text;
            document.body.appendChild(temp);

            const textHeight = temp.offsetHeight;
            const maxHeight = lineHeightPx * maxLines;

            if (textHeight <= maxHeight) {{
                document.body.removeChild(temp);
                return text;
            }}

            let truncated = text;
            temp.textContent = truncated + '...';
            while (temp.offsetHeight > maxHeight && truncated.length > 0) {{
                truncated = truncated.slice(0, -1);
                temp.textContent = truncated.trim() + '...';
            }}
            document.body.removeChild(temp);
            return truncated.trim() + '...';
        }}

        function renderClusters() {{
            const container = document.getElementById('clusters-container');

            let filteredClusters = clusterData.filter(cluster => {{
                const targetCategories = categoryMapping[currentCategoryFilter];
                return cluster.allCategories.some(cat => targetCategories.includes(cat));
            }});

            container.innerHTML = filteredClusters.map(cluster => {{
                return `
                    <div class="cluster-card" data-category="${{cluster.category}}">
                        <div class="cluster-header">
                            <div class="cluster-title">${{cluster.title}}</div>
                            <div class="cluster-percentage">${{cluster.percentage}}%</div>
                        </div>
                        <div class="cluster-summary" data-full-text="${{cluster.summary.replace(/"/g, '&quot;')}}">${{cluster.summary}}</div>
                        <div class="cluster-footer">
                            <button class="click-btn" onclick="openDeepDive(${{cluster.id}})">MORE</button>
                        </div>
                    </div>
                `;
            }}).join('');

            setTimeout(() => {{
                document.querySelectorAll('.cluster-summary').forEach(summaryEl => {{
                    const fullText = summaryEl.getAttribute('data-full-text');
                    const truncated = truncateSummary(fullText, summaryEl);
                    summaryEl.textContent = truncated;
                }});
            }}, 0);

            updateHeaderStats();
        }}

        function updateHeaderStats() {{
            const targetCategories = categoryMapping[currentCategoryFilter];
            const visibleClusters = clusterData.filter(cluster =>
                cluster.allCategories.some(cat => targetCategories.includes(cat))
            );

            const totalMsgs = visibleClusters.reduce((sum, c) => sum + c.messageCount, 0);
            const totalSessions = visibleClusters.reduce((sum, c) => sum + c.sessionCount, 0);

            const msgsEl = document.getElementById('header-messages-count');
            const sessionsEl = document.getElementById('header-sessions-count');
            const conversionEl = document.getElementById('header-conversion-value');

            if (msgsEl) msgsEl.textContent = totalMsgs >= 4000 ? '4000+' : totalMsgs.toLocaleString();
            if (sessionsEl) sessionsEl.textContent = totalSessions.toLocaleString() + ' users';

            if (conversionEl) {{
                if (currentCategoryFilter === 'post-sales') {{
                    conversionEl.textContent = '-';
                }} else if (currentCategoryFilter === 'all') {{
                    conversionEl.textContent = globalMedians.order.toFixed(1) + '%';
                }} else {{
                    const orderVals = visibleClusters.map(c => c.metrics.order).sort((a, b) => a - b);
                    const n = orderVals.length;
                    const median = n > 0
                        ? (n % 2 === 0
                            ? (orderVals[n / 2 - 1] + orderVals[n / 2]) / 2
                            : orderVals[Math.floor(n / 2)])
                        : 0;
                    conversionEl.textContent = median.toFixed(1) + '%';
                }}
            }}
        }}

        function setupFilterButtons() {{
            const _totalMsgsAll = clusterData.reduce((sum, c) => sum + c.messageCount, 0);
            const _categoryShares = {{
                'pre-sales':  _totalMsgsAll > 0
                    ? Math.round(clusterData.filter(c => c.category === 'pre-sales').reduce((sum, c) => sum + c.messageCount, 0) / _totalMsgsAll * 100)
                    : 0,
                'post-sales': _totalMsgsAll > 0
                    ? Math.round(clusterData.filter(c => c.category === 'post-sales').reduce((sum, c) => sum + c.messageCount, 0) / _totalMsgsAll * 100)
                    : 0,
            }};
            const buttons = document.querySelectorAll('.filter-btn[data-filter]');
            buttons.forEach(button => {{
                button.addEventListener('click', () => {{
                    const perfView = document.getElementById('conversion-performance-view');
                    const clustersGrid = document.getElementById('clusters-container');
                    if (perfView.style.display !== 'none') {{
                        perfView.style.display = 'none';
                        clustersGrid.style.display = 'grid';
                    }}

                    const conversionBtn = document.getElementById('conversion-performance-btn');
                    if (conversionBtn) conversionBtn.classList.remove('active');

                    buttons.forEach(btn => {{
                        btn.classList.remove('active');
                        const existing = btn.querySelector('.category-share-badge');
                        if (existing) existing.remove();
                    }});
                    button.classList.add('active');
                    currentCategoryFilter = button.dataset.filter;
                    if (_categoryShares[currentCategoryFilter] !== undefined) {{
                        const badge = document.createElement('span');
                        badge.className = 'category-share-badge';
                        badge.textContent = ' ' + _categoryShares[currentCategoryFilter] + '%';
                        button.appendChild(badge);
                    }}
                    renderClusters();
                }});
            }});
        }}

        let currentSortColumn = 'sessionPercentage';
        let currentSortDirection = 'desc';

        function sortPerformanceClusters(clusters, column, direction) {{
            const sorted = [...clusters];
            sorted.sort((a, b) => {{
                let aValue, bValue;
                switch(column) {{
                    case 'title':
                        aValue = a.title.toLowerCase();
                        bValue = b.title.toLowerCase();
                        return direction === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
                    case 'sessionPercentage':
                        aValue = parseFloat(a.sessionPercentage);
                        bValue = parseFloat(b.sessionPercentage);
                        break;
                    case 'conversion':
                        aValue = a.metrics.order;
                        bValue = b.metrics.order;
                        break;
                    case 'comparison':
                        aValue = a.absoluteDiffVsMedian;
                        bValue = b.absoluteDiffVsMedian;
                        break;
                    default:
                        return 0;
                }}
                return direction === 'asc' ? aValue - bValue : bValue - aValue;
            }});
            return sorted;
        }}

        function renderPerformanceTable(clusters) {{
            const tableBody = document.getElementById('performance-table-body');
            const sortedClusters = sortPerformanceClusters(clusters, currentSortColumn, currentSortDirection);

            tableBody.innerHTML = sortedClusters.map(cluster => {{
                const performanceClass = cluster.absoluteDiffVsMedian >= 0 ? 'good' : 'poor';
                const conversionValue = cluster.metrics.order;
                const absoluteDiff = cluster.absoluteDiffVsMedian;
                const diffIndicator = absoluteDiff >= 0 ? '▲' : '▼';
                const diffSign = absoluteDiff >= 0 ? '+' : '';

                return `
                    <tr>
                        <td>
                            <a href="javascript:void(0)" class="cluster-title-link" onclick="openDeepDive(${{cluster.id}})">
                                ${{cluster.title}}
                            </a>
                        </td>
                        <td>${{cluster.sessionPercentage}}%</td>
                        <td><span class="metric-value">${{conversionValue.toFixed(1)}}%</span></td>
                        <td>
                            <div class="metric-diff ${{performanceClass}}">
                                <span>${{diffIndicator}}</span>
                                <span>${{diffSign}}${{absoluteDiff.toFixed(1)}}%</span>
                            </div>
                        </td>
                    </tr>
                `;
            }}).join('');
        }}

        function setupTableSorting() {{
            const headers = document.querySelectorAll('.performance-table th.sortable');
            headers.forEach(header => {{
                header.addEventListener('click', () => {{
                    const sortKey = header.dataset.sort;
                    if (currentSortColumn === sortKey) {{
                        currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
                    }} else {{
                        currentSortColumn = sortKey;
                        currentSortDirection = 'desc';
                    }}
                    headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                    header.classList.add(`sort-${{currentSortDirection}}`);
                    const performanceClusters = clusterData.filter(cluster => cluster.category === 'pre-sales' && cluster.isSignificant);
                    renderPerformanceTable(performanceClusters);
                }});
            }});
        }}

        function toggleConversionPerformanceView() {{
            const perfView = document.getElementById('conversion-performance-view');
            const clustersGrid = document.getElementById('clusters-container');
            const conversionBtn = document.getElementById('conversion-performance-btn');

            if (perfView.style.display === 'none') {{
                perfView.style.display = 'block';
                clustersGrid.style.display = 'none';

                // Deactivate category filter buttons; activate CP button
                document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {{
                    btn.classList.remove('active');
                    const badge = btn.querySelector('.category-share-badge');
                    if (badge) badge.remove();
                }});
                if (conversionBtn) conversionBtn.classList.add('active');

                const performanceClusters = clusterData.filter(cluster => cluster.category === 'pre-sales' && cluster.isSignificant);
                currentSortColumn = 'comparison';
                currentSortDirection = 'desc';
                renderPerformanceTable(performanceClusters);

                if (!perfView.dataset.sortingSetup) {{
                    setupTableSorting();
                    perfView.dataset.sortingSetup = 'true';
                }}

                const headers = document.querySelectorAll('.performance-table th.sortable');
                headers.forEach(h => {{
                    h.classList.remove('sort-asc', 'sort-desc');
                    if (h.dataset.sort === 'comparison') h.classList.add('sort-desc');
                }});
            }} else {{
                perfView.style.display = 'none';
                clustersGrid.style.display = 'grid';
                if (conversionBtn) conversionBtn.classList.remove('active');
                const allButton = document.querySelector('.filter-btn[data-filter="all"]');
                if (allButton) {{
                    allButton.classList.add('active');
                    currentCategoryFilter = 'all';
                    renderClusters();
                }}
            }}
        }}

        function openDeepDive(clusterId) {{
            const cluster = clusterData.find(c => c.id === clusterId);
            if (!cluster) return;

            document.getElementById('overview-page').style.display = 'none';
            const deepDivePage = document.getElementById('deep-dive-page');
            deepDivePage.style.display = 'block';
            deepDivePage.className = 'deep-dive-page ' + cluster.category;

            document.getElementById('current-cluster-name').textContent = cluster.title;
            document.getElementById('detail-header-title').textContent = cluster.title + ' Deep Dive';
            document.getElementById('detail-summary').textContent = cluster.summary;
            document.getElementById('detail-category').textContent = cluster.categoryDisplay;
            document.getElementById('detail-percentage').textContent = cluster.percentage + '%';
            document.getElementById('detail-avg-session').textContent =
                cluster.avgSessionLength ? cluster.avgSessionLength.toFixed(1) + ' msgs' : 'N/A';

            // Key Phrases — only show phrases with at least one matched message
            const keywordsContainer = document.getElementById('detail-keywords');
            const visibleKeywords = cluster.keywords
                ? cluster.keywords.filter(kw =>
                    !cluster.keyPhraseCounts || (cluster.keyPhraseCounts[kw] || 0) > 0
                  )
                : [];
            if (visibleKeywords.length > 0) {{
                keywordsContainer.innerHTML = visibleKeywords.map(kw =>
                    `<span class="keyword-tag-detail">${{kw}}</span>`
                ).join('');
            }} else {{
                keywordsContainer.innerHTML = '<span class="info-value">No key phrases available</span>';
            }}

            // Performance metrics — pre-sales only
            const metricsSection = document.getElementById('performance-metrics-section');
            if (cluster.category === 'pre-sales' && cluster.metrics && cluster.metrics.order !== undefined) {{
                const value = cluster.metrics.order;
                const median = globalMedians.order;
                const absoluteDiff = value - median;
                const isPositive = absoluteDiff > 0;

                document.getElementById('metric-order-value').textContent = value.toFixed(1) + '%';
                document.getElementById('metric-order-diff').textContent =
                    `${{isPositive ? '▲' : '▼'}} ${{isPositive ? '+' : ''}}${{absoluteDiff.toFixed(1)}}%`;
                document.getElementById('metric-order-diff').className =
                    `metric-comparison ${{isPositive ? 'positive' : 'negative'}}`;
                document.getElementById('metric-order-baseline').textContent = `vs ${{median.toFixed(1)}}%`;
                metricsSection.style.display = 'block';
            }} else {{
                metricsSection.style.display = 'none';
            }}

            // Messages
            const messagesList = document.getElementById('messages-list');
            const messagesTitle = document.querySelector('.messages-title');
            messagesTitle.textContent = `Messages (${{cluster.messages.length}})`;
            messagesList.innerHTML = cluster.messages.map(message => `
                <div class="message-item">${{message}}</div>
            `).join('');

            setTimeout(() => {{
                const keyInfoPanel = document.querySelector('.key-info-panel');
                const messagesSection = document.querySelector('.messages-section');
                if (keyInfoPanel && messagesSection) {{
                    messagesSection.style.height = keyInfoPanel.offsetHeight + 'px';
                }}
            }}, 50);

            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}

        function goBackToOverview() {{
            const deepDivePage = document.getElementById('deep-dive-page');
            deepDivePage.style.display = 'none';
            deepDivePage.className = 'deep-dive-page';
            document.getElementById('overview-page').style.display = 'block';

            const perfView = document.getElementById('conversion-performance-view');
            const clustersGrid = document.getElementById('clusters-container');
            if (perfView.style.display !== 'none') {{
                perfView.style.display = 'none';
                clustersGrid.style.display = 'grid';
            }}
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}

        document.addEventListener('DOMContentLoaded', initializePage);
    </script>
</body>
</html>'''

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"[VOC-REPORT] Saved to: {output_path}")
    return output_path
