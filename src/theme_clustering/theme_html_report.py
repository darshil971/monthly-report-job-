"""
Theme Clustering HTML Report Generator

Glassmorphic design with drill-down capability.
Based on generate_html_report.py styling with theme clustering data.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any


def generate_html_report(
    clusters_data: Dict[str, Any],
    quality_report: Dict[str, Any],
    output_path: str,
    client_name: str = "Client",
) -> str:
    """
    Generate a glassmorphic HTML report from theme clustering results.

    Args:
        clusters_data: Clusters JSON data
        quality_report: Quality report data
        output_path: Path to save HTML file
        client_name: Client name for display

    Returns:
        Path to generated HTML file
    """
    print(f"[HTML] Generating glassmorphic HTML report...")

    # Extract data
    clusters = clusters_data.get('clusters', [])
    total_messages = quality_report.get('total_messages', 0)
    coverage = quality_report.get('coverage_percent', 0)
    num_themes = quality_report.get('num_themes', 0)
    misc_messages = quality_report.get('miscellaneous_messages', 0)

    # Separate regular themes from miscellaneous
    theme_clusters = [c for c in clusters if c.get('cluster_id', -2) >= 0]
    misc_cluster = next((c for c in clusters if c.get('cluster_id') == -1), None)

    # Sort by message count
    theme_clusters.sort(key=lambda x: x.get('message_count', 0), reverse=True)

    # Calculate percentages for each theme
    for cluster in theme_clusters:
        cluster['percentage'] = (cluster.get('message_count', 0) / total_messages * 100) if total_messages > 0 else 0

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Theme Analysis - {client_name}</title>
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
            background: linear-gradient(90deg, #8FADE1 0%, #7A9DD1 100%);
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
            color: #8FADE1;
            transition: all 0.3s ease;
        }}

        .cluster-card:hover .cluster-percentage {{
            font-size: 24px;
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
            background: rgba(143, 173, 225, 0.5);
            color: #3C3C3E;
            border: 1px solid rgba(143, 173, 225, 0.7);
        }}

        .click-btn:hover {{
            background: rgba(143, 173, 225, 0.7);
            transform: scale(1.02);
            box-shadow: 0 6px 24px rgba(143, 173, 225, 0.4);
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
            font-size: 14px;
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

        .message-similarity {{
            color: #8FADE1;
            font-weight: 700;
            margin-right: 8px;
        }}

        .hidden {{
            display: none;
        }}

        .usecase-badge {{
            display: inline-block;
            background: rgba(143, 173, 225, 0.25);
            color: #5A7DB5;
            padding: 3px 10px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
            border: 1px solid rgba(143, 173, 225, 0.4);
            margin-right: 4px;
            margin-bottom: 4px;
        }}

        .borderline-tag {{
            color: #E8A44A;
            font-weight: 700;
            font-size: 14px;
            margin-left: 4px;
            cursor: help;
        }}

        .usecase-rescued-tag {{
            color: #6BBF6B;
            font-weight: 700;
            font-size: 14px;
            margin-left: 4px;
            cursor: help;
        }}

        .fn-rescued-tag {{
            color: #B07ED4;
            font-weight: 700;
            font-size: 14px;
            margin-left: 4px;
            cursor: help;
        }}

        .usecase-boosted-tag {{
            color: #5BB8D4;
            font-weight: 700;
            font-size: 14px;
            margin-left: 4px;
            cursor: help;
        }}

        /* Mobile Responsive */
        @media (max-width: 1024px) {{
            .detail-content {{
                grid-template-columns: 1fr;
            }}

            .key-info-panel {{
                margin-bottom: 20px;
            }}
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 16px;
            }}

            .header-content {{
                flex-direction: column;
                text-align: center;
            }}

            .header-left {{
                min-width: auto;
            }}

            .header-subtitle {{
                justify-content: center;
            }}

            .header-meta {{
                justify-content: center;
                grid-template-columns: repeat(2, 140px);
            }}

            .cluster-title {{
                padding-right: 70px;
            }}

            .cluster-percentage {{
                font-size: 16px;
                padding: 6px 8px;
            }}

            .clusters-grid {{
                grid-template-columns: 1fr;
            }}

            .nav-bar {{
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
            }}

            .breadcrumb {{
                margin-left: 0;
            }}

            .detail-header {{
                margin-left: 0;
                font-size: 20px;
            }}
        }}

        /* Scrollbar styling for glassy effect */
        .messages-list::-webkit-scrollbar {{
            width: 8px;
        }}

        .messages-list::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }}

        .messages-list::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            transition: all 0.3s ease;
        }}

        .messages-list::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.5);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Overview Page -->
        <div id="overview-page">
            <div class="header">
                <div class="header-content">
                    <div class="header-left">
                        <h1 class="header-title">🎯 Theme Analysis</h1>
                        <div class="header-subtitle">
                            <div class="header-byline">Discover what your customers are really saying</div>
                            <div class="header-date">{datetime.now().strftime('%B %Y')}</div>
                        </div>
                    </div>
                    <div class="header-meta">
                        <div class="meta-item">
                            <div class="meta-label">Messages</div>
                            <div class="meta-value">{total_messages if total_messages < 4000 else "4000+"}</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">Themes</div>
                            <div class="meta-value">{num_themes}</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="clusters-grid" id="clusters-container">
                <!-- Clusters will be populated by JavaScript -->
            </div>
        </div>

        <!-- Deep Dive Page -->
        <div id="deep-dive-page" class="deep-dive-page">
            <div class="nav-bar">
                <button class="back-btn" onclick="goBackToOverview()">← Back</button>
                <div class="breadcrumb">Overview > <span id="current-cluster-name">Theme Name</span></div>
            </div>

            <div class="detail-header" id="detail-header-title">
                Theme Deep Dive
            </div>

            <div class="detail-content">
                <div class="key-info-panel">
                    <h3 class="key-info-title">KEY INFO</h3>

                    <div class="info-section">
                        <div class="info-label">Description</div>
                        <div class="info-value" id="detail-summary">Theme description goes here...</div>
                    </div>

                    <div class="info-section keywords">
                        <div class="info-label">Key Phrases</div>
                        <div class="keyword-tags-detail" id="detail-keywords">
                            <!-- Key phrases will be populated by JavaScript -->
                        </div>
                    </div>

                    <div class="info-section" style="display: none;">
                        <div class="info-label">Parent Usecases</div>
                        <div id="detail-usecases">
                            <!-- Usecases will be populated by JavaScript -->
                        </div>
                    </div>

                    <div class="info-section">
                        <div class="info-label">% of Total Messages</div>
                        <div class="info-value large" id="detail-percentage">0%</div>
                    </div>

                    <div class="info-section">
                        <div class="info-label">Number of Messages</div>
                        <div class="info-value large" id="detail-message-count">0</div>
                    </div>
                </div>

                <div class="messages-section">
                    <div class="messages-header">
                        <div class="messages-title">Messages (sorted by similarity)</div>
                    </div>
                    <div class="messages-list" id="messages-list">
                        <!-- Messages will be populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Cluster data from Python
        const clusterData = {json.dumps([{
            'id': cluster.get('cluster_id', 0),
            'title': cluster.get('cluster_title', 'Untitled Theme'),
            'description': cluster.get('description', 'No description available'),
            'percentage': round(cluster['percentage'], 1),
            'messageCount': cluster.get('message_count', 0),
            'keyPhrases': cluster.get('key_phrases', []),
            'parentUsecases': cluster.get('parent_usecases', []),
            'messages': cluster.get('messages', []),
        } for cluster in theme_clusters], indent=8)};

        // Initialize page
        function initializePage() {{
            renderClusters();
        }}

        // Truncate text for summary cards
        function truncateSummary(text, element) {{
            const lineHeight = 1.5;
            const fontSize = 14;
            const cardPadding = 24;
            const bottomSpace = 60; // Space reserved for the MORE button

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

        // Render clusters
        function renderClusters() {{
            const container = document.getElementById('clusters-container');

            container.innerHTML = clusterData.map(cluster => {{
                const usecaseBadges = (cluster.parentUsecases && cluster.parentUsecases.length > 0)
                    ? cluster.parentUsecases.map(uc => `<span class="usecase-badge">${{uc}}</span>`).join('')
                    : '';
                return `
                    <div class="cluster-card">
                        <div class="cluster-header">
                            <div class="cluster-title">${{cluster.title}}</div>
                            <div class="cluster-percentage">${{cluster.percentage}}%</div>
                        </div>
                        ${{usecaseBadges ? `<div style="margin-bottom: 8px;">${{usecaseBadges}}</div>` : ''}}
                        <div class="cluster-summary" data-full-text="${{cluster.description.replace(/"/g, '&quot;')}}">${{cluster.description}}</div>
                        <div class="cluster-footer">
                            <button class="click-btn" onclick="openDeepDive(${{cluster.id}})">MORE</button>
                        </div>
                    </div>
                `;
            }}).join('');

            // Apply smart truncation after rendering
            setTimeout(() => {{
                document.querySelectorAll('.cluster-summary').forEach(summaryEl => {{
                    const fullText = summaryEl.getAttribute('data-full-text');
                    const truncated = truncateSummary(fullText, summaryEl);
                    summaryEl.textContent = truncated;
                }});
            }}, 0);
        }}

        // Open deep dive page
        function openDeepDive(clusterId) {{
            const cluster = clusterData.find(c => c.id === clusterId);
            if (!cluster) return;

            // Hide overview and show deep dive
            document.getElementById('overview-page').style.display = 'none';
            const deepDivePage = document.getElementById('deep-dive-page');
            deepDivePage.style.display = 'block';

            // Update deep dive content
            document.getElementById('current-cluster-name').textContent = cluster.title;
            document.getElementById('detail-header-title').textContent = cluster.title;
            document.getElementById('detail-summary').textContent = cluster.description;
            document.getElementById('detail-percentage').textContent = cluster.percentage + '%';
            document.getElementById('detail-message-count').textContent = cluster.messageCount;

            // Update key phrases
            const keywordsContainer = document.getElementById('detail-keywords');
            if (cluster.keyPhrases && cluster.keyPhrases.length > 0) {{
                keywordsContainer.innerHTML = cluster.keyPhrases.map(phrase =>
                    `<span class="keyword-tag-detail">${{phrase}}</span>`
                ).join('');
            }} else {{
                keywordsContainer.innerHTML = '<span class="info-value">No key phrases available</span>';
            }}

            // Update parent usecases in key info panel
            const usecaseContainer = document.getElementById('detail-usecases');
            if (usecaseContainer) {{
                if (cluster.parentUsecases && cluster.parentUsecases.length > 0) {{
                    usecaseContainer.innerHTML = cluster.parentUsecases.map(uc =>
                        `<span class="usecase-badge">${{uc}}</span>`
                    ).join('');
                    usecaseContainer.closest('.info-section').style.display = 'block';
                }} else {{
                    usecaseContainer.closest('.info-section').style.display = 'none';
                }}
            }}

            // Populate messages with borderline (⚡), usecase-rescued (🔄), and fn-rescued (🎯) symbols
            const messagesList = document.getElementById('messages-list');
            messagesList.innerHTML = cluster.messages.map(message => {{
                let tags = '';
                if (message.borderline) {{
                    tags += '<span class="borderline-tag" title="Borderline reassignment">⚡</span>';
                }}
                if (message.usecase_rescued) {{
                    tags += '<span class="usecase-rescued-tag" title="Usecase-boosted rescue">🔄</span>';
                }}
                if (message.fn_rescued) {{
                    tags += '<span class="fn-rescued-tag" title="False-negative signal rescue">🎯</span>';
                }}
                if (message.usecase_boosted) {{
                    const orig = message.original_cluster ? ` (was: ${{message.original_cluster}})` : '';
                    tags += `<span class="usecase-boosted-tag" title="Usecase boost changed cluster assignment${{orig}}">🏷️</span>`;
                }}
                return `
                    <div class="message-item">
                        <span class="message-similarity">[${{message.similarity.toFixed(3)}}]</span>
                        ${{message.text}}${{tags}}
                    </div>
                `;
            }}).join('');

            // Match messages card height to key info card height
            setTimeout(() => {{
                const keyInfoPanel = document.querySelector('.key-info-panel');
                const messagesSection = document.querySelector('.messages-section');
                if (keyInfoPanel && messagesSection) {{
                    const keyInfoHeight = keyInfoPanel.offsetHeight;
                    messagesSection.style.height = keyInfoHeight + 'px';
                }}
            }}, 50);

            // Scroll to top
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}

        // Go back to overview
        function goBackToOverview() {{
            document.getElementById('deep-dive-page').style.display = 'none';
            document.getElementById('overview-page').style.display = 'block';
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializePage);
    </script>
</body>
</html>'''

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[HTML] Report saved to: {output_path}")

    return output_path


def generate_report_from_files(
    clusters_file: str,
    quality_file: str,
    output_file: str,
    client_name: str = "Client",
) -> str:
    """
    Generate HTML report from JSON files.

    Args:
        clusters_file: Path to clusters JSON file
        quality_file: Path to quality report JSON file
        output_file: Path to save HTML file
        client_name: Client name for display

    Returns:
        Path to generated HTML file
    """
    with open(clusters_file, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)

    with open(quality_file, 'r', encoding='utf-8') as f:
        quality_report = json.load(f)

    return generate_html_report(clusters_data, quality_report, output_file, client_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python theme_html_report.py <clusters_json> <quality_json> [output_html] [client_name]")
        sys.exit(1)

    clusters_file = sys.argv[1]
    quality_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else clusters_file.replace('.json', '.html')
    client_name = sys.argv[4] if len(sys.argv) > 4 else "Client"

    generate_report_from_files(clusters_file, quality_file, output_file, client_name)
    print(f"\n✓ HTML report generated: {output_file}")
