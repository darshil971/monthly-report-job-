#!/usr/bin/env python3
"""
Convert HTML files from new glassmorphic format to old blue gradient format.

Usage:
    python convert_to_old_format.py old.html revised.html
    python convert_to_old_format.py input.html  # overwrites input.html
"""

import re
import sys
from pathlib import Path


def convert_html_to_old_format(html_content):
    """Convert HTML from new glassmorphic format to old blue gradient format."""

    # Remove the truncateSummary JavaScript function (match by signature, not comment)
    html_content = re.sub(
        r'function truncateSummary\(text, element\)\s*\{.*?return truncated\.trim\(\) \+ \'\.\.\.\';\s*\}',
        '',
        html_content,
        flags=re.DOTALL
    )

    # Remove the setTimeout block that calls truncateSummary (match by querySelector, not comment)
    html_content = re.sub(
        r"setTimeout\(\(\) => \{[^}]*document\.querySelectorAll\('\.cluster-summary'\).*?\}, 0\);",
        '',
        html_content,
        flags=re.DOTALL
    )

    # Update cluster-summary in the template to not use data-full-text or truncation
    html_content = re.sub(
        r'<div class="cluster-summary" data-full-text="\$\{cluster\.summary\.replace\(/"/g, \'&quot;\'\)\}">\$\{cluster\.summary\}</div>',
        '<div class="cluster-summary">${cluster.summary}</div>',
        html_content
    )

    # Remove cluster-footer div wrapper from the click button
    html_content = re.sub(
        r'<div class="cluster-footer">\s*<button',
        '<button',
        html_content
    )
    html_content = re.sub(
        r'</button>\s*</div>\s*</div>\s*</div>',
        '</button>\n                    </div>\n                ',
        html_content
    )

    # CSS replacements - mapping new styles to old styles
    css_replacements = [
        # Body styling
        (
            r'body\s*\{[^}]*background:\s*#D2D9E2;[^}]*\}',
            '''body {
            font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
            line-height: 1.5;
        }'''
        ),

        # Header styling
        (
            r'\.header\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.15\);[^}]*\}',
            '''.header {
            background: linear-gradient(135deg, #2563EB 0%, #1d4ed8 100%);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }'''
        ),

        # Remove header:hover
        (r'\.header:hover\s*\{[^}]*\}', ''),

        # Header title
        (
            r'\.header-title\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.header-title {
            font-size: 28px;
            font-weight: 700;
            color: white;
            margin: 0 0 4px 0;
            line-height: 1.2;
        }'''
        ),

        # Header byline
        (
            r'\.header-byline\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.header-byline {
            font-size: 16px;
            font-weight: 400;
            color: rgba(255,255,255,0.9);
            font-style: italic;
            margin: 0;
        }'''
        ),

        # Header date
        (
            r'\.header-date\s*\{[^}]*\}',
            '''.header-date {
            font-size: 14px;
            font-weight: 600;
            color: rgba(255,255,255,0.8);
            background: rgba(255,255,255,0.1);
            padding: 4px 8px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            margin: 0;
        }'''
        ),

        # Remove header-date:hover
        (r'\.header-date:hover\s*\{[^}]*\}', ''),

        # Header meta
        (
            r'\.header-meta\s*\{[^}]*display:\s*grid;[^}]*\}',
            '''.header-meta {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }'''
        ),

        # Meta item
        (
            r'\.meta-item\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.2\);[^}]*\}',
            '''.meta-item {
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
            padding: 10px 16px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            min-width: 80px;
            text-align: center;
        }'''
        ),

        # Remove meta-item:hover
        (r'\.meta-item:hover\s*\{[^}]*\}', ''),

        # Meta label
        (
            r'\.meta-label\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.meta-label {
            font-size: 12px;
            color: #4a5568;
            font-weight: 600;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }'''
        ),

        # Meta value
        (
            r'\.meta-value\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.meta-value {
            font-size: 18px;
            font-weight: 800;
            color: #2c3e50;
            margin-top: 2px;
        }'''
        ),

        # Meta sublabel
        (
            r'\.meta-sublabel\s*\{[^}]*\}',
            '''.meta-sublabel {
            font-size: 8px;
            color: #4a5568;
            font-weight: 600;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }'''
        ),

        # Page link
        (
            r'\.page-link\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.page-link {
            font-size: 14px;
            font-weight: 600;
            color: rgba(255,255,255,0.8);
            background: rgba(255,255,255,0.1);
            padding: 4px 8px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            text-decoration: none;
        }'''
        ),

        # Remove page-link:hover
        (r'\.page-link:hover\s*\{[^}]*\}', ''),

        # Filter buttons
        (
            r'\.filter-buttons\s*\{[^}]*\}',
            '''.filter-buttons {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }'''
        ),

        # Remove filter-separator
        (r'\.filter-separator\s*\{[^}]*\}', ''),

        # Filter button
        (
            r'\.filter-btn\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.15\);[^}]*\}',
            '''.filter-btn {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }'''
        ),

        # Remove filter-btn:hover
        (r'\.filter-btn:hover\s*\{[^}]*\}', ''),

        # Filter button active
        (
            r'\.filter-btn\.active\s*\{[^}]*\}',
            '''.filter-btn.active {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            border-color: #2980b9;
            color: white;
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }

        #conversion-performance-btn.active {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            border-color: #2980b9;
            color: white;
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }'''
        ),

        # Remove filter-btn pre-sales and post-sales active states
        (r'\.filter-btn\[data-filter="pre-sales"\]\.active\s*\{[^}]*\}', ''),
        (r'\.filter-btn\[data-filter="post-sales"\]\.active\s*\{[^}]*\}', ''),

        # Cluster card
        (
            r'\.cluster-card\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.15\);[^}]*\}',
            '''.cluster-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            min-height: 280px;
            display: flex;
            flex-direction: column;
        }'''
        ),

        # Cluster card ::before
        (
            r'\.cluster-card::before\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.3\);[^}]*\}',
            '''.cluster-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: #e9ecef;
        }'''
        ),

        # Cluster card pre-sales ::before
        (
            r'\.cluster-card\[data-category="pre-sales"\]::before\s*\{[^}]*\}',
            '''.cluster-card[data-category="pre-sales"]::before {
            background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        }'''
        ),

        # Cluster card post-sales ::before
        (
            r'\.cluster-card\[data-category="post-sales"\]::before\s*\{[^}]*\}',
            '''.cluster-card[data-category="post-sales"]::before {
            background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        }'''
        ),

        # Cluster card hover
        (
            r'\.cluster-card:hover\s*\{[^}]*\}',
            '''.cluster-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }'''
        ),

        # Cluster title
        (
            r'\.cluster-title\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.cluster-title {
            font-size: 18px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 16px;
            line-height: 1.3;
            padding-right: 80px;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }'''
        ),

        # Cluster percentage
        (
            r'\.cluster-percentage\s*\{[^}]*\}',
            '''.cluster-percentage {
            position: absolute;
            top: 16px;
            right: 16px;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: 800;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }'''
        ),

        # Remove cluster-card:hover .cluster-percentage
        (r'\.cluster-card:hover \.cluster-percentage\s*\{[^}]*\}', ''),

        # Cluster percentage pre-sales
        (
            r'\.cluster-card\[data-category="pre-sales"\] \.cluster-percentage\s*\{[^}]*\}',
            '''.cluster-card[data-category="pre-sales"] .cluster-percentage {
            color: #1d4ed8;
        }'''
        ),

        # Cluster percentage post-sales
        (
            r'\.cluster-card\[data-category="post-sales"\] \.cluster-percentage\s*\{[^}]*\}',
            '''.cluster-card[data-category="post-sales"] .cluster-percentage {
            color: #d97706;
        }'''
        ),

        # Cluster summary
        (
            r'\.cluster-summary\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.cluster-summary {
            font-size: 14px;
            color: #4a5568;
            line-height: 1.5;
            margin-bottom: 12px;
            padding-bottom: 70px;
            flex-grow: 1;
        }'''
        ),

        # Keywords label
        (
            r'\.keywords-label\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.keywords-label {
            font-size: 12px;
            font-weight: 600;
            color: #6c757d;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }'''
        ),

        # Keyword tag
        (
            r'\.keyword-tag\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.2\);[^}]*\}',
            '''.keyword-tag {
            background: rgba(255,255,255,0.8);
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 14px;
            font-weight: 600;
            border: 1px solid rgba(0,0,0,0.1);
            backdrop-filter: blur(5px);
        }'''
        ),

        # Remove keyword-tag:hover
        (r'\.keyword-tag:hover\s*\{[^}]*\}', ''),

        # Keyword tag pre-sales
        (
            r'\.cluster-card\[data-category="pre-sales"\] \.keyword-tag\s*\{[^}]*\}',
            '''.cluster-card[data-category="pre-sales"] .keyword-tag {
            background: rgba(219, 234, 254, 0.8);
            color: #1d4ed8;
            border-color: rgba(29, 78, 216, 0.3);
        }'''
        ),

        # Keyword tag post-sales
        (
            r'\.cluster-card\[data-category="post-sales"\] \.keyword-tag\s*\{[^}]*\}',
            '''.cluster-card[data-category="post-sales"] .keyword-tag {
            background: rgba(254, 243, 199, 0.8);
            color: #d97706;
            border-color: rgba(217, 119, 6, 0.3);
        }'''
        ),

        # Remove cluster-footer and metric-badge-simple
        (r'\.cluster-footer\s*\{[^}]*\}', ''),
        (r'\.metric-badge-simple\s*\{[^}]*\}', ''),
        (r'\.metric-badge-simple\.good\s*\{[^}]*\}', ''),
        (r'\.metric-badge-simple\.poor\s*\{[^}]*\}', ''),

        # Click button
        (
            r'\.click-btn\s*\{[^}]*\}',
            '''.click-btn {
            position: absolute;
            bottom: 16px;
            right: 16px;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }'''
        ),

        # Click button pre-sales
        (
            r'\.cluster-card\[data-category="pre-sales"\] \.click-btn\s*\{[^}]*\}',
            '''.cluster-card[data-category="pre-sales"] .click-btn {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
        }'''
        ),

        # Click button pre-sales hover
        (
            r'\.cluster-card\[data-category="pre-sales"\] \.click-btn:hover\s*\{[^}]*\}',
            '''.cluster-card[data-category="pre-sales"] .click-btn:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }'''
        ),

        # Click button post-sales
        (
            r'\.cluster-card\[data-category="post-sales"\] \.click-btn\s*\{[^}]*\}',
            '''.cluster-card[data-category="post-sales"] .click-btn {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
        }'''
        ),

        # Click button post-sales hover
        (
            r'\.cluster-card\[data-category="post-sales"\] \.click-btn:hover\s*\{[^}]*\}',
            '''.cluster-card[data-category="post-sales"] .click-btn:hover {
            background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
        }'''
        ),

        # Nav bar
        (
            r'\.nav-bar\s*\{[^}]*\}',
            '''.nav-bar {
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
            padding: 16px 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }'''
        ),

        # Remove nav-bar:hover
        (r'\.nav-bar:hover\s*\{[^}]*\}', ''),

        # Back button
        (
            r'\.back-btn\s*\{[^}]*\}',
            '''.back-btn {
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            color: #495057;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 2px solid transparent;
        }'''
        ),

        # Add back button category styles
        (
            r'\.back-btn:hover\s*\{[^}]*\}',
            '''.deep-dive-page.pre-sales .back-btn {
            border-color: #3b82f6;
            color: #1d4ed8;
        }

        .deep-dive-page.post-sales .back-btn {
            border-color: #f59e0b;
            color: #d97706;
        }

        .back-btn:hover {
            background: linear-gradient(135deg, #dee2e6 0%, #ced4da 100%);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }'''
        ),

        # Breadcrumb
        (
            r'\.breadcrumb\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.breadcrumb {
            font-size: 14px;
            color: #6c757d;
        }'''
        ),

        # Detail header
        (
            r'\.detail-header\s*\{[^}]*\}',
            '''.detail-header {
            font-size: 32px;
            font-weight: 700;
            color: #2c3e50;
            margin: 20px 0;
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }'''
        ),

        # Remove detail-header:hover
        (r'\.detail-header:hover\s*\{[^}]*\}', ''),

        # Detail cards (change grid if present)
        (
            r'\.detail-cards\s*\{[^}]*\}',
            '''.detail-cards {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }'''
        ),

        # Detail card
        (
            r'\.detail-card\s*\{[^}]*\}',
            '''.detail-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }'''
        ),

        # Remove detail-card:hover
        (r'\.detail-card:hover\s*\{[^}]*\}', ''),

        # Detail card ::before pre-sales
        (
            r'\.deep-dive-page\.pre-sales \.detail-card::before\s*\{[^}]*\}',
            '''.deep-dive-page.pre-sales .detail-card::before {
            background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        }'''
        ),

        # Detail card ::before post-sales
        (
            r'\.deep-dive-page\.post-sales \.detail-card::before\s*\{[^}]*\}',
            '''.deep-dive-page.post-sales .detail-card::before {
            background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        }'''
        ),

        # Detail card title
        (
            r'\.detail-card-title\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.detail-card-title {
            font-size: 14px;
            font-weight: 600;
            color: #6c757d;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }'''
        ),

        # Detail card content
        (
            r'\.detail-card-content\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.detail-card-content {
            font-size: 16px;
            color: #2c3e50;
            line-height: 1.5;
        }'''
        ),

        # Deep dive keyword tags
        (
            r'\.deep-dive-page\.pre-sales \.keyword-tag\s*\{[^}]*\}',
            '''.deep-dive-page.pre-sales .keyword-tag {
            background: rgba(219, 234, 254, 0.8);
            color: #1d4ed8;
            border-color: rgba(29, 78, 216, 0.3);
        }'''
        ),

        (
            r'\.deep-dive-page\.post-sales \.keyword-tag\s*\{[^}]*\}',
            '''.deep-dive-page.post-sales .keyword-tag {
            background: rgba(254, 243, 199, 0.8);
            color: #d97706;
            border-color: rgba(217, 119, 6, 0.3);
        }'''
        ),

        # Messages section
        (
            r'\.messages-section\s*\{[^}]*\}',
            '''.messages-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            overflow: hidden;
            backdrop-filter: blur(10px);
        }'''
        ),

        # Remove messages-section:hover and ::before
        (r'\.messages-section:hover\s*\{[^}]*\}', ''),
        (r'\.messages-section::before\s*\{[^}]*\}', ''),
        (r'\.deep-dive-page\.post-sales \.messages-section::before\s*\{[^}]*\}', ''),
        (r'\.deep-dive-page\.pre-sales \.messages-section::before\s*\{[^}]*\}', ''),

        # Messages header
        (
            r'\.messages-header\s*\{[^}]*\}',
            '''.messages-header {
            padding: 28px 32px 24px 32px;
            border-bottom: 1px solid rgba(233, 236, 239, 0.5);
        }'''
        ),

        # Messages title
        (
            r'\.messages-title\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.messages-title {
            font-size: 18px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0;
        }

        .deep-dive-page.pre-sales .messages-title {
            padding-left: 8px;
        }

        .deep-dive-page.pre-sales .messages-header {
            padding: 28px 32px 24px 32px;
            border-bottom: 1px solid rgba(233, 236, 239, 0.5);
        }'''
        ),

        # Messages list
        (
            r'\.messages-list\s*\{[^}]*\}',
            '''.messages-list {
            max-height: 400px;
            overflow-y: auto;
            padding: 0;
        }'''
        ),

        # Message item
        (
            r'\.message-item\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.message-item {
            padding: 20px 32px;
            border-bottom: 1px solid #f8f9fa;
            font-size: 16px;
            color: #2c3e50;
            line-height: 1.6;
            font-weight: 500;
        }'''
        ),

        # Message item first child - add extra top padding
        (
            r'(\.message-item:nth-child\(even\)\s*\{)',
            '''.message-item:first-child {
            padding-top: 24px;
        }

        .message-item:last-child {
            padding-bottom: 24px;
            border-bottom: none;
        }

        \\1'''
        ),

        # Remove message-item:hover
        (r'\.message-item:hover\s*\{[^}]*\}', ''),

        # Message item even
        (
            r'\.message-item:nth-child\(even\)\s*\{[^}]*\}',
            '''.message-item:nth-child(even) {
            background: #fafbfc;
        }'''
        ),

        # Remove message-item:nth-child(even):hover
        (r'\.message-item:nth-child\(even\):hover\s*\{[^}]*\}', ''),

        # Scrollbar styling
        (
            r'\.messages-list::-webkit-scrollbar-track\s*\{[^}]*\}',
            '''.messages-list::-webkit-scrollbar-track {
            background: #f8f9fa;
            border-radius: 10px;
        }'''
        ),

        (
            r'\.messages-list::-webkit-scrollbar-thumb\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.3\);[^}]*\}',
            '''.messages-list::-webkit-scrollbar-thumb {
            background: #cbd5e0;
            border-radius: 10px;
            transition: all 0.3s ease;
        }'''
        ),

        (
            r'\.messages-list::-webkit-scrollbar-thumb:hover\s*\{[^}]*\}',
            '''.messages-list::-webkit-scrollbar-thumb:hover {
            background: #a0aec0;
        }'''
        ),

        # Info card
        (
            r'\.info-card\s*\{[^}]*\}',
            '''.info-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }'''
        ),

        # Remove info-card:hover
        (r'\.info-card:hover\s*\{[^}]*\}', ''),

        # Info text
        (
            r'\.info-text\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.info-text {
            font-size: 14px;
            color: #4a5568;
            line-height: 1.6;
            font-weight: 500;
        }'''
        ),

        # Performance table container
        (
            r'\.performance-table-container\s*\{[^}]*\}',
            '''.performance-table-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
            overflow: hidden;
        }'''
        ),

        # Remove performance-table-container:hover
        (r'\.performance-table-container:hover\s*\{[^}]*\}', ''),

        # Performance table thead
        (
            r'\.performance-table thead\s*\{[^}]*\}',
            '''.performance-table thead {
            background: #f8f9fa;
            position: sticky;
            top: 0;
            z-index: 10;
        }'''
        ),

        # Performance table th
        (
            r'\.performance-table th\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.performance-table th {
            padding: 16px 20px;
            text-align: center;
            font-weight: 700;
            color: #2c3e50;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 12px;
            border-bottom: 2px solid #e9ecef;
            cursor: pointer;
            user-select: none;
            transition: all 0.2s ease;
            position: relative;
        }'''
        ),

        # Performance table th:hover
        (
            r'\.performance-table th:hover\s*\{[^}]*\}',
            '''.performance-table th:hover {
            background: #e9ecef;
        }'''
        ),

        # Performance table sort icons
        (
            r'\.performance-table th\.sort-asc::after\s*\{[^}]*color:\s*#8FADE1;[^}]*\}',
            '''.performance-table th.sort-asc::after {
            content: '▲';
            opacity: 1;
            color: #3b82f6;
        }'''
        ),

        (
            r'\.performance-table th\.sort-desc::after\s*\{[^}]*color:\s*#8FADE1;[^}]*\}',
            '''.performance-table th.sort-desc::after {
            content: '▼';
            opacity: 1;
            color: #3b82f6;
        }'''
        ),

        # Performance table tbody tr
        (
            r'\.performance-table tbody tr\s*\{[^}]*\}',
            '''.performance-table tbody tr {
            transition: all 0.2s ease;
            border-bottom: 1px solid #f8f9fa;
        }'''
        ),

        # Performance table tbody tr:nth-child(even)
        (
            r'\.performance-table tbody tr:nth-child\(even\)\s*\{[^}]*\}',
            '''.performance-table tbody tr:nth-child(even) {
            background: #fafbfc;
        }'''
        ),

        # Performance table tbody tr:hover
        (
            r'\.performance-table tbody tr:hover\s*\{[^}]*\}',
            '''.performance-table tbody tr:hover {
            background: #f8f9fa;
        }'''
        ),

        # Performance table td
        (
            r'\.performance-table td\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.performance-table td {
            padding: 14px 20px;
            color: #2c3e50;
            font-weight: 500;
            text-align: center;
        }'''
        ),

        # Cluster title link
        (
            r'\.cluster-title-link\s*\{[^}]*color:\s*#8FADE1;[^}]*\}',
            '''.cluster-title-link {
            color: #3b82f6;
            text-decoration: underline;
            text-underline-offset: 2px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: block;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.4;
            font-weight: 600;
        }'''
        ),

        # Cluster title link hover
        (
            r'\.cluster-title-link:hover\s*\{[^}]*color:\s*#7A9DD1;[^}]*\}',
            '''.cluster-title-link:hover {
            color: #1d4ed8;
            text-decoration: underline;
            text-underline-offset: 2px;
        }'''
        ),

        # Metric value
        (
            r'\.metric-value\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.metric-value {
            font-weight: 700;
            color: #2c3e50;
        }'''
        ),

        # Metric diff good/poor
        (
            r'\.metric-diff\.good\s*\{[^}]*color:\s*#9FC63B;[^}]*\}',
            '''.metric-diff.good {
            color: #38a169;
        }'''
        ),

        (
            r'\.metric-diff\.poor\s*\{[^}]*color:\s*#F01263;[^}]*\}',
            '''.metric-diff.poor {
            color: #c53030;
        }'''
        ),

        # Performance table scrollbar
        (
            r'\.performance-table-container::-webkit-scrollbar-track\s*\{[^}]*\}',
            '''.performance-table-container::-webkit-scrollbar-track {
            background: #f8f9fa;
            border-radius: 10px;
        }'''
        ),

        (
            r'\.performance-table-container::-webkit-scrollbar-thumb\s*\{[^}]*background:\s*rgba\(143,\s*173,\s*225,\s*0\.\d+\);[^}]*\}',
            '''.performance-table-container::-webkit-scrollbar-thumb {
            background: #cbd5e0;
            border-radius: 10px;
            transition: all 0.3s ease;
        }'''
        ),

        (
            r'\.performance-table-container::-webkit-scrollbar-thumb:hover\s*\{[^}]*\}',
            '''.performance-table-container::-webkit-scrollbar-thumb:hover {
            background: #a0aec0;
        }'''
        ),
    ]

    # Apply all CSS replacements
    modified_content = html_content
    for pattern, replacement in css_replacements:
        modified_content = re.sub(pattern, replacement, modified_content, flags=re.DOTALL)

    # Add ID to conversion performance button
    modified_content = re.sub(
        r'<button class="filter-btn" onclick="toggleConversionPerformanceView\(\)">Conversion Performance</button>',
        '<button class="filter-btn" id="conversion-performance-btn" onclick="toggleConversionPerformanceView()">Conversion Performance</button>',
        modified_content
    )

    # Update setupFilterButtons JavaScript function to deactivate conversion button
    modified_content = re.sub(
        r'(function setupFilterButtons\(\) \{\{[\s\S]*?)(// Category filter button)',
        r'''\1// Deactivate conversion button if it exists
                    const conversionBtn = document.getElementById('conversion-performance-btn');
                    if (conversionBtn) {{
                        conversionBtn.classList.remove('active');
                    }}

                    \2''',
        modified_content
    )

    # Update toggleConversionPerformanceView JavaScript function for mutual exclusion.
    # Guard: skip if new-format HTML already has the logic built in.
    if "conversionBtn.classList.add('active')" not in modified_content:
        modified_content = re.sub(
            r'(function toggleConversionPerformanceView\(\) \{\{[\s\S]*?const perfView = document\.getElementById\(\'conversion-performance-view\'\);[\s\S]*?const clustersGrid = document\.getElementById\(\'clusters-container\'\);)',
            r'''\1
            const conversionBtn = document.getElementById('conversion-performance-btn');''',
            modified_content
        )

        modified_content = re.sub(
            r'(if \(perfView\.style\.display === \'none\'\) \{\{[\s\S]*?perfView\.style\.display = \'block\';[\s\S]*?clustersGrid\.style\.display = \'none\';)',
            r'''\1

                // Deactivate all filter buttons and activate conversion button
                document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => btn.classList.remove('active'));
                if (conversionBtn) {{
                    conversionBtn.classList.add('active');
                }}''',
            modified_content
        )

    modified_content = re.sub(
        r'(\}\} else \{\{[\s\S]*?// Hide performance view and show clusters[\s\S]*?perfView\.style\.display = \'none\';[\s\S]*?clustersGrid\.style\.display = \'grid\';)',
        r'''\1

                // Deactivate conversion button and reactivate the "All" filter
                if (conversionBtn) {{
                    conversionBtn.classList.remove('active');
                }}
                const allButton = document.querySelector('.filter-btn[data-filter="all"]');
                if (allButton) {{
                    allButton.classList.add('active');
                    currentCategoryFilter = 'all';
                    renderClusters();
                }}''',
        modified_content
    )

    # -----------------------------------------------------------------------
    # Parity patches — applied to old-format files to bring them in line with
    # the updated theme_voc_report.py output.
    # -----------------------------------------------------------------------

    # 1. Info text update
    modified_content = re.sub(
        r'Shows best and worst significant pre-sales clusters by order conversion\.',
        'Shows significant pre-sales clusters by order conversion.',
        modified_content
    )

    # 2. Performance table: show ALL significant pre-sales (not just showOrder outliers).
    #    Uses isSignificant when available (new data), falls back to showOrder for old data.
    modified_content = re.sub(
        r'clusterData\.filter\(cluster => cluster\.showOrder === true\)',
        r"clusterData.filter(cluster => cluster.isSignificant != null ? (cluster.category === 'pre-sales' && cluster.isSignificant) : cluster.showOrder === true)",
        modified_content
    )

    # 3. Default sort column for performance view: comparison (desc) instead of sessionPercentage.
    modified_content = re.sub(
        r"currentSortColumn = 'sessionPercentage';\s*currentSortDirection = 'desc';",
        "currentSortColumn = 'comparison';\n                currentSortDirection = 'desc';",
        modified_content
    )
    modified_content = re.sub(
        r"if \(h\.dataset\.sort === 'sessionPercentage'\) h\.classList\.add\('sort-desc'\);",
        "if (h.dataset.sort === 'comparison') h.classList.add('sort-desc');",
        modified_content
    )

    # 4. Move sort-desc class from sessionPercentage header to comparison header in the table HTML.
    modified_content = re.sub(
        r'<th class="sortable sort-desc" data-sort="sessionPercentage">',
        '<th class="sortable" data-sort="sessionPercentage">',
        modified_content
    )
    modified_content = re.sub(
        r'<th class="sortable" data-sort="comparison">',
        '<th class="sortable sort-desc" data-sort="comparison">',
        modified_content
    )

    # 5. Inject updateHeaderStats() and wire it up from renderClusters() if not already present.
    if 'function updateHeaderStats()' not in modified_content:
        update_header_stats_fn = r"""
        function updateHeaderStats() {
            const targetCategories = categoryMapping[currentCategoryFilter];
            const visibleClusters = clusterData.filter(cluster =>
                cluster.allCategories.some(cat => targetCategories.includes(cat))
            );
            const totalMsgs = visibleClusters.reduce((sum, c) => sum + c.messageCount, 0);
            const totalSessions = visibleClusters.reduce((sum, c) => sum + (c.sessionCount || 0), 0);
            const msgsEl = document.getElementById('header-messages-count');
            const sessionsEl = document.getElementById('header-sessions-count');
            const conversionEl = document.getElementById('header-conversion-value');
            if (msgsEl) msgsEl.textContent = totalMsgs >= 4000 ? '4000+' : totalMsgs.toLocaleString();
            if (sessionsEl) sessionsEl.textContent = totalSessions.toLocaleString() + ' users';
            if (conversionEl) {
                if (currentCategoryFilter === 'post-sales') {
                    conversionEl.textContent = '-';
                } else if (currentCategoryFilter === 'all') {
                    conversionEl.textContent = globalMedians.order.toFixed(1) + '%';
                } else {
                    const orderVals = visibleClusters.map(c => c.metrics.order).sort((a, b) => a - b);
                    const n = orderVals.length;
                    const median = n > 0
                        ? (n % 2 === 0 ? (orderVals[n/2-1] + orderVals[n/2]) / 2 : orderVals[Math.floor(n/2)])
                        : 0;
                    conversionEl.textContent = median.toFixed(1) + '%';
                }
            }
        }"""
        # Inject before setupFilterButtons
        modified_content = re.sub(
            r'(function setupFilterButtons\(\))',
            update_header_stats_fn + r'\n\n        \1',
            modified_content,
            count=1
        )
        # Wire call from renderClusters — after the setTimeout block that closes renderClusters
        modified_content = re.sub(
            r'(\}, 0\);\s*\}\})\s*\n(\s*function setupFilterButtons)',
            r'\1\n\n            updateHeaderStats();\n        }}\n\n\2',
            modified_content,
            count=1
        )

    # 6. Fix performanceClass in renderPerformanceTable: color by sign of absoluteDiff,
    #    not by orderPerformance (which miscolors 'average' clusters as red).
    modified_content = re.sub(
        r"const performanceClass = cluster\.orderPerformance === 'good' \? 'good' : 'poor';",
        "const performanceClass = cluster.absoluteDiffVsMedian >= 0 ? 'good' : 'poor';",
        modified_content
    )

    # 7. Fix updateHeaderStats conversion median: use 'all' branch for global median,
    #    compute from visibleClusters for any filtered view (pre-sales OR post-sales).
    modified_content = re.sub(
        r"if \(currentCategoryFilter === 'pre-sales'\) \{\s*"
        r"const preSalesClusters = clusterData\.filter\(c => c\.category === 'pre-sales'\);\s*"
        r"const orderVals = preSalesClusters\.map\(c => c\.metrics\.order\)\.sort\(\(a, b\) => a - b\);\s*"
        r"const n = orderVals\.length;\s*"
        r"const median = n > 0.*?: 0;\s*"
        r"conversionEl\.textContent = median\.toFixed\(1\) \+ '%';\s*"
        r"\} else \{\s*"
        r"conversionEl\.textContent = globalMedians\.order\.toFixed\(1\) \+ '%';\s*"
        r"\}",
        "if (currentCategoryFilter === 'all') {\n"
        "                    conversionEl.textContent = globalMedians.order.toFixed(1) + '%';\n"
        "                } else {\n"
        "                    const orderVals = visibleClusters.map(c => c.metrics.order).sort((a, b) => a - b);\n"
        "                    const n = orderVals.length;\n"
        "                    const median = n > 0\n"
        "                        ? (n % 2 === 0 ? (orderVals[n/2-1] + orderVals[n/2]) / 2 : orderVals[Math.floor(n/2)])\n"
        "                        : 0;\n"
        "                    conversionEl.textContent = median.toFixed(1) + '%';\n"
        "                }",
        modified_content,
        flags=re.DOTALL
    )

    # 8b. Fix post-sales dash in already-injected updateHeaderStats (idempotent).
    #     Replaces `if (currentCategoryFilter === 'all') {` with the post-sales guard
    #     ONLY when the post-sales branch is not already present.
    if "currentCategoryFilter === 'post-sales'" not in modified_content:
        modified_content = re.sub(
            r"if \(currentCategoryFilter === 'all'\) \{\s*"
            r"conversionEl\.textContent = globalMedians\.order\.toFixed\(1\) \+ '%';\s*"
            r"\} else \{",
            "if (currentCategoryFilter === 'post-sales') {\n"
            "                    conversionEl.textContent = '-';\n"
            "                } else if (currentCategoryFilter === 'all') {\n"
            "                    conversionEl.textContent = globalMedians.order.toFixed(1) + '%';\n"
            "                } else {",
            modified_content
        )

    # 9. Key phrase filtering: replace any count-badge rendering with filter-only rendering.
    #    Handles both old reports (plain map) and mid-version reports (count badge).
    #    Falls back to showing all phrases if keyPhraseCounts is absent (backward compat).
    if 'visibleKeywords' not in modified_content:
        # Replace count-badge style (mid-version): cluster.keywords.map(kw => { const count = ... })
        modified_content = re.sub(
            r"keywordsContainer\.innerHTML = cluster\.keywords\.map\(kw => \{[\s\S]*?return `<span class=\"keyword-tag-detail\">\$\{kw\}\$\{countBadge\}</span>`;\s*\}\)\.join\(''\);",
            "const visibleKeywords = cluster.keywords\n"
            "                ? cluster.keywords.filter(kw =>\n"
            "                    !cluster.keyPhraseCounts || (cluster.keyPhraseCounts[kw] || 0) > 0\n"
            "                  )\n"
            "                : [];\n"
            "            if (visibleKeywords.length > 0) {\n"
            "                keywordsContainer.innerHTML = visibleKeywords.map(kw =>\n"
            "                    `<span class=\"keyword-tag-detail\">${kw}</span>`\n"
            "                ).join('');\n"
            "            } else {\n"
            "                keywordsContainer.innerHTML = '<span class=\"info-value\">No key phrases available</span>';\n"
            "            }",
            modified_content
        )
        # Replace plain style (no-count): cluster.keywords.map(kw => `<span ...>${kw}</span>`)
        modified_content = re.sub(
            r"keywordsContainer\.innerHTML = cluster\.keywords\.map\(kw =>\s*`<span class=\"keyword-tag-detail\">\$\{kw\}</span>`\s*\)\.join\(''\);",
            "const visibleKeywords = cluster.keywords\n"
            "                ? cluster.keywords.filter(kw =>\n"
            "                    !cluster.keyPhraseCounts || (cluster.keyPhraseCounts[kw] || 0) > 0\n"
            "                  )\n"
            "                : [];\n"
            "            if (visibleKeywords.length > 0) {\n"
            "                keywordsContainer.innerHTML = visibleKeywords.map(kw =>\n"
            "                    `<span class=\"keyword-tag-detail\">${kw}</span>`\n"
            "                ).join('');\n"
            "            } else {\n"
            "                keywordsContainer.innerHTML = '<span class=\"info-value\">No key phrases available</span>';\n"
            "            }",
            modified_content
        )

    # 10. Category share badge: show % of total messages per category on filter button when active.
    if 'category-share-badge' not in modified_content:
        # Inject CSS before </style>
        modified_content = modified_content.replace(
            '</style>',
            '''        .category-share-badge {
            font-size: 12px;
            font-weight: 700;
            opacity: 0.85;
            margin-left: 4px;
        }
</style>''',
            1
        )
        # Inject setupCategoryShareBadges() before function setupFilterButtons
        share_badge_fn = r"""
        function setupCategoryShareBadges() {
            const _totalMsgsAll = clusterData.reduce((sum, c) => sum + c.messageCount, 0);
            const _categoryShares = {
                'pre-sales':  _totalMsgsAll > 0
                    ? Math.round(clusterData.filter(c => c.category === 'pre-sales').reduce((sum, c) => sum + c.messageCount, 0) / _totalMsgsAll * 100)
                    : 0,
                'post-sales': _totalMsgsAll > 0
                    ? Math.round(clusterData.filter(c => c.category === 'post-sales').reduce((sum, c) => sum + c.messageCount, 0) / _totalMsgsAll * 100)
                    : 0,
            };
            const buttons = document.querySelectorAll('.filter-btn[data-filter]');
            buttons.forEach(btn => {
                btn.addEventListener('click', () => {
                    buttons.forEach(b => {
                        const existing = b.querySelector('.category-share-badge');
                        if (existing) existing.remove();
                    });
                    const filter = btn.dataset.filter;
                    if (_categoryShares[filter] !== undefined) {
                        const badge = document.createElement('span');
                        badge.className = 'category-share-badge';
                        badge.textContent = ' ' + _categoryShares[filter] + '%';
                        btn.appendChild(badge);
                    }
                });
            });
        }"""
        modified_content = re.sub(
            r'(function setupFilterButtons\(\))',
            share_badge_fn + r'\n\n        \1',
            modified_content,
            count=1
        )
        # Wire setupCategoryShareBadges() call from initializePage
        modified_content = re.sub(
            r'(function initializePage\(\) \{[^\}]*setupFilterButtons\(\);)',
            r'\1\n            setupCategoryShareBadges();',
            modified_content,
            count=1,
            flags=re.DOTALL
        )

    # 8. Add IDs and sessions byline to Messages meta-item if not already present.
    if 'header-messages-count' not in modified_content:
        modified_content = re.sub(
            r'(<div class="meta-label">Messages</div>\s*)<div class="meta-value">([^<]*)</div>',
            r'\1<div class="meta-value" id="header-messages-count">\2</div>'
            r'\n                            <div class="meta-sublabel" id="header-sessions-count">– users</div>',
            modified_content
        )
    if 'header-conversion-value' not in modified_content:
        modified_content = re.sub(
            r'(<div class="meta-label">Conversion %</div>\s*)<div class="meta-value">([^<]*)</div>',
            r'\1<div class="meta-value" id="header-conversion-value">\2</div>',
            modified_content
        )

    return modified_content


def process_file(file_path):
    """Process a single HTML file."""
    print(f"Processing: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    converted_content = convert_html_to_old_format(html_content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(converted_content)

    print(f"✓ Converted: {file_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_old_format.py <input.html|folder_path> [output.html]")
        print("\nFor single file: Converts input.html to output.html (or overwrites if no output specified)")
        print("For folder: Recursively finds and converts all clusters_*.html files in-place")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: Path '{input_path}' does not exist.")
        sys.exit(1)

    # Handle directory input
    if input_path.is_dir():
        if len(sys.argv) > 2:
            print("Error: Output file cannot be specified when processing a folder.")
            sys.exit(1)

        print(f"Searching for clusters_*.html files in: {input_path}")
        html_files = list(input_path.rglob("clusters_*.html"))

        if not html_files:
            print("No clusters_*.html files found.")
            sys.exit(0)

        print(f"Found {len(html_files)} file(s) to process\n")

        for html_file in html_files:
            process_file(html_file)

        print(f"\n✓ Successfully processed {len(html_files)} file(s)")
        print("- Changed from glassmorphic style to blue gradient style")
        print("- Updated all colors and backgrounds")
        print("- Removed hover animations specific to glassmorphic design")
        print("- Updated scrollbars and interactive elements")

    # Handle single file input
    else:
        output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path

        print(f"Reading from: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        print("Converting format...")

        converted_content = convert_html_to_old_format(html_content)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(converted_content)

        print(f"Successfully converted and saved to: {output_file}")
        print("\nConversion complete!")
        print("- Changed from glassmorphic style to blue gradient style")
        print("- Updated all colors and backgrounds")
        print("- Removed hover animations specific to glassmorphic design")
        print("- Updated scrollbars and interactive elements")


if __name__ == "__main__":
    main()
