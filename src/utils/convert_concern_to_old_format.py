#!/usr/bin/env python3
"""
Convert concern HTML files from glassmorphic format to old blue gradient format.

Usage:
    python convert_concern_to_old_format.py old.html revised.html
    python convert_concern_to_old_format.py input.html  # overwrites input.html
"""

import re
import sys
from pathlib import Path


def convert_concern_html_to_old_format(html_content):
    """Convert concern report HTML from glassmorphic format to old blue gradient format."""

    # CSS replacements - mapping glassmorphic styles to old blue gradient styles
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

        # Container
        (
            r'\.container\s*\{[^}]*\}',
            '''.container {
            max-width: 1400px;
            margin: 0 auto;
        }'''
        ),

        # Title card (main header)
        (
            r'\.title-card\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.15\);[^}]*\}',
            '''.title-card {
            background: linear-gradient(135deg, #2563EB 0%, #1d4ed8 100%);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 40px;
        }'''
        ),

        # Remove title-card:hover
        (r'\.title-card:hover\s*\{[^}]*\}', ''),

        # Title card h1
        (
            r'\.title-card h1\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.title-card h1 {
            font-size: 28px;
            font-weight: 700;
            color: white;
            margin: 0 0 8px 0;
            line-height: 1.2;
        }'''
        ),

        # Date info
        (
            r'\.title-card \.date-info\s*\{[^}]*background:\s*rgba\(255,255,255,0\.2\);[^}]*\}',
            '''.title-card .date-info {
            font-size: 14px;
            font-weight: 600;
            color: white;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 6px 12px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.3);
            display: inline-block;
            margin-top: 8px;
        }'''
        ),

        # Remove date-info:hover
        (r'\.title-card \.date-info:hover\s*\{[^}]*\}', ''),

        # Title card left
        (
            r'\.title-card-left\s*\{[^}]*\}',
            '''.title-card-left {
            flex: 1;
            min-width: 0;
        }'''
        ),

        # Title card metrics
        (
            r'\.title-card-metrics\s*\{[^}]*\}',
            '''.title-card-metrics {
            display: flex;
            align-items: center;
            gap: 20px;
        }'''
        ),

        # Metric item
        (
            r'\.metric-item\s*\{[^}]*\}',
            '''.metric-item {
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
            padding: 12px 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            text-align: center;
            min-width: 100px;
        }'''
        ),

        # Metric label
        (
            r'\.metric-label\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.metric-label {
            font-size: 11px;
            color: #4a5568;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }'''
        ),

        # Metric value
        (
            r'\.metric-value\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.metric-value {
            font-size: 24px;
            font-weight: 800;
            color: #2c3e50;
        }'''
        ),

        # Metric sublabel
        (
            r'\.metric-sublabel\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.metric-sublabel {
            font-size: 10px;
            color: #4a5568;
            font-weight: 500;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }'''
        ),

        # Metric separator
        (
            r'\.metric-separator\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.3\);[^}]*\}',
            '''.metric-separator {
            width: 2px;
            height: 60px;
            background: rgba(255, 255, 255, 0.3);
            flex-shrink: 0;
        }'''
        ),

        # Table container
        (
            r'\.table-container\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.15\);[^}]*\}',
            '''.table-container {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
            overflow-x: auto;
        }'''
        ),

        # Remove table-container:hover
        (r'\.table-container:hover\s*\{[^}]*\}', ''),

        # Table header
        (
            r'\.table-header\s*\{[^}]*\}',
            '''.table-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 2px solid #e9ecef;
        }'''
        ),

        # Table header h2
        (
            r'\.table-header h2\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.table-header h2 {
            font-size: 18px;
            font-weight: 700;
            color: #2c3e50;
        }'''
        ),

        # Table info
        (
            r'\.table-info\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''.table-info {
            font-size: 14px;
            color: #4a5568;
            font-weight: 500;
        }'''
        ),

        # Table thead th
        (
            r'thead th\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.15\);[^}]*\}',
            '''thead th {
            background: #f8f9fa;
            padding: 16px 20px;
            text-align: center;
            font-weight: 700;
            font-size: 12px;
            color: #2c3e50;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #e9ecef;
            cursor: pointer;
            user-select: none;
            position: relative;
            transition: all 0.2s ease;
        }'''
        ),

        # Remove thead th:hover
        (r'thead th:hover\s*\{[^}]*\}', ''),

        # thead th:first-child
        (
            r'thead th:first-child\s*\{[^}]*\}',
            '''thead th:first-child {
            text-align: left;
        }'''
        ),

        # Sorting indicators
        (
            r'thead th\.sortable::after\s*\{[^}]*\}',
            '''thead th.sortable::after {
            content: '⇅';
            position: absolute;
            right: 8px;
            opacity: 0.3;
            font-size: 0.9em;
        }'''
        ),

        (
            r'thead th\.sort-asc::after\s*\{[^}]*\}',
            '''thead th.sort-asc::after {
            content: '↑';
            opacity: 1;
            color: #3b82f6;
        }'''
        ),

        (
            r'thead th\.sort-desc::after\s*\{[^}]*\}',
            '''thead th.sort-desc::after {
            content: '↓';
            opacity: 1;
            color: #3b82f6;
        }'''
        ),

        # Table element
        (
            r'table\s*\{[^}]*\}',
            '''table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14px;
        }'''
        ),

        # tbody tr
        (
            r'tbody tr\s*\{[^}]*\}',
            '''tbody tr {
            transition: all 0.2s ease;
            border-bottom: 1px solid #f8f9fa;
        }'''
        ),

        # tbody tr:hover
        (
            r'tbody tr:hover\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.1[0-9]*\);[^}]*\}',
            '''tbody tr:hover {
            background: #f8f9fa;
        }'''
        ),

        # tbody tr:nth-child(even)
        (
            r'tbody tr:nth-child\(even\)\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.05\);[^}]*\}',
            '''tbody tr:nth-child(even) {
            background: #fafbfc;
        }'''
        ),

        # Remove tbody tr:nth-child(even):hover
        (r'tbody tr:nth-child\(even\):hover\s*\{[^}]*\}', ''),

        # tbody td
        (
            r'tbody td\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''tbody td {
            padding: 14px 20px;
            color: #2c3e50;
            font-weight: 500;
            text-align: center;
        }'''
        ),

        # tbody td:first-child
        (
            r'tbody td:first-child\s*\{[^}]*\}',
            '''tbody td:first-child {
            text-align: left;
        }'''
        ),

        # tbody td.concern
        (
            r'tbody td\.concern\s*\{[^}]*color:\s*#3C3C3E;[^}]*\}',
            '''tbody td.concern {
            font-weight: 600;
            color: #2c3e50;
        }'''
        ),

        # Comparison styles
        (
            r'\.comparison\s*\{[^}]*\}',
            '''.comparison {
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
        }'''
        ),

        (
            r'\.comparison\.positive\s*\{[^}]*\}',
            '''.comparison.positive {
            color: #9FC63B;
        }'''
        ),

        (
            r'\.comparison\.negative\s*\{[^}]*\}',
            '''.comparison.negative {
            color: #F01263;
        }'''
        ),

        (
            r'\.comparison\.neutral\s*\{[^}]*\}',
            '''.comparison.neutral {
            color: #3C3C3E;
            opacity: 0.7;
        }'''
        ),

        (
            r'\.arrow\s*\{[^}]*\}',
            '''.arrow {
            font-size: 0.9em;
        }'''
        ),

        # No data message
        (
            r'\.no-data\s*\{[^}]*\}',
            '''.no-data {
            text-align: center;
            padding: 60px 20px;
            color: #3C3C3E;
            font-size: 1.1em;
            opacity: 0.7;
        }'''
        ),

        # Scrollbar styling
        (
            r'::-webkit-scrollbar\s*\{[^}]*\}',
            '''::-webkit-scrollbar {
            width: 8px;
        }'''
        ),

        (
            r'::-webkit-scrollbar-track\s*\{[^}]*\}',
            '''::-webkit-scrollbar-track {
            background: #f8f9fa;
            border-radius: 10px;
        }'''
        ),

        (
            r'::-webkit-scrollbar-thumb\s*\{[^}]*background:\s*rgba\(255,\s*255,\s*255,\s*0\.3\);[^}]*\}',
            '''::-webkit-scrollbar-thumb {
            background: #cbd5e0;
            border-radius: 10px;
            transition: all 0.3s ease;
        }'''
        ),

        (
            r'::-webkit-scrollbar-thumb:hover\s*\{[^}]*\}',
            '''::-webkit-scrollbar-thumb:hover {
            background: #a0aec0;
        }'''
        ),
    ]

    # Apply all CSS replacements
    modified_content = html_content
    for pattern, replacement in css_replacements:
        modified_content = re.sub(pattern, replacement, modified_content, flags=re.DOTALL)

    return modified_content


def process_file(file_path):
    """Process a single HTML file."""
    print(f"Processing: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    converted_content = convert_concern_html_to_old_format(html_content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(converted_content)

    print(f"✓ Converted: {file_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_concern_to_old_format.py <input.html|folder_path> [output.html]")
        print("\nFor single file: Converts input.html to output.html (or overwrites if no output specified)")
        print("For folder: Recursively finds and converts all concern_*.html files in-place")
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

        print(f"Searching for concern_*.html files in: {input_path}")
        html_files = list(input_path.rglob("concern_*.html"))

        if not html_files:
            print("No concern_*.html files found.")
            sys.exit(0)

        print(f"Found {len(html_files)} file(s) to process\n")

        for html_file in html_files:
            process_file(html_file)

        print(f"\n✓ Successfully processed {len(html_files)} file(s)")
        print("- Changed from glassmorphic style to blue gradient style")
        print("- Updated all colors and backgrounds")
        print("- Removed hover animations specific to glassmorphic design")
        print("- Updated scrollbars and interactive elements")
        print("- All concern data and text preserved unchanged")

    # Handle single file input
    else:
        output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path

        print(f"Reading from: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        print("Converting concern report format...")

        converted_content = convert_concern_html_to_old_format(html_content)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(converted_content)

        print(f"Successfully converted and saved to: {output_file}")
        print("\nConversion complete!")
        print("- Changed from glassmorphic style to blue gradient style")
        print("- Updated all colors and backgrounds")
        print("- Removed hover animations specific to glassmorphic design")
        print("- Updated scrollbars and interactive elements")
        print("- All concern data and text preserved unchanged")


if __name__ == "__main__":
    main()
