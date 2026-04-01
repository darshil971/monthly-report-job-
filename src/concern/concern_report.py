#!/usr/bin/env python3
"""
User Concern Distribution Report Generator
Generates an interactive HTML report from concern clustering JSON data
Filters for pre-sales concerns only (significant clusters: sessions > 10 and % > 2%)
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime


def parse_filename(filename):
    """
    Extract client name and date range from filename.

    Supports two formats:
    1. Standard: concern_clusters_<clientname>_<startdate DDMM>_<enddate DDMM>_sessions.json
    2. Non-standard: concern_clusters_<clientname>_sessions.json (uses default date range)
    """
    # Remove path and extension
    basename = Path(filename).stem

    # Helper function for ordinal suffix
    def get_ordinal(day):
        day = int(day)
        if 10 <= day % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        return f"{day}{suffix}"

    # Try standard pattern first: concern_clusters_<clientname>_<DDMM>_<DDMM>_sessions
    pattern_with_dates = r'concern_clusters_(.+?)_(\d{4})_(\d{4})_sessions'
    match = re.search(pattern_with_dates, basename)

    if match:
        # Standard format with dates
        client_name = match.group(1)
        start_date = match.group(2)
        end_date = match.group(3)

        # Format dates
        start_day = start_date[:2]
        start_month_num = int(start_date[2:])
        end_day = end_date[:2]
        end_month_num = int(end_date[2:])

        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]

        start_month = months[start_month_num - 1]
        end_month = months[end_month_num - 1]

        date_range = f"{get_ordinal(start_day)} {start_month} - {get_ordinal(end_day)} {end_month}"
    else:
        # Non-standard format without dates: extract client name between clusters_ and _sessions
        pattern_without_dates = r'concern_clusters_(.+?)_sessions'
        match = re.search(pattern_without_dates, basename)

        if not match:
            raise ValueError(f"Filename does not match expected pattern: {filename}")

        client_name = match.group(1)
        date_range = ""

    # Format client name (capitalize words, replace underscores/hyphens)
    client_display = client_name.replace('_', ' ').replace('-', ' ').title()

    return client_display, date_range


def load_and_filter_data(filepath):
    """
    Load JSON data and filter for pre-sales concerns only
    Applies additional filters: sessions > 10 and % of total > 2%
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter clusters for pre-sales only
    presales_clusters = [
        cluster for cluster in data['clusters']
        if cluster.get('sales_stage') == 'pre-sales'
        and cluster.get('cluster_title').lower() not in ['other concern', 'other concerns', 'other issues']
    ]

    if not presales_clusters:
        raise ValueError("No pre-sales concerns found in the data")

    # Calculate total sessions across ALL pre-sales concerns (before additional filtering)
    total_presales_sessions = sum(
        cluster['metadata']['unique_session_count']
        for cluster in presales_clusters
    )

    # Apply additional filters: sessions > 10 AND % of total > 2%
    filtered_presales_clusters = []
    for cluster in presales_clusters:
        sessions = cluster['metadata']['unique_session_count']
        percentage = (sessions / total_presales_sessions * 100) if total_presales_sessions > 0 else 0

        if sessions > 10 and percentage > 2.0:
            filtered_presales_clusters.append(cluster)

    if not filtered_presales_clusters:
        raise ValueError("No pre-sales concerns found matching the filters (sessions > 10 and % > 2%)")

    # Calculate totals for FILTERED pre-sales clusters
    filtered_total_sessions = sum(
        cluster['metadata']['unique_session_count']
        for cluster in filtered_presales_clusters
    )

    filtered_total_orders = sum(
        cluster['metadata']['order_sessions_count']
        for cluster in filtered_presales_clusters
    )

    # FIXED: Calculate avg conversion as the mean of individual cluster conversion percentages
    # This gives equal weight to each concern type rather than being volume-weighted
    if filtered_presales_clusters:
        cluster_conversions = [
            cluster['metadata']['order_sessions_percentage']
            for cluster in filtered_presales_clusters
        ]
        avg_conversion = sum(cluster_conversions) / len(cluster_conversions)
    else:
        avg_conversion = 0

    return {
        'metadata': data['metadata'],
        'clusters': filtered_presales_clusters,
        'total_presales_sessions': filtered_total_sessions,
        'total_presales_orders': filtered_total_orders,
        'avg_conversion': avg_conversion,
        'num_presales_clusters': len(filtered_presales_clusters)
    }


def generate_html_report(data, date_range):
    """
    Generate HTML report with embedded CSS and JavaScript
    """
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pre-Sales Concern Distribution</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        body {{
            font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #D2D9E2;
            background-attachment: fixed;
            padding: 20px;
            min-height: 100vh;
            color: #3C3C3E;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        /* Title Card */
        .title-card {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 40px;
        }}

        .title-card:hover {{
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
            transform: translateY(-1px);
        }}

        .title-card-left {{
            flex: 1;
            min-width: 0;
        }}

        .title-card h1 {{
            font-size: 28px;
            font-weight: 700;
            color: #3C3C3E;
            margin-bottom: 12px;
            line-height: 1.2;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}

        .title-card .date-info {{
            font-size: 14px;
            font-weight: 600;
            color: #3C3C3E;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 6px 12px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.3);
            display: inline-block;
            transition: all 0.3s ease;
        }}

        .title-card .date-info:hover {{
            background: rgba(255,255,255,0.25);
            transform: scale(1.02);
        }}

        .title-card-metrics {{
            display: flex;
            align-items: center;
            gap: 24px;
            flex-shrink: 0;
        }}

        .metric-item {{
            text-align: center;
            min-width: 100px;
        }}

        .metric-label {{
            font-size: 11px;
            color: #3C3C3E;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}

        .metric-value {{
            font-size: 24px;
            font-weight: 800;
            color: #3C3C3E;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            line-height: 1;
            margin-bottom: 2px;
        }}

        .metric-sublabel {{
            font-size: 10px;
            color: #3C3C3E;
            font-weight: 500;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .metric-separator {{
            width: 2px;
            height: 60px;
            background: rgba(255, 255, 255, 0.3);
            flex-shrink: 0;
        }}


        /* Table Container */
        .table-container {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            overflow-x: auto;
            transition: all 0.3s ease;
        }}

        .table-container:hover {{
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }}

        .table-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        }}

        .table-header h2 {{
            font-size: 18px;
            font-weight: 700;
            color: #3C3C3E;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .table-info {{
            font-size: 14px;
            color: #3C3C3E;
            font-weight: 500;
        }}

        /* Table */
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14px;
        }}

        thead th {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            padding: 16px 20px;
            text-align: center;
            font-weight: 700;
            font-size: 12px;
            color: #3C3C3E;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            cursor: pointer;
            user-select: none;
            position: relative;
            transition: all 0.2s ease;
        }}

        thead th:first-child {{
            text-align: left;
        }}

        thead th:hover {{
            background: rgba(255, 255, 255, 0.25);
        }}

        thead th.sortable::after {{
            content: '⇅';
            position: absolute;
            right: 8px;
            opacity: 0.3;
            font-size: 0.9em;
        }}

        thead th.sort-asc::after {{
            content: '↑';
            opacity: 1;
            color: #8FADE1;
        }}

        thead th.sort-desc::after {{
            content: '↓';
            opacity: 1;
            color: #8FADE1;
        }}

        tbody tr {{
            transition: all 0.2s ease;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}

        tbody tr:nth-child(even) {{
            background: rgba(255, 255, 255, 0.05);
        }}

        tbody tr:hover {{
            background: rgba(255, 255, 255, 0.15);
        }}

        tbody tr:nth-child(even):hover {{
            background: rgba(255, 255, 255, 0.15);
        }}

        tbody td {{
            padding: 14px 20px;
            color: #3C3C3E;
            font-weight: 500;
            text-align: center;
        }}

        tbody td:first-child {{
            text-align: left;
        }}

        tbody td.concern {{
            font-weight: 600;
            color: #3C3C3E;
        }}

        .comparison {{
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
        }}

        .comparison.positive {{
            color: #9FC63B;
        }}

        .comparison.negative {{
            color: #F01263;
        }}

        .comparison.neutral {{
            color: #3C3C3E;
            opacity: 0.7;
        }}

        .arrow {{
            font-size: 0.9em;
        }}

        /* No data message */
        .no-data {{
            text-align: center;
            padding: 60px 20px;
            color: #3C3C3E;
            font-size: 1.1em;
            opacity: 0.7;
        }}


        /* Scrollbar styling for glassy effect */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}

        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }}

        ::-webkit-scrollbar-thumb {{
            background: rgba(143, 173, 225, 0.4);
            border-radius: 10px;
            transition: all 0.3s ease;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(143, 173, 225, 0.6);
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            body {{
                padding: 16px;
            }}

            .title-card {{
                flex-direction: column;
                align-items: flex-start;
                padding: 20px;
                gap: 20px;
            }}

            .title-card h1 {{
                font-size: 24px;
            }}

            .title-card-metrics {{
                width: 100%;
                justify-content: space-around;
                gap: 12px;
            }}

            .metric-separator {{
                height: 50px;
            }}

            .metric-item {{
                min-width: auto;
            }}

            .metric-value {{
                font-size: 20px;
            }}

            .table-container {{
                padding: 20px;
            }}

            thead th {{
                padding: 12px 10px;
                font-size: 10px;
            }}

            tbody td {{
                padding: 12px 10px;
                font-size: 13px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Title Card -->
        <div class="title-card">
            <div class="title-card-left">
                <h1>Pre-Sales Concern Distribution</h1>
                <div class="date-info">Date Range: {date_range}</div>
            </div>
            <div class="title-card-metrics">
                <div class="metric-item">
                    <div class="metric-label">Concerns</div>
                    <div class="metric-value">{data['num_presales_clusters']}</div>
                    <div class="metric-sublabel">Captured</div>
                </div>
                <div class="metric-separator"></div>
                <div class="metric-item">
                    <div class="metric-label">Sessions</div>
                    <div class="metric-value">{data['total_presales_sessions']}</div>
                    <div class="metric-sublabel">With Concerns</div>
                </div>
                <div class="metric-separator"></div>
                <div class="metric-item">
                    <div class="metric-label">Conversion</div>
                    <div class="metric-value">{data['avg_conversion']:.1f}%</div>
                    <div class="metric-sublabel">Average</div>
                </div>
            </div>
        </div>

        <!-- Table -->
        <div class="table-container">
            <div class="table-header">
                <h2>Concern Breakdown</h2>
                <div class="table-info">Click column headers to sort</div>
            </div>

            <table id="concernTable">
                <thead>
                    <tr>
                        <th class="sortable" data-column="concern">Concern</th>
                        <th class="sortable" data-column="sessions">Sessions With User Concerns</th>
                        <th class="sortable" data-column="percentage">% of Sessions With User Concerns</th>
                        <th class="sortable" data-column="orders">Sessions with Orders</th>
                        <th class="sortable" data-column="conversion">Conversion %</th>
                        <th class="sortable" data-column="comparison">Comparison to Avg</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <!-- Populated by JavaScript -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Data from Python
        const data = {json.dumps({
            'clusters': [
                {
                    'concern': cluster['cluster_title'],
                    'sessions': cluster['metadata']['unique_session_count'],
                    'orders': cluster['metadata']['order_sessions_count'],
                    'conversion': cluster['metadata']['order_sessions_percentage']
                }
                for cluster in data['clusters']
            ],
            'total_sessions': data['total_presales_sessions'],
            'avg_conversion': data['avg_conversion']
        })};

        let sortColumn = null;
        let sortDirection = 'asc';

        function renderTable(sortBy = null) {{
            const tbody = document.getElementById('tableBody');
            let tableData = [...data.clusters];

            // Calculate percentages
            tableData = tableData.map(row => ({{
                ...row,
                percentage: (row.sessions / data.total_sessions * 100),
                comparison: row.conversion - data.avg_conversion
            }}));

            // Sort if needed
            if (sortBy) {{
                tableData.sort((a, b) => {{
                    let aVal = a[sortBy];
                    let bVal = b[sortBy];

                    // Handle string comparison for concern
                    if (sortBy === 'concern') {{
                        aVal = aVal.toLowerCase();
                        bVal = bVal.toLowerCase();
                    }}

                    if (sortDirection === 'asc') {{
                        return aVal > bVal ? 1 : -1;
                    }} else {{
                        return aVal < bVal ? 1 : -1;
                    }}
                }});
            }}

            // Render rows
            tbody.innerHTML = tableData.map(row => {{
                const comparisonClass = row.comparison > 0 ? 'positive' :
                                       row.comparison < 0 ? 'negative' : 'neutral';
                const arrow = row.comparison > 0 ? '↑' :
                             row.comparison < 0 ? '↓' : '−';
                const sign = row.comparison > 0 ? '+' : '';

                return `
                    <tr>
                        <td class="concern">${{row.concern}}</td>
                        <td>${{row.sessions}}</td>
                        <td>${{row.percentage.toFixed(1)}}%</td>
                        <td>${{row.orders}}</td>
                        <td>${{row.conversion.toFixed(1)}}%</td>
                        <td>
                            <span class="comparison ${{comparisonClass}}">
                                <span class="arrow">${{arrow}}</span>
                                ${{sign}}${{Math.abs(row.comparison).toFixed(1)}}%
                            </span>
                        </td>
                    </tr>
                `;
            }}).join('');
        }}

        // Add sorting functionality
        document.querySelectorAll('thead th.sortable').forEach(th => {{
            th.addEventListener('click', () => {{
                const column = th.dataset.column;

                // Toggle direction if same column, else reset to asc
                if (sortColumn === column) {{
                    sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
                }} else {{
                    sortColumn = column;
                    sortDirection = 'asc';
                }}

                // Update UI
                document.querySelectorAll('thead th').forEach(header => {{
                    header.classList.remove('sort-asc', 'sort-desc');
                }});
                th.classList.add(sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');

                // Re-render
                renderTable(column);
            }});
        }});

        // Initial render
        renderTable();
    </script>
</body>
</html>"""

    return html_template


def serve_report(html_content, client_name, date_range, port=8000):
    """
    Serve HTML report on local server and open in browser

    Args:
        html_content: HTML content to write
        client_name: Client name extracted from filename
        date_range: Date range in format DDMM_DDMM
        port: Port for server (unused, kept for backward compatibility)
    """
    # Create concern_reports directory if it doesn't exist
    reports_dir = 'tmp/concern_reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Generate filename: concern_report_<clientname>_<date_range>.html
    filename = f'concern_report_{client_name}_{date_range}.html'
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ Report generated successfully!")
    print(f"📁 Report saved to: {os.path.abspath(filepath)}")


def gen_concern_report(filepath):
    """
    Main function to generate and display report
    """
    try:
        print(f"📂 Loading data from: {filepath}")

        # Parse filename for display
        client_name_display, date_range_display = parse_filename(filepath)
        print(f"📅 Date Range: {date_range_display}")

        # Extract raw client_name and date_range for filename
        basename = Path(filepath).stem
        # Try pattern with dates first
        pattern_with_dates = r'concern_clusters_(.+?)_(\d{4})_(\d{4})_sessions'
        match = re.search(pattern_with_dates, basename)
        if match:
            client_name_raw = match.group(1)
            date_range_raw = f"{match.group(2)}_{match.group(3)}"
        else:
            # Try pattern without dates
            pattern_without_dates = r'concern_clusters_(.+?)_sessions'
            match = re.search(pattern_without_dates, basename)
            if match:
                client_name_raw = match.group(1)
                date_range_raw = "0111_3011"  # Default: 01 Oct to 31 Oct
            else:
                # Fallback to base name if pattern doesn't match
                client_name_raw = basename.replace('concern_clusters_', '').replace('_sessions', '')
                date_range_raw = "0111_3011"

        # Load and filter data
        print(f"🔍 Filtering for significant pre-sales concerns (sessions > 10 and % > 2%)...")
        data = load_and_filter_data(filepath)
        print(f"✓ Found {data['num_presales_clusters']} significant pre-sales concerns")
        print(f"✓ Total pre-sales sessions: {data['total_presales_sessions']}")
        print(f"✓ Average conversion: {data['avg_conversion']:.1f}%")

        # Generate report
        print(f"🎨 Generating HTML report...")
        html_content = generate_html_report(data, date_range_display)

        # Serve and open report
        serve_report(html_content, client_name_raw, date_range_raw)

    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Example usage
    filepath = "/home/anirudh/admin-backend/vec_outs/concern_clusters_dr-vaidya.myshopify.com_sessions.json"

    # You can change this to your actual file path
    # filepath = input("Enter the path to your JSON file: ").strip()

    gen_concern_report(filepath)
