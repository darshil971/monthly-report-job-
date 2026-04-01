"""
Monthly Performance Report Generator
Professional PDF report generation from monthly report JSON data
Replaces Word document generation with modern HTML/PDF workflow
"""

import json
import os
import asyncio
from datetime import datetime
from playwright.async_api import async_playwright


def format_date_range(start_date, end_date):
    """Convert YYYY-MM-DD dates to readable format"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"


def format_currency(value):
    """Format currency values with proper comma separation"""
    return f"₹{int(value):,}"


def format_percentage(value):
    """Format percentage values"""
    return f"{value:.1f}%"


def generate_hourly_chart_svg(hourly_data, peak_hours):
    """Generate inline SVG chart for hourly session distribution"""
    if not hourly_data:
        return ""

    # Find max sessions for scaling
    max_sessions = max([h['hourly_sessions'] for h in hourly_data])

    # Chart dimensions
    width = 760
    height = 200
    padding_left = 50
    padding_right = 20
    padding_top = 20
    padding_bottom = 50

    chart_width = width - padding_left - padding_right
    chart_height = height - padding_top - padding_bottom

    # Calculate bar width and spacing
    bar_width = chart_width / 24

    svg_content = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="{width}" height="{height}" fill="#f9fafb"/>

    <!-- Grid lines -->'''

    # Horizontal grid lines (4 lines)
    for i in range(5):
        y = padding_top + (chart_height * i / 4)
        svg_content += f'''
    <line x1="{padding_left}" y1="{y}" x2="{width - padding_right}" y2="{y}"
          stroke="#e5e7eb" stroke-width="1"/>'''

    # Y-axis labels (session counts)
    for i in range(5):
        y = padding_top + (chart_height * i / 4)
        value = int(max_sessions * (1 - i/4))
        svg_content += f'''
    <text x="{padding_left - 10}" y="{y + 4}" text-anchor="end"
          font-family="sans-serif" font-size="10" fill="#6b7280">{value}</text>'''

    # Bars
    for hour_data in hourly_data:
        hour = hour_data['hour']
        sessions = hour_data['hourly_sessions']

        # Calculate bar height and position
        bar_height = (sessions / max_sessions) * chart_height if max_sessions > 0 else 0
        x = padding_left + (hour * bar_width) + (bar_width * 0.15)
        y = padding_top + chart_height - bar_height
        bar_actual_width = bar_width * 0.7

        # Color: green for top 3 hours, blue for others
        color = "#16a34a" if hour in peak_hours[:3] else "#93c5fd"

        svg_content += f'''
    <rect x="{x}" y="{y}" width="{bar_actual_width}" height="{bar_height}"
          fill="{color}" rx="2"/>'''

    # X-axis labels (hours in 12-hour format)
    for hour in range(0, 24, 3):  # Show every 3rd hour
        x = padding_left + (hour * bar_width) + (bar_width / 2)
        y = height - padding_bottom + 20

        # Convert to 12-hour format
        if hour == 0:
            hour_label = "12 AM"
        elif hour < 12:
            hour_label = f"{hour} AM"
        elif hour == 12:
            hour_label = "12 PM"
        else:
            hour_label = f"{hour - 12} PM"

        svg_content += f'''
    <text x="{x}" y="{y}" text-anchor="middle"
          font-family="sans-serif" font-size="10" fill="#6b7280">{hour_label}</text>'''

    # Axis labels (moved further from axes)
    svg_content += f'''
    <!-- Y-axis label -->
    <text x="10" y="{height/2}" text-anchor="middle"
          font-family="sans-serif" font-size="11" fill="#374151" font-weight="600"
          transform="rotate(-90, 10, {height/2})">Sessions</text>

    <!-- X-axis label -->
    <text x="{width/2}" y="{height - 3}" text-anchor="middle"
          font-family="sans-serif" font-size="11" fill="#374151" font-weight="600">Hour of Day (IST)</text>

</svg>'''

    return svg_content


def generate_report(json_path, output_folder=None):
    """Generate monthly performance report with modern design"""

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load and encode logo
    import base64
    logo_path = os.getenv('VERIFAST_LOGO_PATH', './Verifast_logo_HD.png')
    with open(logo_path, 'rb') as logo_file:
        logo_base64 = base64.b64encode(logo_file.read()).decode('utf-8')

    # Extract data sections
    metadata = data.get('metadata', {})
    interactions = data.get('interactions', {})
    sales = data.get('sales', {})
    support = data.get('support', {}).get('metrics', {})
    interaction_trend = data.get('interaction_trend', {})
    user_query_dist = data.get('user_query_distribution', {})
    utm_contribution = data.get('utm_contribution', {})

    client_name = metadata.get('client_name', 'Unknown')
    client_id = metadata.get('client_id', 'unknown')
    is_special_client = metadata.get('is_special_client', False)
    start_date = metadata.get('report_start_date', '')
    end_date = metadata.get('report_end_date', '')

    date_range = format_date_range(start_date, end_date)

    # Generate output path if not provided
    if output_folder is None:
        output_folder = f"./new_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Calculate date strings for filename
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ddmm = start_dt.strftime("%d%m")
    end_ddmm = end_dt.strftime("%d%m")

    output_path = os.path.join(output_folder, f"{client_id}_monthly_report_{start_ddmm}_{end_ddmm}")

    # Check data availability
    has_location_data = len(user_query_dist.get('by_location', [])) > 0
    has_utm_data = len(user_query_dist.get('by_utm_source', [])) > 0
    has_hourly_data = len(interaction_trend.get('hourly_data', [])) > 0

    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monthly Performance Report</title>
    <style>
        @page {{
            size: A4;
            margin: 12mm;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
            font-size: 11px;
            line-height: 1.5;
            color: #1f2937;
            background: #fff;
        }}

        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding-bottom: 12px;
            border-bottom: 2px solid #111827;
            margin-bottom: 16px;
        }}

        .header-left h1 {{
            font-size: 20px;
            font-weight: 700;
            color: #111827;
            letter-spacing: -0.5px;
            margin-bottom: 6px;
        }}

        .client-name {{
            font-size: 13px;
            color: #6b7280;
            font-weight: 500;
        }}

        .header-right {{
            text-align: right;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 8px;
        }}

        .logo {{
            height: 40px;
            width: auto;
        }}

        .date-range {{
            font-size: 12px;
            font-weight: 600;
            color: #374151;
        }}

        /* Section */
        .section {{
            margin-bottom: 16px;
            page-break-inside: avoid;
        }}

        .page-1-section {{
            page-break-after: avoid;
            margin-bottom: 14px;
        }}

        .section-header {{
            display: flex;
            align-items: center;
            margin-bottom: 14px;
            page-break-after: avoid;
        }}

        .section-number {{
            font-size: 10px;
            font-weight: 700;
            color: #fff;
            background: #111827;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 12px;
        }}

        .section-title {{
            font-size: 14px;
            font-weight: 700;
            color: #111827;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}

        .metrics-grid-2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}

        .metrics-grid-2x2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}

        .metric-card {{
            background: #f9fafb;
            border: 1.5px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 14px;
            page-break-inside: avoid;
        }}

        .metric-label {{
            font-size: 10px;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-bottom: 6px;
        }}

        .metric-value {{
            font-size: 22px;
            font-weight: 700;
            color: #111827;
            line-height: 1.2;
        }}

        .metric-subtext {{
            font-size: 9px;
            color: #9ca3af;
            margin-top: 3px;
        }}

        /* Info Card */
        .info-card {{
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 1.5px solid #93c5fd;
            border-radius: 10px;
            padding: 16px 18px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin-top: 16px;
        }}

        .info-card-text {{
            font-size: 13px;
            color: #1e3a8a;
            line-height: 1.6;
            font-weight: 500;
        }}

        .info-card-icon {{
            font-size: 20px;
            color: #2563eb;
            margin-right: 10px;
            font-weight: 700;
        }}

        .info-card-highlight {{
            font-weight: 700;
            color: #2563eb;
            font-size: 14px;
        }}

        /* Table Container */
        .table-container {{
            background: #f9fafb;
            border: 1.5px solid #e5e7eb;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 16px;
            page-break-inside: avoid;
        }}

        .table-title {{
            font-size: 11px;
            font-weight: 600;
            color: #374151;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid #e5e7eb;
        }}

        /* Cluster Items (VoC Style) */
        .cluster-item {{
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            page-break-inside: avoid;
        }}

        .cluster-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 4px;
        }}

        .cluster-title {{
            font-size: 11px;
            font-weight: 600;
            color: #111827;
            flex: 1;
        }}

        .cluster-percentage {{
            font-size: 10px;
            color: #2563eb;
            font-weight: 600;
            margin-left: 10px;
            background: #dbeafe;
            padding: 3px 8px;
            border-radius: 10px;
        }}

        .cluster-summary {{
            font-size: 10px;
            color: #6b7280;
            line-height: 1.5;
            margin-bottom: 6px;
        }}

        .cluster-messages {{
            background: #f9fafb;
            border-radius: 6px;
            padding: 8px 10px;
        }}

        .cluster-messages-label {{
            font-size: 9px;
            font-weight: 600;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-bottom: 6px;
        }}

        .message-item {{
            font-size: 10px;
            color: #4b5563;
            line-height: 1.4;
            padding: 4px 0;
            padding-left: 10px;
            border-left: 2px solid #16a34a;
            margin-bottom: 4px;
        }}

        .message-item:last-child {{
            margin-bottom: 0;
        }}

        /* Chart Container */
        .chart-container {{
            background: #f9fafb;
            border: 1.5px solid #e5e7eb;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
            text-align: center;
        }}

        .chart-title {{
            font-size: 11px;
            font-weight: 600;
            color: #374151;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 12px;
            text-align: left;
        }}

        /* Page Breaks */
        .force-page-break {{
            page-break-before: always;
        }}

        /* Footer */
        .footer {{
            margin-top: 24px;
            padding-top: 14px;
            border-top: 1px solid #e5e7eb;
        }}

        .footer-note {{
            font-size: 9px;
            color: #9ca3af;
            line-height: 1.6;
        }}

        /* UTM Contribution Metrics */
        .utm-contribution-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}

        .utm-source-card {{
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px;
        }}

        .utm-source-name {{
            font-size: 11px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}

        .utm-metric {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 10px;
        }}

        .utm-metric-label {{
            color: #6b7280;
        }}

        .utm-metric-value {{
            font-weight: 600;
            color: #111827;
        }}
    </style>
</head>
<body>
    <!-- PAGE 1 -->
    <div class="header">
        <div class="header-left">
            <h1>Monthly Performance Report</h1>
        </div>
        <div class="header-right">
            <img src="data:image/png;base64,{logo_base64}" alt="Verifast Logo" class="logo">
            <div class="date-range">{date_range}</div>
        </div>
    </div>

    <!-- SECTION 1: CUSTOMER INTERACTIONS -->
    <div class="section page-1-section">
        <div class="section-header">
            <div class="section-number">1</div>
            <div class="section-title">Customer Interactions</div>
        </div>

        <div class="metrics-grid{'' if not is_special_client else '-2'}">
            <div class="metric-card">
                <div class="metric-label">Total Sessions</div>
                <div class="metric-value">{int(interactions.get('sessions', 0)):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Messages</div>
                <div class="metric-value">{int(interactions.get('total_human_messages', 0)):,}</div>
            </div>"""

    # Conditionally add presales sessions card (only if not special client)
    if not is_special_client:
        product_sessions = int(interactions.get('product_sessions', 0))
        total_sessions = int(interactions.get('sessions', 1))
        presales_percentage = (product_sessions / total_sessions * 100) if total_sessions > 0 else 0

        html_content += f"""
            <div class="metric-card">
                <div class="metric-label">Pre-Sales Sessions</div>
                <div class="metric-value">{presales_percentage:.1f}%</div>
                <div class="metric-subtext">{product_sessions:,} sessions</div>
            </div>"""

    html_content += """
        </div>"""

    # Free-text info card (only if not special client)
    if not is_special_client:
        free_text_pct = interactions.get('free_text_percentage', 0)
        ai_prompt_pct = interactions.get('ai_prompt_percentage', 0)

        html_content += f"""
        <div class="info-card">
            <div class="info-card-text">
                <span class="info-card-icon">ⓘ</span>
                Did you know <span class="info-card-highlight">{free_text_pct:.1f}%</span> of your customers personally type their queries instead of using conversation starters? This demonstrates genuine engagement with your AI sales agent.
            </div>
        </div>"""

    html_content += """
    </div>

    <!-- SECTION 2: CONVERSIONS & REVENUE -->
    <div class="section page-1-section">
        <div class="section-header">
            <div class="section-number">2</div>
            <div class="section-title">Sales Performance</div>
        </div>

        <div class="metrics-grid-2x2">"""

    # Sales metrics
    total_orders_assisted = sales.get('total_orders_assisted', {})
    total_orders_utm = sales.get('total_orders_utm', {})
    add_to_cart_rate = sales.get('add_to_cart_rate', {})
    aov = sales.get('aov', {})

    html_content += f"""
            <div class="metric-card">
                <div class="metric-label">UTM Order Value</div>
                <div class="metric-value">{format_currency(total_orders_utm.get('value', 0))}</div>
                <div class="metric-subtext">{int(total_orders_utm.get('orders', 0)):,} orders</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Assisted Order Value</div>
                <div class="metric-value">{format_currency(total_orders_assisted.get('value', 0))}</div>
                <div class="metric-subtext">{int(total_orders_assisted.get('orders', 0)):,} orders</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Order Value</div>
                <div class="metric-value">{format_currency(aov.get('value', 0))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Add to Carts</div>
                <div class="metric-value">{int(add_to_cart_rate.get('value', 0)):,}</div>
            </div>
        </div>
    </div>

    <!-- SECTION 3: CUSTOMER SUPPORT -->
    <div class="section page-1-section">
        <div class="section-header">
            <div class="section-number">3</div>
            <div class="section-title">Customer Support</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Support Queries</div>
                <div class="metric-value">{int(support.get('total_support_queries', 0)):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Time Saved</div>
                <div class="metric-value">{support.get('time_saved_hours', '0.0h')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cost Saved</div>
                <div class="metric-value">{format_currency(support.get('cost_saved_rupees', 0))}</div>
            </div>
        </div>"""

    # Calculate after-hours percentage from hourly data
    # Office hours: 10 AM - 7 PM (hours 10-18), After hours: before 10 AM or after 7 PM (hours 0-9, 19-23)
    has_after_hours_data = False
    after_hours_percentage = 0

    if has_hourly_data:
        hourly_data_temp = interaction_trend.get('hourly_data', [])

        if hourly_data_temp:
            # Calculate total from hourly data (not from interactions.sessions)
            total_hourly_sessions = sum(h['hourly_sessions'] for h in hourly_data_temp)

            # Sum sessions from after-hours (0-9 and 19-23)
            after_hours_sessions = sum(
                h['hourly_sessions'] for h in hourly_data_temp
                if h['hour'] < 10 or h['hour'] >= 19
            )

            if after_hours_sessions > 0 and total_hourly_sessions > 0:
                after_hours_percentage = (after_hours_sessions / total_hourly_sessions * 100)
                has_after_hours_data = True

    # Add after-hours info card if we have data
    if has_after_hours_data:
        html_content += f"""
        <div class="info-card">
            <div class="info-card-text">
                <span class="info-card-icon">ⓘ</span>
                <span class="info-card-highlight">{after_hours_percentage:.1f}%</span>
                of customers chat with your sales agent <em style="font-style: italic;">after</em> office hours
            </div>
        </div>"""

    html_content += """
    </div>"""

    # SECTION 4: HOURLY INTERACTION PATTERN (Hourly Chart)
    if has_hourly_data:
        hourly_data = interaction_trend.get('hourly_data', [])
        peak_hours = interaction_trend.get('peak_chat_hours', [])
        peak_percentage = interaction_trend.get('peak_percentage', 0)

        # Generate SVG chart
        chart_svg = generate_hourly_chart_svg(hourly_data, peak_hours)

        html_content += f"""
    <!-- SECTION 4: HOURLY INTERACTION PATTERN -->
    <div class="section">
        <div class="section-header">
            <div class="section-number">4</div>
            <div class="section-title">Hourly Interaction Pattern</div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Hourly Session Distribution</div>
            {chart_svg}
            <div style="margin-top: 12px; font-size: 10px; color: #6b7280; text-align: left;">
                <strong style="color: #111827;">Peak Hours:</strong> {', '.join([f'{h}:00' for h in peak_hours[:3]])}
                ({peak_percentage:.1f}% of total sessions)
            </div>
        </div>
    </div>"""

    # SECTION 5: USER QUERY DISTRIBUTION
    html_content += """
    <!-- PAGE 2: User Query Distribution -->
    <div class="force-page-break"></div>

    <div class="section">
        <div class="section-header">
            <div class="section-number">5</div>
            <div class="section-title">User Query Distribution</div>
        </div>"""

    # 5.1: By Use Case
    usecase_data = user_query_dist.get('by_usecase', [])
    if usecase_data:
        html_content += """
        <div class="table-container">
            <div class="table-title">By Use Case</div>"""

        for idx, usecase_item in enumerate(usecase_data, 1):
            usecase_name = usecase_item.get('usecase', 'Unknown')
            message_count = usecase_item.get('message_count', 0)
            percentage = usecase_item.get('percentage', 0)
            top_themes = usecase_item.get('top_themes', [])

            # Format themes as quoted string
            themes_str = ', '.join([f'"{theme.strip()}"' for theme in top_themes if theme.strip()])

            html_content += f"""
            <div class="cluster-item">
                <div class="cluster-header">
                    <div class="cluster-title">{idx}. {usecase_name}</div>
                    <div class="cluster-percentage">{percentage:.1f}%</div>
                </div>
                <div class="cluster-messages">
                    <div class="cluster-messages-label">Top Themes</div>"""

            for theme in top_themes[:3]:
                if theme.strip():
                    html_content += f"""
                    <div class="message-item">{theme.strip()}</div>"""

            html_content += """
                </div>
            </div>"""

        html_content += """
        </div>"""

    # 5.2: By Location (only if data available)
    if has_location_data:
        location_data = user_query_dist.get('by_location', [])

        html_content += """
        <div class="table-container">
            <div class="table-title">By Location</div>"""

        for idx, location_item in enumerate(location_data, 1):
            location_name = location_item.get('location', 'Unknown')
            message_count = location_item.get('message_count', 0)
            percentage = location_item.get('percentage', 0)
            top_themes = location_item.get('top_themes', [])

            themes_str = ', '.join([f'"{theme.strip()}"' for theme in top_themes if theme.strip()])

            html_content += f"""
            <div class="cluster-item">
                <div class="cluster-header">
                    <div class="cluster-title">{idx}. {location_name}</div>
                    <div class="cluster-percentage">{percentage:.1f}%</div>
                </div>
                <div class="cluster-messages">
                    <div class="cluster-messages-label">Top Themes</div>"""

            for theme in top_themes[:3]:
                if theme.strip():
                    html_content += f"""
                    <div class="message-item">{theme.strip()}</div>"""

            html_content += """
                </div>
            </div>"""

        html_content += """
        </div>"""

    # 5.3: By UTM Source (only if data available)
    if has_utm_data:
        utm_data = user_query_dist.get('by_utm_source', [])

        html_content += """
        <div class="table-container">
            <div class="table-title">By UTM Source</div>"""

        for idx, utm_item in enumerate(utm_data, 1):
            utm_name = utm_item.get('utm_source', 'Unknown')
            message_count = utm_item.get('message_count', 0)
            percentage = utm_item.get('percentage', 0)
            top_themes = utm_item.get('top_themes', [])

            themes_str = ', '.join([f'"{theme.strip()}"' for theme in top_themes if theme.strip()])

            html_content += f"""
            <div class="cluster-item">
                <div class="cluster-header">
                    <div class="cluster-title">{idx}. {utm_name}</div>
                </div>
                <div class="cluster-messages">
                    <div class="cluster-messages-label">Top Themes</div>"""

            for theme in top_themes[:3]:
                if theme.strip():
                    html_content += f"""
                    <div class="message-item">{theme.strip()}</div>"""

            html_content += """
                </div>"""

            # Add UTM contribution metrics if available for this source
            if utm_contribution and utm_name in utm_contribution:
                contrib_data = utm_contribution[utm_name]
                sessions = contrib_data.get('sessions', 0)
                utm_orders = contrib_data.get('utm_attributed_order', {})
                a2c_data = contrib_data.get('add_to_cart_msg_count', {})

                html_content += f"""
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #e5e7eb;">
                    <div style="font-size: 9px; color: #6b7280; margin-bottom: 4px; font-weight: 600;">Performance Metrics</div>
                    <div style="display: flex; gap: 12px; flex-wrap: wrap; font-size: 9px;">
                        <div><span style="color: #6b7280;">Sessions Identified:</span> <strong>{sessions}</strong></div>
                        <div><span style="color: #6b7280;">Orders:</span> <strong>{utm_orders.get('count', 0)}</strong> <span style="color: #9ca3af;">({utm_orders.get('percentage', 0):.1f}%)</span></div>
                        <div><span style="color: #6b7280;">Add to Cart:</span> <strong>{a2c_data.get('count', 0)}</strong> <span style="color: #9ca3af;">({a2c_data.get('percentage', 0):.1f}%)</span></div>
                    </div>
                </div>"""

            html_content += """
            </div>"""

        html_content += """
        </div>"""

    html_content += """
</body>
</html>
"""

    # Save HTML
    html_path = os.path.abspath(f"{output_path}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Generate PDF using Playwright
    pdf_path = os.path.abspath(f"{output_path}.pdf")
    asyncio.run(generate_pdf_from_html(html_path, pdf_path))

    # Delete HTML file after PDF generation
    if os.path.exists(html_path):
        os.remove(html_path)

    print(f"\n✅ Monthly report PDF generated!")
    print(f"   Client: {client_name}")
    print(f"   Period: {date_range}")
    print(f"   PDF: {pdf_path}")

    return pdf_path


async def generate_pdf_from_html(html_path, pdf_path):
    """Generate PDF from HTML file using Playwright"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f'file://{html_path}')
        await page.pdf(path=pdf_path, format='A4', print_background=True)
        await browser.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        json_path = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) >= 3 else None
    else:
        # Default for testing
        json_path = 'new_outputs/palaknots_report_data_0111_0311.json'
        output_folder = None

    generate_report(json_path, output_folder)
