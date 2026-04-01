"""
Report builder: orchestrates data collection, analysis, and PDF generation.
Extracted from admin-backend/report.py new_report() function.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.config import MonthlyReportJobConfig
from src.data.raw_data_client import RawDataClient
from src.data.chat_data_processor import (
    process_chat_data,
    build_bubble_dataframe,
    load_data_into_memory,
    get_hourly_distribution,
)
from src.report.analysis import (
    get_top_queries,
    geo_cohort,
    utm_cohort,
    has_insufficient_data_themes,
)


class ReportBuilder:
    """Builds the monthly report JSON and generates PDF for a single client."""

    def __init__(self, config: MonthlyReportJobConfig, data_client: RawDataClient):
        self._config = config
        self._data_client = data_client

    def build_report(
        self,
        client_name: str,
        start_date: str,
        end_date: str,
        json_path: str,
        output_folder: str,
    ) -> Optional[str]:
        """
        End-to-end report generation for a single client.
        Returns path to generated PDF, or None if generation failed.
        """
        is_special_client = (
            client_name.startswith('wa_') or
            client_name.startswith('email_') or
            client_name.startswith('app_')
        )

        out_dir = self._config.new_outputs_dir

        # ================================================================
        # STEP 1: Fetch API data
        # ================================================================
        print(f"\n[ReportBuilder] Fetching data from admin dashboard APIs...")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        com_start_dt = start_dt - timedelta(days=30)
        com_end_dt = start_dt - timedelta(days=1)
        com_start = com_start_dt.strftime("%Y-%m-%d")
        com_end = com_end_dt.strftime("%Y-%m-%d")

        # Sales API
        try:
            sales_data = self._data_client.get_sales_summary(
                client_name, start_date, end_date
            )
        except Exception as e:
            print(f"[ReportBuilder] Sales API error: {e}")
            sales_data = {}

        # Support API
        try:
            support_data = self._data_client.get_support_stats(
                client_name, start_date, end_date
            )
        except Exception as e:
            print(f"[ReportBuilder] Support API error: {e}")
            support_data = {}

        # Internal Feedback API (cached in-memory across clients)
        internal_feedback_data = self._data_client.get_internal_feedback(
            client_name, start_date, end_date
        )

        # ================================================================
        # STEP 2: Fetch raw session data
        # ================================================================
        if not os.path.exists(json_path):
            print(f"[ReportBuilder] Fetching raw session data from analytics...")
            self._data_client.fetch_and_save_session_data(
                client_name, start_dt, end_dt, json_path
            )

        # Load raw data for hourly analysis
        raw_data = load_data_into_memory(json_path)
        hourly_counts = get_hourly_distribution(raw_data) if raw_data else {}

        # ================================================================
        # STEP 3: Process chat data
        # ================================================================
        df = process_chat_data(json_path)
        if df.empty:
            print(f"[ReportBuilder] No chat data found for {client_name}. Cannot generate report.")
            return None

        total_sessions = df['session_id'].nunique()
        total_messages = len(df)

        # ================================================================
        # STEP 4: Bubble analysis
        # ================================================================
        try:
            bubble_df, clean_df = build_bubble_dataframe(df, min_clicks=7, min_words=3)
        except Exception:
            is_special_client = True
            bubble_df = pd.DataFrame(columns=[
                "bubble_message", "clicks", "%_of_total_clicks",
                "avg_click_position", "avg_msgs_after_click",
                "%_did_a2c", "%_placed_order"
            ])
            clean_df = df.copy()

        # ================================================================
        # STEP 5: Cohort analyses
        # ================================================================
        client_id = client_name.split('.')[0]
        start_ddmm = start_dt.strftime("%d%m")
        end_ddmm = end_dt.strftime("%d%m")

        md_geo = os.path.join(out_dir, f"geo_cohort_{client_id}_{start_ddmm}_{end_ddmm}.md")
        md_utm = os.path.join(out_dir, f"utm_cohort_{client_id}_{start_ddmm}_{end_ddmm}.md")

        geo_cohort(df, clean_df, md_path=md_geo)
        _, top_utm_sources = utm_cohort(df, clean_df, md_path=md_utm)

        # ================================================================
        # STEP 6: UTM contribution API
        # ================================================================
        utm_contribution = {}
        if top_utm_sources:
            try:
                utm_contrib = self._data_client.get_utm_source_contribution(
                    client_name, start_date, end_date, com_start, com_end
                )
                current_data = utm_contrib.get("current", [])
                for utm_source in top_utm_sources:
                    source_data = next(
                        (s for s in current_data
                         if s.get("source", "").lower() == utm_source.lower()),
                        None
                    )
                    if source_data:
                        utm_contribution[utm_source] = {
                            "source": source_data.get("source"),
                            "sessions": source_data.get("sessions"),
                            "utm_attributed_order": source_data.get("utm_attributed_order"),
                            "add_to_cart_msg_count": source_data.get("add_to_cart_msg_count"),
                        }
            except Exception as e:
                print(f"[ReportBuilder] UTM contribution API error: {e}")


        # ================================================================
        # STEP 8: AI prompt % calculation
        # ================================================================
        total_bubble_clicks = bubble_df['clicks'].sum() if not bubble_df.empty else 0
        if total_messages > 0:
            ai_prompt_percentage = round((total_bubble_clicks / total_messages) * 100, 2)
            free_text_percentage = round(100 - ai_prompt_percentage, 2)
        else:
            ai_prompt_percentage = 0
            free_text_percentage = 0

        # ================================================================
        # STEP 9: Interaction trend
        # ================================================================
        interaction_trend = {}
        total_sessions_from_api = int(internal_feedback_data.get("number_of_sessions") or 0)
        if hourly_counts and total_sessions_from_api > 0:
            hourly_data = []
            for hour in range(24):
                sessions = hourly_counts.get(hour, 0)
                percentage = round((sessions / total_sessions_from_api * 100), 2)
                hourly_data.append({
                    "hour": int(hour),
                    "hourly_sessions": int(sessions),
                    "percentage": percentage
                })
            sorted_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)
            peak_hours_with_counts = sorted_hours[:3]
            peak_hours = [int(h) for h, _ in peak_hours_with_counts]
            peak_sessions = sum(c for _, c in peak_hours_with_counts)
            peak_percentage = round((peak_sessions / total_sessions_from_api * 100), 2)
            interaction_trend = {
                "hourly_data": hourly_data,
                "peak_chat_hours": peak_hours,
                "peak_percentage": peak_percentage,
            }

        # ================================================================
        # STEP 10: Distribution analyses (usecase, location, UTM)
        # ================================================================
        usecase_distribution = self._build_distribution(
            df, 'secondary_usecase', threshold_pct=0.05, top_n=5, label_key='usecase'
        )
        location_distribution = self._build_distribution(
            df, 'location', threshold_pct=0.02, top_n=3, label_key='location'
        )
        utm_distribution = self._build_distribution(
            df, 'utm_source', threshold_pct=0.01, top_n=3, label_key='utm_source',
            skip_insufficient=False
        )

        # ================================================================
        # STEP 11: Support metrics
        # ================================================================
        total_human_messages_api = int(internal_feedback_data.get("total_human_messages") or 0)
        support_cost_saved_rupees = round(total_human_messages_api * 3.79, 2)
        time_saved_hours = round(total_human_messages_api / 60, 1)

        # ================================================================
        # STEP 12: Build JSON report
        # ================================================================
        order_value_by_utm = internal_feedback_data.get("order_value_by_utm", 0)
        number_of_orders_by_utm = internal_feedback_data.get("number_of_orders_by_utm", 0)
        aov_client = round(order_value_by_utm / number_of_orders_by_utm, 2) if number_of_orders_by_utm > 0 else 0

        report_data = {
            "metadata": {
                "client_name": client_name,
                "client_id": client_id,
                "report_start_date": start_date,
                "report_end_date": end_date,
                "generated_at": datetime.now().isoformat(),
                "is_special_client": is_special_client,
            },
            "interactions": {
                "sessions": int(internal_feedback_data.get("number_of_sessions") or 0),
                "total_human_messages": int(internal_feedback_data.get("total_human_messages") or 0),
                "product_sessions": int(internal_feedback_data.get("product_sessions") or 0),
                "avg_human_messages": round(float(internal_feedback_data.get("avg_human_messages") or 0), 2),
                "ai_prompt_percentage": ai_prompt_percentage,
                "free_text_percentage": free_text_percentage,
                "order_value_by_utm": round(float(order_value_by_utm), 2),
                "number_of_orders_by_utm": int(number_of_orders_by_utm),
                "aov_client": aov_client,
            },
            "sales": {
                "total_orders_assisted": {
                    "value": sales_data.get("total_orders_assisted", {}).get("value", 0),
                    "orders": sales_data.get("total_orders_assisted", {}).get("orders", 0),
                },
                "total_orders_utm": {
                    "value": sales_data.get("total_orders_utm", {}).get("value", 0),
                    "orders": sales_data.get("total_orders_utm", {}).get("orders", 0),
                },
                "add_to_cart_rate": {
                    "value": sales_data.get("add_to_cart_rate", {}).get("value", 0),
                },
                "aov": {
                    "value": sales_data.get("aov", {}).get("value", 0),
                },
            },
            "support": {
                "metrics": {
                    "total_support_queries": int(support_data.get("metrics", {}).get("total_support_queries", 0)),
                    "time_saved_hours": f"{time_saved_hours}h",
                    "avg_human_messages": support_data.get("metrics", {}).get("avg_human_messages", 0),
                    "cost_saved_rupees": support_cost_saved_rupees,
                }
            },
            "interaction_trend": interaction_trend,
            "user_query_distribution": {
                "by_usecase": usecase_distribution,
                "by_location": location_distribution,
                "by_utm_source": utm_distribution,
            },
            "utm_contribution": utm_contribution,
        }

        # Save JSON
        json_filename = f"{client_id}_report_data_{start_ddmm}_{end_ddmm}.json"
        json_save_path = os.path.join(out_dir, json_filename)
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"[ReportBuilder] Report data saved to: {json_save_path}")

        # ================================================================
        # STEP 13: Generate PDF
        # ================================================================
        try:
            from src.report.monthly_pdf import generate_report
            pdf_path = generate_report(json_save_path, output_folder=output_folder)
            print(f"[ReportBuilder] PDF generated: {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"[ReportBuilder] Error generating PDF: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_distribution(
        self,
        df: pd.DataFrame,
        column: str,
        threshold_pct: float,
        top_n: int,
        label_key: str,
        skip_insufficient: bool = True,
    ) -> List[Dict]:
        """Build distribution with GPT theme analysis for a given column."""
        distribution = []
        if column not in df.columns:
            return distribution

        total_msgs = len(df)
        if column == 'location':
            counts_all = df[column].dropna().value_counts()
        elif column == 'utm_source':
            counts_all = df[column].dropna().value_counts()
        else:
            counts_all = df[column].value_counts()

        min_threshold = total_msgs * threshold_pct
        if column == 'secondary_usecase':
            filtered = counts_all[
                (counts_all.index != 'None') & (counts_all >= min_threshold)
            ].head(top_n)
        else:
            filtered = counts_all[counts_all >= min_threshold].head(top_n)

        total_filtered = filtered.sum()

        print(f"[ReportBuilder] Analyzing themes for {len(filtered)} {column} values...")
        for idx, (label, count) in enumerate(filtered.items(), 1):
            percentage = round((count / total_filtered * 100), 2) if total_filtered > 0 else 0
            msgs = df[df[column] == label]['message_text'].astype(str).tolist()
            print(f"  [{idx}/{len(filtered)}] Analyzing {label}: {len(msgs)} messages")

            top_themes_str = get_top_queries(msgs)
            if top_themes_str:
                top_themes_list = [t.strip() for t in top_themes_str.split(',') if t.strip()]
            else:
                top_themes_list = ["Insufficient data for theme extraction"]

            if skip_insufficient and has_insufficient_data_themes(top_themes_list):
                print(f"    Skipping {label} - insufficient data")
                continue

            if not skip_insufficient and has_insufficient_data_themes(top_themes_list):
                top_themes_list = ["Query themes unavailable"]

            distribution.append({
                label_key: label,
                "message_count": int(count),
                "percentage": percentage,
                "top_themes": top_themes_list[:3],
            })

        return distribution
