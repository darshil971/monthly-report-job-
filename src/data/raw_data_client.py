"""
Raw data client for fetching data from admin-backend and analytics APIs.
Extracted from admin-backend/report.py API wrapper functions.
No file-based caching — production Cloud Run job runs fresh each time.
"""

import json
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.config import MonthlyReportJobConfig


class RawDataClient:
    """Handles all API data fetching."""

    def __init__(self, config: MonthlyReportJobConfig, jwt_token: str, auth_manager=None):
        self._config = config
        self._jwt_token = jwt_token
        self._auth_manager = auth_manager
        self._internal_feedback_cache: Dict[str, list] = {}

    def _refresh_token(self):
        """Refresh JWT token using the auth manager."""
        if not self._auth_manager:
            print("[RawDataClient] No auth manager available — cannot refresh token")
            return False
        try:
            self._jwt_token = self._auth_manager.create_custom_token(
                self._config.firebase_auth_email, "monthly-report-job"
            )
            print("[RawDataClient] JWT token refreshed successfully")
            return True
        except Exception as e:
            print(f"[RawDataClient] Token refresh failed: {e}")
            return False

    def _make_api_request(
        self, payload: dict, endpoint: str = None, url: str = None, max_retries: int = 3
    ) -> dict:
        """Make POST request to admin dashboard API with retry."""
        import time
        base_url = self._config.dashboard_api_url
        if not url and endpoint:
            url = f"{base_url}{endpoint}"

        for attempt in range(1, max_retries + 1):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._jwt_token}"
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                if response.status_code == 401 and self._refresh_token():
                    # Retry immediately with the new token
                    headers["Authorization"] = f"Bearer {self._jwt_token}"
                    response = requests.post(url, headers=headers, json=payload, timeout=180)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"[RawDataClient] API request failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
        return {}

    def get_sales_summary(self, client_name: str, start_date: str, end_date: str) -> dict:
        """Fetch sales summary. Note: end_date + 1 day for complete data."""
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        api_end_date = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        payload = {
            "index_name": client_name,
            "startDate": start_date,
            "endDate": api_end_date
        }
        return self._make_api_request(payload, "/sales")

    def get_support_stats(self, client_name: str, start_date: str, end_date: str) -> dict:
        """Fetch support statistics. Note: end_date + 1 day for complete data."""
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        api_end_date = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        payload = {
            "index_name": client_name,
            "startDate": start_date,
            "endDate": api_end_date
        }
        return self._make_api_request(payload, "/support")

    def get_internal_feedback(self, client_name: str, start_date: str, end_date: str) -> dict:
        """Fetch internal feedback data (returns all clients, filter needed).
        Cached in-memory per process run to avoid duplicate calls for 150+ clients."""
        cache_key = f"{start_date}_{end_date}"

        if cache_key not in self._internal_feedback_cache:
            payload = {
                "index_name": "all",
                "start_date": start_date,
                "end_date": end_date,
                "is_admin": True
            }
            all_feedback = self._make_api_request(
                payload=payload,
                url=self._config.internal_feedback_url,
            )
            self._internal_feedback_cache[cache_key] = all_feedback
            print(f"[RawDataClient] Internal feedback fetched and cached in memory")
        else:
            print(f"[RawDataClient] Using in-memory cached internal feedback")

        # Filter for this client
        all_feedback = self._internal_feedback_cache[cache_key]
        if isinstance(all_feedback, list):
            for client_data in all_feedback:
                if client_data.get("index_name", "").lower() == client_name.lower():
                    return client_data

        print(f"[RawDataClient] No internal feedback data found for {client_name}")
        return {}

    def get_utm_source_contribution(
        self,
        client_name: str,
        start_date: str,
        end_date: str,
        com_start: str,
        com_end: str,
    ) -> dict:
        """Fetch UTM source contribution data with comparison period."""
        payload = {
            "index_name": client_name,
            "startDate": start_date,
            "endDate": end_date,
            "comStartDate": com_start,
            "comEndDate": com_end
        }
        return self._make_api_request(payload, "/sales/utm-source-contribution")

    def fetch_and_save_session_data(
        self, client_name: str, start_date: datetime, end_date: datetime, json_path: str
    ):
        """Fetch raw session data day-by-day from analytics server and save to JSON.
        This file IS needed on disk because concern_cluster + theme_clustering read from it."""
        if os.path.exists(json_path):
            print(f"[RawDataClient] Data file already exists: {json_path}")
            return

        combined_data = []
        current_date = start_date
        day_delta = timedelta(days=1)

        while current_date <= end_date:
            day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = current_date.replace(hour=23, minute=59, second=59, microsecond=999999)

            daily_data = self._fetch_from_analytics(client_name, day_start, day_end)
            print(f"[RawDataClient] Got data for {day_start.date()}")

            if daily_data:
                if len(combined_data) == 0:
                    combined_data.append(daily_data)
                else:
                    combined_data[0].update(daily_data)

            current_date += day_delta

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as fp:
            json.dump(combined_data, fp, indent=4)
        print(f"[RawDataClient] Data saved to {json_path}")

    def _fetch_from_analytics(
        self, index_name: str, day_start: datetime, day_end: datetime,
        max_retries: int = 3
    ) -> dict:
        """Fetch session data from analytics server for a single day with retry."""
        import time
        headers = {'Content-Type': 'application/json'}
        payload = {
            'index_name': index_name,
            'start_date': day_start.isoformat(),
            'end_date': day_end.isoformat(),
        }

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    self._config.analytics_fetch_url,
                    headers=headers,
                    json=payload,
                    timeout=180
                )
                if resp.status_code == 200:
                    return resp.json().get('data', {})
                else:
                    print(f"[RawDataClient] Analytics fetch failed (status={resp.status_code})")
            except Exception as e:
                print(f"[RawDataClient] Analytics fetch error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
        return {}
