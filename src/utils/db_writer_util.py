"""
Database writer utility for job status tracking.
Invokes the db-sql-writer Cloud Run service to execute SQL queries.
Pattern from onboarding-report-job/db_writer_util.py.
"""

import os
import requests
import traceback
from typing import Dict, Any, Optional


DB_WRITER_URL = "https://db-sql-writer-306034828043.asia-south1.run.app"


def _get_id_token_from_metadata() -> Optional[str]:
    """Get ID token from GCP metadata server."""
    try:
        metadata_url = (
            "http://metadata.google.internal/computeMetadata/v1/"
            "instance/service-accounts/default/identity?audience=" + DB_WRITER_URL
        )
        response = requests.get(
            metadata_url,
            headers={"Metadata-Flavor": "Google"},
            timeout=5
        )
        response.raise_for_status()
        return response.text
    except Exception:
        return None


def _get_id_token_from_service_account() -> Optional[str]:
    """Get ID token using service account credentials file."""
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        service_account_path = os.environ.get(
            'GOOGLE_APPLICATION_CREDENTIALS',
            './ecom-review-app-8eb6e1990945.json'
        )

        if not os.path.exists(service_account_path):
            # Try common locations
            for path in ['/app/ecom-review-app-8eb6e1990945.json', './ecom-review-app-8eb6e1990945.json']:
                if os.path.exists(path):
                    service_account_path = path
                    break

        credentials = service_account.IDTokenCredentials.from_service_account_file(
            service_account_path,
            target_audience=DB_WRITER_URL
        )
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        print(f"[db_writer] Service account token failed: {e}")
        return None


def db_writer_sql_invoker(query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Invoke the db-writer-sql Cloud Run service to execute SQL queries.
    Tries metadata server first, then service account file for ID token.
    """
    payload = {"query": query, "params": params}

    # Try metadata server first
    id_token = _get_id_token_from_metadata()

    # Fallback to service account file
    if not id_token:
        print("[db_writer] Metadata server failed, trying service account...")
        id_token = _get_id_token_from_service_account()

    if not id_token:
        print("[db_writer] Could not obtain ID token from any source")
        return None

    try:
        headers = {
            "Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(DB_WRITER_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 401:
            # Token from metadata rejected — retry with service account
            print("[db_writer] Metadata token got 401, retrying with service account...")
            sa_token = _get_id_token_from_service_account()
            if sa_token and sa_token != id_token:
                try:
                    headers["Authorization"] = f"Bearer {sa_token}"
                    response = requests.post(DB_WRITER_URL, headers=headers, json=payload, timeout=15)
                    response.raise_for_status()
                    return response.json()
                except Exception as retry_err:
                    print(f"[db_writer] Retry also failed: {retry_err}")
            else:
                print(f"[db_writer] No alternative token available")
        print(f"[db_writer] Error: {e}")
        return None
    except Exception as e:
        print(f"[db_writer] Error: {e}")
        print(traceback.format_exc())
        return None
