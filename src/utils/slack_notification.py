"""
Slack notification utility for job status updates.
"""

import os
import requests
from typing import Optional


def send_slack_notification(
    message: str,
    webhook_url: Optional[str] = None
):
    """Send a notification to Slack via webhook."""
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
    if not url:
        print("[slack] No webhook URL configured, skipping notification")
        return

    try:
        response = requests.post(url, json={"text": message}, timeout=10)
        if response.status_code == 200:
            print(f"[slack] Notification sent successfully")
        else:
            print(f"[slack] Failed to send notification: {response.status_code}")
    except Exception as e:
        print(f"[slack] Error sending notification: {e}")
