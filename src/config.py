"""
Configuration management for the monthly report job.
All hardcoded values from the original pipeline are centralized here
and loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MonthlyReportJobConfig:
    """Configuration for the monthly report pipeline."""

    # Firebase Authentication
    firebase_credentials_path: str = os.getenv(
        "FIREBASE_ADMIN_CREDENTIALS_PATH",
        "./ecom-review-app-8eb6e1990945.json"
    )
    firebase_auth_uid: str = os.getenv(
        "FIREBASE_AUTH_UID",
        "ThpJw2tctVZEjW5OQpBTwQrznBp1"
    )
    firebase_auth_email: str = os.getenv(
        "FIREBASE_AUTH_EMAIL",
        "tanmay@verifast.tech"
    )
    google_identity_api_key: str = os.getenv(
        "GOOGLE_IDENTITY_API_KEY",
        "AIzaSyAUfm7T8PhzJ7qSdLyExZxo_0NHUXzwUGo"
    )

    # Azure OpenAI GPT-4.1
    azure_gpt41_api_key: str = os.getenv(
        "AZURE_GPT41_API_KEY",
        "8268d8ed232440be940c3f0e4f8b05d3"
    )
    azure_gpt41_endpoint: str = os.getenv(
        "AZURE_GPT41_ENDPOINT",
        "https://openai-sweden-central-deployment.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
    )

    # Azure OpenAI GPT-4o
    azure_gpt4o_api_key: str = os.getenv(
        "AZURE_GPT4O_API_KEY",
        "17121590f68d4b16b3f70af0e1a8b326"
    )
    azure_gpt4o_endpoint: str = os.getenv(
        "AZURE_GPT4O_ENDPOINT",
        "https://verifast-east-us.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview"
    )

    # Azure OpenAI Embeddings
    azure_embedding_api_key: str = os.getenv(
        "AZURE_EMBEDDING_API_KEY",
        "f339ca0a17e943df9f2ed92be64dadcb"
    )
    azure_embedding_endpoint: str = os.getenv(
        "AZURE_EMBEDDING_ENDPOINT",
        "https://openai-sweden-central-deployment.openai.azure.com"
    )
    azure_embedding_deployment: str = os.getenv(
        "AZURE_EMBEDDING_DEPLOYMENT",
        "text-embedding-3-large"
    )

    # API URLs
    admin_backend_base_url: str = os.getenv(
        "ADMIN_BACKEND_BASE_URL",
        "https://admin-backend.verifast.ai"
    )
    analytics_base_url: str = os.getenv(
        "ANALYTICS_BASE_URL",
        "https://admin-analytics.verifast.ai"
    )

    # Verifast Logo
    verifast_logo_path: str = os.getenv(
        "VERIFAST_LOGO_PATH",
        "./Verifast_logo_HD.png"
    )

    # Google Cloud Storage
    storage_bucket_name: str = os.getenv(
        "STORAGE_BUCKET_NAME",
        "monthly_reports"
    )
    storage_gcs_prefix: str = os.getenv(
        "STORAGE_GCS_PREFIX",
        "monthly_report"
    )

    # Slack
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")

    # Temp working directory (cleaned between clients)
    tmp_dir: str = "tmp"
    vec_outs_dir: str = "tmp/vec_outs"
    concern_reports_dir: str = "tmp/concern_reports"
    voc_reports_dir: str = "tmp/voc_reports"
    new_outputs_dir: str = "tmp/new_outputs"

    @property
    def dashboard_api_url(self) -> str:
        return f"{self.admin_backend_base_url}/api/admin/dashboard"

    @property
    def internal_feedback_url(self) -> str:
        return f"{self.admin_backend_base_url}/internal/feedback"

    @property
    def analytics_fetch_url(self) -> str:
        return f"{self.analytics_base_url}/nlp/fetch_data_of_index_by_date"

    def setup_client_tmp(self, client_name: str):
        """Create fresh tmp dirs for a client. Cleans previous client data."""
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        for d in [self.tmp_dir, self.vec_outs_dir, self.concern_reports_dir,
                  self.voc_reports_dir, self.new_outputs_dir]:
            os.makedirs(d, exist_ok=True)

    def get_client_json_path(self, client_name: str) -> str:
        """Path for raw session JSON (only file that must be on disk)."""
        return os.path.join(self.tmp_dir, f"{client_name}.json")
