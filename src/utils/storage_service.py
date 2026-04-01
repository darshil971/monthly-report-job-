"""
Google Cloud Storage service for uploading report outputs.
Includes encrypted upload support from admin-backend/bulk_report_uploader.py.
"""

import base64
import datetime
import json
import os
import re
from urllib.parse import quote

from google.cloud import storage
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class CloudStorage:
    def __init__(self, bucket_name: str):
        key_file_path = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "./ecom-review-app-8eb6e1990945.json"
        )
        self.storage_client = storage.Client.from_service_account_json(key_file_path)
        self.bucket_name = bucket_name
        self.bucket = self.storage_client.get_bucket(self.bucket_name)

    def upload_file(self, local_path: str, gcs_path: str) -> str:
        """Upload a local file to GCS. Returns the GCS path."""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"[CloudStorage] Uploaded {local_path} -> gs://{self.bucket_name}/{gcs_path}")
        return gcs_path

    def write_object(self, object_name: str, content: bytes):
        """Upload raw bytes to GCS with auto-detected content type."""
        blob = self.bucket.blob(object_name)
        if object_name.lower().endswith('.pdf'):
            content_type = 'application/pdf'
        elif object_name.lower().endswith(('.html', '.htm')):
            content_type = 'text/html'
        elif object_name.lower().endswith('.json'):
            content_type = 'application/json'
        else:
            content_type = 'application/octet-stream'
        blob.upload_from_string(content, content_type=content_type)

    def generate_signed_url(self, blob_name: str, expiration=datetime.timedelta(days=10)) -> str:
        """Generate a signed URL for temporary access."""
        blob = self.bucket.blob(blob_name)
        return blob.generate_signed_url(expiration=expiration)

    def path_of_object(self, blob_name: str) -> str:
        """Generate a public GCS URL for a blob path."""
        blob = self.bucket.blob(blob_name)
        encoded_path = quote(blob.name)
        return f"https://storage.googleapis.com/{self.bucket_name}/{encoded_path}"


# ============================================================
# Encrypted report uploader (from bulk_report_uploader.py)
# ============================================================

ENCRYPTION_KEY = os.getenv("REPORT_ENCRYPTION_KEY", "russian price - 6000")


def _setup_cipher():
    """Setup Fernet cipher for path encryption."""
    key_bytes = ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b'verifast_salt_2025',
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(key_bytes))
    return Fernet(key)


def _encrypt_path(cipher, path: str) -> str:
    """Encrypt a path string and return URL-safe encoded result."""
    encrypted = cipher.encrypt(path.encode('utf-8'))
    return encrypted.decode('utf-8').replace('/', '_').replace('+', '-').replace('=', '')


def _extract_shopify_url_and_handle(html_text: str):
    """Extract URL from <a class="page-link" ...> and derive Shopify product handle."""
    if not html_text:
        return None, None
    tag_match = re.search(
        r'<a[^>]*class=["\'][^"\']*\bpage-link\b[^"\']*["\'][^>]*>',
        html_text, flags=re.IGNORECASE
    )
    if not tag_match:
        return None, None
    href_match = re.search(
        r'(?:href|data-href)\s*=\s*["\']([^"\']+)["\']',
        tag_match.group(0), flags=re.IGNORECASE
    )
    if not href_match:
        return None, None
    url = href_match.group(1)
    h = re.search(r'/(?:products|collections)/([^/?#]+)', url, flags=re.IGNORECASE)
    return url, (h.group(1) if h else None)


def upload_client_reports(
    storage_service: CloudStorage,
    index_name: str,
    date_path: str,
    html_files: list,
    pdf_path: str = None,
):
    """
    Upload a client's reports (HTML + PDF) to GCS with encrypted filenames.
    Writes a keys_map JSON for the frontend to resolve paths.

    Args:
        storage_service: CloudStorage instance
        index_name: Client shop domain (e.g. zanducare.myshopify.com)
        date_path: YYYY_MM string (e.g. "2026_02")
        html_files: List of dicts [{"path": "/local/path.html", "key": "concern-report"|"page-level-report"}]
        pdf_path: Local path to the monthly PDF report (optional)
    """
    cipher = _setup_cipher()
    html_file_entries = []

    for html_entry in html_files:
        local_path = html_entry.get("path")
        configured_key = html_entry.get("key", "page-level-report")

        if not local_path or not os.path.exists(local_path):
            print(f"    - Skipping: File not found at '{local_path}'")
            continue

        # Read HTML to extract source_url for page-level reports
        source_url = None
        if "page-level-report" in configured_key:
            try:
                with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_url, _ = _extract_shopify_url_and_handle(f.read())
            except Exception:
                pass

        html_filename = os.path.basename(local_path)
        name, ext = os.path.splitext(html_filename)
        unique_name = f"{name}-{datetime.datetime.now().timestamp()}"
        encrypted_name = _encrypt_path(cipher, unique_name)
        destination = f"{index_name}/{date_path}/{encrypted_name}{ext}"

        with open(local_path, 'rb') as f:
            storage_service.write_object(destination, f.read())

        print(f"    - Uploaded HTML: {destination}")
        html_file_entries.append({
            "key": configured_key,
            "source_url": source_url,
            "object_path": destination,
        })

    pdf_file_entry = None
    if pdf_path and os.path.exists(pdf_path):
        pdf_filename = os.path.basename(pdf_path)
        name, ext = os.path.splitext(pdf_filename)
        unique_name = f"{name}-{datetime.datetime.now().timestamp()}"
        encrypted_name = _encrypt_path(cipher, unique_name)
        destination = f"{index_name}/{date_path}/{encrypted_name}{ext}"

        with open(pdf_path, 'rb') as f:
            storage_service.write_object(destination, f.read())

        print(f"    - Uploaded PDF: {destination}")
        pdf_file_entry = {"object_path": destination}

    # Write keys_map JSON
    if html_file_entries or pdf_file_entry:
        keys_payload = {
            "index_name": index_name,
            "date_path": date_path,
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "html_files": html_file_entries,
        }
        if pdf_file_entry:
            keys_payload["pdf_file"] = pdf_file_entry

        keys_bytes = json.dumps(keys_payload, indent=2).encode('utf-8')
        unique_name = f"keys_map-{datetime.datetime.now().timestamp()}"
        encrypted_name = _encrypt_path(cipher, unique_name)
        keys_destination = f"{index_name}/{date_path}/json/{encrypted_name}.json"
        storage_service.write_object(keys_destination, keys_bytes)
        print(f"    - Uploaded keys_map: {keys_destination}")
