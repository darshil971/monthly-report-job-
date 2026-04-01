"""
Firebase authentication manager for JWT token generation.
Extracted from admin-backend/report.py FirebaseAuthManager class.
"""

import os
import time
import requests
import firebase_admin
from firebase_admin import credentials, auth

from src.config import MonthlyReportJobConfig


class FirebaseAuthManager:
    """Singleton Firebase authentication manager for JWT token generation."""
    _instance = None

    def __new__(cls, config: MonthlyReportJobConfig = None):
        if cls._instance is None:
            cls._instance = super(FirebaseAuthManager, cls).__new__(cls)
            cls._instance._config = config or MonthlyReportJobConfig()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(self._config.firebase_credentials_path)
            firebase_admin.initialize_app(cred)

    def create_custom_token(self, email: str, client_name: str) -> str:
        """Create Firebase custom token and exchange for JWT ID token."""
        try:
            additional_claims = {
                "email": email,
                "email_verified": True,
                "role": "client_admin",
                "client_name": client_name
            }

            custom_token = auth.create_custom_token(
                uid=self._config.firebase_auth_uid,
                developer_claims=additional_claims
            )
            # firebase-admin 6.x returns str; older versions return bytes
            if isinstance(custom_token, bytes):
                custom_token = custom_token.decode('utf-8')

            url = (
                f"https://identitytoolkit.googleapis.com/v1/"
                f"accounts:signInWithCustomToken?key={self._config.google_identity_api_key}"
            )

            for attempt in range(1, 4):
                try:
                    response = requests.post(url, json={
                        'token': custom_token,
                        'returnSecureToken': True
                    }, timeout=15)
                    response.raise_for_status()
                    return response.json()['idToken']
                except requests.exceptions.RequestException as req_err:
                    print(f"[FirebaseAuthManager] Token exchange failed (attempt {attempt}/3): {req_err}")
                    if attempt < 3:
                        time.sleep(2 * attempt)
            raise Exception("Failed to exchange Firebase token after 3 attempts")

        except Exception as e:
            print(f"[FirebaseAuthManager] Error generating JWT token: {str(e)}")
            raise

