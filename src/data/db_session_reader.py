"""
Direct DB reader for session data — replaces the slow API-based day-by-day fetch.
Same day-by-day approach to avoid DB overload, but skips HTTP round-trips.
Output format matches fetch_index_data_for_date_range / to_dict() exactly.

Queries copied from admin-backend/data_layer_migration.py:
  - ChatHistory:        line 860   — SELECT * → [id, created_at, actor, text, index_name, session]
  - SessionMetadata:    line 2658  — SELECT session_id, session_number, index_name, bot_page, session_user_state
  - UseCase:            line 3779  — SELECT id, created_at, index_name, session_id, use_case_name, primary_usecase, secondary_usecase
  - UserField:          line 1627  — SELECT id, created_at, index_name, session_id, key_field, val_field
  - ChatLinkInteractionEvents: line 3400 — SELECT * → [id, created_at, link, session_id, index_name, event]
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict
from urllib.parse import quote

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


class DBSessionReader:
    """Reads chat session data directly from the DB, day by day."""

    def __init__(self, host: str, user: str, password: str, database: str, port: int = 3306):
        encoded_pw = quote(password)
        conn_str = f"mysql+pymysql://{user}:{encoded_pw}@{host}:{port}/{database}"
        self.engine = create_engine(conn_str, poolclass=NullPool, echo=False)

    @classmethod
    def from_env(cls) -> "DBSessionReader":
        return cls(
            host=os.getenv("DB_READ_HOST", "10.72.224.201"),
            user=os.getenv("DB_READ_USER", "dev"),
            password=os.getenv("DB_READ_PASSWORD", "Info@1234"),
            database=os.getenv("DB_READ_DATABASE", "app-backend-db"),
            port=int(os.getenv("DB_READ_PORT", "3306")),
        )

    def _execute(self, query: str, params: dict) -> list:
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            return list(result.fetchall())

    def _fetch_day(self, index_name: str, day_start: datetime, day_end: datetime) -> dict:
        """Fetch all session data for a single day. Returns {session_id: {...}} dict."""

        # 1. Get session IDs from ChatHistory
        # Same as data_layer_migration.py line 2661
        session_rows = self._execute(
            "SELECT DISTINCT session FROM ChatHistory "
            "WHERE index_name = :index_name AND created_at BETWEEN :start AND :end",
            {"index_name": index_name, "start": day_start, "end": day_end}
        )
        session_ids = [r[0] for r in session_rows]

        if not session_ids:
            return {}

        s_params = {f"s_{i}": sid for i, sid in enumerate(session_ids)}
        s_ph = ", ".join(f":s_{i}" for i in range(len(session_ids)))
        base = {"index_name": index_name, **s_params}

        # 2. ChatHistory — exact query from data_layer_migration.py line 860
        #    SELECT * FROM ChatHistory → columns: id, created_at, actor, text, index_name, session
        #    ChatMessage.to_dict() = asdict() with created_at.isoformat()
        msg_rows = self._execute(
            f"SELECT * FROM ChatHistory "
            f"WHERE session IN ({s_ph}) AND index_name = :index_name "
            f"AND text NOT LIKE :img_data_pattern ORDER BY created_at ASC",
            {**base, "img_data_pattern": "HACKY_IMG_DATA%"}
        )
        msg_dict: Dict[str, list] = {}
        for r in msg_rows:
            sid = r[5]  # session
            if sid not in msg_dict:
                msg_dict[sid] = []
            msg_dict[sid].append({
                "id": r[0],
                "created_at": r[1].isoformat() if r[1] else None,
                "actor": r[2],
                "text": r[3],
                "index_name": r[4],
                "session": r[5],
            })

        # 3. SessionMetadata — exact query from data_layer_migration.py line 2658
        #    SELECT sm.session_id, sm.session_number, sm.index_name, sm.bot_page, sm.session_user_state
        #    SessionMetadata.to_dict() = asdict() → {session_id, index_name, bot_page, session_number, session_user_state}
        meta_rows = self._execute(
            f"SELECT session_id, session_number, index_name, bot_page, session_user_state "
            f"FROM SessionMetadata "
            f"WHERE index_name = :index_name AND session_id IN ({s_ph})",
            base
        )
        meta_dict: Dict[str, list] = {}
        for r in meta_rows:
            sid = r[0]  # session_id
            if sid not in meta_dict:
                meta_dict[sid] = []
            meta_dict[sid].append({
                "session_id": r[0],
                "index_name": r[2],
                "bot_page": r[3],
                "session_number": r[1],
                "session_user_state": r[4],
            })

        # 4. UseCase — exact query from data_layer_migration.py line 3779
        #    SELECT id, created_at, index_name, session_id, use_case_name, primary_usecase, secondary_usecase
        #    UseCaseEntity.to_dict() = asdict() with created_at.isoformat()
        #    → {index_name, session_id, use_case, created_at, primary_usecase, secondary_usecase}
        uc_rows = self._execute(
            f"SELECT id, created_at, index_name, session_id, use_case_name, primary_usecase, secondary_usecase "
            f"FROM UseCase "
            f"WHERE index_name = :index_name AND session_id IN ({s_ph})",
            base
        )
        uc_dict: Dict[str, list] = {}
        for r in uc_rows:
            sid = r[3]  # session_id
            if sid not in uc_dict:
                uc_dict[sid] = []
            uc_dict[sid].append({
                "index_name": r[2],
                "session_id": r[3],
                "use_case": r[4],
                "created_at": r[1].isoformat() if r[1] else None,
                "primary_usecase": r[5],
                "secondary_usecase": r[6],
            })

        # 5. UserField — exact query from data_layer_migration.py line 1627
        #    SELECT DISTINCT uf.id, uf.created_at, uf.index_name, uf.session_id, uf.key_field, uf.val_field
        #    UserField.to_dict() = asdict() with created_at.isoformat()
        #    → {index_name, session_id, key_field, val_field, id, session_number(None), created_at}
        uf_rows = self._execute(
            f"SELECT DISTINCT id, created_at, index_name, session_id, key_field, val_field "
            f"FROM UserField "
            f"WHERE index_name = :index_name AND session_id IN ({s_ph})",
            base
        )
        uf_dict: Dict[str, list] = {}
        for r in uf_rows:
            sid = r[3]  # session_id
            if sid not in uf_dict:
                uf_dict[sid] = []
            uf_dict[sid].append({
                "index_name": r[2],
                "session_id": r[3],
                "key_field": r[4],
                "val_field": r[5],
                "id": r[0],
                "session_number": None,
                "created_at": r[1].isoformat() if r[1] else None,
            })

        # 6. ChatLinkInteractionEvents — exact query from data_layer_migration.py line 3400
        #    SELECT * → columns: id, created_at, link, session_id, index_name, event
        #    But fetch_index_data_for_date_range only keeps: {created_at, link}
        link_rows = self._execute(
            f"SELECT * FROM ChatLinkInteractionEvents "
            f"WHERE index_name = :index_name AND session_id IN ({s_ph}) ORDER BY created_at ASC",
            base
        )
        link_dict: Dict[str, list] = {}
        for r in link_rows:
            sid = r[3]  # session_id
            if sid not in link_dict:
                link_dict[sid] = []
            link_dict[sid].append({"created_at": str(r[1]), "link": r[2]})

        # 7. Build per-session dict — exact same keys as fetch_index_data_for_date_range
        #    (admin_screen_service.py line 1414-1422)
        day_data = {}
        for sid in session_ids:
            uf_list = uf_dict.get(sid)

            # shopify_data_for_session: filtered from user_fields where key_field == "shopify_order_details"
            shopify_data = [f for f in (uf_list or []) if f.get("key_field") == "shopify_order_details"] or None
            # product_added_to_cart: filtered from user_fields where key_field == "product_added_to_cart"
            pac_data = [f for f in (uf_list or []) if f.get("key_field") == "product_added_to_cart"] or None

            day_data[sid] = {
                "chat": msg_dict.get(sid),
                "use_cases": uc_dict.get(sid),
                "user_field": uf_list,
                "metadata_for_session": meta_dict.get(sid),
                "each_session_link_interaction": link_dict.get(sid),
                "shopify_data_for_session": shopify_data,
                "product_added_to_cart": pac_data,
            }

        return day_data

    def fetch_and_save_session_data(
        self, index_name: str, start_date: datetime, end_date: datetime, json_path: str
    ):
        """Fetch session data day-by-day from DB and save to JSON."""
        if os.path.exists(json_path):
            print(f"[DBSessionReader] Data file already exists: {json_path}")
            return

        combined_data: dict = {}
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_delta = timedelta(days=1)

        while current <= end_date:
            day_start = current
            day_end = current.replace(hour=23, minute=59, second=59, microsecond=999999)

            try:
                daily = self._fetch_day(index_name, day_start, day_end)
                print(f"[DBSessionReader] {day_start.date()}: {len(daily)} sessions")
                combined_data.update(daily)
            except Exception as e:
                print(f"[DBSessionReader] Error fetching {day_start.date()}: {e}")

            current += day_delta

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump([combined_data], f, indent=4)
        print(f"[DBSessionReader] Saved {len(combined_data)} total sessions to {json_path}")
