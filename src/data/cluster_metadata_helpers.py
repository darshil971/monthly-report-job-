"""
Helper function extracted from admin-backend/cluster_metadata.py.
Loads chat data from JSON file supporting both daily and report formats.
"""

import json
from typing import Dict


def load_chat_data(chat_data_file_path: str, report: bool = False) -> Dict[str, Dict]:
    """
    Load chat data from JSON file. Supports both daily format and report format.

    Args:
        chat_data_file_path: Path to the chat data JSON file
        report: If True, use report format (array with 'chat' field),
                else use daily format (dict with 'messages' field)

    Returns:
        Dictionary mapping session_id to session data with normalized field names.
    """
    print(f"[CLUSTER_METADATA] Loading chat data from: {chat_data_file_path}")
    print(f"[CLUSTER_METADATA] Format: {'Report' if report else 'Daily'}")

    with open(chat_data_file_path, 'r') as f:
        raw_data = json.load(f)

    if report:
        chat_data = {}

        # Check for single-dict format (all sessions in one dict)
        if isinstance(raw_data, list) and len(raw_data) == 1 and isinstance(raw_data[0], dict):
            first_dict = raw_data[0]
            if len(first_dict) > 10:
                print(f"[CLUSTER_METADATA] Detected single-dict report format")
                raw_data = [
                    {session_id: session_data}
                    for session_id, session_data in first_dict.items()
                ]

        for session_obj in raw_data:
            for session_id, session_data in session_obj.items():
                # Convert 'chat' to 'messages'
                if 'chat' in session_data:
                    session_data['messages'] = session_data.pop('chat')

                # Convert 'user_field' (singular) to 'user_fields' (plural)
                if 'user_field' in session_data:
                    user_field = session_data.pop('user_field')
                    if user_field:
                        normalized_user_fields = []
                        for field in user_field:
                            normalized_field = {}
                            if 'key_field' in field:
                                normalized_field['key'] = field['key_field']
                            if 'val_field' in field:
                                normalized_field['value'] = field['val_field']
                            for k, v in field.items():
                                if k not in ['key_field', 'val_field']:
                                    normalized_field[k] = v
                            normalized_user_fields.append(normalized_field)
                        session_data['user_fields'] = normalized_user_fields
                    else:
                        session_data['user_fields'] = []

                # Convert 'use_cases' to 'usecases'
                if 'use_cases' in session_data:
                    session_data['usecases'] = session_data.pop('use_cases')

                chat_data[session_id] = session_data

        print(f"[CLUSTER_METADATA] Loaded {len(chat_data)} sessions from report format")
    else:
        chat_data = raw_data
        print(f"[CLUSTER_METADATA] Loaded {len(chat_data)} sessions from daily format")

    return chat_data
