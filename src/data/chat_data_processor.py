"""
Chat data processing: transforms raw session JSON into analysis-ready DataFrame.
Extracted from admin-backend/report.py — process_chat_data, build_bubble_dataframe,
and helper functions.
"""

import json
import os
import pandas as pd
import pytz
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.timezone("UTC")

# Use case mapping: raw secondary_usecase -> user-friendly name
SECONDARY_USECASE_MAPPING = {
    "Product Info": "Product details",
    "Contact Details": "Customer care details",
    "Seeking Solution": "Product to solve specific issue",
    "Order Status": "Order status",
    "Unsure About Effectiveness": "Product efficacy doubts",
    "Differentiation": "Comparison with another brand",
    "Place Order": "Place an order",
    "Product Ingredients": "Product ingredients",
    "Delivery Time": "Delivery timeline",
    "Customer Information": "Personal information",
    "Pricing": "Pricing",
    "Cancellation": "Order cancellation",
    "Payment": "Payment process",
    "Return/Refund Policy": "Return or refund policy",
    "Address Change": "Change delivery address",
    "Wrong/Damaged Order": "Wrong or damaged order",
    "Certification": "Product certification",
    "Discount": "Discounts or offers",
    "Talk to Agent": "Speak to a human agent",
    "E-commerce/Quick Commerce": "Availability on other platforms",
    "Comparison": "Comparison between products",
    "Out of Stock": "Product availability",
    "Feedback": "Feedback or complaint",
    "B2B Queries": "Bulk or distribution order",
    "About Team": "Company or team information",
    "Benefits": "Product benefits",
    "Book Consultation": "Book a consultation",
    "Combined Usage": "Using products together",
    "Competitor": "Competitor products",
    "Customer Product Requirement": "Product needs or requirements",
    "Durability and Quality": "Durability or quality",
    "Energy Efficiency": "Energy efficiency",
    "Flavor Options": "Flavor options",
    "Greetings": "Greeting or courtesy",
    "How to Use": "Product usage",
    "Longevity": "Product longevity",
    "Material and Finish": "Material or finish",
    "No Query": "No specific query",
    "Nutrition Info": "Nutrition information",
    "Pairing Info": "Product pairing",
    "Physical Store": "Physical store locations",
    "Product Features": "Product features",
    "Product Installation": "Product installation",
    "Product Not Working": "Product not working",
    "Product Suitability": "Suitability for specific needs or skin type",
    "Safety Features": "Safety features",
    "Shelf Life Storage": "Shelf life or storage",
    "Similar Product": "Similar products",
    "Sizing Chart": "Sizing or size chart",
    "Space and Size Fit": "Space or size compatibility",
    "Specialty": "Specialty products",
    "Warranty": "Warranty information",
    "Website Issue": "Website issue",
}


def convert_utc_to_ist(utc_timestamp: str) -> Optional[str]:
    """Convert a UTC ISO-8601 timestamp to IST formatted string."""
    try:
        utc_dt = datetime.fromisoformat(
            utc_timestamp.replace("Z", "+00:00")
        ).replace(tzinfo=UTC)
        ist_dt = utc_dt.astimezone(IST)
        return ist_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        print(f"[convert_utc_to_ist] Error for {utc_timestamp}: {exc}")
        return None


def extract_utm_data(user_state_str: str) -> Dict[str, Optional[str]]:
    """Return utm_source / utm_medium / utm_campaign values (if any)."""
    default = {"utm_source": None, "utm_medium": None, "utm_campaign": None}
    try:
        if user_state_str and user_state_str != "{}":
            user_state = json.loads(user_state_str)
            for k in default:
                if k in user_state:
                    val = user_state[k]
                    default[k] = val[0] if isinstance(val, list) and val else val
    except Exception as exc:
        print(f"[extract_utm_data] Parse error: {exc}")
    return default


def extract_scroll_percentage(scroll_position_str: str) -> Optional[int]:
    """Grab scrollPercentage value from scroll_position JSON."""
    try:
        if scroll_position_str:
            return json.loads(scroll_position_str).get("scrollPercentage")
    except Exception:
        pass
    return None


def extract_visited_days_count(visited_days_str: str) -> Optional[int]:
    """Return length of visited_days JSON array."""
    try:
        if visited_days_str:
            visited_days = json.loads(visited_days_str)
            if isinstance(visited_days, list):
                return len(visited_days)
    except Exception:
        pass
    return None


def check_verifast_order(shopify_order_str: str) -> Optional[int]:
    """Detect Verifast UTM flag inside Shopify order details."""
    try:
        if not shopify_order_str:
            return 0
        order_data = json.loads(shopify_order_str)
        if isinstance(order_data, str):
            order_data = json.loads(order_data)
        if not isinstance(order_data, dict):
            return 0
        return 1 if order_data.get("has_verifast_utm") else 0
    except Exception:
        return 0


def load_data_into_memory(json_path: str) -> dict:
    """Load combined session data from JSON file, filter out empty sessions."""
    try:
        with open(json_path, 'r') as handle:
            combined_data = json.load(handle)
            if isinstance(combined_data, list) and len(combined_data) > 0:
                data = combined_data[0]
            else:
                data = {}

        sessions_to_remove = [
            sid for sid, sval in data.items()
            if len(sval.get('chat', [])) == 0
        ]
        for s in sessions_to_remove:
            del data[s]

        print(f"[load_data_into_memory] Loaded {len(data)} sessions")
        return data
    except Exception as e:
        print(f"[load_data_into_memory] Error: {e}")
        return {}


def get_hourly_distribution(data: dict) -> dict:
    """Calculate hourly distribution of sessions (IST). Returns hour -> count dict."""
    ist = pytz.timezone("Asia/Kolkata")
    session_hours = defaultdict(set)

    for session_id, session_data in data.items():
        for message in session_data.get('chat', []):
            if message.get('actor') == 'customer' and message.get('created_at'):
                dt_utc = datetime.fromisoformat(message['created_at'])
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                dt_tz = dt_utc.astimezone(ist)
                session_hours[session_id].add(dt_tz.hour)

    hour_session_count = defaultdict(int)
    for hours in session_hours.values():
        for hr in hours:
            hour_session_count[hr] += 1

    return dict(hour_session_count)


def process_chat_data(json_file_path: str) -> pd.DataFrame:
    """
    Transform raw chat-session JSON into analysis-ready DataFrame.
    One row per customer message with session metadata, usecases, etc.
    """
    print(f"[process_chat_data] Loading: {json_file_path}")
    with open(json_file_path, "r", encoding="utf-8") as fh:
        raw_data = json.load(fh)
    if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict):
        raw_data = raw_data[0]

    if not raw_data or isinstance(raw_data, list):
        print(f"[process_chat_data] No session data found or invalid format")
        return pd.DataFrame()

    print(f"[process_chat_data] Sessions found: {len(raw_data)}")

    records = []
    for s_idx, (session_id, s_data) in enumerate(raw_data.items(), start=1):
        if s_idx % 1000 == 0:
            print(f"[process_chat_data] ...{s_idx} sessions parsed")

        messages = s_data.get("chat") or []
        usecases = s_data.get("use_cases") or []
        user_fields = s_data.get("user_field") or []
        user_attrs = s_data.get("user_attributes") or []
        metadata = s_data.get("metadata_for_session") or []

        bot_page = metadata[0].get("bot_page") if metadata else None
        utm = extract_utm_data(
            metadata[0].get("session_user_state", "{}") if metadata else "{}"
        )

        # Extract location
        location = next(
            (uf["val_field"] for uf in user_fields
             if uf.get("key_field", "").lower() == "location"),
            None
        )
        if location is not None and str(location).strip().lower() == "unknown":
            location = None

        # a2c / orders
        has_a2c = any(uf.get("key_field") == "product_added_to_cart" for uf in user_fields)
        order_placed = any(uf.get("key_field") == "shopify_order_details" for uf in user_fields)
        verifast_flag = 0
        for uf in user_fields:
            if uf.get("key_field") == "shopify_order_details":
                verifast_flag = check_verifast_order(uf.get("val_field", ""))
                break

        # Build time-indexed lookups
        attr_by_time = defaultdict(dict)
        for attr in user_attrs:
            attr_by_time[attr["created_at"]][attr["key"]] = attr.get("value")

        usecase_by_time = {}
        for uc in usecases:
            try:
                uc_json = json.loads(uc.get("use_case", "{}"))
                usecase_by_time[uc["created_at"]] = {
                    "primary": uc_json.get("primary"),
                    "secondary": uc_json.get("secondary")
                }
            except Exception:
                continue

        # Iterate customer messages
        customer_msg_counter = 0
        for idx, msg in enumerate(messages):
            if msg.get("actor") != "customer":
                continue
            customer_msg_counter += 1
            ts = msg.get("created_at")
            record = {
                "session_id": session_id,
                "message_text": msg.get("text", ""),
                "message_timestamp_ist": convert_utc_to_ist(ts),
                "msg_number_in_session": customer_msg_counter,
                "bot_page": bot_page,
                "location": location,
                "utm_source": utm.get("utm_source"),
                "utm_medium": utm.get("utm_medium"),
                "utm_campaign": utm.get("utm_campaign"),
                "did_a2c": int(has_a2c),
                "order_placed": int(order_placed),
                "verifast_order": verifast_flag if order_placed else 0,
                "primary_usecase": None,
                "secondary_usecase": None,
                "events_counter": None,
                "event_counter_diff": None,
                "has_talked_to_bot": None,
                "scroll_percentage": None,
                "visited_days_count": None,
                "talk_to_agent": 0,
            }

            # Assign closest use case
            uc_ts = min((t for t in usecase_by_time if t >= ts), default=None)
            if uc_ts:
                primary = usecase_by_time[uc_ts]["primary"]
                secondary = usecase_by_time[uc_ts]["secondary"]

                # Merge / Clean use-cases
                if secondary in ["Seeking Solution", "Customer Problem"]:
                    secondary = "Seeking Solution"
                elif secondary in ["Feedback", "Complaint"]:
                    secondary = "Feedback"
                elif secondary in ["Product Info", "All Products", "product_benefits"]:
                    secondary = "Product Info"
                elif secondary in ["ecommerce_quick_commerce", "E-commerce/Quick Commerce"]:
                    secondary = "E-commerce/Quick Commerce"
                elif secondary in ["side_effect", "side_effects", "Unsure About Effectiveness"]:
                    secondary = "Unsure About Effectiveness or Side Effects"
                elif secondary in ["store", "physical_store", "Inquiry about availability on other platforms"]:
                    secondary = "Inquiry about stores or availability on other platforms"
                elif secondary in ["durability", "durability_and_quality"]:
                    secondary = "Inquiry about durability or quality"
                elif secondary == "No More Product Query":
                    secondary = None

                record["primary_usecase"] = primary
                record["secondary_usecase"] = secondary

            # Assign closest attribute snapshot
            attr_ts = min((t for t in attr_by_time if t >= ts), default=None)
            if attr_ts:
                d = attr_by_time[attr_ts]
                record["events_counter"] = d.get("events_counter")
                record["event_counter_diff"] = d.get("event_counter_diff")
                record["has_talked_to_bot"] = d.get("has_talked_to_bot")
                record["scroll_percentage"] = extract_scroll_percentage(
                    d.get("scroll_position", "")
                )
                record["visited_days_count"] = extract_visited_days_count(
                    d.get("visited_days", "")
                )

            # Detect "Talk to Agent"
            if idx + 1 < len(messages):
                nxt = messages[idx + 1]
                if nxt.get("actor") == "AI" and "Talk to Agent" in nxt.get("text", ""):
                    record["talk_to_agent"] = 1

            records.append(record)

    df_out = pd.DataFrame(records)
    df_out.dropna(subset=['secondary_usecase'], inplace=True)

    # Map to user-friendly names
    df_out['secondary_usecase'] = df_out['secondary_usecase'].apply(
        lambda x: SECONDARY_USECASE_MAPPING.get(x, x)
    )

    print(f"[process_chat_data] Processed customer messages: {len(df_out)}")
    return df_out


def build_bubble_dataframe(
    df: pd.DataFrame,
    min_clicks: int = 7,
    min_words: int = 3,
    skip_duplicates: bool = True,
    first_only: bool = False,
    calc_ei: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive bubble-message (AI prompt) statistics and return (bubble_df, cleaned_df).
    cleaned_df has bubble messages removed for GPT analysis.
    """
    txt_counts = Counter(df["message_text"])
    candidate_bubbles = {
        t for t, c in txt_counts.items()
        if c >= min_clicks and len(t.split()) >= min_words
    }

    bubble_stats = defaultdict(list)
    bubble_orders = defaultdict(int)
    bubble_a2c = defaultdict(int)
    bubble_click_n = Counter()

    work = df.copy()
    if first_only:
        work = work[work["msg_number_in_session"] == 1]

    for sid, grp in work.groupby("session_id"):
        seen = set()
        for _, row in grp.iterrows():
            txt = row["message_text"]
            if txt not in candidate_bubbles:
                continue
            if skip_duplicates and txt in seen:
                continue
            seen.add(txt)

            bubble_click_n[txt] += 1
            click_pos = row["msg_number_in_session"]
            msgs_after = grp["msg_number_in_session"].max() - click_pos
            bubble_stats[txt].append((click_pos, msgs_after))

            if row["did_a2c"]:
                bubble_a2c[txt] += 1
            if row["order_placed"]:
                bubble_orders[txt] += 1

    rows: List[Dict[str, Any]] = []
    total_clicks = sum(bubble_click_n.values()) or 1
    for txt, n in bubble_click_n.most_common():
        positions = [p for p, _ in bubble_stats[txt]]
        afters = [a for _, a in bubble_stats[txt]]

        row = {
            "bubble_message": txt,
            "clicks": n,
            "%_of_total_clicks": round(n / total_clicks * 100, 2),
            "avg_click_position": round(mean(positions), 1),
            "avg_msgs_after_click": round(mean(afters), 1),
            "%_did_a2c": round(bubble_a2c[txt] / n * 100, 1),
            "%_placed_order": round(bubble_orders[txt] / n * 100, 1),
        }
        if calc_ei:
            expected = 100 / len(bubble_click_n)
            row["EI"] = round((row["%_of_total_clicks"] / expected - 1) * 100, 2)
        rows.append(row)

    bubble_df = (
        pd.DataFrame(rows).sort_values("clicks", ascending=False)
        if rows
        else pd.DataFrame(columns=[
            "bubble_message", "clicks", "%_of_total_clicks",
            "avg_click_position", "avg_msgs_after_click",
            "%_did_a2c", "%_placed_order"
        ])
    )
    if not calc_ei and "EI" in bubble_df.columns:
        bubble_df.drop(columns=["EI"], inplace=True)

    clean_df = df[~df["message_text"].isin(bubble_click_n)].copy()
    return bubble_df, clean_df
