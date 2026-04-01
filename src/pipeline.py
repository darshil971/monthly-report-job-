"""
Monthly Report Pipeline — Cloud Run Job Entry Point

Replaces og_report.py from admin-backend. Orchestrates the full monthly report
generation pipeline for a list of Shopify clients:

1. For each client:
   a. Fetch raw session data from analytics API (day-by-day, cached)
   b. Fetch dashboard metrics (sales, support, feedback, UTM)
   c. Process chat data and run GPT analyses
   d. Generate monthly PDF report
   e. Run concern clustering (for non-special clients)
   f. Track job status in DB
   g. Upload outputs to GCS (optional)

2. After all clients: organize reports into monthly_report/ structure

Usage:
    python -m src.pipeline --start-date 2026-02-01 --end-date 2026-02-28
    python -m src.pipeline --start-date 2026-02-01 --end-date 2026-02-28 --client zanducare.myshopify.com
    python -m src.pipeline --start-date 2026-02-01 --end-date 2026-02-28 --folder february_all_report
"""

import argparse
import os
import sys
import traceback
from datetime import datetime

import requests as _requests

from src.config import MonthlyReportJobConfig
from src.data.firebase_manager import FirebaseAuthManager
from src.data.raw_data_client import RawDataClient
from src.report.report_builder import ReportBuilder
from src.utils.job_status_tracker import insert_job_status, update_job_status
from src.utils.slack_notification import send_slack_notification


# ============================================================
# Dynamic client list fetching (from auto-onboarding pattern)
# ============================================================

SHOPIFY_APP_URL = os.getenv(
    "SHOPIFY_APP_URL",
    "https://shopify-app.verifast.ai/api"
)
SHOPIFY_APP_AUTH_TOKEN = os.getenv(
    "SHOPIFY_APP_AUTH_TOKEN",
    "3af@1s42a"
)


def fetch_all_active_clients() -> list:
    """
    Fetch all active clients from the Shopify app API.
    Same as auto-onboarding/manage_client_key.py ClientKeyHandler.get_all_clients_with_keys.

    Returns list of client shop domains (e.g. ['zanducare.myshopify.com', ...]).
    """
    import time
    url = f"{SHOPIFY_APP_URL}/onboarding/get_all_clients"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SHOPIFY_APP_AUTH_TOKEN}",
    }

    for attempt in range(1, 4):
        try:
            response = _requests.post(url, headers=headers, data={}, timeout=30)
            response.raise_for_status()
            all_stores_data = response.json()
            all_clients = all_stores_data.get("all_clients", [])

            active_clients = [
                client["shop"].lower()
                for client in all_clients
                if client.get("shop")
            ]

            print(f"[fetch_clients] Fetched {len(active_clients)} clients from API")
            return active_clients
        except Exception as e:
            print(f"[fetch_clients] Failed (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(2 * attempt)

    raise Exception("Failed to fetch client list after 3 attempts")



def is_special_client(client_name: str) -> bool:
    """Check if client should skip concern clustering."""
    return (
        client_name.startswith('wa_') or
        client_name.startswith('email_') or
        client_name.startswith('app_')
    )


def run_concern_clustering_for_client(
    json_path: str,
    client_name: str,
    start_date: str,
    end_date: str,
    vec_outs_dir: str = "vec_outs",
):
    """Run concern clustering for a client."""
    from src.concern.concern_cluster import run_concern_clustering

    start_ddmm = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d%m")
    end_ddmm = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d%m")
    client_id = client_name.split('.')[0] if '.' in client_name else client_name

    run_concern_clustering(
        chat_data_file_path=json_path,
        output_dir=vec_outs_dir,
        remove_bubbles=False,
        merge_enabled=True,
        min_message_threshold=5,
        round2_excluded_targets=['other concern', 'other concerns'],
        report=True,
        client_name_override=client_id,
        date_range_override=f"{start_ddmm}_{end_ddmm}",
    )


def run_theme_clustering_for_client(
    json_path: str,
    client_name: str,
    end_date: str,
    is_special: bool,
    vec_outs_dir: str = "vec_outs",
):
    """
    Run theme clustering (VoC) pipeline for a client.
    For regular clients: auto-detect top 3 product pages, run per page.
    For special clients: run once on all messages (no page filter).
    """
    from src.theme_clustering.theme_clustering import run_pipeline
    from src.data.vector_helpers import load_messages_from_report, get_top_frequent_pages
    from src.data.cluster_metadata_helpers import load_chat_data

    report_month = datetime.strptime(end_date, '%Y-%m-%d').strftime('%B %Y')

    # Load chat data once (shared across all page runs for metadata)
    chat_data = None
    try:
        chat_data = load_chat_data(json_path, report=True)
    except Exception as e:
        print(f"[theme_clustering] Warning: Failed to load chat data for metadata: {e}")

    if is_special:
        # Special clients: run once on all messages, no page filtering
        df = load_messages_from_report(
            json_path,
            remove_bubbles=True,
            filter_pages=None,
            max_messages=10000,
        )
        if df is None or df.empty:
            print(f"[theme_clustering] No messages found for {client_name}")
            return
        run_pipeline(
            messages=df['text'].tolist(),
            session_ids=df['session_id'].tolist(),
            client_name=client_name,
            page_name=None,
            chat_data=chat_data,
            save_outputs=True,
            output_dir_override=f"./{vec_outs_dir}",
            report_month=report_month,
        )
    else:
        # Regular clients: auto-detect top 3 product pages, run per page
        top_pages = get_top_frequent_pages(json_path, report=True, top_k=3)
        if not top_pages:
            print(f"[theme_clustering] No pages found for {client_name}")
            return

        print(f"[theme_clustering] Found {len(top_pages)} top page(s) to process")
        for i, page_url in enumerate(top_pages, 1):
            print(f"\n[{i}/{len(top_pages)}] Processing page: {page_url[:80]}...")
            try:
                df_page = load_messages_from_report(
                    json_path,
                    remove_bubbles=True,
                    filter_pages=[page_url],
                    max_messages=10000,
                )
                if df_page is None or df_page.empty:
                    print(f"  No messages for this page, skipping")
                    continue
                run_pipeline(
                    messages=df_page['text'].tolist(),
                    session_ids=df_page['session_id'].tolist(),
                    client_name=client_name,
                    page_name=page_url,
                    chat_data=chat_data,
                    save_outputs=True,
                    output_dir_override=f"./{vec_outs_dir}",
                    report_month=report_month,
                )
                print(f"  Theme clustering completed for page {i}")
            except Exception as e:
                print(f"  Error clustering page {i}: {e}")
                traceback.print_exc()


def _convert_html_to_old_format(file_path: str, is_concern: bool) -> str:
    """Convert HTML report from glassmorphic to old blue gradient format. Returns converted path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        if is_concern:
            from src.utils.convert_concern_to_old_format import convert_concern_html_to_old_format
            converted = convert_concern_html_to_old_format(html_content)
        else:
            from src.utils.convert_voc_to_old_format import convert_html_to_old_format
            converted = convert_html_to_old_format(html_content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(converted)

        print(f"    Converted to old format: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"    Warning: Format conversion failed for {os.path.basename(file_path)}: {e}")

    return file_path


def _collect_html_reports(client_name: str, config: MonthlyReportJobConfig) -> list:
    """Collect all HTML reports, convert to old format, return for GCS upload."""
    import glob
    html_files = []

    # Concern reports — convert to old format
    # Filename uses client_id (slurrpfarm), not full domain (slurrpfarm.myshopify.com)
    client_id = client_name.split('.')[0] if '.' in client_name else client_name
    pattern = os.path.join(config.concern_reports_dir, f"concern_report_{client_id}_*.html")
    for f in glob.glob(pattern):
        _convert_html_to_old_format(f, is_concern=True)
        html_files.append({"path": f, "key": "concern-report"})

    # VoC reports (from theme clustering) — convert to old format
    client_slug = client_name.replace('.myshopify.com', '')
    pattern = os.path.join(config.voc_reports_dir, f"clusters_{client_slug}*_voc_report.html")
    for f in glob.glob(pattern):
        _convert_html_to_old_format(f, is_concern=False)
        html_files.append({"path": f, "key": "page-level-report"})

    return html_files


def process_single_client(
    client_name: str,
    start_date: str,
    end_date: str,
    config: MonthlyReportJobConfig,
    data_client: RawDataClient,
    report_builder: ReportBuilder,
    date_path: str,
    run_concern: bool = True,
    run_theme: bool = True,
) -> bool:
    """
    Process a single client: generate PDF report + concern clustering + theme clustering.
    Uploads all outputs to GCS with encrypted filenames.
    Cleans up tmp/ between clients so disk doesn't fill up.
    Returns True on success (PDF generated), False otherwise.
    """
    print(f"\n{'='*80}")
    print(f"Starting analysis for {client_name}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*80}\n")

    # Clean previous client data and set up fresh tmp dirs
    config.setup_client_tmp(client_name)
    json_path = config.get_client_json_path(client_name)

    # Insert job status
    job_id = None
    try:
        job_id = insert_job_status(client_name, start_date, end_date, status="IN_PROGRESS")
    except Exception as e:
        print(f"[pipeline] Warning: Could not insert job status: {e}")

    pdf_success = False
    pdf_path = None

    try:
        # Step 1: Generate monthly PDF report
        pdf_path = report_builder.build_report(
            client_name=client_name,
            start_date=start_date,
            end_date=end_date,
            json_path=json_path,
            output_folder=config.tmp_dir,
        )

        if pdf_path:
            print(f"\nPDF report generated: {pdf_path}")
            pdf_success = True
        else:
            print(f"\nPDF report generation failed for {client_name}")

        # Step 2: Concern clustering (for non-special clients)
        if run_concern and not is_special_client(client_name):
            print(f"\nStarting concern clustering for {client_name}...")
            try:
                run_concern_clustering_for_client(
                    json_path=json_path,
                    client_name=client_name,
                    start_date=start_date,
                    end_date=end_date,
                    vec_outs_dir=config.vec_outs_dir,
                )
                print(f"Concern clustering completed for {client_name}")
            except Exception as e:
                print(f"Error during concern clustering for {client_name}: {e}")
                traceback.print_exc()
        elif is_special_client(client_name):
            print(f"\nSkipping concern clustering for special client: {client_name}")

        # Step 3: Theme clustering (VoC reports)
        if run_theme:
            print(f"\nStarting theme clustering for {client_name}...")
            try:
                run_theme_clustering_for_client(
                    json_path=json_path,
                    client_name=client_name,
                    end_date=end_date,
                    is_special=is_special_client(client_name),
                    vec_outs_dir=config.vec_outs_dir,
                )
                print(f"Theme clustering completed for {client_name}")
            except Exception as e:
                print(f"Error during theme clustering for {client_name}: {e}")
                traceback.print_exc()

        # Step 4: Convert HTML reports to old format
        print(f"\nConverting HTML reports to old format for {client_name}...")
        html_reports = _collect_html_reports(client_name, config)
        print(f"Converted {len(html_reports)} HTML report(s)")

        # Step 5: Upload all outputs to GCS (encrypted, bulk_report_uploader pattern)
        # TODO: Uncomment after manual verification of reports
        # print(f"\nUploading reports to GCS for {client_name}...")
        # try:
        #     from src.utils.storage_service import CloudStorage, upload_client_reports
        #     storage = CloudStorage(config.storage_bucket_name)
        #     html_files = _collect_html_reports(client_name, config)
        #     upload_client_reports(
        #         storage_service=storage,
        #         index_name=client_name,
        #         date_path=date_path,
        #         html_files=html_files,
        #         pdf_path=pdf_path,
        #     )
        #     print(f"GCS upload completed for {client_name}")
        # except Exception as e:
        #     print(f"[pipeline] Warning: GCS upload failed: {e}")
        #     traceback.print_exc()

        # Update job status
        if job_id:
            try:
                status = "COMPLETED" if pdf_success else "FAILED"
                update_job_status(job_id, status)
            except Exception:
                pass

        return pdf_success

    except Exception as e:
        print(f"\nError during analysis for {client_name}: {e}")
        traceback.print_exc()

        if job_id:
            try:
                update_job_status(job_id, "FAILED")
            except Exception:
                pass

        return False


def main():
    parser = argparse.ArgumentParser(description="Monthly Report Generation Pipeline")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--client",
        help="Comma-separated list of client domains, or omit to fetch all from API"
    )
    parser.add_argument(
        "--skip-concern",
        action="store_true",
        help="Skip concern clustering"
    )
    args = parser.parse_args()

    # Validate dates
    try:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. ({e})")
        sys.exit(1)

    # Build client list
    if args.client and args.client != 'all':
        client_list = [c.strip().lower() for c in args.client.split(',')]
    else:
        # Dynamically fetch active clients from the Shopify app API
        print("Fetching active client list from API...")
        client_list = fetch_all_active_clients()

    # GCS date_path derived from start date (e.g. "2026_02")
    date_path = start_dt.strftime("%Y_%m")

    # Initialize config
    config = MonthlyReportJobConfig()

    print("\n" + "=" * 80)
    print("MONTHLY REPORT PIPELINE STARTED")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Total clients: {len(client_list)}")
    print(f"Concern clustering: {'DISABLED' if args.skip_concern else 'ENABLED'}")
    print(f"Theme clustering: ENABLED")
    print(f"GCS bucket: {config.storage_bucket_name}")
    print("=" * 80 + "\n")

    # Send start notification
    send_slack_notification(
        f"Monthly report pipeline started\n"
        f"Period: {args.start_date} to {args.end_date}\n"
        f"Clients: {len(client_list)}"
    )

    # Initialize Firebase auth and generate JWT once (reused for all clients)
    print("Initializing Firebase authentication...")
    auth_manager = FirebaseAuthManager(config)
    jwt_token = auth_manager.create_custom_token(config.firebase_auth_email, "monthly-report-job")
    data_client = RawDataClient(config, jwt_token)
    report_builder = ReportBuilder(config, data_client)

    # Process clients
    total_clients = len(client_list)
    successful = 0
    failed = 0
    failed_clients = []

    for i, client_name in enumerate(client_list, start=1):
        try:

            success = process_single_client(
                client_name=client_name,
                start_date=args.start_date,
                end_date=args.end_date,
                config=config,
                data_client=data_client,
                report_builder=report_builder,
                date_path=date_path,
                run_concern=not args.skip_concern,
                run_theme=True,
            )

            if success:
                successful += 1
            else:
                failed += 1
                failed_clients.append(client_name)

        except Exception as e:
            print(f"\nFATAL ERROR for {client_name}: {e}")
            traceback.print_exc()
            failed += 1
            failed_clients.append(client_name)

        remaining = total_clients - i
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{total_clients} clients processed")
        print(f"Successful: {successful} | Failed: {failed} | Remaining: {remaining}")
        print(f"{'='*80}\n")

    # Final summary
    print("\n" + "=" * 80)
    print("MONTHLY REPORT PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Total clients: {total_clients}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_clients:
        print(f"\nFailed clients:")
        for c in failed_clients:
            print(f"  - {c}")
    print("=" * 80 + "\n")

    send_slack_notification(
        f"Monthly report pipeline completed\n"
        f"Period: {args.start_date} to {args.end_date}\n"
        f"Successful: {successful}/{total_clients}\n"
        f"Failed: {failed}"
        + (f"\nFailed clients: {', '.join(failed_clients)}" if failed_clients else "")
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()



                                                                                                         
#   Run for a single client (test):                                                                         
#   PYTHONPATH=. python3 -m src.pipeline --start-date 2026-03-01 --end-date 2026-03-31 --client             
#   zanducare.myshopify.com --skip-concern --skip-theme                                                     
                                                                                                          
#   Run for a single client (full pipeline):                                                              
#   PYTHONPATH=. python3 -m src.pipeline --start-date 2026-03-01 --end-date 2026-03-31 --client             
#   zanducare.myshopify.com                                                                                 
                                                                                                          
#   Run for all clients (fetches from API):                                                                 
#   PYTHONPATH=. python3 -m src.pipeline --start-date 2026-03-01 --end-date 2026-03-31          