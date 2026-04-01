"""
Post-run consolidation: collects all outputs (VoC HTML, monthly PDF, concern HTML)
and organizes them into a single directory per client.
Extracted from admin-backend/organize_voc_reports.py with configurable paths.
"""

import os
import shutil
import glob
from pathlib import Path
from collections import defaultdict


def extract_client_from_filename(filename, known_clients):
    """
    Extract client name from cluster filename: clusters_{client}_{page}_voc_report.html
    Uses known client list to determine where client name ends and page name begins.
    """
    if not filename.startswith("clusters_") or not filename.endswith("_voc_report.html"):
        return None, None

    middle = filename[9:-16]

    sorted_clients = sorted(known_clients, key=len, reverse=True)
    for client_slug in sorted_clients:
        if middle == client_slug:
            return client_slug, "all"
        elif middle.startswith(client_slug + "_"):
            page_name = middle[len(client_slug) + 1:]
            return client_slug, page_name

    if middle.startswith("wa_") or middle.startswith("email_"):
        parts = middle.split("_", 2)
        if len(parts) >= 2:
            client_slug = f"{parts[0]}_{parts[1]}"
            page_name = parts[2] if len(parts) == 3 else "all"
            return client_slug, page_name

    parts = middle.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return parts[0], "all"


def is_special_client(client_name):
    """Check if client has special naming (wa_, email_, or no .myshopify.com suffix)."""
    return (
        client_name.startswith('wa_') or
        client_name.startswith('email_') or
        (not client_name.endswith('.myshopify.com') and '.' not in client_name)
    )


def organize_reports(
    voc_reports_dir: str,
    output_dir: str,
    concern_reports_dir: str,
    monthly_report_dir: str,
):
    """
    Organize monthly reports into client-specific folders.
    All paths are passed as arguments (no hardcoded paths).
    """
    print(f"\n{'='*70}")
    print(f"MONTHLY REPORT ORGANIZATION")
    print(f"{'='*70}\n")

    # Step 1: Build known clients from output directory
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Error: {output_dir} does not exist!")
        return

    client_folders = [f for f in output_path.iterdir() if f.is_dir()]
    known_clients = set()
    client_slug_to_full_name = {}

    for folder in client_folders:
        client_full = folder.name
        if client_full.endswith('.myshopify.com'):
            client_slug = client_full.replace('.myshopify.com', '')
        else:
            client_slug = client_full
        known_clients.add(client_slug)
        client_slug_to_full_name[client_slug] = client_full

    print(f"Found {len(known_clients)} known clients in output folder")

    # Step 2: Scan voc_reports
    voc_reports_path = Path(voc_reports_dir)
    client_voc_files = defaultdict(list)

    if voc_reports_path.exists():
        cluster_files = list(voc_reports_path.glob("clusters_*_voc_report.html"))
        for cluster_file in cluster_files:
            client_slug, page_name = extract_client_from_filename(
                cluster_file.name, known_clients
            )
            if client_slug:
                client_voc_files[client_slug].append((cluster_file, page_name))
        print(f"Found {len(cluster_files)} VoC HTML files grouped into {len(client_voc_files)} clients")

    # Step 3: Create output and process
    monthly_report_path = Path(monthly_report_dir)
    monthly_report_path.mkdir(parents=True, exist_ok=True)

    stats = {
        'voc_html_copied': 0,
        'monthly_pdf_copied': 0,
        'concern_reports_copied': 0,
    }

    all_client_slugs = known_clients | set(client_voc_files.keys())
    for client_slug in sorted(all_client_slugs):
        if client_slug in client_slug_to_full_name:
            client_full_name = client_slug_to_full_name[client_slug]
        else:
            client_full_name = f"{client_slug}.myshopify.com"

        client_dest_folder = monthly_report_path / client_full_name
        client_dest_folder.mkdir(parents=True, exist_ok=True)

        # Copy VoC HTML files
        for voc_file, page_name in client_voc_files.get(client_slug, []):
            dest_file = client_dest_folder / voc_file.name
            shutil.copy2(str(voc_file), str(dest_file))
            stats['voc_html_copied'] += 1

        # Copy monthly report PDFs
        client_folder = output_path / client_full_name
        if client_folder.exists():
            for pdf_file in client_folder.glob("*_monthly_report_*.pdf"):
                dest_file = client_dest_folder / pdf_file.name
                shutil.copy2(str(pdf_file), str(dest_file))
                stats['monthly_pdf_copied'] += 1

        # Copy concern reports
        concern_pattern = os.path.join(
            concern_reports_dir, f"concern_report_{client_full_name}_*.html"
        )
        for concern_file in glob.glob(concern_pattern):
            concern_filename = os.path.basename(concern_file)
            dest_file = client_dest_folder / concern_filename
            shutil.copy2(concern_file, str(dest_file))
            stats['concern_reports_copied'] += 1

    print(f"\nFiles Copied:")
    print(f"  VoC HTML reports: {stats['voc_html_copied']}")
    print(f"  Monthly report PDFs: {stats['monthly_pdf_copied']}")
    print(f"  Concern reports: {stats['concern_reports_copied']}")
    print(f"\nOrganized reports into {monthly_report_dir}")
