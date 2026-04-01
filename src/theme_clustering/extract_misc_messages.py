"""
Extract all messages from the Miscellaneous cluster in a clustering output file
and write just the message texts to a plain text file (one per line).
"""

import json
import os
from datetime import datetime


def extract_misc_messages(clusters_path: str, output_path: str | None = None):
    with open(clusters_path) as f:
        data = json.load(f)

    # Find the miscellaneous cluster (cluster_id == -1 or title match)
    misc_cluster = None
    for cluster in data["clusters"]:
        if cluster["cluster_id"] == -1 or cluster["cluster_title"].lower() == "miscellaneous":
            misc_cluster = cluster
            break

    if misc_cluster is None:
        print("No miscellaneous cluster found.")
        return

    messages = [msg["text"] for msg in misc_cluster["messages"]]

    if output_path is None:
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/misc_messages_{timestamp}.txt"

    with open(output_path, "w") as f:
        f.write("\n".join(messages))

    print(f"Wrote {len(messages)} miscellaneous messages to {output_path}")


if __name__ == "__main__":
    # ===== EDIT THESE =====
    clusters_path = "/home/anirudh/admin-backend/new_clustering/outputs/themes_sheopals_20260219_065727.json"
    output_path = None  # Set to a string path to use custom filename, or None for auto-generated
    # ======================

    extract_misc_messages(clusters_path, output_path)
