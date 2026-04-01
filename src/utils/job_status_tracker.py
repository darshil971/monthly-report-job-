"""
Job status tracking via the db-sql-writer Cloud Run service.
Tracks MONTHLY_REPORT jobs in the shared `jobs` table.
"""

from datetime import datetime, timezone
from typing import Optional
from src.utils.db_writer_util import db_writer_sql_invoker


def insert_job_status(
    index_name: str,
    start_date: str,
    end_date: str,
    status: str = "IN_PROGRESS",
    report_type: str = "MONTHLY_PDF"
) -> Optional[int]:
    """Insert a new job status record. Returns the job ID or None."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    query = """
        INSERT INTO jobs
            (index_name, start_date, end_date, job_type, status, report_type, created_at, updated_at)
        VALUES
            (:index_name, :start_date, :end_date, :job_type, :status, :report_type, :created_at, :updated_at)
    """
    params = {
        "index_name": index_name,
        "start_date": start_date,
        "end_date": end_date,
        "job_type": "MONTHLY_REPORT",
        "status": status,
        "report_type": report_type,
        "created_at": now,
        "updated_at": now,
    }
    result = db_writer_sql_invoker(query, params)
    if result:
        return result.get("lastrowid")
    return None


def update_job_status(
    job_id: int,
    status: str,
    report_url: str = None
):
    """Update an existing job status."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if report_url:
        query = """
            UPDATE jobs
            SET status = :status, report_url = :report_url,
                updated_at = :updated_at, completed_at = :completed_at
            WHERE id = :job_id
        """
        params = {
            "job_id": job_id,
            "status": status,
            "report_url": report_url,
            "updated_at": now,
            "completed_at": now,
        }
    else:
        query = """
            UPDATE jobs
            SET status = :status, updated_at = :updated_at
            WHERE id = :job_id
        """
        params = {"job_id": job_id, "status": status, "updated_at": now}

        if status in ("COMPLETED", "FAILED"):
            query = """
                UPDATE jobs
                SET status = :status, updated_at = :updated_at, completed_at = :completed_at
                WHERE id = :job_id
            """
            params["completed_at"] = now

    db_writer_sql_invoker(query, params)
