"""
Job status tracking via the db-sql-writer Cloud Run service.
Tracks MONTHLY_REPORT jobs in the shared `jobs` table.
Uses index_name + date range + report_type to identify rows (no lastrowid needed).
"""

from datetime import datetime, timezone
from src.utils.db_writer_util import db_writer_sql_invoker


def insert_job_status(
    index_name: str,
    start_date: str,
    end_date: str,
    status: str = "IN_PROGRESS",
    report_type: str = "MONTHLY_PDF"
):
    """Insert a new job status record."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    start_timestamp = f"{start_date} 00:00:00"
    end_timestamp = f"{end_date} 23:59:59"

    query = """
        INSERT INTO jobs
            (index_name, start_date, end_date, job_type, status, report_type, created_at, updated_at)
        VALUES
            (:index_name, :start_date, :end_date, :job_type, :status, :report_type, :created_at, :updated_at)
    """
    params = {
        "index_name": index_name,
        "start_date": start_timestamp,
        "end_date": end_timestamp,
        "job_type": "MONTHLY_REPORT",
        "status": status,
        "report_type": report_type,
        "created_at": now,
        "updated_at": now,
    }
    db_writer_sql_invoker(query, params)


def update_job_status(
    index_name: str,
    start_date: str,
    end_date: str,
    report_type: str,
    status: str,
):
    """Update job status by matching index_name + dates + report_type."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    start_timestamp = f"{start_date} 00:00:00"
    end_timestamp = f"{end_date} 23:59:59"
    completed_at = now if status in ("COMPLETED", "FAILED") else None

    query = """
        UPDATE jobs
        SET status = :status,
            updated_at = :updated_at,
            completed_at = :completed_at
        WHERE index_name = :index_name
          AND start_date = :start_date
          AND end_date = :end_date
          AND report_type = :report_type
          AND job_type = :job_type
    """
    params = {
        "index_name": index_name,
        "status": status,
        "updated_at": now,
        "completed_at": completed_at,
        "start_date": start_timestamp,
        "end_date": end_timestamp,
        "report_type": report_type,
        "job_type": "MONTHLY_REPORT",
    }
    db_writer_sql_invoker(query, params)
