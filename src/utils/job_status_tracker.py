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
    """Update existing PENDING row to IN_PROGRESS, or insert if no row exists."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    start_timestamp = f"{start_date} 00:00:00"
    end_timestamp = f"{end_date} 23:59:59"

    base_params = {
        "index_name": index_name,
        "start_date": start_timestamp,
        "end_date": end_timestamp,
        "job_type": "MONTHLY_REPORT",
        "report_type": report_type,
    }

    # Step 1: Update any existing PENDING row (created by admin API on trigger)
    update_query = """
        UPDATE jobs
        SET status = :status, updated_at = :updated_at
        WHERE index_name = :index_name
          AND start_date = :start_date
          AND end_date = :end_date
          AND report_type = :report_type
          AND job_type = :job_type
          AND status = 'PENDING'
    """
    db_writer_sql_invoker(update_query, {**base_params, "status": status, "updated_at": now})

    # Step 2: Insert only if no matching row exists (covers direct trigger without admin API)
    insert_query = """
        INSERT INTO jobs
            (index_name, start_date, end_date, job_type, status, report_type, created_at, updated_at)
        SELECT :index_name, :start_date, :end_date, :job_type, :status, :report_type, :created_at, :updated_at
        FROM DUAL
        WHERE NOT EXISTS (
            SELECT 1 FROM jobs
            WHERE index_name = :index_name AND start_date = :start_date AND end_date = :end_date
              AND report_type = :report_type AND job_type = :job_type
        )
    """
    db_writer_sql_invoker(insert_query, {**base_params, "status": status, "created_at": now, "updated_at": now})


def delete_pending_job(
    index_name: str,
    start_date: str,
    end_date: str,
    report_type: str,
):
    """Delete a PENDING job row that won't be processed (e.g. concern skipped for special clients)."""
    start_timestamp = f"{start_date} 00:00:00"
    end_timestamp = f"{end_date} 23:59:59"

    query = """
        DELETE FROM jobs
        WHERE index_name = :index_name
          AND start_date = :start_date
          AND end_date = :end_date
          AND report_type = :report_type
          AND job_type = :job_type
          AND status = 'PENDING'
    """
    db_writer_sql_invoker(query, {
        "index_name": index_name,
        "start_date": start_timestamp,
        "end_date": end_timestamp,
        "report_type": report_type,
        "job_type": "MONTHLY_REPORT",
    })


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
