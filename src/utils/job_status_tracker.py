"""
Job status tracking via the db-sql-writer Cloud Run service.
Tracks MONTHLY_REPORT jobs in the shared `jobs` table.
Pattern from onboarding-report-job/master.py.
"""

from datetime import datetime
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
    query = """
        INSERT INTO jobs (index_name, start_date, end_date, job_type, status, report_type, created_at, updated_at)
        VALUES (%(index_name)s, %(start_date)s, %(end_date)s, 'MONTHLY_REPORT', %(status)s, %(report_type)s, NOW(), NOW())
    """
    params = {
        "index_name": index_name,
        "start_date": start_date,
        "end_date": end_date,
        "status": status,
        "report_type": report_type,
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
    if report_url:
        query = """
            UPDATE jobs
            SET status = %(status)s, report_url = %(report_url)s,
                updated_at = NOW(), completed_at = NOW()
            WHERE id = %(job_id)s
        """
        params = {"job_id": job_id, "status": status, "report_url": report_url}
    else:
        query = """
            UPDATE jobs
            SET status = %(status)s, updated_at = NOW()
            WHERE id = %(job_id)s
        """
        params = {"job_id": job_id, "status": status}

    db_writer_sql_invoker(query, params)
