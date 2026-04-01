-- Job tracking table (shared across all Verifast jobs)
-- This table already exists in app-backend-db; included here for reference.

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    index_name VARCHAR(255) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    job_type ENUM('ONBOARDING_REPORT','CLUSTER_OPTIMIZE','DAY3','MONTHLY_REPORT') NOT NULL,
    status ENUM('PENDING','IN_PROGRESS','COMPLETED','FAILED') NOT NULL DEFAULT 'PENDING',
    report_url TEXT,
    report_type ENUM('INTERNAL','CLIENT','MONTHLY_PDF','CONCERN_HTML') DEFAULT NULL,
    bubble_success BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL
);
