# Monthly Report Job

Cloud Run job for generating monthly performance reports for Verifast clients.

## What It Does

For each client, the pipeline produces:
1. **Monthly PDF Report** — key metrics: sessions, orders, UTM contribution, support cost savings, AI prompt %, hourly trend, user intent breakdown
2. **Concern Clustering** — semantic clustering of customer concerns with HDBSCAN pipeline, Excel export, and interactive HTML report
3. **Organized Output** — all reports consolidated into `monthly_report/{client}/` directories

## Quick Start

### Local Development

```bash
# Install dependencies
poetry install

# Install Playwright browser
playwright install chromium

# Copy environment file
cp .env.example .env
# Edit .env with your credentials

# Run for specific clients
python -m src.pipeline --start-date 2026-02-01 --end-date 2026-02-28 --client zanducare.myshopify.com

# Run for all clients
python -m src.pipeline --start-date 2026-02-01 --end-date 2026-02-28

# PDF only (skip concern clustering)
python -m src.pipeline --start-date 2026-02-01 --end-date 2026-02-28 --skip-concern

# With GCS upload and report organization
python -m src.pipeline --start-date 2026-02-01 --end-date 2026-02-28 --upload-gcs --organize
```

### Docker

```bash
# Build
./build.sh

# Run locally via Docker
./run_job.sh 2026-02-01 2026-02-28 --local

# Run specific client
./run_job.sh 2026-02-01 2026-02-28 --client zanducare.myshopify.com --local
```

### Cloud Run

```bash
# Build and push
./build.sh

# Execute on Cloud Run
./run_job.sh 2026-02-01 2026-02-28

# With specific client
./run_job.sh 2026-02-01 2026-02-28 --client zanducare.myshopify.com
```

## CLI Arguments

| Argument | Required | Description |
|---|---|---|
| `--start-date` | Yes | Start date (YYYY-MM-DD) |
| `--end-date` | Yes | End date (YYYY-MM-DD) |
| `--client` | No | Comma-separated client domains, or omit for all |
| `--folder` | No | Output folder name (auto-generated from month if omitted) |
| `--skip-concern` | No | Skip concern clustering (PDF only) |
| `--upload-gcs` | No | Upload outputs to Google Cloud Storage |
| `--organize` | No | Organize all reports into monthly_report/ after processing |

## Project Structure

```
monthly-report-job/
├── Dockerfile                  # Multi-stage Docker build
├── entrypoint.sh               # Container entrypoint
├── build.sh                    # Docker build & push to GCP
├── run_job.sh                  # Job execution (local or Cloud Run)
├── pyproject.toml              # Poetry dependencies
├── schema.sql                  # Jobs table schema
├── .env.example                # Environment variables template
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration (env vars, paths)
│   ├── pipeline.py             # Main entry point & orchestrator
│   │
│   ├── data/
│   │   ├── firebase_manager.py # Firebase JWT auth
│   │   ├── raw_data_client.py  # API data fetching with caching
│   │   └── chat_data_processor.py  # Raw JSON → DataFrame ETL
│   │
│   ├── report/
│   │   ├── report_builder.py   # Report orchestration & JSON assembly
│   │   ├── monthly_pdf.py      # HTML → PDF via Playwright
│   │   └── analysis.py         # GPT-based theme analysis
│   │
│   ├── concern/
│   │   ├── concern_cluster.py  # HDBSCAN concern clustering engine
│   │   └── concern_report.py   # Interactive HTML concern report
│   │
│   ├── organize/
│   │   └── organize_reports.py # Post-run file consolidation
│   │
│   └── utils/
│       ├── openai_utils.py     # Azure OpenAI wrappers
│       ├── storage_service.py  # Google Cloud Storage
│       ├── db_writer_util.py   # Cloud Run DB writer service
│       ├── job_status_tracker.py  # Job status in MySQL
│       └── slack_notification.py  # Slack webhooks
```

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description |
|---|---|
| `FIREBASE_ADMIN_CREDENTIALS_PATH` | Path to Firebase service account JSON |
| `AZURE_GPT41_API_KEY` | Azure OpenAI GPT-4.1 API key |
| `AZURE_EMBEDDING_API_KEY` | Azure OpenAI embeddings API key |
| `VERIFAST_LOGO_PATH` | Path to Verifast logo PNG for PDF reports |
| `STORAGE_BUCKET_NAME` | GCS bucket for report uploads |
| `SLACK_WEBHOOK_URL` | Slack webhook for notifications |

## Output Files

```
output/{folder}/{client}/
├── raw/
│   ├── {client}.json                          # Raw session data
│   ├── sales_{id}_{ddmm}_{ddmm}.json         # Cached API response
│   └── ...
└── {id}_monthly_report_{ddmm}_{ddmm}.pdf     # PDF report

new_outputs/
├── {id}_report_data_{ddmm}_{ddmm}.json       # Structured report data
└── chat_data_{id}_{ddmm}_{ddmm}.csv          # Chat data export

vec_outs/
├── concern_clusters_{base}_sessions.json      # Concern clusters
└── concern_clusters_{base}_export.xlsx        # Excel export

concern_reports/
└── concern_report_{client}_{range}.html       # Interactive HTML
```

## Idempotency

| Step | Idempotent? |
|---|---|
| Raw session data fetch | Yes (skips if file exists) |
| API data fetch | Yes (cached to files) |
| Chat data processing | Yes (pure read) |
| GPT analyses | No (re-runs each time) |
| Embedding generation | Yes (cached) |
| PDF generation | No (overwrites) |

## Cloud Run Deployment

```bash
# Image location
asia-south1-docker.pkg.dev/ecom-review-app/jobs/monthly-report/main/v1

# Cloud Run job config
Region: asia-south1
Memory: 4Gi
CPU: 2
Timeout: 3600s (1 hour per client)
```
