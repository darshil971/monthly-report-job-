#!/bin/bash
set -e

# Usage: ./run_job.sh <start-date> <end-date> [--client client1,client2,...] [--local]
# Example: ./run_job.sh 2026-02-01 2026-02-28
# Example: ./run_job.sh 2026-02-01 2026-02-28 --client zanducare.myshopify.com
# Example: ./run_job.sh 2026-02-01 2026-02-28 --local

JOB_NAME="monthly-report"
REGION="asia-south1"

START_DATE="${1:?'Start date required (YYYY-MM-DD)'}"
END_DATE="${2:?'End date required (YYYY-MM-DD)'}"
shift 2

# Parse remaining args
LOCAL_MODE=false
CLIENT_ARG=""
FOLDER_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --client)
            CLIENT_ARG="--client,$2"
            shift 2
            ;;
        --folder)
            FOLDER_ARG="--folder,$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

ARGS="--start-date,${START_DATE},--end-date,${END_DATE}"
if [ -n "${CLIENT_ARG}" ]; then
    ARGS="${ARGS},${CLIENT_ARG}"
fi
if [ -n "${FOLDER_ARG}" ]; then
    ARGS="${ARGS},${FOLDER_ARG}"
fi

if [ "${LOCAL_MODE}" = true ]; then
    echo "Running locally with Docker..."
    docker run --rm \
        --env-file .env \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/new_outputs:/app/new_outputs" \
        -v "$(pwd)/vec_outs:/app/vec_outs" \
        -v "$(pwd)/concern_reports:/app/concern_reports" \
        -v "$(pwd)/monthly_report:/app/monthly_report" \
        asia-south1-docker.pkg.dev/ecom-review-app/jobs/monthly-report/main/v1 \
        python -m src.pipeline --start-date "${START_DATE}" --end-date "${END_DATE}" ${CLIENT_ARG:+--client ${CLIENT_ARG#--client,}} ${FOLDER_ARG:+--folder ${FOLDER_ARG#--folder,}}
else
    echo "Executing Cloud Run job: ${JOB_NAME}"
    echo "Args: ${ARGS}"
    gcloud run jobs execute "${JOB_NAME}" \
        --region="${REGION}" \
        --args="${ARGS}"
fi
