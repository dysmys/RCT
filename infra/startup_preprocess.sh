#!/bin/bash
# Startup script for belief-preprocessor VM.
# Installs dependencies, waits for files to be uploaded, then runs preprocessing.
set -euo pipefail

LOG=/var/log/preprocess-startup.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) Startup started ==="

# System deps
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl

# Python environment
VENV="/opt/preprocess_venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
    chromadb \
    sentence-transformers \
    requests \
    numpy \
    python-dotenv \
    tqdm

echo "=== $(date) Dependencies installed ==="

# Signal that VM is ready for file uploads
touch /tmp/startup-complete
echo "=== $(date) VM ready — waiting for run trigger ==="

# The orchestration script (run_preprocess.sh) will:
#   1. SCP the dataset, scripts, and HF key
#   2. Touch /tmp/start-preprocessing to trigger this watcher
nohup bash -c '
  while [ ! -f /tmp/start-preprocessing ]; do sleep 5; done
  echo "=== $(date) Trigger received — starting preprocessing ==="
  cd /opt/preprocess
  export DB_PATH=/opt/preprocess/database/results.db
  if /opt/preprocess_venv/bin/python3 scripts/preprocess_beliefs.py \
      --repos dataset/test.json \
      --resume \
      2>&1 | tee /var/log/preprocess.log; then
    echo "=== $(date) Preprocessing complete ==="
    # Upload results to GCS
    GCS_BUCKET=$(cat /tmp/gcs_bucket 2>/dev/null || echo "gs://seng-models/belief_preprocess")
    echo "Uploading to $GCS_BUCKET ..."
    gsutil -m cp /opt/preprocess/database/results.db   "$GCS_BUCKET/results.db"
    gsutil -m cp -r /opt/preprocess/database/chroma_db "$GCS_BUCKET/chroma_db/"
    echo "=== $(date) Upload complete ==="
    touch /tmp/preprocess-complete
  else
    echo "=== $(date) Preprocessing FAILED — check /var/log/preprocess.log ==="
    echo "FAILED" > /tmp/preprocess-status
    # Do NOT touch /tmp/preprocess-complete so the orchestrator keeps the VM alive for debugging
  fi
' &

echo "=== $(date) Startup script done ==="
