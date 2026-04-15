#!/bin/bash
# Orchestrate the overnight belief preprocessing run.
#
# Usage:
#   ./infra/run_preprocess.sh              # provision VM, upload files, start run
#   ./infra/run_preprocess.sh --no-destroy # keep VM after run (for inspection)
#   ./infra/run_preprocess.sh --status     # check status of running job
#   ./infra/run_preprocess.sh --download   # download results from GCS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ZONE="us-east4-c"
VM_NAME="belief-preprocessor"
GCS_BUCKET="gs://seng-models/belief_preprocess"
HF_KEY="$HOME/.ssh/hf_dsy.key"

NO_DESTROY=false
STATUS_ONLY=false
DOWNLOAD_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --no-destroy)   NO_DESTROY=true ;;
    --status)       STATUS_ONLY=true ;;
    --download)     DOWNLOAD_ONLY=true ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------
if [ "$STATUS_ONLY" = true ]; then
  echo "=== Preprocessing status ==="
  gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    echo '--- Process ---'
    pgrep -f preprocess_beliefs.py && echo 'RUNNING' || echo 'NOT RUNNING'
    echo '--- Last 20 lines ---'
    tail -20 /var/log/preprocess.log 2>/dev/null || echo 'No log yet'
    echo '--- Complete? ---'
    test -f /tmp/preprocess-complete && echo 'YES' || echo 'not yet'
  " 2>&1
  exit 0
fi

# ---------------------------------------------------------------------------
# Download results
# ---------------------------------------------------------------------------
if [ "$DOWNLOAD_ONLY" = true ]; then
  echo "=== Downloading results from $GCS_BUCKET ==="
  mkdir -p "$PROJECT_ROOT/database"
  gsutil cp "$GCS_BUCKET/results.db" "$PROJECT_ROOT/database/results_preprocess.db"
  gsutil -m cp -r "$GCS_BUCKET/chroma_db" "$PROJECT_ROOT/database/"
  echo "Downloaded to $PROJECT_ROOT/database/"
  exit 0
fi

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "=== Pre-flight checks ==="
for cmd in terraform gcloud gsutil; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: $cmd not found"
    exit 1
  fi
done

if [ ! -f "$HF_KEY" ]; then
  echo "ERROR: HF key not found at $HF_KEY"
  exit 1
fi

if ! gcloud auth print-access-token &>/dev/null; then
  echo "ERROR: gcloud not authenticated. Run: gcloud auth login"
  exit 1
fi

echo "All checks passed."

# ---------------------------------------------------------------------------
# Provision VM
# ---------------------------------------------------------------------------
echo ""
echo "=== Provisioning belief-preprocessor VM ==="
cd "$SCRIPT_DIR"
terraform init -input=false -upgrade >/dev/null
terraform apply -auto-approve -target=google_compute_instance.belief_preprocessor

# ---------------------------------------------------------------------------
# Wait for VM startup
# ---------------------------------------------------------------------------
echo ""
echo "=== Waiting for VM startup (up to 5 min) ==="
MAX_WAIT=300
ELAPSED=0
while true; do
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "ERROR: VM not ready after ${MAX_WAIT}s"
    exit 1
  fi
  if gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
      --command="test -f /tmp/startup-complete" 2>/dev/null; then
    echo "VM ready (${ELAPSED}s)"
    break
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
  echo "  ...waiting (${ELAPSED}s)"
done

# ---------------------------------------------------------------------------
# Upload files
# ---------------------------------------------------------------------------
echo ""
echo "=== Uploading files ==="

# Create directory structure on VM
gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
  --command="sudo mkdir -p /opt/preprocess/{scripts/database,scripts/utils,dataset,database/chroma_db} && sudo chown -R \$USER:\$USER /opt/preprocess"

# Upload scripts
gcloud compute scp \
  "$PROJECT_ROOT/scripts/preprocess_repo/extract_beliefs_from_repo.py" \
  "$PROJECT_ROOT/scripts/preprocess_repo/label_belief_edges.py" \
  "$PROJECT_ROOT/scripts/preprocess_repo/stage4_explore.py" \
  "$VM_NAME":/opt/preprocess/scripts/ --zone="$ZONE"

# Upload supporting modules — copy individual files to avoid scp double-nesting
gcloud compute scp \
  "$PROJECT_ROOT/scripts/database/__init__.py" \
  "$PROJECT_ROOT/scripts/database/db.py" \
  "$PROJECT_ROOT/scripts/database/schema.py" \
  "$VM_NAME":/opt/preprocess/scripts/database/ --zone="$ZONE"
gcloud compute scp \
  "$PROJECT_ROOT/scripts/utils/__init__.py" \
  "$PROJECT_ROOT/scripts/utils/logger.py" \
  "$VM_NAME":/opt/preprocess/scripts/utils/ --zone="$ZONE"

# Startup script expects entrypoint named preprocess_beliefs.py
gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
  --command="cp /opt/preprocess/scripts/extract_beliefs_from_repo.py /opt/preprocess/scripts/preprocess_beliefs.py"
echo "  Uploaded scripts"

# Upload dataset
gcloud compute scp \
  "$PROJECT_ROOT/dataset/test.json" \
  "$VM_NAME":/opt/preprocess/dataset/ --zone="$ZONE"
echo "  Uploaded dataset/test.json"

# Upload HF token to both expected locations
gcloud compute scp "$HF_KEY" "$VM_NAME":/opt/preprocess/.hf_token --zone="$ZONE"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
  --command="mkdir -p ~/.ssh && cp /opt/preprocess/.hf_token ~/.ssh/hf_dsy.key"
echo "  Uploaded HF token"

# Write GCS bucket path for upload step
gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
  --command="echo '$GCS_BUCKET' > /tmp/gcs_bucket"

# ---------------------------------------------------------------------------
# Trigger preprocessing
# ---------------------------------------------------------------------------
echo ""
echo "=== Starting preprocessing ==="
gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
  --command="touch /tmp/start-preprocessing && echo 'Preprocessing triggered'"

echo ""
echo "=== Preprocessing running in background ==="
echo ""
echo "Monitor with:"
echo "  ./infra/run_preprocess.sh --status"
echo ""
echo "Download results when done:"
echo "  ./infra/run_preprocess.sh --download"

# ---------------------------------------------------------------------------
# Optionally destroy VM after completion
# ---------------------------------------------------------------------------
if [ "$NO_DESTROY" = false ]; then
  echo ""
  echo "Waiting for preprocessing to complete before destroying VM..."
  MAX_TRAIN=36000  # 10 hours max
  ELAPSED=0
  while true; do
    if [ "$ELAPSED" -ge "$MAX_TRAIN" ]; then
      echo "WARNING: Timed out waiting for completion — VM left running"
      exit 0
    fi
    sleep 120
    ELAPSED=$((ELAPSED + 120))
    if gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
        --command="test -f /tmp/preprocess-complete" 2>/dev/null; then
      echo "Preprocessing complete (${ELAPSED}s)"
      break
    fi
    PROGRESS=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
      --command="tail -1 /var/log/preprocess.log 2>/dev/null | cut -c1-80" 2>/dev/null || true)
    echo "  [${ELAPSED}s] $PROGRESS"
  done

  echo "=== Destroying VM ==="
  cd "$SCRIPT_DIR"
  terraform destroy -auto-approve -target=google_compute_instance.belief_preprocessor
  echo "VM destroyed."
fi

echo ""
echo "=== Done ==="
echo "Results in GCS: $GCS_BUCKET"
echo "Download with: ./infra/run_preprocess.sh --download"
