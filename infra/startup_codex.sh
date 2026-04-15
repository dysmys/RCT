#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Startup script — codex experiment runner VM
# ---------------------------------------------------------------------------

apt-get update -qq
apt-get install -y -qq git python3 python3-pip python3-venv curl jq

# Node.js (LTS) — required for Codex CLI
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt-get install -y -qq nodejs

# Codex CLI
npm i -g @openai/codex

# Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash

# Mount the persistent repo disk
DISK_DEV="/dev/disk/by-id/google-codex-repo-store"
MOUNT_POINT="/mnt/repos"

mkdir -p "$MOUNT_POINT"

# Format only if not already formatted
if ! blkid "$DISK_DEV" > /dev/null 2>&1; then
    mkfs.ext4 -F "$DISK_DEV"
fi

mount -o discard,defaults "$DISK_DEV" "$MOUNT_POINT"

# Persist mount across reboots
DISK_UUID=$(blkid -s UUID -o value "$DISK_DEV")
echo "UUID=$DISK_UUID $MOUNT_POINT ext4 discard,defaults,nofail 0 2" >> /etc/fstab

chmod 777 "$MOUNT_POINT"

# Python environment (for experiment runner scripts)
VENV="/opt/experiment_venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
    requests \
    python-dotenv \
    tqdm \
    openai \
    anthropic \
    chromadb \
    sentence-transformers

# Pull config from instance metadata
SENG_URL=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/seng_inference_url" -H "Metadata-Flavor: Google")

cat > /opt/experiment.env <<EOF
SENG_INFERENCE_URL=${SENG_URL}
REPO_DIR=${MOUNT_POINT}
VENV=${VENV}
AGENTS=claude,codex
EOF

echo "[startup] Multi-agent experiment VM ready (Claude + Codex). Repos at $MOUNT_POINT"
