#!/bin/bash
set -euo pipefail

LOG=/var/log/rct-startup.log
exec > >(tee -a "$LOG") 2>&1

echo "[startup] $(date -u) — RCT runner VM starting"

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------

apt-get update -qq
apt-get install -y -qq \
    git python3 python3-pip python3-venv \
    curl jq ca-certificates gnupg \
    build-essential libssl-dev

# ---------------------------------------------------------------------------
# Docker — isolated test execution
# ---------------------------------------------------------------------------

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin
systemctl enable --now docker

# ---------------------------------------------------------------------------
# Node.js (LTS) + Codex CLI + Claude Code CLI
# ---------------------------------------------------------------------------

curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt-get install -y -qq nodejs

npm install -g @openai/codex
curl -fsSL https://claude.ai/install.sh | bash

echo "[startup] Node $(node --version), Codex and Claude CLI installed"

# ---------------------------------------------------------------------------
# Mount persistent repo disk
# ---------------------------------------------------------------------------

DISK_DEV="/dev/disk/by-id/google-repo-store"
MOUNT_POINT="/mnt/repos"

mkdir -p "$MOUNT_POINT"

if ! blkid "$DISK_DEV" > /dev/null 2>&1; then
    echo "[startup] Formatting repo disk..."
    mkfs.ext4 -F "$DISK_DEV"
fi

mount -o discard,defaults "$DISK_DEV" "$MOUNT_POINT"

DISK_UUID=$(blkid -s UUID -o value "$DISK_DEV")
echo "UUID=$DISK_UUID $MOUNT_POINT ext4 discard,defaults,nofail 0 2" >> /etc/fstab

chmod 777 "$MOUNT_POINT"
echo "[startup] Repo disk mounted at $MOUNT_POINT"

# ---------------------------------------------------------------------------
# Python virtualenv
# ---------------------------------------------------------------------------

VENV="/opt/experiment_venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
    requests \
    python-dotenv \
    tqdm \
    anthropic \
    openai \
    chromadb \
    sentence-transformers \
    pytest

echo "[startup] Python venv ready at $VENV"

# ---------------------------------------------------------------------------
# /opt/experiment.env — picked up by run scripts
# ---------------------------------------------------------------------------

cat > /opt/experiment.env <<EOF
REPO_DIR=${MOUNT_POINT}
VENV=${VENV}
WORKERS=16
EOF

echo "[startup] $(date -u) — RCT runner VM ready. Repos at $MOUNT_POINT, workers=16"
