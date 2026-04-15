#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Startup script — belief query VM
# ---------------------------------------------------------------------------

apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv curl

# Python environment
VENV="/opt/belief_query_venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
    fastapi \
    uvicorn \
    chromadb \
    sentence-transformers \
    requests

# Read config from instance metadata
PORT=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/belief_query_port" -H "Metadata-Flavor: Google")

# Copy server code (embedded here for self-contained startup)
mkdir -p /opt/belief_query

# server.py is deployed separately via scp or provisioner — see README

# Write systemd service so server survives reboots
cat > /etc/systemd/system/belief-query.service << EOF
[Unit]
Description=Belief Query API Server
After=network.target

[Service]
ExecStart=$VENV/bin/uvicorn server:app --host 0.0.0.0 --port $PORT
WorkingDirectory=/opt/belief_query
Restart=always
RestartSec=5
Environment=CHROMADB_PATH=/opt/belief_query/chromadb_data

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable belief-query
systemctl start belief-query

echo "[startup] Belief query VM ready on port $PORT"
