#!/bin/bash

sudo -i <<EOF
echo "[INFO] Sourcing PYNQ environment..."
source /etc/profile.d/pynq_venv.sh

echo "[INFO] Changing to demo directory..."
cd /home/ubuntu/demo

echo "[INFO] Running classifier..."
python classifier.py test.jpg
EOF