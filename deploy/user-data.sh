#!/bin/bash
# EC2 user-data: prepares a fresh Ubuntu 24.04 host to run the Trinetra stack.
#
# Runs once as root on first boot. Output goes to /var/log/cloud-init-output.log.
#
# This installs the runtime only. It deliberately does not fetch the application,
# because two things it needs are gitignored and must be copied from a developer
# machine: .env (secrets) and models/diagnostics.joblib (the trained bundle,
# pinned to the scikit-learn version in requirements.txt).
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io docker-compose-v2

systemctl enable --now docker
usermod -aG docker ubuntu

# A t3.small has 2 GB, and the image build peaks higher than that when the Node
# stage and the scikit-learn install overlap. Swap turns a hard OOM kill into a
# slow build. Harmless on larger instance types.
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

install -o ubuntu -g ubuntu -d /home/ubuntu/Trinetra_AI

# Marker the deploy steps can poll, so you do not scp into a half-built host.
touch /home/ubuntu/.provisioned
chown ubuntu:ubuntu /home/ubuntu/.provisioned
echo "user-data complete"
