#!/usr/bin/env bash
set -e

export http_proxy=http://proxy-dmz.intel.com:912
export https_proxy=http://proxy-dmz.intel.com:912
export no_proxy=localhost,127.0.0.1,::1

echo "==> [1/4] Adding Docker GPG key..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL -x "$https_proxy" https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "==> [2/4] Adding Docker apt repository..."
ARCH=$(dpkg --print-architecture)
CODENAME=$(lsb_release -cs)
echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "==> [3/4] Installing Docker Engine..."
sudo -E apt-get update -qq
sudo -E apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> [4/4] Starting Docker daemon..."
sudo service docker start

echo ""
echo "Docker installed successfully:"
docker --version

echo ""
echo "==> Configuring Docker daemon to use proxy..."
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<EOF
[Service]
Environment="HTTP_PROXY=http://proxy-dmz.intel.com:912"
Environment="HTTPS_PROXY=http://proxy-dmz.intel.com:912"
Environment="NO_PROXY=localhost,127.0.0.1"
EOF

echo "Done! Run 'sudo usermod -aG docker \$USER && newgrp docker' to use Docker without sudo."
