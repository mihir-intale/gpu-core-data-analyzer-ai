#!/usr/bin/env bash
set -e

# /etc/default/docker — picked up by sysvinit service script
cat > /tmp/docker_default <<'EOF'
export http_proxy="http://proxy-dmz.intel.com:912"
export https_proxy="http://proxy-dmz.intel.com:912"
export no_proxy="localhost,127.0.0.1,::1"
EOF
sudo cp /tmp/docker_default /etc/default/docker

# /etc/docker/daemon.json — proxy block for dockerd itself
sudo mkdir -p /etc/docker
cat > /tmp/daemon.json <<'EOF'
{
  "proxies": {
    "http-proxy": "http://proxy-dmz.intel.com:912",
    "https-proxy": "http://proxy-dmz.intel.com:912",
    "no-proxy": "localhost,127.0.0.1"
  }
}
EOF
sudo cp /tmp/daemon.json /etc/docker/daemon.json

# Restart daemon picking up env proxy
sudo service docker stop 2>/dev/null || true
sudo HTTP_PROXY=http://proxy-dmz.intel.com:912 \
     HTTPS_PROXY=http://proxy-dmz.intel.com:912 \
     NO_PROXY=localhost,127.0.0.1 \
     service docker start
sleep 2

echo "==> Testing docker pull (hello-world)..."
sudo docker pull hello-world
echo "==> Proxy configured successfully."
