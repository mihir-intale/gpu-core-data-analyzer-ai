# gpu-core-data-analyzer-ai

## Docker Setup (WSL Ubuntu)

### 1. Install Docker Engine in WSL

Run the provided install script from the project root inside your WSL terminal.  
It handles adding the Docker apt repository and starting the daemon.

```bash
chmod +x install_docker_wsl.sh
./install_docker_wsl.sh
```

Grant your user Docker access (takes effect on next WSL session):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Configure Docker Daemon Proxy

Required on Intel corporate network. Run the proxy config script once:

```bash
chmod +x configure_docker_proxy.sh
./configure_docker_proxy.sh
```

This sets `proxy-dmz.intel.com:912` for all Docker daemon pulls.

### 3. Build the Image

From the project root (`c:\Code\gpu-core-data-analyzer-ai` or its WSL mount path):

```bash
cd /mnt/c/Code/gpu-core-data-analyzer-ai

sudo DOCKER_BUILDKIT=0 docker build \
  --build-arg http_proxy=http://proxy-dmz.intel.com:912 \
  --build-arg https_proxy=http://proxy-dmz.intel.com:912 \
  --build-arg HTTP_PROXY=http://proxy-dmz.intel.com:912 \
  --build-arg HTTPS_PROXY=http://proxy-dmz.intel.com:912 \
  --build-arg no_proxy=localhost,127.0.0.1 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t gpu-core-analyzer .
```

> `DOCKER_BUILDKIT=0` is required so the legacy builder inherits the daemon's proxy environment for pulling base images.

### 4. Run the Container

```bash
sudo docker run -d --name gpu-core-analyzer -p 5000:5000 gpu-core-analyzer
```

Verify it is healthy:

```bash
curl http://localhost:5000/health
# {"status":"healthy","ollama":"unreachable","model":"gpt-oss:20b"}
```

### Useful Commands

| Action | Command |
|---|---|
| View logs | `sudo docker logs gpu-core-analyzer` |
| Stop container | `sudo docker stop gpu-core-analyzer` |
| Remove container | `sudo docker rm gpu-core-analyzer` |
| Rebuild image | `sudo docker rm -f gpu-core-analyzer && docker build ...` |
| Start Docker daemon (WSL reboot) | `sudo service docker start` |
