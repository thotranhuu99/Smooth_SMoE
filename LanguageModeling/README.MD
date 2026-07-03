# Language Modeling with Smooth MoE

## Prerequisites

- Docker with NVIDIA runtime support
- Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

## Quick Start

### 1. Setup Docker Environment

Run the containerized environment using Docker Compose:

```yaml
version: '3.8'

services:
  moe_nfn:
    image: anonymizedacamedia/smooth_moe:latest
    container_name: smooth_moe
    runtime: nvidia  # Use the NVIDIA runtime for GPU support
    environment:
      - NVIDIA_VISIBLE_DEVICES=all  # Make sure all GPUs are visible inside the container
    command: bash  # Command to run inside the container
    stdin_open: true  # Enable interactive mode
    tty: true  # Allocate a pseudo-TTY
    restart: always  # Optionally, restart the container if it stops
```

### 2. Start the Container

```bash
docker-compose up -d
```

### 3. Access the Container

```bash
docker exec -it smooth_moe bash
```

### 4. Run Experiments

Once inside the container, follow the instructions at `/workspace/Smooth_MoE_Private/README.md` to run the experiments. All datasets and environment dependencies are automatically downloaded and configured.