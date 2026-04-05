# label-studio-compose

Production-ready [Label Studio](https://labelstud.io) stack: PostgreSQL · Redis · MinIO (S3) · Nginx · Cloudflare Tunnel · SAM3 interactive segmentation.

> **繁體中文說明** → [README.zh-TW.md](README.zh-TW.md)

## Stack

| Service | Image | Role |
|---------|-------|------|
| `label-studio` | heartexlabs/label-studio:latest | Labeling UI + API |
| `db` | postgres:15-alpine | Metadata store |
| `redis` | redis:7-alpine | Task queue / cache |
| `minio` | minio/minio:latest | S3-compatible object storage |
| `nginx` | nginx:1.27-alpine | Reverse proxy |
| `cloudflared` | cloudflare/cloudflared:latest | Zero Trust tunnel |
| `sam3-ml-backend` | (custom build) | SAM3 interactive segmentation *(GPU, optional)* |

## Prerequisites

- Docker Engine ≥ 26 + Docker Compose v2
- NVIDIA GPU + `nvidia-container-toolkit` (SAM3 backend only)
- Cloudflare account with Zero Trust enabled
- HuggingFace account — Meta `facebook/sam3` license accepted

## Quick Start

```bash
git clone https://github.com/felimet/label-studio-compose
cd label-studio-compose
cp .env.example .env
$EDITOR .env           # fill every <PLACEHOLDER>

make up                # start core stack
make init-minio        # create S3 bucket + policies
make create-admin      # create first superuser

make gpu               # (optional) start SAM3 ML backend on GPU
```

Connect MinIO storage in Label Studio:
**Project → Settings → Cloud Storage → Add Source Storage → S3**
(endpoint: `http://minio:9000`, use `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`).

## Makefile Reference

| Target | Description |
|--------|-------------|
| `up / down / restart / logs / ps` | Core stack lifecycle |
| `gpu / gpu-down` | SAM3 GPU overlay |
| `init-minio` | One-time bucket initialisation |
| `create-admin` | Create superuser |
| `health` | Check all services |
| `build-sam3 / test-sam3` | Build image / run pytest |
| `push` | git add + commit + push |

## Documentation

| Guide | Contents |
|-------|----------|
| [docs/configuration.md](docs/configuration.md) | `.env` variable reference |
| [docs/cloudflare-tunnel.md](docs/cloudflare-tunnel.md) | Zero Trust setup + WAF rules |
| [docs/sam3-backend.md](docs/sam3-backend.md) | SAM3 model setup + annotation workflow |
| [docs/architecture.md](docs/architecture.md) | Service topology, volumes, networking |

## License

Apache-2.0 © 2026 Jia-Ming Zhou
