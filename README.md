# label-studio-compose

Production-ready [Label Studio](https://labelstud.io) stack: PostgreSQL · Redis · MinIO (S3) · Nginx · Cloudflare Tunnel · SAM3 interactive segmentation.

> **繁體中文說明** → [README.zh-TW.md](README.zh-TW.md)

## Stack

| Service | Image | Role |
|---------|-------|------|
| `label-studio` | `heartexlabs/label-studio:20260404.151117-fb-bros-956-f3692362` | Labeling UI + API |
| `db` | `postgres:15.17-alpine3.21` | Metadata store |
| `redis` | `redis:7.4.5-alpine3.21` | Task queue / cache |
| `minio` | `minio/minio:RELEASE.2025-10-15T17-29-55Z` ⚠️ | S3-compatible object storage |
| `minio-init` | `minio/mc:RELEASE.2025-08-13T08-35-41Z` | One-shot bucket + CORS setup |
| `nginx` | `nginx:1.28.3-alpine3.23` | Reverse proxy |
| `cloudflared` | `cloudflare/cloudflared:2026.3.0` | Zero Trust tunnel |
| `sam3-ml-backend` | (custom build) | SAM3 interactive segmentation *(GPU, optional)* |

> ⚠️ `minio/minio` repository archived 2026-02-13. `RELEASE.2025-10-15T17-29-55Z` is the final release (CVE fix). Evaluate migration to Cloudflare R2 / AWS S3 for long-term use.

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
                       # set LABEL_STUDIO_USER_TOKEN=<openssl rand -hex 32>
                       # set LABEL_STUDIO_API_KEY separately (dedicated service token)

make up                # start core stack (admin account auto-created on first boot)
make init-minio        # create S3 bucket + policies

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
