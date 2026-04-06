# label-studio-compose

Production-ready [Label Studio](https://labelstud.io) stack: PostgreSQL · Redis · MinIO (S3) · Nginx · Cloudflare Tunnel · SAM3 interactive segmentation.

> **繁體中文說明** → [README.zh-TW.md](README.zh-TW.md)

## Stack

| Service | Image | Role |
|---------|-------|------|
| `label-studio` | `heartexlabs/label-studio:20260404.151117-fb-bros-956-f3692362` | Labeling UI + API |
| `db` | `postgres:17` | Metadata store |
| `redis` | `redis:8.6.2` | Task queue / cache |
| `minio` | `minio/minio:RELEASE.2025-04-22T22-12-26Z` ⚠️ | S3-compatible object storage |
| `minio-init` | `minio/mc:RELEASE.2025-08-13T08-35-41Z` | One-shot bucket + CORS setup |
| `nginx` | `nginx:1.28.3-alpine3.23` | Reverse proxy |
| `cloudflared` | `cloudflare/cloudflared:2026.3.0` | Zero Trust tunnel |
| `sam3-image-backend` | (custom build) | SAM3 image segmentation → BrushLabels *(GPU, optional)* |
| `sam3-video-backend` | (custom build) | SAM3 video object tracking → VideoRectangle *(GPU, optional)* |

> ⚠️ `minio/minio` repository archived 2026-02-13. `RELEASE.2025-04-22T22-12-26Z` patches a privilege escalation CVE; no further updates expected. Evaluate migration to Cloudflare R2 / AWS S3 for long-term use.

## Prerequisites

- Docker Engine ≥ 26 + Docker Compose v2
- NVIDIA GPU + `nvidia-container-toolkit` (SAM3 backend only)
- Cloudflare account with Zero Trust enabled
- HuggingFace account — Meta `facebook/sam3.1` license accepted

## Quick Start

```bash
git clone https://github.com/felimet/label-studio-compose
cd label-studio-compose

# 1. Core stack
cp .env.example .env
$EDITOR .env           # fill every <PLACEHOLDER>
                       # LABEL_STUDIO_USER_TOKEN: openssl rand -hex 20  (must be ≤40 chars)

make up                # start core stack (admin account auto-created on first boot)
make init-minio        # create S3 bucket + policies

# 2. Get the Label Studio API token (needed for SAM3 backends)
#    Login → Avatar (top-right) → Account & Settings → Legacy Token → Copy
#    ⚠ Must use Legacy Token (NOT Personal Access Token) — ML SDK sends
#      "Authorization: Token <key>"; PAT uses JWT Bearer → 401 Unauthorized.

# 3. SAM3 ML backends (optional, requires NVIDIA GPU)
cp .env.ml.example .env.ml
$EDITOR .env.ml        # set LABEL_STUDIO_API_KEY (from step 2) and HF_TOKEN

make ml-up
```

Connect MinIO storage in Label Studio:
**Project → Settings → Cloud Storage → Add Source Storage → S3**
(endpoint: `http://minio:9000`, use `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`).

## Makefile Reference

| Target | Description |
|--------|-------------|
| `up / down / restart / logs / ps` | Core stack lifecycle |
| `ml-up / ml-down` | SAM3 ML overlay (image + video) |
| `build-sam3-image / build-sam3-video` | Build ML backend images |
| `test-sam3-image / test-sam3-video` | Run pytest in containers |
| `init-minio` | One-time bucket initialisation |
| `create-admin` | Create superuser |
| `health` | Check all services |
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
