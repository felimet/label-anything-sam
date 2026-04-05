# Runbook

Operations reference for the Label Studio production stack.

## Health Checks

```bash
make health          # full stack check (PostgreSQL · Redis · MinIO · LS · Nginx · SAM3)
```

Individual service health endpoints:

| Service | Endpoint | Expected |
|---------|----------|---------- |
| Label Studio | `http://localhost:8085/health` *(dev port)* | HTTP 200 |
| nginx | `http://localhost:8090/health` *(dev port)* | `OK` |
| sam3-image-backend | `http://sam3-image-backend:9090/health` *(internal)* | HTTP 200 |
| sam3-video-backend | `http://sam3-video-backend:9090/health` *(internal)* | HTTP 200 |
| MinIO | `mc ready local` *(via docker exec)* | `The cluster is ready` |

## Deployment

### First-time deployment

```bash
cp .env.example .env
# Fill in all <PLACEHOLDER> values — see docs/configuration.md
# Generate secrets:
openssl rand -hex 32    # LABEL_STUDIO_SECRET_KEY, LABEL_STUDIO_USER_TOKEN
openssl rand -base64 24 # POSTGRES_PASSWORD, REDIS_PASSWORD, MINIO_ROOT_PASSWORD

docker compose up -d
make init-minio         # create bucket + CORS (one-time)
make health
```

### Upgrade services

```bash
# 1. Update version pins in .env (LABEL_STUDIO_VERSION, POSTGRES_VERSION, etc.)
# 2. Pull new images
docker compose pull

# 3. Rolling restart (LS → nginx → cloudflared)
docker compose up -d --no-deps label-studio
docker compose up -d --no-deps nginx
docker compose up -d --no-deps cloudflared

# 4. Verify
make health
```

### Start ML backends (GPU)

```bash
make ml-up             # core stack + sam3-image-backend + sam3-video-backend
make health            # includes SAM3 checks
```

After startup, register each backend in Label Studio:

1. **Project → Settings → Machine Learning → Add Model**
2. Image backend URL: `http://sam3-image-backend:9090`
3. Video backend URL: `http://sam3-video-backend:9090`
4. Click **Validate and Save**, enable **Auto-Annotation**

## Rollback

### Application rollback

```bash
# Revert LABEL_STUDIO_VERSION in .env to previous value, then:
docker compose up -d --no-deps label-studio
```

> PostgreSQL schema migrations are not automatically reversed. If the new version ran `migrate`, rolling back may require a database restore.

### Database restore from backup

```bash
docker compose stop label-studio
docker compose exec db psql -U labelstudio -c "DROP DATABASE labelstudio;"
docker compose exec db psql -U labelstudio -c "CREATE DATABASE labelstudio;"
docker compose exec -T db psql -U labelstudio labelstudio < backup.sql
docker compose start label-studio
```

## Common Issues

### Label Studio won't start — "database does not exist"

```bash
# Check postgres health
docker compose exec db pg_isready -U labelstudio

# Inspect init log
docker compose logs db | grep -i error
```

If the database was never created, re-run init:
```bash
docker compose down
docker volume rm label-studio_postgres-data   # ⚠️ deletes all data
docker compose up -d db
docker compose logs -f db  # wait for "database system is ready"
```

### MinIO bucket not found

```bash
make init-minio
# Or manually:
docker compose run --rm minio-init
```

### SAM3 backend — model download fails

Common causes:
- `HF_TOKEN` not set or expired
- Meta license not accepted at https://huggingface.co/facebook/sam3.1

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  logs sam3-image-backend | grep -i error
```

Fix: update `HF_TOKEN` in `.env`, then:
```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  restart sam3-image-backend sam3-video-backend
```

### SAM3 backend — CUDA out of memory

Reduce concurrent workers (already set to 1 in Dockerfile CMD). If OOM still occurs:

1. Close other GPU workloads
2. Use `DEVICE=cpu` in `.env` (very slow, no GPU required)
3. Try a smaller model: `SAM3_IMAGE_MODEL_ID=facebook/sam3`

### Redis connection refused

```bash
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" ping
# Expected: PONG
docker compose logs redis | tail -20
```

### nginx 502 Bad Gateway

```bash
docker compose ps label-studio          # check healthy status
docker compose logs label-studio --tail=50
```

Label Studio typically takes 60–90 s to start on first boot.

## Log Access

```bash
make logs                                     # all services, last 100 lines
docker compose logs -f label-studio           # LS only
docker compose logs -f db redis               # infra only

# ML backends
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  logs -f sam3-image-backend sam3-video-backend
```

Structured JSON logs are enabled (`JSON_LOG=1`). Use `jq` to filter:

```bash
docker compose logs label-studio 2>&1 | jq 'select(.level=="ERROR")'
```

## Backup

### Data volumes

```bash
# Label Studio media / exports (bind mount — already on host)
tar -czf ls-data-$(date +%Y%m%d).tar.gz ./label-studio-data/

# PostgreSQL
docker compose exec db pg_dump -U labelstudio labelstudio \
  > backup-$(date +%Y%m%d).sql

# MinIO (via mc mirror)
docker compose exec minio mc mirror local/label-studio-bucket /backup/minio/
```

### Exclude from backup

- `redis-data` — ephemeral task queue; tasks re-queued on restart
- `hf-cache`, `sam3-image-models`, `sam3-video-models` — re-downloadable from HuggingFace

## Monitoring

No bundled monitoring stack. Recommended additions:

| Tool | Purpose |
|------|---------|
| Prometheus + cAdvisor | Container resource metrics |
| Loki + Promtail | Log aggregation (forward docker json logs) |
| Grafana | Dashboard for above |
| Uptime Kuma | External endpoint health checks |

Cloudflare provides basic WAF analytics and request metrics for public endpoints.
