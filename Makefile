.PHONY: up down restart logs ps \
        ml-up ml-down \
        build-sam3-image build-sam3-video \
        test-sam3-image test-sam3-video \
        init-minio health create-admin \
        push

# ─── Core stack ─────────────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

# ─── ML Backends (SAM3 image + video) ───────────────────────
# override.yml must be included explicitly when using -f flags
# (Docker Compose only auto-loads override.yml when no -f is specified)
ML_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml

ml-up:
	$(ML_COMPOSE) up -d

ml-down:
	$(ML_COMPOSE) down

build-sam3-image:
	$(ML_COMPOSE) build sam3-image-backend

build-sam3-video:
	$(ML_COMPOSE) build sam3-video-backend

test-sam3-image:
	$(ML_COMPOSE) exec sam3-image-backend python -m pytest tests/ --tb=short -v

test-sam3-video:
	$(ML_COMPOSE) exec sam3-video-backend python -m pytest tests/ --tb=short -v

# ─── Initialisation ──────────────────────────────────────────
init-minio:
	docker compose run --rm minio-init

create-admin:
	@bash scripts/create-admin.sh

# ─── Health check ────────────────────────────────────────────
health:
	@bash scripts/healthcheck.sh

# ─── Git ─────────────────────────────────────────────────────
push:
	git add -A
	@read -p "Commit message: " msg; git commit -m "$$msg"
	git push origin main
