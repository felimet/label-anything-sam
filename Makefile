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
ml-up:
	docker compose -f docker-compose.yml -f docker-compose.ml.yml up -d

ml-down:
	docker compose -f docker-compose.yml -f docker-compose.ml.yml down

build-sam3-image:
	docker compose -f docker-compose.yml -f docker-compose.ml.yml \
		build sam3-image-backend

build-sam3-video:
	docker compose -f docker-compose.yml -f docker-compose.ml.yml \
		build sam3-video-backend

test-sam3-image:
	docker compose -f docker-compose.yml -f docker-compose.ml.yml \
		exec sam3-image-backend python -m pytest tests/ --tb=short -v

test-sam3-video:
	docker compose -f docker-compose.yml -f docker-compose.ml.yml \
		exec sam3-video-backend python -m pytest tests/ --tb=short -v

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
