.PHONY: up down restart logs ps gpu gpu-down \
        init-minio health create-admin \
        test-sam3 build-sam3 \
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

# ─── GPU overlay (SAM3 backend) ──────────────────────────────
gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

gpu-down:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml down

# ─── Initialisation ──────────────────────────────────────────
init-minio:
	docker compose run --rm minio-init

create-admin:
	@bash scripts/create-admin.sh

# ─── Health check ────────────────────────────────────────────
health:
	@bash scripts/healthcheck.sh

# ─── SAM3 backend ────────────────────────────────────────────
build-sam3:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
		build sam3-ml-backend

test-sam3:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
		exec sam3-ml-backend python -m pytest tests/ --tb=short -v

# ─── Git ─────────────────────────────────────────────────────
push:
	git add -A
	@read -p "Commit message: " msg; git commit -m "$$msg"
	git push origin main
