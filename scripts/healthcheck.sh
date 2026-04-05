#!/bin/bash
# Full stack health check — validates all services are responsive.
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILED=1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

FAILED=0

echo "═══════════════════════════════════"
echo "  Label Studio Stack Health Check"
echo "═══════════════════════════════════"

# ── PostgreSQL ──────────────────────────────────────────────
echo ""
echo "── PostgreSQL ──"
if docker compose exec -T db pg_isready -q 2>/dev/null; then
    pass "PostgreSQL accepting connections"
else
    fail "PostgreSQL not ready"
fi

# ── Redis ───────────────────────────────────────────────────
echo ""
echo "── Redis ──"
REDIS_PASS=$(grep REDIS_PASSWORD .env 2>/dev/null | cut -d= -f2 || echo "")
if docker compose exec -T redis redis-cli -a "$REDIS_PASS" ping 2>/dev/null | grep -q PONG; then
    pass "Redis PONG"
else
    fail "Redis not responding"
fi

# ── MinIO ───────────────────────────────────────────────────
echo ""
echo "── MinIO ──"
if docker compose exec -T minio mc ready local 2>/dev/null; then
    pass "MinIO API healthy"
else
    fail "MinIO not ready"
fi

BUCKET=$(grep MINIO_BUCKET .env 2>/dev/null | cut -d= -f2 || echo "label-studio-bucket")
if docker compose exec -T minio mc ls "local/${BUCKET}" 2>/dev/null; then
    pass "MinIO bucket '${BUCKET}' accessible"
else
    warn "MinIO bucket '${BUCKET}' not found — run: make init-minio"
fi

# ── Label Studio ────────────────────────────────────────────
echo ""
echo "── Label Studio ──"
if docker compose exec -T label-studio curl -sf http://localhost:8080/health 2>/dev/null; then
    pass "Label Studio /health OK"
else
    fail "Label Studio not responding"
fi

# ── Nginx ───────────────────────────────────────────────────
echo ""
echo "── Nginx ──"
if docker compose exec -T nginx curl -sf http://localhost/health 2>/dev/null | grep -q OK; then
    pass "Nginx proxy healthy"
else
    fail "Nginx not responding"
fi

# ── SAM3 ML Backend (optional — only when GPU stack running) ─
echo ""
echo "── SAM3 ML Backend (GPU stack) ──"
if docker compose ps sam3-ml-backend 2>/dev/null | grep -q running; then
    if docker compose exec -T sam3-ml-backend curl -sf http://localhost:9090/health 2>/dev/null; then
        pass "SAM3 backend /health OK"
    else
        fail "SAM3 backend not responding"
    fi
else
    warn "SAM3 backend not running (start with: make gpu)"
fi

# ── Cloudflared ─────────────────────────────────────────────
echo ""
echo "── Cloudflare Tunnel ──"
if docker compose ps cloudflared 2>/dev/null | grep -q running; then
    pass "cloudflared container running"
else
    warn "cloudflared not running"
fi

echo ""
echo "═══════════════════════════════════"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}All checks passed.${NC}"
else
    echo -e "${RED}${FAILED} check(s) failed — see above.${NC}"
    exit 1
fi
