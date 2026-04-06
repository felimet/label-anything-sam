# label-studio-compose

生產就緒的 [Label Studio](https://labelstud.io) 完整部署方案：PostgreSQL · Redis · MinIO (S3) · Nginx · Cloudflare Tunnel · SAM3 互動式影像分割。

> **English documentation** → [README.md](README.md)

## 服務架構

| 服務 | 映像 | 用途 |
|------|------|------|
| `label-studio` | `heartexlabs/label-studio:20260404.151117-fb-bros-956-f3692362` | 標注 UI + API |
| `db` | `postgres:17` | 資料庫 |
| `redis` | `redis:8.6.2` | 任務佇列 / 快取 |
| `minio` | `minio/minio:RELEASE.2025-04-22T22-12-26Z` ⚠️ | S3 相容物件儲存 |
| `minio-init` | `minio/mc:RELEASE.2025-08-13T08-35-41Z` | 一次性 bucket + CORS 初始化 |
| `nginx` | `nginx:1.28.3-alpine3.23` | 反向代理 |
| `cloudflared` | `cloudflare/cloudflared:2026.3.0` | Zero Trust Tunnel |
| `sam3-image-backend` | (自訂建置) | SAM3 影像分割 → BrushLabels *(需 GPU，可選)* |
| `sam3-video-backend` | (自訂建置) | SAM3 影片物件追蹤 → VideoRectangle *(需 GPU，可選)* |

> ⚠️ `minio/minio` 儲存庫已於 2026-02-13 封存，不再更新。`RELEASE.2025-04-22T22-12-26Z` 修補一個權限提升 CVE；長期使用建議評估遷移至 Cloudflare R2 或 AWS S3。

## 前置需求

- Docker Engine ≥ 26 + Docker Compose v2
- NVIDIA GPU + `nvidia-container-toolkit`（僅 SAM3 後端需要）
- Cloudflare 帳號，已開啟 Zero Trust
- HuggingFace 帳號，已同意 Meta `facebook/sam3.1` 使用條款

## 快速開始

```bash
git clone https://github.com/felimet/label-studio-compose
cd label-studio-compose

# 1. 核心服務
cp .env.example .env
$EDITOR .env           # 填入所有 <PLACEHOLDER> 值
                       # LABEL_STUDIO_USER_TOKEN: openssl rand -hex 20（必須 ≤40 字元）

make up                # 啟動核心服務（管理員帳號於首次啟動時自動建立）
make init-minio        # 建立 S3 儲存桶 + 存取政策

# 2. 取得 Label Studio API Token（SAM3 後端需要）
#    登入 → 右上角頭像 → Account & Settings → Legacy Token → 複製
#    ⚠ 必須使用 Legacy Token（不可用 Personal Access Token）——ML SDK 傳送
#      "Authorization: Token <key>"；PAT 使用 JWT Bearer → 401 Unauthorized。

# 3. SAM3 ML 後端（可選，需 NVIDIA GPU）
cp .env.ml.example .env.ml
$EDITOR .env.ml        # 填入 LABEL_STUDIO_API_KEY（步驟 2）及 HF_TOKEN

make ml-up
```

在 Label Studio 中連接 MinIO 儲存：
**專案 → Settings → Cloud Storage → Add Source Storage → S3**
（endpoint: `http://minio:9000`，使用 `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`）

## Makefile 指令

| 指令 | 說明 |
|------|------|
| `up / down / restart / logs / ps` | 核心服務生命週期管理 |
| `ml-up / ml-down` | SAM3 ML 疊加層（影像 + 影片） |
| `build-sam3-image / build-sam3-video` | 建置 ML 後端映像 |
| `test-sam3-image / test-sam3-video` | 在容器內執行 pytest |
| `init-minio` | 一次性儲存桶初始化 |
| `create-admin` | 建立管理員帳號 |
| `health` | 檢查所有服務狀態 |
| `push` | git add + commit + push |

## 文件

| 文件 | 內容 |
|------|------|
| [docs/configuration.md](docs/configuration.md) | `.env` 環境變數說明 |
| [docs/cloudflare-tunnel.md](docs/cloudflare-tunnel.md) | Zero Trust 設定 + WAF 規則 |
| [docs/sam3-backend.md](docs/sam3-backend.md) | SAM3 模型設定 + 標注流程 |
| [docs/architecture.md](docs/architecture.md) | 服務拓撲、Volume、網路 |

## 授權

Apache-2.0 © 2026 Jia-Ming Zhou
