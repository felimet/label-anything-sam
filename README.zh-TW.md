# label-studio-compose

生產就緒的 [Label Studio](https://labelstud.io) 完整部署方案：PostgreSQL · Redis · MinIO (S3) · Nginx · Cloudflare Tunnel · SAM3 互動式影像分割。

> **English documentation** → [README.md](README.md)

## 服務架構

| 服務 | 映像 | 用途 |
|------|------|------|
| `label-studio` | heartexlabs/label-studio:latest | 標注 UI + API |
| `db` | postgres:15-alpine | 資料庫 |
| `redis` | redis:7-alpine | 任務佇列 / 快取 |
| `minio` | minio/minio:latest | S3 相容物件儲存 |
| `nginx` | nginx:1.27-alpine | 反向代理 |
| `cloudflared` | cloudflare/cloudflared:latest | Zero Trust Tunnel |
| `sam3-ml-backend` | (自訂建置) | SAM3 互動分割 *(需 GPU，可選)* |

## 前置需求

- Docker Engine ≥ 26 + Docker Compose v2
- NVIDIA GPU + `nvidia-container-toolkit`（僅 SAM3 後端需要）
- Cloudflare 帳號，已開啟 Zero Trust
- HuggingFace 帳號，已同意 Meta `facebook/sam3` 使用條款

## 快速開始

```bash
git clone https://github.com/felimet/label-studio-compose
cd label-studio-compose
cp .env.example .env
$EDITOR .env           # 填入所有 <PLACEHOLDER> 值

make up                # 啟動核心服務
make init-minio        # 建立 S3 儲存桶 + 存取政策
make create-admin      # 建立第一位管理員

make gpu               # （可選）啟動 SAM3 GPU ML 後端
```

在 Label Studio 中連接 MinIO 儲存：
**專案 → Settings → Cloud Storage → Add Source Storage → S3**
（endpoint: `http://minio:9000`，使用 `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`）

## Makefile 指令

| 指令 | 說明 |
|------|------|
| `up / down / restart / logs / ps` | 核心服務生命週期管理 |
| `gpu / gpu-down` | SAM3 GPU 疊加層 |
| `init-minio` | 一次性儲存桶初始化 |
| `create-admin` | 建立管理員帳號 |
| `health` | 檢查所有服務狀態 |
| `build-sam3 / test-sam3` | 建置映像 / 執行測試 |
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
