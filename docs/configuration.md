# 環境變數設定說明

所有變數定義於 `.env`（從 `.env.example` 複製後填入）。

## PostgreSQL

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `POSTGRES_USER` | `labelstudio` | 資料庫使用者名稱 |
| `POSTGRES_PASSWORD` | — | **必填。** 強隨機密碼 |
| `POSTGRES_DB` | `labelstudio` | 資料庫名稱 |

## Redis

| 變數 | 說明 |
|------|------|
| `REDIS_PASSWORD` | **必填。** Redis AUTH 密碼 |

## MinIO

| 變數 | 範例 | 說明 |
|------|------|------|
| `MINIO_ROOT_USER` | — | 管理員帳號 |
| `MINIO_ROOT_PASSWORD` | — | 管理員密碼（≥8 字元） |
| `MINIO_BUCKET` | `label-studio-bucket` | 儲存桶名稱；由 `make init-minio` 自動建立 |
| `MINIO_EXTERNAL_HOST` | `minio.example.com` | 對外公開網域；嵌入 Presigned URL |

> **重要：** `MINIO_EXTERNAL_HOST` 必須可從瀏覽器端解析。MinIO 用此值產生 Presigned URL，Label Studio 內部請求仍走 `http://minio:9000`。

## Label Studio

| 變數 | 範例 | 說明 |
|------|------|------|
| `LABEL_STUDIO_HOST` | `https://label-studio.example.com` | 對外公開 URL；用於 CSRF 信任來源 |
| `LABEL_STUDIO_SECRET_KEY` | `openssl rand -hex 32` | Django Session 金鑰 |
| `LABEL_STUDIO_USERNAME` | `admin@example.com` | 初始管理員 Email |
| `LABEL_STUDIO_PASSWORD` | — | 初始管理員密碼 |
| `LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK` | `true` | 關閉公開註冊（需邀請連結） |

## Cloudflare Tunnel

| 變數 | 說明 |
|------|------|
| `CLOUDFLARE_TUNNEL_TOKEN` | Zero Trust 儀表板產生的 Tunnel Token |
| `MINIO_EXTERNAL_HOST` | MinIO 公開網域（同上，雙重用途） |

詳細設定步驟見 [cloudflare-tunnel.md](cloudflare-tunnel.md)。

## SAM3 ML 後端

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `HF_TOKEN` | — | HuggingFace Token；下載 `facebook/sam3` 必填 |
| `LABEL_STUDIO_API_KEY` | — | LS API 金鑰；首次啟動後從 Settings → Access Tokens 取得 |
| `SAM3_MODEL_ID` | `facebook/sam3` | HuggingFace Hub 模型 ID |
| `DEVICE` | `cuda` | `cuda`（GPU）或 `cpu`（備援） |
| `EMBED_CACHE_SIZE` | `50` | 記憶體中最大快取影像數 |
| `EMBED_CACHE_TTL` | `300` | 快取 TTL（秒） |

## 產生強密碼

```bash
# Django Secret Key
openssl rand -hex 32

# 資料庫 / Redis / MinIO 密碼
openssl rand -base64 24
```
