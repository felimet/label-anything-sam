# 系統架構

## 服務拓撲

```mermaid
graph TD
    Internet([外部網際網路])
    CFEdge[Cloudflare Edge<br/>WAF · DDoS 防護]
    CF[cloudflared 容器<br/>主動出站連線]
    Nginx[nginx:80<br/>反向代理]
    MinioAPI[minio:9000<br/>S3 API + Presigned URL]
    LS[label-studio:8080<br/>Django 應用]
    DB[(db:5432<br/>PostgreSQL)]
    Redis[(redis:6379<br/>任務佇列)]
    SAM3I[sam3-image-backend:9090<br/>SAM3 影像分割]
    SAM3V[sam3-video-backend:9090<br/>SAM3 影片追蹤]

    Internet --> CFEdge
    CFEdge -- "label-studio.example.com" --> CF
    CFEdge -- "minio.example.com<br/>WAF: GET/HEAD only" --> CF
    CF --> Nginx
    CF --> MinioAPI
    Nginx --> LS
    LS --> DB
    LS --> Redis
    LS -. "內部 S3 呼叫" .-> MinioAPI
    SAM3I -. "ML overlay<br/>opt-in" .-> LS
    SAM3V -. "ML overlay<br/>opt-in" .-> LS
```

## 服務啟動相依關係

```mermaid
graph LR
    DB[db<br/>healthy]
    Redis[redis<br/>healthy]
    Minio[minio<br/>healthy]
    MinioInit[minio-init<br/>手動 make init-minio]
    LS[label-studio<br/>healthy]
    SAM3I[sam3-image-backend<br/>ML overlay]
    SAM3V[sam3-video-backend<br/>ML overlay]
    Nginx[nginx]
    CF[cloudflared]

    DB --> LS
    Redis --> LS
    Minio --> LS
    Minio -.-> MinioInit
    LS --> SAM3I
    LS --> SAM3V
    LS --> Nginx
    Nginx --> CF
```

## Presigned URL 資料流

```mermaid
sequenceDiagram
    participant B as 瀏覽器
    participant CF as Cloudflare Edge<br/>(WAF: GET/HEAD)
    participant M as minio:9000
    participant LS as label-studio

    LS->>M: 內部 S3 PUT（上傳標注資料）
    M-->>LS: OK + Presigned URL<br/>(HMAC-SHA256, 時效限制)
    LS-->>B: 回傳含 Presigned URL 的回應

    B->>CF: GET minio.example.com/bucket/file?X-Amz-Signature=...
    CF->>M: 轉發（WAF 驗證通過）
    M-->>B: 檔案內容
```

## Docker Volumes

| Volume / 路徑 | 類型 | 掛載服務 | 內容 |
|---------------|------|----------|------|
| `postgres-data` | named volume | db | PostgreSQL 資料檔 |
| `redis-data` | named volume | redis | Redis AOF / RDB |
| `minio-data` | named volume | minio | 物件儲存資料 |
| `./label-studio-data` | bind mount | label-studio | 媒體檔、匯出、上傳；host 端可直接觀察 |
| `hf-cache` | named volume | sam3-image-backend, sam3-video-backend | HuggingFace Hub 快取（`~/.cache/huggingface`） |
| `sam3-image-models` | named volume | sam3-image-backend | SAM3 影像模型權重（`/data/models`） |
| `sam3-video-models` | named volume | sam3-video-backend | SAM3 影片模型權重（`/data/models`） |

## 內部網路

所有服務共用 `internal` bridge 網路（`172.20.0.0/16`）。正式環境無任何埠號暴露於主機，所有流量由 cloudflared 進入。

本機開發（`docker-compose.override.yml`）額外暴露：

| 服務 | 主機埠號 |
|------|----------|
| nginx | 8090 |
| label-studio | 8085 |
| minio API | 19000 |
| minio console | 19001 |
| postgres | 5433 |
| redis | 6380 |

## SAM3 ML 疊加層

SAM3 後端為**選用疊加層**，定義於 `docker-compose.ml.yml`。兩個後端共用同一個 `internal` bridge 網路（由 base compose 建立，project name `label-studio`）：

```bash
make up       # 核心服務（不含 SAM3，無需 GPU）
make ml-up    # 核心服務 + SAM3 影像 + 影片後端（需 NVIDIA GPU）
make ml-down  # 停止所有服務（含核心 + SAM3）
```

`docker-compose.ml.yml` 不設定 `name:` 欄位，避免覆蓋 base project name 而造成網路隔離。

## 安全設計決策

| 決策 | 理由 |
|------|------|
| MinIO WAF：僅 GET/HEAD | Presigned URL 已有 HMAC 驗證；防止資料竄改與儲存桶列舉 |
| MinIO 不使用 CF Access | CF Access 會攔截 Presigned URL，破壞瀏覽器直接存取 |
| `SSRF_PROTECTION_ENABLED=false` | Label Studio 需呼叫內部 `minio:9000`；僅對可信內部網段放行 |
| 非 root 使用者（uid 1001） | SAM3 容器與 Label Studio 容器均以非 root 身份執行 |
