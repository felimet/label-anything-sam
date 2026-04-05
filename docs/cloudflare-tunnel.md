# Cloudflare Tunnel 設定

本專案透過單一 `cloudflared` 容器對外暴露兩條路由。

## 路由規劃

| 公開網域 | 內部目標 | 存取控制 |
|----------|----------|----------|
| `label-studio.example.com` | `http://nginx:80` | 可選加 CF Access（SSO） |
| `minio.example.com` | `http://minio:9000` | WAF 規則：僅允許 GET/HEAD |

> **不對 MinIO 使用 CF Access 的原因：** Presigned URL 已內嵌 HMAC-SHA256 驗證且有時效限制，若再套 CF Access 登入牆，瀏覽器將無法直接存取 Presigned 下載 / 上傳連結。WAF 規則已提供足夠防護。

## 步驟一：建立 Tunnel

1. 登入 [Cloudflare Zero Trust](https://one.dash.cloudflare.com) → **Networks → Tunnels → Create a tunnel**
2. 選擇 **Cloudflared** connector 類型
3. 為 Tunnel 命名（例如 `label-studio`）
4. 複製 Tunnel Token → 貼入 `.env` 的 `CLOUDFLARE_TUNNEL_TOKEN`
5. 儲存後不需要在本機安裝 cloudflared（容器會自動連線）

## 步驟二：設定 Public Hostnames

在 Tunnel 的 **Public Hostnames** 頁籤新增：

| Subdomain | Domain | Service |
|-----------|--------|---------|
| `label-studio` | `example.com` | `http://nginx:80` |
| `minio` | `example.com` | `http://minio:9000` |

> cloudflared 容器以**主動出站**方式連線至 Cloudflare Edge，主機防火牆無需開放入站埠號。

## 步驟三：設定 MinIO WAF 規則

前往 **Security → WAF → Custom Rules**，針對 `minio.example.com` 新增以下規則：

### 規則 1：封鎖非 GET/HEAD 請求

```
(http.host eq "minio.example.com") and not (http.request.method in {"GET" "HEAD"})
```

動作：**Block**

### 規則 2：封鎖儲存桶列舉

```
(http.host eq "minio.example.com") and (
  (http.request.uri.query contains "list-type") or
  (http.request.uri.query contains "list-objects") or
  (http.request.uri.path eq "/") or
  (http.request.uri.path matches "^/[^/]+/?$" and not http.request.uri.query contains "X-Amz-Signature")
)
```

動作：**Block**

## 步驟四（可選）：以 CF Access 保護 Label Studio

如需在 Label Studio 前加 SSO 登入牆：

1. **Access → Applications → Add an Application → Self-hosted**
2. Domain：`label-studio.example.com`
3. 設定 Policy（對應你的 IdP）
4. 新增 **Service Auth Bypass** 規則（允許 API Token 呼叫，避免 ML 後端被擋）：
   - Rule type：`Service Token`
   - 建立一組 Service Token 供 `sam3-ml-backend` 使用，或改用 IP CIDR 放行內部容器網段

## 故障排除

| 症狀 | 可能原因 | 解決方式 |
|------|----------|----------|
| Label Studio 回傳 CSRF 403 | `LABEL_STUDIO_HOST` 未正確設定 | 確認 `.env` 中 `LABEL_STUDIO_HOST` 與實際網域一致 |
| Presigned URL 無法在瀏覽器存取 | `MINIO_EXTERNAL_HOST` 設定錯誤 | 確認與 CF Tunnel Public Hostname 相符 |
| cloudflared 容器持續重啟 | Token 無效或 Tunnel 已刪除 | 至 CF Zero Trust 重新產生 Token |
| MinIO Presigned URL 422 | CF WAF 規則過於嚴格 | 檢查規則是否誤擋帶有 `X-Amz-Signature` 的 GET 請求 |
