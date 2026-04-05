# SAM3 ML 後端

SAM3（Segment Anything Model 3）是 Meta 於 2025 年 11 月釋出的下一代分割模型，支援影像分割（Image）與影片物件追蹤（Video）。本專案以兩個獨立 ML 後端服務的形式整合進 Label Studio：

| 服務 | 路徑 | 監聽埠 | 功能 |
|------|------|--------|------|
| `sam3-image-backend` | `ml-backends/sam3-image/` | `:9090` | 靜態影像分割 → BrushLabels RLE 遮罩 |
| `sam3-video-backend` | `ml-backends/sam3-video/` | `:9090` | 影片物件追蹤 → VideoRectangle 序列 |

## 前置需求

1. **同意 Meta 使用條款**：前往 [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) → *Agree and access repository*
2. 產生 HuggingFace **Read Token** → 填入 `.env` 的 `HF_TOKEN`
3. NVIDIA GPU，VRAM ≥ 8 GB（bfloat16 推論，torch 2.7 + CUDA 12.6）
4. 主機已安裝 `nvidia-container-toolkit`

> **SAM3 與 SAM2 的差異**：SAM3 使用 `facebookresearch/sam3`（源碼安裝，非 HuggingFace transformers），支援 PCS（Promptable Concept Segmentation，文字概念提示），影像端呼叫 `build_sam3_image_model()` + `Sam3Processor`，影片端呼叫 `build_sam3_video_predictor()`。

## 啟動

```bash
make ml-up              # 建置映像 + 以 ML Compose overlay 啟動（含核心服務）
make ml-down            # 停止所有服務（含核心）

make build-sam3-image   # 僅建置影像後端映像
make build-sam3-video   # 僅建置影片後端映像

make test-sam3-image    # 在容器內執行影像後端 pytest
make test-sam3-video    # 在容器內執行影片後端 pytest
```

首次啟動時，容器從 HuggingFace Hub 下載 `facebook/sam3.1` 權重（約 3.5 GB）至 `sam3-image-models` / `sam3-video-models` Docker Volume。健康檢查 `start_period: 300s`，下載期間不觸發重啟。

## 連接至 Label Studio

**影像後端**

1. Label Studio → 專案 → **Settings → Machine Learning → Add Model**
2. URL：`http://sam3-image-backend:9090`
3. 點選 **Validate and Save**，開啟 **Auto-Annotation**

**影片後端**

1. 同上，URL：`http://sam3-video-backend:9090`
2. 需要含有 `<Video>` 與 `<VideoRectangle>` 標籤的標注配置

> 兩個後端共用同一個 `LABEL_STUDIO_API_KEY`。建議在 LS UI（Settings → Access Tokens）建立獨立的 token 後填入 `.env`。

## 影像後端

### 標注配置（image）

將 [ml-backends/sam3-image/labeling_config.xml](../ml-backends/sam3-image/labeling_config.xml) 匯入專案：

```
Settings → Labeling Interface → Code → 貼上 XML
```

| 控制項 | 類型 | 用途 |
|--------|------|------|
| `<KeyPointLabels smart="true">` | 點擊提示 | 正向（Object）或負向（Background）點 |
| `<RectangleLabels smart="true">` | 框選提示 | SAM3 邊界框約束 |
| `<BrushLabels>` | 輸出 | SAM3 遮罩（Label Studio RLE 格式） |

### Prompt 優先順序

```
文字提示（標籤名稱）> 矩形框（RectangleLabels）> 點擊（KeyPointLabels）
```

SAM3 PCS 模式（文字提示）可回傳 N 個實例遮罩；每個實例輸出為獨立的 `BrushLabels` 結果。

### 推論流程（image）

```
Label Studio（點擊事件）
    │  POST /predict  { task, context: {keypointlabels | rectanglelabels} }
    ▼
NewModel.predict()
    ├── _load_image()          ← PIL 影像，附 LRU + TTL 快取
    ├── _parse_context()       ← 百分比座標 → 像素座標；正向/負向標籤解析
    │                             取第一個正向標籤名稱作為文字 concept prompt
    ├── Sam3Processor.set_image()   → 影像編碼
    ├── set_text_prompt() | set_box_prompt() | set_point_prompt()
    └── mask2rle()             ← Label Studio RLE（label_studio_converter.brush）
    │  ModelResponse { brushlabels[], rle }
    ▼
Label Studio（渲染遮罩覆蓋層）
```

## 影片後端

### 標注配置（video）

```xml
<View>
  <Video name="video" value="$video"/>
  <VideoRectangle name="box" toName="video"/>
  <Labels name="labels" toName="video">
    <Label value="Person"/>
    <Label value="Vehicle"/>
  </Labels>
</View>
```

### 推論流程（video）

```
Label Studio（使用者在影格 N 畫框）
    │  POST /predict  { task, context: {videorectangle: {sequence, labels}} }
    ▼
NewModel.predict()
    ├── get_local_path()              ← 從 LS 下載影片
    ├── _get_prompts()                ← 提取第一個啟用畫格的框 + 標籤
    ├── predictor.handle_request({ type: "start_session" })
    ├── handle_request({ type: "add_prompt", frame_index, box_pct, text })
    └── handle_request({ type: "get_output" }) × MAX_FRAMES_TO_TRACK
    │  ModelResponse { videorectangle: {sequence: [{frame, x, y, w, h}...]} }
    ▼
Label Studio（渲染多畫格追蹤框）
```

### 已知限制（骨架版本）

| 限制 | 說明 |
|------|------|
| Session 無過期機制 | `_sessions` dict 只加不刪；長時間運行會緩慢洩漏記憶體 |
| 追蹤長度固定 | 最多 `MAX_FRAMES_TO_TRACK` 畫格（env 預設 10） |
| 單物件追蹤 | 每次 predict 只處理第一個提示畫格的第一個框 |
| 無遮罩輸出 | 僅輸出 VideoRectangle（bbox），不輸出影片逐格遮罩 |

## 執行測試

```bash
# 不需要 GPU 或真實模型——使用 mock 進行 CPU 測試
cd ml-backends/sam3-image
DEVICE=cpu python -m pytest tests/ --tb=short -v

cd ml-backends/sam3-video
DEVICE=cpu python -m pytest tests/ --tb=short -v
```

影像後端測試：座標轉換、標注配置解析、多實例遮罩輸出、prompt 優先順序、mocked `Sam3Processor` 的完整 `predict()` 路徑。  
影片後端測試：`_get_prompts` 提取、VideoRectangle 序列輸出、session 建立、mocked `build_sam3_video_predictor` 的完整路徑。

## 環境變數

詳細說明見 [docs/configuration.md](configuration.md#sam3-ml-後端)。
