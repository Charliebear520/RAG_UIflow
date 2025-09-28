# Embedding 模型配置指南

## 概述

本系統支持多種 embedding 模型，按優先級排序：

1. **Gemini Embedding-001** (主要選項)
2. **BGE-M3** (備用選項)
3. **TF-IDF** (最後備選)

## 配置方法

### 1. Gemini Embedding-001 (推薦)

設置環境變量：

```bash
export USE_GEMINI_EMBEDDING=true
export GOOGLE_API_KEY=your_google_api_key_here
export GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
```

或在 `.env` 文件中：

```
USE_GEMINI_EMBEDDING=true
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
```

### 2. BGE-M3 (備用選項)

設置環境變量：

```bash
export USE_BGE_M3_EMBEDDING=true
```

或在 `.env` 文件中：

```
USE_BGE_M3_EMBEDDING=true
```

**注意**: BGE-M3 需要安裝額外依賴：

```bash
pip install sentence-transformers torch
```

### 3. TF-IDF (自動備選)

如果上述兩種方法都不可用，系統會自動使用 TF-IDF 作為備選方案。

## 模型特性

### Gemini Embedding-001

- **優點**: 高質量向量，支持多語言，API 穩定
- **缺點**: 需要 Google API Key，有使用限制
- **維度**: 768
- **適用**: 生產環境，高質量檢索

### BGE-M3

- **優點**: 本地運行，無 API 限制，支持多語言
- **缺點**: 需要較多內存，首次下載模型較大
- **維度**: 1024
- **適用**: 本地部署，無 API 限制需求

### TF-IDF

- **優點**: 輕量級，無外部依賴
- **缺點**: 檢索質量較低
- **維度**: 可變 (max_features=4096)
- **適用**: 快速測試，資源受限環境

## 自動降級機制

系統會按以下順序嘗試：

1. 如果 `USE_GEMINI_EMBEDDING=true` 且 `GOOGLE_API_KEY` 存在 → 使用 Gemini
2. 如果 Gemini 失敗且 `USE_BGE_M3_EMBEDDING=true` → 使用 BGE-M3
3. 如果都失敗 → 使用 TF-IDF

## 性能建議

- **生產環境**: 使用 Gemini Embedding-001
- **開發測試**: 使用 BGE-M3 或 TF-IDF
- **資源受限**: 使用 TF-IDF

## 故障排除

### Gemini API 錯誤

- 檢查 API Key 是否正確
- 確認 API 配額是否充足
- 檢查網絡連接

### BGE-M3 錯誤

- 確認已安裝 `sentence-transformers`
- 檢查磁盤空間（模型約 2GB）
- 確認內存充足（建議 8GB+）

### 性能優化

- Gemini: 使用批量處理 (batch_size=64)
- BGE-M3: 調整 batch_size 根據內存情況
- TF-IDF: 調整 max_features 參數
