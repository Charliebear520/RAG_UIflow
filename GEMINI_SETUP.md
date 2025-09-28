# Gemini API 設置指南

## 問題描述

如果您遇到以下錯誤：

```
Gemini調用失敗: GOOGLE_API_KEY not set. 回退到提取式回答。
```

這表示系統無法找到有效的 Google API Key 來調用 Gemini 模型。

## 解決方案

### 方法 1：設置 GOOGLE_API_KEY（推薦）

在您的環境中設置 `GOOGLE_API_KEY`：

```bash
export GOOGLE_API_KEY=your_google_api_key_here
```

或在 `.env` 文件中添加：

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 方法 2：設置 GEMINI_API_KEY（備用）

如果沒有 `GOOGLE_API_KEY`，系統會自動嘗試使用 `GEMINI_API_KEY`：

```bash
export GEMINI_API_KEY=your_google_api_key_here
```

或在 `.env` 文件中添加：

```
GEMINI_API_KEY=your_google_api_key_here
```

## 獲取 API Key

1. 前往 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 登入您的 Google 帳戶
3. 點擊 "Create API Key"
4. 複製生成的 API Key
5. 按照上述方法設置環境變量

## 啟用 Gemini 生成功能

設置 API Key 後，還需要啟用 Gemini 生成功能：

```bash
export USE_GEMINI_COMPLETION=true
```

或在 `.env` 文件中添加：

```
USE_GEMINI_COMPLETION=true
```

## 驗證設置

重啟後端服務，您應該看到類似以下的調試信息：

```
🔧 Embedding 配置:
   USE_GEMINI_EMBEDDING: True
   GOOGLE_API_KEY: 已設置
   GEMINI_API_KEY: 未設置
   USE_BGE_M3_EMBEDDING: False
   GOOGLE_EMBEDDING_MODEL: gemini-embedding-001
   USE_GEMINI_COMPLETION: True
```

## 注意事項

- 同一個 Google API Key 可以用於 embedding 和生成功能
- 系統會優先使用 `GOOGLE_API_KEY`，如果沒有則使用 `GEMINI_API_KEY`
- 如果兩個都沒有設置，系統會回退到提取式回答
- 確保 API Key 有足夠的配額和權限

## 故障排除

如果仍然遇到問題：

1. 檢查 API Key 是否正確複製（沒有多餘的空格）
2. 確認 API Key 有 Gemini API 的訪問權限
3. 檢查網絡連接是否正常
4. 查看後端日誌中的詳細錯誤信息
