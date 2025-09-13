# RAG 系統 - 文檔分割策略評測平台

## 🎯 項目概述

本項目是一個基於 FastAPI 和 React 的 RAG (Retrieval-Augmented Generation) 系統，專門用於文檔分割策略的評測和優化。系統支持多種分割算法、繁體中文問題生成、以及全面的評估指標。

## ✨ 主要功能

### 📄 文檔處理

- **PDF 轉換**: 支持 PDF 文檔轉換為文本格式
- **多格式支持**: 使用 pdfplumber 和 PyPDF2 雙重保障
- **元數據提取**: 提取頁面信息、表格、圖表等結構化數據

### 🔪 分割策略

- **固定大小分割**: 滑動窗口分割算法
- **層次化分割**: 按段落和句子層次分割
- **自適應分割**: 根據語義邊界自適應分割
- **混合分割**: 根據內容特徵動態選擇分割大小
- **語義分割**: 基於句子語義的分割策略

### 🤖 智能問題生成

- **繁體中文支持**: 專門針對繁體中文法律文檔優化
- **多樣化問題類型**: 案例應用、情境分析、實務處理、法律後果、合規判斷
- **Gemini AI 集成**: 使用 Google Gemini API 生成高質量問題
- **難度分級**: 基礎、進階、應用三個難度等級

### 📊 評估指標

- **Precision Omega**: 理想情況下可達到的最大精度
- **Precision@K**: 前 K 個結果中相關文檔的比例 (K=1,3,5,10)
- **Recall@K**: 相關文檔中被檢索到的比例 (K=1,3,5,10)
- **TF-IDF 檢索**: 使用 TF-IDF 向量化和餘弦相似度
- **字符級匹配**: 基於字符重疊的相關性評估

### 🎨 用戶界面

- **響應式設計**: 支持桌面和移動設備
- **實時監控**: 評估進度實時顯示
- **結果可視化**: 表格化展示和顏色編碼
- **配置比較**: 多配置橫向比較分析

## 🏗 技術架構

### 後端技術棧

- **FastAPI**: Python Web 框架，高性能和自動文檔生成
- **Pydantic**: 數據驗證和設置管理
- **PyPDF2 & pdfplumber**: PDF 文檔處理
- **scikit-learn**: TF-IDF 向量化和相似度計算
- **Google Generative AI**: 問題生成

### 前端技術棧

- **React 18**: 現代化用戶界面框架
- **TypeScript**: 類型安全的 JavaScript
- **Vite**: 快速構建工具
- **Tailwind CSS**: 實用優先的 CSS 框架

### 評估算法

- **TF-IDF 檢索**: 基於詞頻-逆文檔頻率的檢索算法
- **餘弦相似度**: 計算查詢和文檔的相似度
- **字符級匹配**: 50% 字符重疊閾值的相關性判斷
- **綜合評分**: 多指標加權平均評分

## 🚀 快速開始

### 環境要求

- Python 3.8+
- Node.js 16+
- Google Gemini API Key

### 後端設置

1. **克隆項目**

```bash
git clone <repository-url>
cd RAG
```

2. **安裝依賴**

```bash
cd backend
pip install -r requirements.txt
```

3. **環境變量配置**

```bash
# 創建 .env 文件
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

4. **啟動後端服務**

```bash
python -m uvicorn app.main_new:app --reload --host 0.0.0.0 --port 8000
```

### 前端設置

1. **安裝依賴**

```bash
cd frontend
npm install
```

2. **啟動開發服務器**

```bash
npm run dev
```

3. **訪問應用**

```
http://localhost:5173
```

## 📖 使用指南

### 基本工作流程

1. **上傳文檔**

   - 在 Upload 頁面選擇 PDF 文件
   - 配置元數據提取選項
   - 等待轉換完成

2. **配置分割策略**

   - 在 Chunk 頁面選擇分割策略
   - 調整策略參數
   - 預覽分割結果

3. **啟用評測模式**

   - 開啟「啟用評測模式」開關
   - 配置評估參數（分塊大小、重疊比例）
   - 設置問題生成參數

4. **執行評測**

   - 點擊「開始評測」按鈕
   - 監控評測進度
   - 查看評測結果

5. **分析結果**
   - 查看評測結果表格
   - 分析最佳配置推薦
   - 導出評測報告

### 高級功能

- **批量評測**: 支持多個配置組合的批量評測
- **結果比較**: 不同評測結果的橫向比較
- **配置優化**: 基於評測結果的智能配置建議
- **報告導出**: 完整的評測報告 JSON 導出

## 📁 項目結構

```
RAG/
├── backend/                 # 後端代碼
│   ├── app/
│   │   ├── models.py       # 數據模型
│   │   ├── store.py        # 數據存儲
│   │   ├── chunking.py     # 分割算法
│   │   ├── evaluation.py   # 評估邏輯
│   │   ├── pdf_processor.py # PDF處理
│   │   ├── question_generator.py # 問題生成
│   │   ├── routes.py       # API路由
│   │   └── main_new.py     # 主應用
│   └── requirements.txt    # Python依賴
├── frontend/               # 前端代碼
│   ├── src/
│   │   ├── components/     # React組件
│   │   ├── routes/         # 頁面路由
│   │   ├── lib/           # 工具庫
│   │   └── main.tsx       # 入口文件
│   └── package.json       # Node依賴
├── docs/                  # 文檔
│   ├── 功能完成總結.md
│   ├── 視覺化改進指南.md
│   └── K值選擇與一致性修復指南.md
└── corpus/               # 測試文檔
    └── 著作權法.pdf
```

## 🔧 配置說明

### 評估參數

- **Chunk Sizes**: [300, 500, 800] (可自定義)
- **Overlap Ratios**: [0.0, 0.1, 0.2] (可自定義)
- **K Values**: [1, 3, 5, 10] (可自定義)
- **Relevance Threshold**: 50% 字符重疊 (可調整)

### 環境變量

```bash
GEMINI_API_KEY=your_gemini_api_key_here
USE_GEMINI_EMBEDDING=false
USE_GEMINI_COMPLETION=false
```

## 📊 性能指標

### 實際測試結果示例

```
最佳配置：chunk_size=800, overlap_ratio=0.0
- Precision Omega: 1.000
- Precision@5: 0.493
- Recall@5: 0.203
- Chunk Count: 25
- Average Chunk Length: 796.9
```

### 系統性能

- **PDF 轉換**: 平均 2-5 秒 (取決於文檔大小)
- **分割處理**: 平均 1-3 秒
- **評估執行**: 平均 30-60 秒 (取決於配置數量)
- **問題生成**: 平均 10-20 秒 (取決於問題數量)

## 🛠 開發指南

### 添加新的分割策略

1. 在 `backend/app/chunking.py` 中實現新的策略類
2. 繼承 `ChunkingStrategy` 基類
3. 實現 `chunk` 方法
4. 在 `get_chunking_strategy` 函數中註冊新策略

### 添加新的評估指標

1. 在 `backend/app/evaluation.py` 中實現新的指標函數
2. 在 `EvaluationMetrics` 類中添加新字段
3. 在 `evaluate_chunk_config` 函數中計算新指標

### 自定義前端組件

1. 在 `frontend/src/components/` 中創建新組件
2. 使用 TypeScript 定義接口
3. 遵循現有的設計模式

## 🐛 故障排除

### 常見問題

1. **PDF 轉換失敗**

   - 檢查 PDF 文件是否損壞
   - 嘗試使用不同的 PDF 文件
   - 檢查文件大小限制

2. **Gemini API 錯誤**

   - 驗證 API Key 是否正確
   - 檢查 API 配額限制
   - 確認網絡連接正常

3. **評估結果為 0**
   - 檢查問題生成是否成功
   - 驗證分割配置是否合理
   - 確認相關性閾值設置

### 調試模式

啟用詳細日誌：

```bash
export LOG_LEVEL=DEBUG
python -m uvicorn app.main_new:app --reload
```

## 📈 未來規劃

### 短期改進

- [ ] 修復前後端 K 值一致性問題
- [ ] 添加更多視覺化圖表
- [ ] 優化評估算法性能
- [ ] 改進用戶界面體驗

### 長期規劃

- [ ] 支持更多文檔格式 (Word, TXT, HTML)
- [ ] 實現分布式評估
- [ ] 添加機器學習模型集成
- [ ] 支持多語言問題生成

## 📄 許可證

本項目採用 MIT 許可證。詳見 [LICENSE](LICENSE) 文件。

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

1. Fork 本項目
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📞 聯繫方式

如有問題或建議，請通過以下方式聯繫：

- 提交 Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 電子郵件: your-email@example.com

---

**注意**: 本系統主要用於研究和教育目的。在生產環境中使用前，請進行充分的測試和評估。
