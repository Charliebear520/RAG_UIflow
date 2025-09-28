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

## 🚀 快速開始

### 環境要求

- Python 3.8+
- Node.js 16+
- Google Gemini API Key

### 後端設置

```bash
cd backend
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
echo "USE_GEMINI_COMPLETION=true" >> .env
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**注意**: 如果您遇到 "GOOGLE_API_KEY not set" 錯誤，請參考 [Gemini API 設置指南](GEMINI_SETUP.md)

### 前端設置

```bash
cd frontend
npm install
npm run dev
```

### 訪問應用

```
http://localhost:5173
```

## 📖 使用指南

1. **上傳文檔**: 在 Upload 頁面選擇 PDF 文件
2. **配置分割**: 在 Chunk 頁面選擇分割策略和參數
3. **啟用評測**: 開啟評測模式並配置評估參數
4. **執行評測**: 點擊開始評測，監控進度
5. **分析結果**: 查看評測結果和最佳配置推薦

## 📁 項目結構

```
RAG/
├── backend/                 # 後端代碼（模組化重構）
│   ├── app/
│   │   ├── models.py       # 數據模型
│   │   ├── store.py        # 數據存儲
│   │   ├── chunking.py     # 分割算法
│   │   ├── evaluation.py   # 評估邏輯
│   │   ├── pdf_processor.py # PDF處理
│   │   ├── question_generator.py # 問題生成
│   │   ├── routes.py       # API路由
│   │   └── main.py         # 主應用文件
│   └── requirements.txt    # Python依賴
├── frontend/               # 前端代碼（組件化）
│   ├── src/
│   │   ├── components/     # React組件
│   │   │   ├── ChunkStrategySelector.tsx
│   │   │   └── EvaluationPanel.tsx
│   │   ├── routes/         # 頁面路由
│   │   ├── lib/           # 工具庫
│   │   └── main.tsx       # 入口文件
│   └── package.json       # Node依賴
├── docs/                  # 完整文檔
│   ├── README.md          # 詳細使用指南
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
```

## 📊 實際測試結果

```
最佳配置：chunk_size=800, overlap_ratio=0.0
- Precision Omega: 1.000
- Precision@5: 0.493
- Recall@5: 0.203
- Chunk Count: 25
- Average Chunk Length: 796.9
```

## 📈 系統改進

### ✅ 已完成的改進

- 模組化後端代碼重構
- 前端組件化設計
- 完整的文檔整理
- K 值一致性修復
- 視覺化改進建議
- 清理臨時測試檔案
- 整合重複文檔

### 🔮 未來規劃

- 修復前後端 K 值一致性問題
- 添加更多視覺化圖表
- 支持更多文檔格式
- 實現分布式評估

## 📚 詳細文檔

完整的使用指南和技術文檔請參考：

- [詳細 README](docs/README.md)
- [功能完成總結](docs/功能完成總結.md)
- [視覺化改進指南](docs/視覺化改進指南.md)
- [K 值選擇與一致性修復指南](docs/K值選擇與一致性修復指南.md)

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

## 📄 許可證

MIT License

---

**注意**: 本系統主要用於研究和教育目的。在生產環境中使用前，請進行充分的測試和評估。
