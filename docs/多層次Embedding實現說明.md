# 多層次 Embedding 實現說明

## 概述

基於論文《Unlocking Legal Knowledge with Multi-Layered Embedding-Based Retrieval》的理論基礎，我們實現了完整的多層次 embedding 檢索系統。該系統能夠為法律文檔的不同語義層次創建獨立的 embedding，並根據查詢類型智能選擇合適的檢索策略。

## 核心功能

### 1. 多層次語義分塊

- **概念層 (conceptual)**：500-1500 字符，適合定義性查詢
- **程序層 (procedural)**：300-800 字符，適合流程性查詢
- **規範層 (normative)**：200-500 字符，適合條文性查詢

### 2. 查詢分類器

自動識別查詢類型並推薦合適的 embedding 層次：

- **概念性查詢**：什麼是、定義、概念、含義等
- **程序性查詢**：如何、程序、流程、申請等
- **規範性查詢**：第 X 條、規定、禁止、應、得等

### 3. 多層次 Embedding

為每個語義層次創建獨立的 embedding 向量，支持：

- Gemini text-embedding-004
- BGE-M3 embedding

### 4. 智能檢索策略

- **單層次檢索**：根據查詢類型選擇最合適的層次
- **多層次融合檢索**：從所有層次檢索並融合結果

### 5. 結果融合算法

支持多種融合策略：

- **加權求和 (weighted_sum)**：根據層次權重融合
- **倒數排名 (reciprocal_rank)**：基於排名位置融合
- **CombSUM**：簡單求和融合
- **CombANZ**：平均分數融合
- **CombMNZ**：分數乘以出現次數融合

## API 端點

### 1. 多層次 Embedding

```http
POST /api/multi-level-embed
```

**功能**：為每個語義層次創建獨立的 embedding

**請求體**：

```json
{
  "doc_ids": ["doc1", "doc2"] // 可選，不指定則處理所有文檔
}
```

**響應**：

```json
{
  "message": "Multi-level embeddings created successfully",
  "total_vectors": 1500,
  "levels": {
    "conceptual": {
      "provider": "gemini",
      "model": "text-embedding-004",
      "num_vectors": 500,
      "dimension": 768,
      "num_chunks": 500
    },
    "procedural": {
      "provider": "gemini",
      "model": "text-embedding-004",
      "num_vectors": 400,
      "dimension": 768,
      "num_chunks": 400
    },
    "normative": {
      "provider": "gemini",
      "model": "text-embedding-004",
      "num_vectors": 600,
      "dimension": 768,
      "num_chunks": 600
    }
  },
  "available_levels": ["conceptual", "procedural", "normative"]
}
```

### 2. 查詢分析

```http
POST /api/query-analysis
```

**功能**：分析查詢類型並推薦檢索策略

**請求體**：

```json
{
  "query": "什麼是著作權？"
}
```

**響應**：

```json
{
  "query_analysis": {
    "query": "什麼是著作權？",
    "query_type": "conceptual",
    "confidence": 0.85,
    "scores": {
      "conceptual": 0.85,
      "procedural": 0.1,
      "normative": 0.05
    },
    "recommended_level": "conceptual",
    "analysis": {
      "is_conceptual": true,
      "is_procedural": false,
      "is_normative": false,
      "is_mixed": false
    }
  },
  "retrieval_suggestions": {
    "recommended_method": "multi-level",
    "recommended_level": "conceptual",
    "available_levels": ["conceptual", "procedural", "normative"],
    "alternative_levels": ["procedural", "normative"]
  },
  "system_status": {
    "has_multi_level_embeddings": true,
    "has_standard_embeddings": false
  }
}
```

### 3. 單層次檢索

```http
POST /api/multi-level-retrieve
```

**功能**：根據查詢類型智能選擇層次進行檢索

**請求體**：

```json
{
  "query": "如何申請著作權？",
  "k": 10
}
```

**響應**：

```json
{
  "results": [
    {
      "rank": 1,
      "content": "申請著作權的程序包括...",
      "similarity": 0.85,
      "doc_id": "doc1",
      "doc_name": "著作權法.pdf",
      "chunk_index": 15,
      "metadata": {
        "level": "procedural",
        "query_type": "procedural",
        "confidence": 0.9
      }
    }
  ],
  "metrics": {
    "total_chunks_searched": 400,
    "query_type": "procedural",
    "recommended_level": "procedural",
    "classification_confidence": 0.9,
    "embedding_provider": "gemini",
    "embedding_model": "text-embedding-004"
  },
  "query_analysis": {
    // 查詢分析結果
  }
}
```

### 4. 多層次融合檢索

```http
POST /api/multi-level-fusion-retrieve
```

**功能**：從所有層次檢索並融合結果

**請求體**：

```json
{
  "query": "著作權的相關規定",
  "k": 10,
  "fusion_strategy": "weighted_sum",
  "level_weights": {
    "normative": 1.0,
    "procedural": 0.8,
    "conceptual": 0.6
  },
  "similarity_threshold": 0.3,
  "max_results": 10,
  "normalize_scores": true
}
```

**響應**：

```json
{
  "results": [
    {
      "rank": 1,
      "content": "第X條規定...",
      "similarity": 0.92,
      "doc_id": "doc1",
      "doc_name": "著作權法.pdf",
      "chunk_index": 25,
      "level": "normative",
      "metadata": {
        "level_scores": {
          "normative": 0.92,
          "procedural": 0.75,
          "conceptual": 0.68
        },
        "fusion_method": "weighted_sum"
      }
    }
  ],
  "metrics": {
    "total_chunks_searched": 1500,
    "levels_searched": ["conceptual", "procedural", "normative"],
    "fusion_strategy": "weighted_sum",
    "level_weights": {
      "normative": 1.0,
      "procedural": 0.8,
      "conceptual": 0.6
    },
    "level_contributions": {
      "conceptual": {
        "num_results": 8,
        "avg_similarity": 0.65,
        "max_similarity": 0.78
      },
      "procedural": {
        "num_results": 6,
        "avg_similarity": 0.72,
        "max_similarity": 0.85
      },
      "normative": {
        "num_results": 10,
        "avg_similarity": 0.88,
        "max_similarity": 0.95
      }
    }
  },
  "query_analysis": {
    // 查詢分析結果
  },
  "level_results": {
    // 各層次的原始檢索結果
  }
}
```

## 使用流程

### 1. 準備階段

```bash
# 1. 上傳文檔
POST /api/upload

# 2. 進行多層次語義分塊
POST /api/multi-level-semantic-chunk
{
  "doc_id": "doc1",
  "chunk_size": 1000,
  "overlap_ratio": 0.1
}

# 3. 創建多層次embedding
POST /api/multi-level-embed
{
  "doc_ids": ["doc1"]
}
```

### 2. 檢索階段

```bash
# 方式1：智能單層次檢索
POST /api/multi-level-retrieve
{
  "query": "什麼是著作權？",
  "k": 10
}

# 方式2：多層次融合檢索
POST /api/multi-level-fusion-retrieve
{
  "query": "著作權的相關規定",
  "k": 10,
  "fusion_strategy": "weighted_sum"
}
```

## 技術優勢

### 1. 語義層次化

- 不同粒度的語義表示
- 針對性的檢索策略
- 更精確的結果匹配

### 2. 智能查詢分類

- 自動識別查詢意圖
- 動態選擇檢索層次
- 提高檢索準確性

### 3. 多種融合策略

- 靈活的結果融合算法
- 可配置的權重設置
- 適應不同檢索需求

### 4. 完整的評估指標

- 詳細的檢索統計
- 層次貢獻分析
- 性能監控支持

## 與論文的對應關係

| 論文概念                 | 實現對應              |
| ------------------------ | --------------------- |
| Multi-layered embeddings | 多層次 embedding 存儲 |
| Semantic chunking        | 語義分塊策略          |
| Hierarchical structure   | 層次化檢索            |
| Aboutness                | 查詢分類器            |
| Result fusion            | 多種融合算法          |

## 注意事項

1. **依賴關係**：使用多層次檢索前必須先進行多層次分塊和 embedding
2. **性能考慮**：多層次 embedding 會增加存儲和計算開銷
3. **配置調優**：可根據具體需求調整層次權重和融合策略
4. **兼容性**：與現有的標準檢索 API 完全兼容

## 未來改進方向

1. **動態權重調整**：根據查詢歷史自動調整層次權重
2. **跨層次關聯**：建立層次間的語義關聯
3. **個性化檢索**：基於用戶偏好自定義檢索策略
4. **實時學習**：根據用戶反饋持續優化分類器
