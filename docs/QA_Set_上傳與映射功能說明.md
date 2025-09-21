# QA Set 上傳與映射功能說明

## 功能概述

本功能實現了完整的 QA set 上傳、chunk 映射和評測流程，讓使用者可以上傳自定義的 QA set，自動進行不同分塊策略的 span 到 chunk ID 映射，並基於映射結果進行 P@K/R@K 評測。

## 核心流程

### 1. 上傳 QA Set

- 用戶選擇 JSON 格式的 QA set 文件
- 配置分塊參數（chunk sizes, overlap ratios, strategy）
- 系統驗證文件格式和內容

### 2. 分塊與映射

- 根據配置生成不同組合的分塊
- 使用 IoU > 0.5 的閾值將 QA set 中的 spans 映射到 chunks
- 為每個配置生成映射結果

### 3. 結果確認

- 顯示映射結果（預設前 3 個配置）
- 用戶可以展開查看所有配置的映射結果
- 確認無誤後開始評測

### 4. P@K/R@K 評測

- 使用映射後的 QA set 作為 ground truth
- 進行不同 k 值的 Precision@K 和 Recall@K 評估
- 生成評測報告和推薦配置

## 技術實現

### 後端 API 端點

#### 1. 上傳 QA Set

```http
POST /api/upload-qa-set
Content-Type: multipart/form-data

Parameters:
- file: JSON格式的QA set文件
- doc_id: 文檔ID
- chunk_sizes: JSON數組，分塊大小選項
- overlap_ratios: JSON數組，重疊比例選項
- strategy: 分塊策略名稱
```

#### 2. 查詢映射狀態

```http
GET /api/qa-mapping/status/{task_id}
```

#### 3. 獲取映射結果

```http
GET /api/qa-mapping/result/{task_id}
```

### 核心算法

#### IoU 計算

```python
def calculate_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    """計算兩個span的IoU (Intersection over Union)"""
    start1, end1 = span1
    start2, end2 = span2

    # 計算交集
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_length = max(0, intersection_end - intersection_start)

    # 計算聯集
    union_length = (end1 - start1) + (end2 - start2) - intersection_length

    if union_length == 0:
        return 0.0

    return intersection_length / union_length
```

#### Span 到 Chunk 映射

```python
def map_spans_to_chunks(qa_items, chunks, chunk_size, overlap, strategy):
    """
    將QA set中的spans映射到chunks

    使用IoU > 0.5的閾值進行映射：
    - 正例：找到IoU > 0.5的chunks，可能一對多
    - 負例：relevant_chunks為空列表
    """
```

### 前端組件

#### 1. QASetUploader 組件

- 文件上傳界面
- 分塊參數配置
- 映射進度顯示
- 映射結果預覽（可展開）

#### 2. EvaluationPanel 組件更新

- 新增評估模式選擇（QA Set vs 自動生成問題）
- 整合 QA set 映射結果
- 基於映射結果的評測流程

## 數據格式

### QA Set JSON 格式

```json
[
  {
    "query": "問題內容",
    "label": "Yes/No",
    "answer": "答案內容",
    "spans": [
      {
        "start_char": 1000,
        "end_char": 1500,
        "text": "相關文本片段"
      }
    ]
  }
]
```

### 映射結果格式

```json
{
  "task_id": "uuid",
  "doc_id": "doc_id",
  "original_qa_set": [...],
  "configs": [...],
  "mapping_results": {
    "config_001_800_0.1": {
      "config": {...},
      "chunks": [...],
      "mapped_qa_set": [
        {
          "query": "問題",
          "label": "Yes",
          "answer": "答案",
          "relevant_chunks": ["chunk_003"]
        }
      ]
    }
  }
}
```

## 使用示例

### 1. 準備 QA Set 文件

```json
[
  {
    "query": "Consider ROC著作權法第3章; 侵權處罰是否存在？",
    "snippets": [
      {
        "file_path": "copyright.json",
        "span": [1000, 1500]
      }
    ],
    "label": "Yes",
    "answer": "第88條：故意侵害著作權者，處三年以下有期徒刑..."
  }
]
```

### 2. 配置分塊參數

- Chunk Sizes: [300, 500, 800]
- Overlap Ratios: [0.0, 0.1, 0.2]
- Strategy: "fixed_size"

### 3. 映射結果示例

```json
[
  {
    "query": "Consider ROC著作權法第3章; 侵權處罰是否存在？",
    "snippets": [
      {
        "file_path": "copyright.json",
        "span": [1000, 1500]
      }
    ],
    "label": "Yes",
    "answer": "第88條：故意侵害著作權者，處三年以下有期徒刑...",
    "relevant_chunks": ["chunk_003"]
  }
]
```

## 優勢特點

1. **靈活性**：支持多種分塊策略和參數組合
2. **準確性**：使用 IoU 閾值確保映射精度
3. **可視化**：清晰的映射結果展示和確認流程
4. **完整性**：從上傳到評測的完整工作流程
5. **可擴展性**：易於添加新的分塊策略和評估指標

## 注意事項

1. QA set 文件必須是 JSON 格式
2. 每個問題需要包含有效的 spans 或 snippets 信息
3. IoU 閾值設為 0.5，可根據需要調整
4. 映射過程是異步的，需要輪詢查詢狀態
5. 建議先檢查映射結果的正確性再進行評測

## 測試驗證

已通過以下測試：

- IoU 計算功能測試
- Span 映射功能測試
- 真實 QA 數據測試

所有測試均通過，功能運行正常。
