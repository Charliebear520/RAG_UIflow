# HopRAG 層級命名更新說明

## 概述

已將 HopRAG 系統中的層級命名從舊版本更新為新的統一命名方式，以配合多層級分塊和多層級 embedding 的新架構。

## 層級命名映射

### 完整的六層級映射

| 舊名稱            | 新名稱               | 說明                   |
| ----------------- | -------------------- | ---------------------- |
| law               | document             | 法規文件層級           |
| chapter           | document_component   | 章節層級               |
| section           | basic_unit_hierarchy | 節層級                 |
| article           | basic_unit           | 條文層級（基本單元）   |
| paragraph         | basic_unit_component | 項層級（基本單元組件） |
| subparagraph/item | enumeration          | 款/目層級（枚舉）      |

### HopRAG 使用的層級

HopRAG 系統主要使用以下兩個層級建立節點：

1. **basic_unit**（原 article）- 條文層級節點
2. **basic_unit_component**（原 item/paragraph）- 項層級節點

## 修改的文件清單

### 1. `hoprag_config.py`

**修改內容：**

- 更新 `NodeType` 枚舉類別
  - `ARTICLE` → `BASIC_UNIT`
  - `ITEM` → `BASIC_UNIT_COMPONENT`
- 更新 `EdgeType` 枚舉類別
  - `ARTICLE_TO_ARTICLE` → `BASIC_UNIT_TO_BASIC_UNIT`
  - `ARTICLE_TO_ITEM` → `BASIC_UNIT_TO_COMPONENT`
  - `ITEM_TO_ARTICLE` → `COMPONENT_TO_BASIC_UNIT`
  - `ITEM_TO_ITEM` → `COMPONENT_TO_COMPONENT`
- 新增 `from_legacy_name()` 方法支援向後兼容

**程式碼範例：**

```python
class NodeType(Enum):
    """節點類型枚舉 - 對應新的層級命名"""
    BASIC_UNIT = "basic_unit"  # 原 article
    BASIC_UNIT_COMPONENT = "basic_unit_component"  # 原 item

    @classmethod
    def from_legacy_name(cls, legacy_name: str):
        """從舊名稱轉換為新的NodeType"""
        mapping = {
            "article": cls.BASIC_UNIT,
            "item": cls.BASIC_UNIT_COMPONENT,
        }
        return mapping.get(legacy_name, cls.BASIC_UNIT)
```

### 2. `hoprag_graph_builder.py`

**修改內容：**

- 節點創建時使用新的 `NodeType.BASIC_UNIT` 和 `NodeType.BASIC_UNIT_COMPONENT`
- 更新變數命名：
  - `article_node` → `basic_unit_node`
  - `item_node` → `component_node`
- 更新 `_determine_edge_type()` 方法以識別新的層級名稱

**關鍵修改：**

```python
# 創建basic_unit節點
basic_unit_node = LegalNode(
    node_id=node_id,
    node_type=NodeType.BASIC_UNIT,
    content=chunk['content'],
    # ...
)

# 創建basic_unit_component節點
component_node = LegalNode(
    node_id=node_id,
    node_type=NodeType.BASIC_UNIT_COMPONENT,
    content=chunk['content'],
    # ...
)
```

### 3. `hoprag_system.py`

**修改內容：**

- 更新 `LegalNode` 數據結構註釋
- 更新字符串類型值為新命名
  - `'article'` → `'basic_unit'`
  - `'item'` → `'basic_unit_component'`
- 更新邊類型判定邏輯
- 更新統計節點計數的變數名稱

**關鍵修改：**

```python
# 節點類型註釋
node_type: str  # "basic_unit" 或 "basic_unit_component" (原 "article" 或 "item")

# 邊類型判定
if from_type == 'basic_unit' and to_type == 'basic_unit':
    return 'basic_unit_to_basic_unit'
elif from_type == 'basic_unit' and to_type == 'basic_unit_component':
    return 'basic_unit_to_component'
# ...
```

### 4. `hoprag_system_modular.py`

**修改內容：**

- 更新統計節點類型的變數名稱
- 更新 `get_graph_statistics()` 方法返回的鍵名

**關鍵修改：**

```python
# 統計節點類型
basic_unit_count = sum(1 for node in self.nodes.values()
                       if node.node_type.value == 'basic_unit')
component_count = sum(1 for node in self.nodes.values()
                      if node.node_type.value == 'basic_unit_component')

return {
    "basic_unit_nodes": basic_unit_count,
    "basic_unit_component_nodes": component_count,
    # ...
}
```

### 5. `hoprag_result_processor.py`

**修改內容：**

- 更新 `_get_type_bonus()` 方法中的節點類型判斷
- 更新 `_get_node_type_weight()` 方法中的權重映射

**關鍵修改：**

```python
def _get_type_bonus(self, node_type: str) -> float:
    """根據節點類型獲取分數加成 - 使用新的層級命名"""
    if node_type == "basic_unit":
        return 1.0
    elif node_type == "basic_unit_component":
        return 0.9
    else:
        return 0.8
```

## 向後兼容性

為確保向後兼容，我們在 `hoprag_config.py` 中新增了 `from_legacy_name()` 方法，可以將舊的命名轉換為新的 `NodeType`。

## 影響範圍

### 不受影響的部分

以下欄位名稱**不受影響**，保持不變：

- `article_number` - 條文編號
- `item_number` - 項次編號
- `parent_article_id` - 父條文 ID
- `article_label` - 條文標籤

這些是**數據欄位名稱**，不是**層級類型名稱**，因此不需要修改。

### 受影響的 API

如果有外部系統使用 HopRAG 的 API，需要注意以下變更：

1. **圖統計 API** - 返回的鍵名已更新：

   - `article_nodes` → `basic_unit_nodes`
   - `item_nodes` → `basic_unit_component_nodes`

2. **節點類型值** - 在 JSON 響應中的值已更新：

   - `"article"` → `"basic_unit"`
   - `"item"` → `"basic_unit_component"`

3. **邊類型值** - 在 JSON 響應中的值已更新：
   - `"article_to_article"` → `"basic_unit_to_basic_unit"`
   - `"article_to_item"` → `"basic_unit_to_component"`
   - `"item_to_article"` → `"component_to_basic_unit"`
   - `"item_to_item"` → `"component_to_component"`

## 測試建議

更新後建議進行以下測試：

1. **單元測試**

   - 測試節點創建是否正確使用新的類型
   - 測試邊類型判定是否正確

2. **集成測試**

   - 測試從 multi_level_chunks 構建圖是否正常
   - 測試 HopRAG 檢索功能是否正常
   - 測試圖統計 API 是否返回正確的鍵名

3. **數據驗證**
   - 驗證節點類型在數據庫中正確存儲
   - 驗證邊類型在圖結構中正確表示

## 更新日期

2025-10-06

## 相關文檔

- [多層次 Embedding 實現說明](./多層次Embedding實現說明.md)
- [HopRAG 模組化架構說明](./HopRAG模組化架構說明.md)
- [HopRAG 系統實作完成說明](./HopRAG系統實作完成說明.md)
