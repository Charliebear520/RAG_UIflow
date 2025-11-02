# 章、節層級 Metadata 處理策略

## 🎯 概述

對於章、節層級的 chunks，我採用**中等強度增強 + 結構性分析**的策略，既保證了 metadata 的豐富性，又避免了過度的 LLM 調用成本。

## 🏗️ 處理策略

### 1. **層級識別**

系統會自動識別章、節層級的 chunks：

```python
def _is_chapter_section_level_chunk(self, chunk: Dict[str, Any]) -> bool:
    """判斷是否為「章、節」層級的chunk"""
    # 方法1：檢查metadata中的層級信息
    if metadata.get("level") in ["document_component", "basic_unit_hierarchy"]:
        return True

    # 方法2：檢查內容中是否包含章節模式
    patterns = [
        r"第[一二三四五六七八九十\d]+章",
        r"第[一二三四五六七八九十\d]+節",
        r"總則|分則|附則"
    ]

    # 方法3：檢查chunk_id是否包含章節層級標識
    if any(keyword in chunk_id.lower() for keyword in ["chapter", "section", "章", "節"]):
        return True
```

### 2. **中等強度 Metadata 增強**

章、節層級會進行中等強度的 metadata 增強，包含以下字段：

```json
{
  "semantic_keywords": {
    "primary_keywords": ["總則", "權利", "義務"],
    "structural_keywords": ["章", "節", "規定"],
    "legal_terms": ["權利", "義務", "責任"],
    "keyword_weights": {...}
  },
  "legal_domain": {
    "legal_domain": "著作權法",
    "confidence": 0.8
  },
  "chapter_section_type": {
    "chapter_section_type": "總則性章節",
    "type_description": "定義總則性章節的相關規定",
    "confidence": 0.9
  },
  "legal_concepts": [
    {
      "concept_name": "總則",
      "concept_type": "結構性概念",
      "legal_domain": "程序法",
      "importance_score": 0.8
    }
  ],
  "scope_keywords": {
    "scope_types": ["適用範圍", "定義範圍"],
    "scope_description": "定義適用範圍和限制條件"
  },
  "enhancement_level": "medium",
  "is_chapter_section_level": true
}
```

## 📊 不同層級的處理對比

| 層級           | 增強強度 | 主要功能     | Metadata 字段 | LLM 調用 |
| -------------- | -------- | ------------ | ------------- | -------- |
| **條層級**     | 完整增強 | 核心法律內容 | 全字段        | 是       |
| **章節層級**   | 中等增強 | 結構性內容   | 核心字段      | 否       |
| **項款目層級** | 輕量增強 | 細節內容     | 基本字段      | 否       |
| **其他層級**   | 輕量增強 | 輔助內容     | 基本字段      | 否       |

## 🔍 章、節層級專用分析

### 1. **結構性關鍵詞提取**

```python
def _extract_chapter_section_keywords(self, content: str) -> Dict[str, Any]:
    """提取章、節層級的關鍵詞"""
    # 分類關鍵詞
    structural_keywords = ["章", "節", "總則", "分則", "附則", "規定", "原則"]
    legal_terms = ["權利", "義務", "責任", "處罰", "程序", "適用"]

    return {
        "primary_keywords": [...],           # 主要關鍵詞
        "structural_keywords": [...],        # 結構性關鍵詞
        "legal_terms": [...],               # 法律術語
        "keyword_weights": {...}            # 關鍵詞權重
    }
```

### 2. **章節類型分類**

```python
def _classify_chapter_section_type(self, content: str) -> Dict[str, Any]:
    """分類章、節類型"""
    chapter_types = {
        "總則性章節": ["總則", "一般"],
        "分則性章節": ["分則", "特別"],
        "附則性章節": ["附則", "附"],
        "罰則性章節": ["罰則", "處罰"],
        "程序性章節": ["程序", "手續"]
    }

    return {
        "chapter_section_type": "總則性章節",
        "type_description": "定義總則性章節的相關規定",
        "confidence": 0.9
    }
```

### 3. **結構性法律概念提取**

```python
def _extract_chapter_section_concepts(self, content: str) -> List[Dict[str, Any]]:
    """提取章、節層級的法律概念（簡化版）"""
    structural_concepts = {
        "總則": {"importance": 0.8, "type": "結構性概念"},
        "分則": {"importance": 0.8, "type": "結構性概念"},
        "罰則": {"importance": 0.9, "type": "結構性概念"}
    }

    # 返回結構性概念列表
```

### 4. **範圍關鍵詞提取**

```python
def _extract_scope_keywords(self, content: str) -> Dict[str, Any]:
    """提取範圍關鍵詞"""
    scope_patterns = {
        "適用範圍": ["適用", "範圍", "適用於"],
        "定義範圍": ["定義", "指", "謂"],
        "例外範圍": ["例外", "除外", "不適用"],
        "程序範圍": ["程序", "手續", "方式"]
    }

    return {
        "scope_types": ["適用範圍", "定義範圍"],
        "scope_description": "定義適用範圍和限制條件"
    }
```

## 🚀 實際範例

### 範例：《著作權法》第一章總則

#### 原始內容

```
第一章 總則

第一條 為保障著作人著作權益，調和社會公共利益，促進國家文化發展，特制定本法。

第二條 本法所稱著作，指屬於文學、科學、藝術或其他學術範圍之創作。

第三條 本法所稱著作人，指創作著作之人。
```

#### 增強後的 Metadata

```json
{
  "semantic_keywords": {
    "primary_keywords": ["著作人", "著作權益", "社會公共利益", "國家文化發展"],
    "structural_keywords": ["章", "總則", "規定"],
    "legal_terms": ["著作權", "創作", "文學", "科學", "藝術"],
    "keyword_weights": {
      "著作人": 0.8,
      "著作權益": 0.9,
      "社會公共利益": 0.7
    }
  },
  "legal_domain": {
    "legal_domain": "著作權法",
    "confidence": 0.9
  },
  "chapter_section_type": {
    "chapter_section_type": "總則性章節",
    "type_description": "定義總則性章節的相關規定",
    "confidence": 0.95
  },
  "legal_concepts": [
    {
      "concept_name": "總則",
      "concept_type": "結構性概念",
      "legal_domain": "程序法",
      "importance_score": 0.8,
      "confidence": 0.9
    },
    {
      "concept_name": "著作權益",
      "concept_type": "權利概念",
      "legal_domain": "著作權法",
      "importance_score": 0.9,
      "confidence": 0.95
    }
  ],
  "scope_keywords": {
    "scope_types": ["適用範圍", "定義範圍"],
    "scope_description": "定義適用範圍和限制條件"
  },
  "enhancement_level": "medium",
  "is_chapter_section_level": true
}
```

## 🔄 檢索中的應用

### 1. **查詢匹配**

當用戶查詢「著作權法的總則規定」時：

```python
# 查詢關鍵詞
query_keywords = ["著作權法", "總則", "規定"]

# 章節層級匹配
chapter_section_match_score = 0.85  # 高匹配分數

# 匹配到第一章總則的metadata
matched_chapter = "著作權法_第一章_總則"
```

### 2. **候選擴展**

系統會將該章節及其下的所有條文都納入候選範圍：

```python
# 擴展候選
candidate_chunks = [
    "著作權法_第一章_總則",           # 章節層級
    "著作權法_第1條",               # 條層級
    "著作權法_第2條",               # 條層級
    "著作權法_第3條",               # 條層級
    "著作權法_第1條_第1項",         # 項層級（繼承）
    "著作權法_第1條_第2項"          # 項層級（繼承）
]
```

### 3. **分數計算**

章節層級的 chunks 會獲得結構性加分：

```python
final_score = (
    vector_score * 0.6 +
    bm25_score * 0.25 +
    metadata_bonus * 0.15 +
    structural_bonus * 0.05  # 章節結構加分
)
```

## 📈 優勢效果

### 1. **效率優化**

- 章節層級不需要 LLM 調用，使用規則和模式匹配
- 大幅降低計算成本和時間

### 2. **結構性理解**

- 識別法律文檔的層次結構
- 理解章節的功能和範圍

### 3. **檢索精度**

- 通過結構性關鍵詞提升匹配精度
- 支持「總則」、「分則」等結構性查詢

### 4. **法律專業性**

- 體現法律文檔的邏輯結構
- 支持法律專業的查詢模式

## 🎯 總結

章、節層級的 metadata 處理策略：

✅ **中等強度增強** - 平衡效果與效率  
✅ **結構性分析** - 識別章節類型和功能  
✅ **專業關鍵詞** - 提取法律結構性術語  
✅ **範圍識別** - 理解適用範圍和限制  
✅ **無 LLM 調用** - 使用規則和模式匹配  
✅ **檢索支持** - 支持結構性查詢

這個策略既保證了章節層級 metadata 的豐富性，又避免了過度的計算成本，完美平衡了效果與效率！
