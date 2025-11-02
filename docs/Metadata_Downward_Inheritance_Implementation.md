# Metadata 向下繼承實現說明

## 🎯 概述

根據您的需求，我已經實現了**Metadata 的向下繼承性 (Downward Inheritance)**，這是最重要的法律 RAG 策略。在「條」層級生成的 metadata，會自動附加到其下所有的子層級（項、款、目）chunks 上。

## 🔄 運作流程

### 1. **查詢階段 - Metadata 關鍵字匹配**

當使用者查詢時，系統首先對「條」層級的 metadata 進行關鍵字匹配：

```python
# 查詢：「老師上課可以傳輸影片嗎？」
query_keywords = ["老師", "上課", "傳輸", "影片"]

# 匹配到《著作權法》第46條的metadata
matched_article = "著作權法_第46條"
metadata_match_score = 0.85  # 高匹配分數
```

### 2. **繼承階段 - 擴展候選範圍**

系統會將該「條」以及其下的所有「項」、「款」、「目」chunks 都視為候選對象：

```python
# 找到第46條的所有子chunks
inherited_candidates = [
    "著作權法_第46條_第1項",
    "著作權法_第46條_第2項",
    "著作權法_第46條_第3項",
    "著作權法_第46條_第1款",
    "著作權法_第46條_第2款"
]
```

### 3. **檢索階段 - 向量相似度計算**

對所有候選 chunks 進行向量相似度計算，最終可能因為語義更接近而返回第 2 項的內容：

```python
final_results = [
    {
        "content": "第46條第2項：為學校授課需要...",
        "hybrid_score": 0.92,
        "inherited_from": "著作權法_第46條",
        "inheritance_bonus": 0.1,
        "metadata_match_reason": "繼承自條層級 著作權法_第46條"
    }
]
```

## 🏗️ 技術實現

### 1. **Metadata 增強器改進**

#### 條層級識別

```python
def _is_article_level_chunk(self, chunk: Dict[str, Any]) -> bool:
    """判斷是否為「條」層級的chunk"""
    # 方法1：檢查metadata中的層級信息
    if metadata.get("level") == "basic_unit":
        return True

    # 方法2：檢查內容中是否包含條號模式
    if re.search(r"第\d+條", content):
        return True

    # 方法3：檢查chunk_id是否包含條層級標識
    if "article" in chunk_id.lower():
        return True
```

#### 完整 metadata 生成

```python
def _enhance_article_level_chunk(self, content: str, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """為「條」層級chunk進行完整metadata增強"""
    enhanced = {
        **cleaned_metadata,
        "legal_concepts": [...],      # 法律概念
        "semantic_keywords": {...},   # 語義關鍵詞
        "article_type": {...},        # 條文類型
        "legal_domain": {...},        # 法律領域
        "legal_relations": [...],     # 法律關係
        "query_intent_tags": [...],   # 查詢意圖標籤
        "enhancement_level": "full",  # 標記為完整增強
        "is_article_level": True
    }
```

#### 輕量級增強

```python
def _enhance_lightweight_chunk(self, content: str, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """為非條層級chunk進行輕量級metadata增強"""
    enhanced = {
        **cleaned_metadata,
        "semantic_keywords": {...},   # 基本關鍵詞
        "legal_domain": {...},        # 基本法律領域
        "enhancement_level": "lightweight",  # 標記為輕量級增強
        "is_article_level": False
    }
```

### 2. **繼承關係映射**

#### 條層級 metadata 存儲

```python
self.article_metadata_map = {
    "著作權法_第46條": {
        "legal_concepts": [...],
        "semantic_keywords": {...},
        "enhancement_level": "full"
    }
}
```

#### 繼承關係映射

```python
self.inheritance_hierarchy = {
    "著作權法_第46條_第1項": "著作權法_第46條",
    "著作權法_第46條_第2項": "著作權法_第46條",
    "著作權法_第46條_第3項": "著作權法_第46條"
}
```

### 3. **檢索策略實現**

#### Metadata 關鍵字匹配

```python
def _metadata_keyword_match(self, query: str) -> List[str]:
    """通過metadata關鍵字匹配找到相關的條層級"""
    matched_articles = []
    article_metadata_map = self.metadata_enhancer.get_article_metadata_map()
    query_keywords = self._extract_query_keywords(query)

    for article_id, metadata in article_metadata_map.items():
        match_score = self._calculate_metadata_match_score(query_keywords, metadata)
        if match_score > 0.3:  # 可調整閾值
            matched_articles.append(article_id)

    return matched_articles
```

#### 繼承策略應用

```python
def _apply_inheritance_strategy(self, candidate_nodes, matched_articles, query, config):
    """應用metadata向下繼承策略"""
    inherited_candidates = []
    inheritance_hierarchy = self.metadata_enhancer.get_inheritance_hierarchy()

    for article_id in matched_articles:
        # 找到該條層級的所有子chunks
        child_chunks = [child_id for child_id, parent_id in inheritance_hierarchy.items()
                       if parent_id == article_id]

        for child_chunk_id in child_chunks:
            chunk_info = self._get_chunk_info_by_id(child_chunk_id)
            if chunk_info:
                # 添加繼承標記和額外加分
                chunk_info["inherited_from"] = article_id
                chunk_info["inheritance_bonus"] = config.inheritance_bonus
                chunk_info["metadata_match_reason"] = f"繼承自條層級 {article_id}"
                inherited_candidates.append(chunk_info)

    return candidate_nodes + inherited_candidates
```

## 🎛️ 配置選項

### EnhancedHybridConfig 新增配置

```python
config = EnhancedHybridConfig(
    # ... 原有配置 ...

    # Metadata向下繼承配置
    enable_inheritance_strategy=True,    # 啟用繼承策略
    metadata_match_threshold=0.3,        # metadata匹配閾值
    inheritance_bonus=0.1,               # 繼承加分
    inheritance_boost_factor=1.2         # 繼承提升係數
)
```

## 📊 實際範例

### 範例：著作權法第 46 條

#### 1. **條層級 metadata 生成**

```json
{
  "article_id": "著作權法_第46條",
  "legal_concepts": [
    {
      "concept_name": "合理使用",
      "concept_type": "權利例外",
      "legal_domain": "著作權法",
      "importance_score": 0.9,
      "synonyms": ["公平使用", "fair use"]
    }
  ],
  "semantic_keywords": {
    "primary_keywords": ["學校", "教師", "授課目的", "重製", "公開傳輸"],
    "keyword_weights": {
      "學校": 0.8,
      "教師": 0.7,
      "授課目的": 0.6,
      "重製": 0.5,
      "公開傳輸": 0.4
    }
  },
  "query_intent_tags": ["例外查詢", "權利查詢"],
  "enhancement_level": "full"
}
```

#### 2. **查詢匹配過程**

```python
# 查詢：「老師上課可以傳輸影片嗎？」
query = "老師上課可以傳輸影片嗎？"

# 關鍵詞提取
query_keywords = ["老師", "上課", "傳輸", "影片"]

# Metadata匹配
match_score = 0.85  # 高匹配分數
matched_article = "著作權法_第46條"
```

#### 3. **繼承候選擴展**

```python
# 繼承的chunks
inherited_candidates = [
    {
        "chunk_id": "著作權法_第46條_第1項",
        "content": "為學校授課需要，在合理範圍內...",
        "inherited_from": "著作權法_第46條",
        "inheritance_bonus": 0.1
    },
    {
        "chunk_id": "著作權法_第46條_第2項",
        "content": "為學校授課需要，得重製他人已公開發表之著作...",
        "inherited_from": "著作權法_第46條",
        "inheritance_bonus": 0.1
    }
]
```

#### 4. **最終檢索結果**

```python
final_results = [
    {
        "content": "第46條第2項：為學校授課需要，得重製他人已公開發表之著作...",
        "hybrid_score": 0.92,
        "vector_score": 0.85,
        "bm25_score": 0.78,
        "metadata_bonus": 0.15,
        "inheritance_bonus": 0.1,
        "inherited_from": "著作權法_第46條",
        "metadata_match_reason": "繼承自條層級 著作權法_第46條"
    }
]
```

## 🚀 優勢效果

### 1. **檢索廣度提升**

- 原本只檢索到條層級，現在可以檢索到項、款、目層級
- 大幅提升檢索的覆蓋範圍

### 2. **檢索精度提升**

- 通過 metadata 關鍵字匹配，精確定位相關條文
- 繼承策略確保相關子條文也被納入考慮

### 3. **法律專業性**

- 符合法律文檔的層次結構
- 體現法律條文的邏輯關係

### 4. **效率優化**

- 只對條層級進行完整 metadata 增強
- 子層級通過繼承獲得 metadata，避免重複計算

## 📈 統計信息

系統提供詳細的統計信息：

```python
stats = enhanced_hybrid_rag.get_retrieval_stats()
print(stats["inheritance_stats"])

# 輸出：
{
    "total_articles": 150,           # 總條層級數量
    "total_inheritance_relations": 450,  # 總繼承關係數量
    "avg_children_per_article": 3.0      # 平均每條的子層級數量
}
```

## 🎯 總結

Metadata 向下繼承實現了您要求的所有功能：

✅ **條層級作為 metadata 生成重心**  
✅ **項、款、目層級自動繼承 metadata**  
✅ **查詢時先進行 metadata 關鍵字匹配**  
✅ **匹配到條層級後擴展到所有子層級**  
✅ **最終通過向量相似度確定最佳結果**

這個策略完美體現了法律文檔的層次結構，讓檢索既廣又精，大大提升了法律 RAG 的專業性和實用性！
