# Metadata 查看與編輯界面說明

## 🎯 **功能概述**

我已經為您創建了一個完整的 **Enhanced Metadata 查看與編輯界面**，讓您可以：

✅ **查看所有 chunks 的 enhanced metadata**  
✅ **按層級分類顯示（條層級、章節層級、繼承層級）**  
✅ **編輯任何 metadata 字段**  
✅ **生成新的 enhanced metadata**  
✅ **實時更新和保存**

## 🔧 **界面組成**

### 1. **MetadataViewer 組件**

```typescript
// 位置：frontend/src/components/MetadataViewer.tsx
interface EnhancedMetadata {
  legal_concepts?: Array<{
    concept_name: string;
    concept_type: string;
    legal_domain: string;
    importance_score: number;
    synonyms: string[];
    confidence: number;
  }>;
  semantic_keywords?: {
    primary_keywords: string[];
    keyword_weights: Record<string, number>;
  };
  article_type?: {
    article_type: string;
    confidence: number;
  };
  // ... 更多字段
}
```

### 2. **後端 API 端點**

```python
# 新增的API端點
GET  /api/enhanced-metadata-stats      # 獲取統計信息
GET  /api/enhanced-metadata-list       # 獲取metadata列表
POST /api/update-enhanced-metadata     # 更新特定chunk的metadata
POST /api/generate-enhanced-metadata   # 生成新的enhanced metadata
```

## 🚀 **使用流程**

### 步驟 1：**進入 Metadata 界面**

1. 在 ChunkPage 完成分塊後
2. 點擊步驟 2 卡片右上角的 **"Metadata"** 按鈕
3. 打開 Metadata Viewer 界面

### 步驟 2：**生成 Enhanced Metadata**

```typescript
// 如果還沒有enhanced metadata，會顯示生成按鈕
<button onClick={generateEnhancedMetadata}>生成 Enhanced Metadata</button>

// 生成完成後會顯示統計信息：
// - 總chunks: 4212
// - 條層級chunks: 156
// - 章節層級chunks: 23
// - 繼承chunks: 4033
```

### 步驟 3：**查看 Metadata 內容**

```typescript
// 左側：Chunk列表
<div className="list-group">
  {chunks.map(chunk => (
    <button key={chunkId}>
      <small>{chunkId}</small>
      {metadata.is_article_level && <span className="badge bg-success">條層級</span>}
      {metadata.is_chapter_section_level && <span className="badge bg-warning">章節層級</span>}
      {metadata.inherited_from && <span className="badge bg-info">繼承</span>}
    </button>
  ))}
</div>

// 右側：Metadata詳細內容
<div className="metadata-display">
  {Object.entries(metadata).map(([key, value]) =>
    renderMetadataField(key, value)
  )}
</div>
```

### 步驟 4：**編輯 Metadata**

```typescript
// 點擊「編輯」按鈕進入編輯模式
<button onClick={() => setIsEditing(true)}>編輯</button>

// 編輯界面支持：
// - 文本字段：直接輸入
// - 數字字段：數字輸入框
// - 數組字段：JSON格式編輯
// - 對象字段：嵌套編輯
```

## 📊 **界面功能詳解**

### 1. **統計信息顯示**

```typescript
// 頂部統計信息
<span className="badge bg-info">
  總計 {Object.keys(enhancedMetadata).length} 個chunks
</span>

// 詳細統計（生成後顯示）
{
  "total_chunks": 4212,
  "article_level_chunks": 156,
  "chapter_section_chunks": 23,
  "inherited_chunks": 4033,
  "enhancement_levels": {
    "full": 156,      // 條層級：完整增強
    "medium": 23,     // 章節層級：中等增強
    "lightweight": 4033, // 其他層級：輕量增強
    "none": 0
  }
}
```

### 2. **Chunk 分類標識**

```typescript
// 不同層級的標識
{
  metadata.is_article_level && <span className="badge bg-success">條層級</span>;
}
{
  metadata.is_chapter_section_level && (
    <span className="badge bg-warning">章節層級</span>
  );
}
{
  metadata.inherited_from && <span className="badge bg-info">繼承</span>;
}
```

### 3. **Metadata 字段顯示**

```typescript
// 支持嵌套結構的顯示
const renderMetadataField = (key: string, value: any, level: number = 0) => {
  if (typeof value === "object" && !Array.isArray(value)) {
    // 對象：遞歸顯示
    return Object.entries(value).map(([subKey, subValue]) =>
      renderMetadataField(subKey, subValue, level + 1)
    );
  }

  if (Array.isArray(value)) {
    // 數組：逐項顯示
    return value.map((item, index) => (
      <div key={index}>
        {typeof item === "object" ? (
          Object.entries(item).map(([itemKey, itemValue]) =>
            renderMetadataField(itemKey, itemValue, level + 2)
          )
        ) : (
          <span>{JSON.stringify(item)}</span>
        )}
      </div>
    ));
  }

  // 基本類型：直接顯示
  return (
    <div>
      <strong>{key}:</strong> {JSON.stringify(value)}
    </div>
  );
};
```

### 4. **編輯功能**

```typescript
// 文本字段編輯
<input
  type="text"
  value={value}
  onChange={(e) => updateMetadata(path, e.target.value)}
/>

// 數字字段編輯
<input
  type="number"
  value={value}
  onChange={(e) => updateMetadata(path, parseFloat(e.target.value))}
/>

// 數組/對象字段編輯
<textarea
  value={JSON.stringify(value, null, 2)}
  onChange={(e) => {
    try {
      const parsed = JSON.parse(e.target.value);
      updateMetadata(path, parsed);
    } catch (err) {
      // 忽略JSON解析錯誤
    }
  }}
/>
```

## 🎨 **界面特色**

### 1. **響應式設計**

- 左側：Chunk 列表（25%寬度）
- 右側：Metadata 詳細內容（75%寬度）
- 支持滾動查看長內容

### 2. **直觀的分類**

- 🟢 **條層級**：完整增強，包含所有 metadata 字段
- 🟡 **章節層級**：中等增強，包含結構性字段
- 🔵 **繼承**：從父級條層級繼承 metadata

### 3. **實時編輯**

- 點擊「編輯」按鈕進入編輯模式
- 支持嵌套字段的編輯
- 實時預覽和驗證

### 4. **數據持久化**

```typescript
// 保存時同時更新兩個存儲
await api.post("/update-enhanced-metadata", {
  chunk_id: selectedChunkId,
  enhanced_metadata: editingMetadata,
});

// 後端會更新：
// 1. store.enhanced_metadata
// 2. faiss_store.enhanced_metadata
// 3. 持久化保存
```

## 🔄 **與現有系統的整合**

### 1. **ChunkPage 整合**

```typescript
// 在ChunkPage中添加Metadata按鈕
{
  chunkingResults.length > 0 && (
    <button
      className="btn btn-sm btn-outline-light"
      onClick={() => setShowMetadataViewer(true)}
    >
      <i className="bi bi-tags"></i> Metadata
    </button>
  );
}
```

### 2. **HybridRAG 整合**

```typescript
// 編輯後的metadata會自動用於HybridRAG檢索
// 在enhanced_hybrid_rag.py中：
metadata_bonus = self._calculate_metadata_bonus(
  query,
  enhanced_metadata,
  config
);
```

### 3. **數據流整合**

```
分塊階段 → 生成Enhanced Metadata → 查看/編輯 → 保存 → HybridRAG檢索使用
```

## 💡 **使用建議**

### 1. **查看優先級**

1. 先查看「條層級」chunks（最重要）
2. 再查看「章節層級」chunks（結構性）
3. 最後查看「繼承」chunks（細節性）

### 2. **編輯建議**

- 重點編輯 `legal_concepts` 和 `semantic_keywords`
- 調整 `importance_score` 和 `confidence`
- 添加或修改 `synonyms` 同義詞

### 3. **性能優化**

- 大量 chunks 時，使用搜索功能快速定位
- 批量編輯相似 chunks 的 metadata
- 定期保存避免數據丟失

## 🎯 **總結**

現在您有了完整的 Enhanced Metadata 查看與編輯界面：

✅ **完整的查看功能** - 所有 metadata 一目了然  
✅ **直觀的編輯功能** - 支持所有字段類型的編輯  
✅ **智能的分類顯示** - 按層級和類型分類  
✅ **實時的數據同步** - 編輯後立即生效  
✅ **完美的系統整合** - 與 HybridRAG 無縫配合

這個界面讓您可以精細化控制每個 chunk 的 metadata，從而優化 HybridRAG 的檢索效果！🚀
