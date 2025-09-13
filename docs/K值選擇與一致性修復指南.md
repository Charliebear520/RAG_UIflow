# K 值選擇與一致性修復指南

## 🔍 問題分析

### 發現的不一致問題

- **後端推薦算法**: 使用 k=5 ✅（更合理）
- **前端顯示**: 使用 k=3 ❌（不一致）
- **結果**: 用戶看到的指標與推薦算法使用的指標不一致

### 為什麼後端使用 k=5 更合理

從實際數據看：

```
Precision@3: 0.711 (71.1%) - 過於樂觀
Precision@5: 0.493 (49.3%) - 更現實 ✅
Precision@10: 0.267 (26.7%) - 過於保守
```

## 📊 不同 K 值的實際意義

### Precision@1

- **用途**: 評估最佳匹配的準確性
- **適用場景**: 精確問答、事實查詢
- **特點**: 通常很高，但不代表整體性能

### Precision@3

- **用途**: 評估前 3 個結果的相關性
- **適用場景**: 需要多個相關結果的查詢
- **特點**: 平衡點，但可能過於樂觀

### Precision@5 ⭐ **推薦**

- **用途**: 評估前 5 個結果的相關性
- **適用場景**: 大多數 RAG 應用
- **特點**: 更接近實際使用體驗

### Precision@10

- **用途**: 評估前 10 個結果的相關性
- **適用場景**: 需要大量相關文檔的場景
- **特點**: 更保守，但更全面

## 🛠 修復方案

### 方案 1: 將前端改為 k=5（推薦）

保持與後端一致，使用 k=5 作為主要指標

### 方案 2: 將後端改為 k=3

不推薦，因為 k=5 更符合實際使用場景

### 方案 3: 提供多 k 值選擇

最靈活，但增加複雜度

## 📝 具體修改建議

### 需要修改的前端文件

`frontend/src/routes/ChunkPage.tsx`

### 修改點 1: 推薦算法評分

```typescript
// 當前代碼 (第528行)
(current.metrics.precision_at_k[3] || 0) * 0.3 +
(current.metrics.recall_at_k[3] || 0) * 0.3 +

// 建議改為
(current.metrics.precision_at_k[5] || 0) * 0.3 +
(current.metrics.recall_at_k[5] || 0) * 0.3 +
```

### 修改點 2: 結果表格顯示

```typescript
// 當前代碼 (第1192行)
result.metrics.precision_at_k[3] > 0.7;

// 建議改為
result.metrics.precision_at_k[5] > 0.5; // 調整閾值
```

### 修改點 3: 對比分析表格

```typescript
// 當前代碼 (第1316行)
metrics.precision_at_k?.[3]?.toFixed(3);

// 建議改為
metrics.precision_at_k?.[5]?.toFixed(3);
```

### 修改點 4: 顏色編碼閾值調整

```typescript
// 由於k=5的數值通常比k=3低，需要調整閾值
const getPerformanceColor = (value: number, metric: string) => {
  if (metric === "precision") {
    // k=5的閾值調整
    return value > 0.5
      ? "text-green-600" // 高 (原來0.7)
      : value > 0.3
      ? "text-yellow-600" // 中 (原來0.5)
      : "text-red-600"; // 低
  }
  return value > 0.8
    ? "text-green-600"
    : value > 0.6
    ? "text-yellow-600"
    : "text-red-600";
};
```

## 🎯 建議的改進方案

### 方案 1: 使用 k=5 作為主要指標

**理由**：

- k=5 更接近實際 RAG 應用場景
- 平衡了 precision 和 recall
- 避免了 k=3 過於樂觀的問題

### 方案 2: 多 k 值綜合評分

**理由**：

- 提供更全面的性能評估
- 不同應用場景可能需要不同的 k 值
- 避免單一 k 值的偏見

### 方案 3: 自適應 k 值選擇

**理由**：

- 根據文檔長度和查詢複雜度動態調整
- 更智能的評估方式

## 🛠 具體實現建議

### 1. 修改前端主要顯示為 k=5

```typescript
// 當前代碼
precision_at_k[3]?.toFixed(3);

// 建議改為
precision_at_k[5]?.toFixed(3);
```

### 2. 添加多 k 值比較視圖

```typescript
const MultiKComparison = ({ result }) => (
  <div className="grid grid-cols-4 gap-2 text-sm">
    <div className="text-center">
      <div className="font-bold">P@1</div>
      <div>{result.metrics.precision_at_k[1]?.toFixed(3)}</div>
    </div>
    <div className="text-center">
      <div className="font-bold">P@3</div>
      <div>{result.metrics.precision_at_k[3]?.toFixed(3)}</div>
    </div>
    <div className="text-center">
      <div className="font-bold">P@5</div>
      <div className="text-blue-600 font-bold">
        {result.metrics.precision_at_k[5]?.toFixed(3)}
      </div>
    </div>
    <div className="text-center">
      <div className="font-bold">P@10</div>
      <div>{result.metrics.precision_at_k[10]?.toFixed(3)}</div>
    </div>
  </div>
);
```

### 3. 智能 k 值選擇邏輯

```typescript
const getOptimalK = (chunkCount: number, queryType: string) => {
  // 根據chunk數量調整k值
  if (chunkCount < 10) return 3;
  if (chunkCount < 50) return 5;
  if (chunkCount < 100) return 10;
  return 15;

  // 或者根據查詢類型
  // if (queryType === 'specific') return 3;
  // if (queryType === 'general') return 5;
  // return 10;
};
```

### 4. 綜合評分算法改進

```typescript
const calculateCompositeScore = (metrics) => {
  // 加權平均，更重視實際使用場景
  const weights = {
    precision_at_k: { 1: 0.1, 3: 0.2, 5: 0.4, 10: 0.3 },
    recall_at_k: { 1: 0.1, 3: 0.2, 5: 0.4, 10: 0.3 },
    precision_omega: 0.3,
  };

  const precisionScore = Object.entries(weights.precision_at_k).reduce(
    (sum, [k, weight]) => sum + (metrics.precision_at_k[k] || 0) * weight,
    0
  );

  const recallScore = Object.entries(weights.recall_at_k).reduce(
    (sum, [k, weight]) => sum + (metrics.recall_at_k[k] || 0) * weight,
    0
  );

  return (
    precisionScore * 0.4 +
    recallScore * 0.3 +
    (metrics.precision_omega || 0) * 0.3
  );
};
```

## 📋 實施優先級

### 高優先級（立即修復）

1. 修改推薦算法評分使用 k=5
2. 修改對比分析表格顯示 k=5
3. 調整顏色編碼閾值

### 中優先級（短期改進）

1. 添加 k 值選擇器
2. 提供多 k 值比較視圖
3. 添加 k 值說明文檔

### 低優先級（長期規劃）

1. 實現自適應 k 值選擇
2. 基於用戶反饋優化 k 值
3. 添加 k 值敏感性分析

## 📊 預期效果

### 修復後的一致性

- 前端顯示與後端推薦算法使用相同的 k 值
- 用戶看到的指標與推薦理由一致
- 更符合實際 RAG 應用場景的評估

### 數值變化預期

```
修復前 (k=3): Precision = 0.711 (71.1%)
修復後 (k=5): Precision = 0.493 (49.3%)
```

數值會降低，但更真實反映實際性能。

## 🎯 最終建議

### 立即可實施

1. **將主要顯示指標改為 k=5**
2. **保留 k=3 作為對比參考**
3. **添加 k 值選擇說明**

### 中期改進

1. **實現多 k 值綜合評分**
2. **添加 k 值選擇器**
3. **提供不同場景的 k 值建議**

### 長期規劃

1. **自適應 k 值選擇**
2. **基於用戶行為的 k 值優化**
3. **動態權重調整**
