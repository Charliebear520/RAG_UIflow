# RCTS 層次分割實現說明

## 概述

您說得很對！之前的實現確實和原本的差不多。現在我已經實現了真正的 RCTS（RecursiveCharacterTextSplitter）層次分割，這是一個真正改進的版本。

## RCTS 層次分割的優勢

### 1. 智能分隔符識別

RCTS 使用遞歸字符分割器，按優先級識別分隔符：

```
分隔符優先級（針對中文法律文檔優化）：
1. "\n\n" - 段落分隔
2. "\n" - 行分隔
3. "。" - 句號
4. "；" - 分號
5. "，" - 逗號
6. "、" - 頓號
7. " " - 空格
8. "" - 字符級別
```

### 2. 結合層次結構識別

- 在條文邊界強制分割（保持法律邏輯完整性）
- 對於過大的 chunk，使用 RCTS 進一步智能分割
- 保持結構信息前綴，確保上下文完整性

### 3. 中文法律文檔優化

- 專門針對中文標點符號優化
- 支持法律文檔的特殊格式
- 處理複雜的層次結構

## 實現細節

### 1. 核心類：`RCTSHierarchicalChunking`

```python
class RCTSHierarchicalChunking(ChunkingStrategy):
    def __init__(self):
        # 初始化RCTS分割器
        self.rcts_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "；", "，", "、", " ", ""]
        )

    def chunk(self, text, max_chunk_size=1000, overlap_ratio=0.1,
              preserve_structure=True, **kwargs):
        # 結合層次結構的RCTS分割
```

### 2. 智能分割邏輯

```python
def _hierarchical_rcts_split(self, text, max_chunk_size, overlap_ratio):
    # 1. 檢測層次結構（章、條、項）
    # 2. 在條文邊界強制分割
    # 3. 對於過大的chunk，使用RCTS進一步分割
    # 4. 保持結構信息前綴
```

### 3. 結構信息保持

```python
def _rcts_split_with_structure(self, text, max_size, overlap_ratio, structure):
    # 構建結構前綴：章、條、項信息
    structure_prefix = self._build_structure_prefix(structure)

    # 使用RCTS分割
    sub_chunks = self.rcts_splitter.split_text(text)

    # 為每個sub-chunk添加結構前綴
    # 第一個chunk：完整結構信息
    # 後續chunk：基本結構信息
```

## 前端集成

### 1. 新增策略選項

```typescript
export type ChunkStrategy =
  | "fixed_size"
  | "hierarchical"
  | "rcts_hierarchical" // 新增
  | "structured_hierarchical";
// ...
```

### 2. 參數配置

```typescript
rcts_hierarchical: {
  max_chunk_size: number; // 最大分塊大小
  overlap_ratio: number; // 重疊比例
  preserve_structure: boolean; // 是否保持層次結構
}
```

### 3. UI 界面

- 最大分塊大小設置
- 重疊比例調整
- 保持層次結構開關
- 詳細說明文字

## 與其他策略的對比

### 1. vs 固定大小分割

- **固定大小**：簡單粗暴，可能切斷語義
- **RCTS 層次**：智能識別分隔符，保持語義完整性

### 2. vs 基本層次分割

- **基本層次**：只按結構分割，長度不均勻
- **RCTS 層次**：結合智能分隔符，處理長文本更優雅

### 3. vs 結構化層次分割

- **結構化層次**：需要 JSON 數據，依賴 PDF 解析
- **RCTS 層次**：純文本處理，更通用

## 使用場景

### 1. 法律文檔處理

- 保持條文完整性
- 處理複雜的法律邏輯
- 支持引用關係

### 2. 長文本處理

- 智能處理超長條文
- 保持語義連貫性
- 優化檢索效果

### 3. 通用文檔處理

- 不依賴特定格式
- 適應各種文檔類型
- 自動優化分割點

## 測試建議

### 1. 基本功能測試

```bash
# 測試RCTS層次分割
strategy: "rcts_hierarchical"
max_chunk_size: 1000
overlap_ratio: 0.1
preserve_structure: true
```

### 2. 對比測試

- RCTS 層次分割 vs 固定大小分割
- RCTS 層次分割 vs 基本層次分割
- 觀察分割質量和檢索效果

### 3. 參數調優

- 調整 max_chunk_size（500-2000）
- 調整 overlap_ratio（0.05-0.3）
- 測試 preserve_structure 開關效果

## 預期效果

### 1. 分割質量提升

- 更智能的分割點選擇
- 更好的語義完整性
- 更均勻的 chunk 長度

### 2. 檢索效果改善

- 提高 precision@K
- 改善 recall@K
- 更好的上下文保持

### 3. 法律邏輯保持

- 條文完整性
- 引用關係保持
- 結構信息保留

## 總結

RCTS 層次分割真正結合了：

- **智能分隔符識別**：RCTS 的遞歸分割能力
- **層次結構保持**：法律文檔的結構邏輯
- **中文優化**：針對中文法律文檔的特殊處理

這是一個真正改進的實現，不再是"和原本的差不多"，而是真正利用了 RCTS 的優勢來提升層次分割的效果！
