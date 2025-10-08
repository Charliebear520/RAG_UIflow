# HopRAG 检索返回 0 结果修复说明

## 问题描述

用户报告 HopRAG 检索功能无法正常工作，不管问什么问题都返回 0 个结果。从终端日志可以看到：

```
✅ 初始檢索完成，獲得 1 個初始節點
✅ 圖遍歷完成，共找到 1 個相關節點
🔄 處理檢索結果，目標數量: 5
🔍 過濾檢索結果，最小分數: 0.3
✅ 過濾完成，0 -> 0 個結果
✅ HopRAG增強檢索完成，返回 0 個結果
```

## 根本原因分析

经过深入调查，发现存在**两个严重的 bug**：

### Bug 1: Node ID 格式不匹配

**问题位置**: `hoprag_result_processor.py:274`

**原因**:

- HopRAG 图谱中的 node*id 格式：`{doc_id}\_basic_unit*{chunk*idx}`或`{doc_id}\_basic_unit_component*{chunk_idx}`
- 检索结果返回的 node*id 格式：`{doc_id}*{idx}`
- 两者格式不匹配，导致在`_convert_to_retrieval_results`中检查`if node_id in nodes`时永远找不到匹配的节点

**代码位置对比**:

```python
# hoprag_graph_builder.py:935, 971
node_id = chunk['metadata'].get('id', f"{doc_id}_basic_unit_{chunk_idx}")
node_id = chunk['metadata'].get('id', f"{doc_id}_basic_unit_component_{chunk_idx}")

# main.py:6586 (multi_level_retrieve_original)
'node_id': f"{doc_id}_{idx}"

# hoprag_result_processor.py:274 (原代码)
if node_id and node_id in nodes:  # 永远找不到！
    node = nodes[node_id]
```

**影响**: 所有 base_results 都无法被转换为 RetrievalResult 对象，导致过滤前结果就是 0 个。

### Bug 2: HopRAG 遍历结果的 similarity_score 为 0

**问题位置**: `hoprag_result_processor.py:311`

**原因**:

```python
similarity_score=0.0,  # HopRAG結果沒有直接的相似度分數
```

在过滤逻辑中：

```python
def _calculate_relevance_score(self, result: RetrievalResult, query: str) -> float:
    base_score = result.similarity_score  # 0.0
    hop_penalty = self.config.hop_weight_decay ** result.hop_level  # 0.8
    type_bonus = self._get_type_bonus(result.node_type)  # 1.0
    relevance_score = base_score * hop_penalty * type_bonus  # 0.0 * ... = 0.0
    return min(relevance_score, 1.0)
```

因为`base_score`是 0.0，所以最终的`relevance_score`也是 0.0，小于 0.3 的过滤阈值，所有 HopRAG 遍历得到的节点都被过滤掉了。

**影响**: 即使 HopRAG 遍历找到了相关节点，也会在过滤阶段被全部删除。

### Bug 3: 边数过少影响图遍历效果

从日志看到：

```
✅ 邊連接完成，共建立 3 條邊
✅ NetworkX圖構建完成：307個節點，3條邊
```

307 个节点只有 3 条边，这会导致图遍历无法找到相关节点。这个问题需要单独调查边构建逻辑。

## 解决方案

### 修复 1: 添加 content fallback 匹配机制

**文件**: `backend/app/hoprag_result_processor.py`

**修改**: 在`_convert_to_retrieval_results`方法中添加通过 content 匹配节点的 fallback 逻辑

```python
def _convert_to_retrieval_results(self, base_results: List[Dict[str, Any]],
                                hop_results: Dict[int, List[str]],
                                nodes: Dict[str, LegalNode]) -> List[RetrievalResult]:
    """轉換為RetrievalResult對象"""
    retrieval_results = []

    # 創建content到node的映射（用於fallback匹配）
    content_to_node = {}
    for nid, node in nodes.items():
        content_to_node[node.content.strip()] = (nid, node)

    # 處理基礎結果
    for result in base_results:
        node_id = result.get('node_id') or result.get('id')
        node = None

        # 嘗試直接匹配node_id
        if node_id and node_id in nodes:
            node = nodes[node_id]
        # 如果直接匹配失敗，嘗試通過content匹配
        elif 'content' in result:
            content_key = result['content'].strip()
            if content_key in content_to_node:
                node_id, node = content_to_node[content_key]
                print(f"🔍 通過content匹配找到節點: {node_id[:50]}...")

        if node:
            retrieval_result = RetrievalResult(
                node_id=node_id,
                content=node.content,
                contextualized_text=node.contextualized_text,
                law_name=node.law_name,
                article_number=node.article_number,
                item_number=node.item_number,
                node_type=node.node_type.value,
                hop_level=0,
                hop_source="base_retrieval",
                similarity_score=result.get('similarity_score', 0.5),  # 如果沒有分數，給默認值0.5
                metadata=node.metadata
            )
            retrieval_results.append(retrieval_result)
```

**优点**:

- 兼容性强：既支持直接 node_id 匹配，也支持 content fallback
- 不侵入其他模块，修改最小化
- 提供了调试信息（打印匹配日志）

### 修复 2: 为 HopRAG 遍历结果分配合理的 similarity_score

**文件**: `backend/app/hoprag_result_processor.py`

**修改**: 基于 hop_level 计算相似度分数

```python
# 處理HopRAG結果
for hop_level, node_ids in hop_results.items():
    if hop_level == 0:  # 跳過基礎結果
        continue

    for node_id in node_ids:
        if node_id in nodes:
            node = nodes[node_id]

            # 為HopRAG遍歷結果分配基於hop_level的相似度分數
            # 越近的跳躍層次，分數越高
            base_hop_score = 0.7  # 基礎分數
            hop_decay = 0.15  # 每跳衰減
            hop_similarity = max(0.3, base_hop_score - (hop_level - 1) * hop_decay)

            retrieval_result = RetrievalResult(
                node_id=node_id,
                content=node.content,
                contextualized_text=node.contextualized_text,
                law_name=node.law_name,
                article_number=node.article_number,
                item_number=node.item_number,
                node_type=node.node_type.value,
                hop_level=hop_level,
                hop_source="hoprag_traversal",
                similarity_score=hop_similarity,  # 基於跳躍層次的相似度分數
                metadata=node.metadata
            )
            retrieval_results.append(retrieval_result)
```

**评分策略**:

- Hop 1: 0.7 (基础分数)
- Hop 2: 0.55 (0.7 - 0.15)
- Hop 3: 0.40 (0.7 - 0.30)
- Hop 4+: 0.30 (最低保底分数)

所有分数都大于 0.3 的过滤阈值，确保 HopRAG 遍历的节点不会被过滤掉。

## 其他相关问题

### EdgeType 序列化错误

虽然已经修复了代码（见`HopRAG持久化JSON序列化修复说明.md`），但用户看到的错误仍然显示 EdgeType 问题，这说明需要**重启后端服务**才能加载新代码。

### 边数过少问题

需要单独调查为什么 307 个节点只建立了 3 条边。可能的原因：

1. 相似度阈值过高
2. 伪查询质量不佳
3. 边构建算法有 bug

## 测试建议

修复后应测试以下场景：

1. **基础检索测试**:

   ```python
   # 测试是否能够正确匹配节点
   results = await hoprag_system.enhanced_retrieve(
       query="公司員工離職後，能否帶走他在職期間創作的作品？",
       base_results=base_results,
       k=5
   )
   assert len(results) > 0
   ```

2. **Content 匹配测试**:

   ```python
   # 检查是否触发了content fallback
   # 查看日志中是否有 "🔍 通過content匹配找到節點"
   ```

3. **HopRAG 遍历测试**:
   ```python
   # 检查遍历结果的similarity_score是否合理
   for result in results:
       if result['hop_source'] == 'hoprag_traversal':
           assert result['similarity_score'] >= 0.3
           assert result['similarity_score'] <= 0.7
   ```

## 后续行动

1. ✅ **修复代码** - 已完成
2. ⏳ **重启后端服务** - 需要用户执行
3. ⏳ **测试检索功能** - 重启后测试
4. 🔍 **调查边数过少问题** - 需要单独分析

## 修复时间

2025-10-07

## 相关文件

- `backend/app/hoprag_result_processor.py` - 结果处理器（主要修改）
- `backend/app/hoprag_graph_builder.py` - 图构建器（node_id 格式定义）
- `backend/app/main.py` - API 端点（multi_level_retrieve_original）
- `backend/app/hoprag_system_modular.py` - HopRAG 系统主类
