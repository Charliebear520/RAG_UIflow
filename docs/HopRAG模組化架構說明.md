# HopRAG 模組化架構說明

## 概述

按照原本設計的架構，我們將 HopRAG 系統重新組織為模組化管理，實現更好的可維護性、可擴展性和測試性。

## 模組化架構

```
HopRAG系統架構：
├── HopRAGConfig (配置管理)
│   ├── HopRAGConfig類
│   ├── 預設配置
│   └── 配置驗證
├── GraphBuilder (圖構建器)
│   ├── PseudoQueryGenerator - 偽查詢生成器
│   ├── EdgeConnector - 邊連接器
│   └── PassageGraphBuilder - 段落圖構建器
├── HopRetriever (多跳檢索器)
│   ├── InitialRetriever - 初始檢索器
│   ├── GraphTraverser - 圖遍歷器
│   └── LLMReasoner - LLM推理器
├── ResultProcessor (結果處理器)
│   ├── RelevanceFilter - 相關性過濾器
│   └── ResultRanker - 結果排序器
└── HopRAGSystem (主系統類)
    ├── 模組協調
    ├── 狀態管理
    └── API接口
```

## 模組詳細說明

### 1. HopRAGConfig (配置管理)

**檔案**: `hoprag_config.py`

**功能**:

- 統一管理所有配置參數
- 提供配置驗證功能
- 支持多種預設配置

**核心類**:

```python
@dataclass
class HopRAGConfig:
    similarity_threshold: float = 0.7
    max_edges_per_node: int = 10
    max_hops: int = 4
    top_k_per_hop: int = 20
    # ... 更多配置參數
```

**預設配置**:

- `DEFAULT_CONFIG`: 標準配置
- `HIGH_PERFORMANCE_CONFIG`: 高性能配置
- `HIGH_ACCURACY_CONFIG`: 高精度配置

### 2. GraphBuilder (圖構建器)

**檔案**: `hoprag_graph_builder.py`

#### 2.1 PseudoQueryGenerator (偽查詢生成器)

**功能**:

- 為法律節點生成內向和外向問題
- 使用 LLM 進行智能問題生成
- 支持問題質量驗證

**核心方法**:

```python
async def generate_pseudo_queries_for_node(self, node: LegalNode) -> LegalNode
async def _generate_incoming_questions(self, node: LegalNode) -> List[str]
async def _generate_outgoing_questions(self, node: LegalNode) -> List[str]
```

#### 2.2 EdgeConnector (邊連接器)

**功能**:

- 執行邊匹配算法
- 計算偽查詢相似度
- 建立圖節點間的邏輯連接

**核心方法**:

```python
async def connect_edges(self, nodes: Dict[str, LegalNode], embedding_model) -> Dict[str, List[Dict]]
async def _perform_edge_matching(self, nodes: Dict[str, LegalNode]) -> Dict[str, List[Dict]]
```

#### 2.3 PassageGraphBuilder (段落圖構建器)

**功能**:

- 協調圖構建流程
- 從多層次 chunks 創建節點
- 管理整個圖構建過程

**核心方法**:

```python
async def build_graph(self, multi_level_chunks) -> Tuple[Dict[str, LegalNode], Dict[str, List[Dict]]]
```

### 3. HopRetriever (多跳檢索器)

**檔案**: `hoprag_hop_retriever.py`

#### 3.1 InitialRetriever (初始檢索器)

**功能**:

- 從基礎檢索結果提取初始節點
- 管理初始檢索統計信息

**核心方法**:

```python
async def retrieve_initial_nodes(self, query: str, base_results: List[Dict]) -> List[str]
```

#### 3.2 GraphTraverser (圖遍歷器)

**功能**:

- 執行多跳圖遍歷
- 管理跳躍過程和統計信息

**核心方法**:

```python
async def traverse_graph(self, query: str, initial_nodes: List[str], graph: nx.DiGraph, nodes: Dict[str, LegalNode]) -> Dict[int, List[str]]
```

#### 3.3 LLMReasoner (LLM 推理器)

**功能**:

- 使用 LLM 判斷節點相關性
- 支持批量推理處理

**核心方法**:

```python
async def reason_about_relevance(self, query: str, current_node: str, neighbor: Dict[str, Any]) -> bool
async def batch_reason_about_relevance(self, query: str, current_node: str, neighbors: List[Dict[str, Any]]) -> List[bool]
```

### 4. ResultProcessor (結果處理器)

**檔案**: `hoprag_result_processor.py`

#### 4.1 RelevanceFilter (相關性過濾器)

**功能**:

- 過濾相關性較低的結果
- 支持多種過濾策略

**核心方法**:

```python
def filter_results(self, results: List[RetrievalResult], query: str, min_score: float = 0.3) -> List[RetrievalResult]
```

#### 4.2 ResultRanker (結果排序器)

**功能**:

- 對檢索結果進行排序
- 支持多種排序策略

**排序策略**:

- `weighted_merge`: 加權排序
- `simple_merge`: 簡單排序
- `hop_aware_rank`: 跳躍感知排序

**核心方法**:

```python
def rank_results(self, results: List[RetrievalResult], query: str, strategy: str = None) -> List[RetrievalResult]
```

### 5. HopRAGSystem (主系統類)

**檔案**: `hoprag_system_modular.py`

**功能**:

- 協調所有模組的工作
- 管理系統狀態
- 提供統一的 API 接口

**核心方法**:

```python
async def build_graph_from_multi_level_chunks(self, multi_level_chunks)
async def enhanced_retrieve(self, query: str, base_results: List[Dict], k: int = 5)
def get_graph_statistics(self) -> Dict[str, Any]
def get_module_status(self) -> Dict[str, Any]
def update_config(self, new_config: HopRAGConfig)
def export_graph_data(self) -> Dict[str, Any]
def import_graph_data(self, graph_data: Dict[str, Any])
```

## API 端點

### 基礎端點

```http
POST /api/build-hoprag-graph          # 構建HopRAG圖譜
GET  /api/hoprag-status               # 獲取系統狀態
POST /api/hoprag-enhanced-retrieve    # HopRAG增強檢索
```

### 配置管理端點

```http
GET  /api/hoprag-config               # 獲取當前配置
POST /api/hoprag-config               # 更新配置
```

### 數據管理端點

```http
POST /api/hoprag-export               # 導出圖數據
POST /api/hoprag-import               # 導入圖數據
POST /api/hoprag-reset                # 重置系統
```

## 使用範例

### 1. 基本使用

```python
from backend.app.hoprag_system_modular import HopRAGSystem
from backend.app.hoprag_config import DEFAULT_CONFIG

# 初始化系統
hoprag_system = HopRAGSystem(
    llm_client=llm_client,
    embedding_model=embedding_model,
    config=DEFAULT_CONFIG
)

# 構建圖譜
await hoprag_system.build_graph_from_multi_level_chunks(multi_level_chunks)

# 執行檢索
results = await hoprag_system.enhanced_retrieve(query, base_results, k=5)
```

### 2. 配置管理

```python
from backend.app.hoprag_config import HopRAGConfig

# 創建自定義配置
custom_config = HopRAGConfig(
    similarity_threshold=0.8,
    max_hops=3,
    top_k_per_hop=15
)

# 更新配置
hoprag_system.update_config(custom_config)
```

### 3. 模組狀態檢查

```python
# 獲取模組狀態
module_status = hoprag_system.get_module_status()
print(module_status)

# 獲取圖統計信息
graph_stats = hoprag_system.get_graph_statistics()
print(graph_stats)
```

## 測試

### 運行模組化測試

```bash
cd /Users/charliebear/Desktop/code/RAG
python test_hoprag_modular.py
```

### 測試內容

1. **系統測試**: 完整的工作流程測試
2. **配置測試**: 不同配置的效果測試
3. **模組測試**: 各個模組的獨立測試
4. **API 測試**: 所有 API 端點的測試

## 優勢

### 1. 模組化設計

✅ **清晰分離**: 每個模組職責明確
✅ **易於維護**: 可以獨立修改和測試各個模組
✅ **可重用性**: 模組可以在其他項目中重用

### 2. 配置管理

✅ **統一配置**: 所有配置參數集中管理
✅ **靈活調整**: 支持運行時配置更新
✅ **預設配置**: 提供多種預設配置選項

### 3. 狀態管理

✅ **完整狀態**: 提供詳細的系統狀態信息
✅ **模組狀態**: 可以檢查各個模組的狀態
✅ **統計信息**: 豐富的統計和監控信息

### 4. 數據管理

✅ **導出/導入**: 支持圖數據的導出和導入
✅ **持久化**: 可以保存和恢復系統狀態
✅ **重置功能**: 支持系統重置和重新初始化

### 5. 擴展性

✅ **插件化**: 可以輕鬆添加新的模組
✅ **策略模式**: 支持多種處理策略
✅ **接口標準**: 統一的模組接口標準

## 與原架構的對比

| 方面         | 原架構       | 模組化架構   |
| ------------ | ------------ | ------------ |
| **組織方式** | 單一大類     | 多個專門模組 |
| **配置管理** | 分散在各處   | 統一配置管理 |
| **狀態監控** | 基本統計     | 詳細模組狀態 |
| **測試性**   | 整體測試     | 模組獨立測試 |
| **維護性**   | 較難維護     | 易於維護     |
| **擴展性**   | 需要修改主類 | 插件化擴展   |

## 總結

模組化架構使 HopRAG 系統更加：

- **可維護**: 清晰的模組分離
- **可測試**: 獨立的模組測試
- **可配置**: 統一的配置管理
- **可監控**: 詳細的狀態信息
- **可擴展**: 插件化的設計

這種架構更適合生產環境的使用，也為未來的功能擴展奠定了良好的基礎。
