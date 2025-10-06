"""
HopRAG配置管理模組
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class NodeType(Enum):
    """節點類型枚舉"""
    ARTICLE = "article"
    ITEM = "item"

class EdgeType(Enum):
    """邊類型枚舉"""
    ARTICLE_TO_ARTICLE = "article_to_article"
    ARTICLE_TO_ITEM = "article_to_item"
    ITEM_TO_ARTICLE = "item_to_article"
    ITEM_TO_ITEM = "item_to_item"

class QueryType(Enum):
    """查詢類型枚舉"""
    INCOMING = "incoming"
    OUTGOING = "outgoing"

@dataclass
class HopRAGConfig:
    """HopRAG系統配置"""
    
    # 圖構建配置 - 基於論文4.1 Settings
    similarity_threshold: float = 0.7
    max_edges_per_node: int = 10  # 固定限制（向後兼容）
    
    # 動態邊數限制 - 基於論文O(n log n)要求
    use_dynamic_edge_limit: bool = True  # 是否使用動態邊數限制
    edge_limit_factor: float = 1.0  # O(n log n)的係數因子
    
    # 動態問題生成配置
    use_dynamic_question_count: bool = True  # 是否使用動態問題數量
    min_incoming_questions: int = 2  # 最少內向問題數量
    min_outgoing_questions: int = 4  # 最少外向問題數量
    max_incoming_questions: int = 8  # 最多內向問題數量
    max_outgoing_questions: int = 12  # 最多外向問題數量
    
    # 向後兼容的固定數量配置（當use_dynamic_question_count=False時使用）
    max_pseudo_queries_per_node: int = 5
    
    # 檢索配置 - 基於論文4.1 Settings和Table 4優化
    max_hops: int = 4  # 論文建議：基於小世界理論，4跳足以涵蓋大多數邏輯鄰域
    top_k_per_hop: int = 20  # 每跳保留的節點數
    initial_retrieve_k: int = 20  # 初始檢索節點數
    
    # nhop性能分析配置
    enable_nhop_analysis: bool = True  # 是否啟用nhop分析
    nhop_cost_tracking: bool = True  # 是否追蹤LLM呼叫成本
    queue_length_threshold: float = 1.5  # 隊列長度閾值（論文：第5跳為1.23）
    
    # LLM配置 - 基於論文4.1 Settings優化
    # 問題生成LLM配置（論文：GPT-4o-mini, temp=0.1, max_tokens=2048）
    question_generation_temperature: float = 0.1  # 論文建議：低溫度確保一致性
    question_generation_max_tokens: int = 2048  # 論文設置
    question_generation_model: str = "gpt-4o-mini"  # 論文推薦模型
    
    # 推理LLM配置（論文：GPT-4o 或 GPT-3.5-turbo）
    reasoning_temperature: float = 0.7  # 推理可以稍高溫度
    reasoning_max_tokens: int = 1000
    reasoning_model: str = "gpt-3.5-turbo"  # 論文推薦模型
    
    # 通用LLM配置
    llm_temperature: float = 0.7  # 向後兼容
    llm_max_tokens: int = 1000  # 向後兼容
    llm_max_retries: int = 3
    
    # Embedding配置 - 基於論文4.1 Settings
    embedding_dimension: int = 768  # 論文：BGE-base (768維)
    embedding_batch_size: int = 32
    embedding_model_name: str = "BAAI/bge-base-zh-v1.5"  # 論文推薦：BGE-base
    
    # 關鍵字提取配置 - 基於論文4.1 Settings
    # 論文使用：PaddleNLP POS tagging，我們使用jieba作為替代
    keyword_extraction_method: str = "jieba"  # jieba, paddlenlp, spacy
    keyword_min_length: int = 2  # 最小關鍵詞長度
    keyword_max_length: int = 10  # 最大關鍵詞長度
    enable_pos_filtering: bool = True  # 是否啟用詞性過濾
    
    # 檢索策略配置
    base_strategy: str = "multi_level"  # multi_level, single_level, hybrid
    use_hoprag: bool = True
    
    # Ablation研究配置 - 基於論文Table 5
    enable_ablation_mode: bool = False  # 是否啟用Ablation模式
    ablation_traversal_method: str = "similarity_only"  # similarity_only, llm_reasoning
    ablation_llm_model: str = "Qwen2.5-1.5B-Instruct"  # 論文推薦的低成本替代模型
    
    # 結果處理配置
    result_merge_strategy: str = "weighted_merge"  # weighted_merge, simple_merge
    hop_weight_decay: float = 0.8  # 每跳權重衰減因子
    
    # 混合檢索配置
    use_hybrid_retrieval: bool = True  # 是否使用混合檢索
    jaccard_weight: float = 0.5  # Jaccard相似度權重
    cosine_weight: float = 0.5  # 餘弦相似度權重
    lexical_threshold: float = 0.3  # 詞彙相似度閾值
    semantic_threshold: float = 0.5  # 語義相似度閾值
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            'similarity_threshold': self.similarity_threshold,
            'max_edges_per_node': self.max_edges_per_node,
            'use_dynamic_edge_limit': self.use_dynamic_edge_limit,
            'edge_limit_factor': self.edge_limit_factor,
            'use_dynamic_question_count': self.use_dynamic_question_count,
            'min_incoming_questions': self.min_incoming_questions,
            'min_outgoing_questions': self.min_outgoing_questions,
            'max_incoming_questions': self.max_incoming_questions,
            'max_outgoing_questions': self.max_outgoing_questions,
            'max_pseudo_queries_per_node': self.max_pseudo_queries_per_node,
            'max_hops': self.max_hops,
            'top_k_per_hop': self.top_k_per_hop,
            'initial_retrieve_k': self.initial_retrieve_k,
            'enable_nhop_analysis': self.enable_nhop_analysis,
            'nhop_cost_tracking': self.nhop_cost_tracking,
            'queue_length_threshold': self.queue_length_threshold,
            'question_generation_temperature': self.question_generation_temperature,
            'question_generation_max_tokens': self.question_generation_max_tokens,
            'question_generation_model': self.question_generation_model,
            'reasoning_temperature': self.reasoning_temperature,
            'reasoning_max_tokens': self.reasoning_max_tokens,
            'reasoning_model': self.reasoning_model,
            'llm_temperature': self.llm_temperature,
            'llm_max_tokens': self.llm_max_tokens,
            'llm_max_retries': self.llm_max_retries,
            'embedding_dimension': self.embedding_dimension,
            'embedding_batch_size': self.embedding_batch_size,
            'embedding_model_name': self.embedding_model_name,
            'keyword_extraction_method': self.keyword_extraction_method,
            'keyword_min_length': self.keyword_min_length,
            'keyword_max_length': self.keyword_max_length,
            'enable_pos_filtering': self.enable_pos_filtering,
            'base_strategy': self.base_strategy,
            'use_hoprag': self.use_hoprag,
            'result_merge_strategy': self.result_merge_strategy,
            'hop_weight_decay': self.hop_weight_decay,
            'use_hybrid_retrieval': self.use_hybrid_retrieval,
            'jaccard_weight': self.jaccard_weight,
            'cosine_weight': self.cosine_weight,
            'lexical_threshold': self.lexical_threshold,
            'semantic_threshold': self.semantic_threshold,
            'enable_ablation_mode': self.enable_ablation_mode,
            'ablation_traversal_method': self.ablation_traversal_method,
            'ablation_llm_model': self.ablation_llm_model
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HopRAGConfig':
        """從字典創建配置"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """驗證配置有效性"""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        if self.max_hops < 1:
            raise ValueError("max_hops must be at least 1")
        
        if self.top_k_per_hop < 1:
            raise ValueError("top_k_per_hop must be at least 1")
        
        if self.max_edges_per_node < 1:
            raise ValueError("max_edges_per_node must be at least 1")
        
        return True

# 預設配置
DEFAULT_CONFIG = HopRAGConfig()

# 高性能配置
HIGH_PERFORMANCE_CONFIG = HopRAGConfig(
    similarity_threshold=0.8,
    max_edges_per_node=5,
    max_hops=3,
    top_k_per_hop=15,
    embedding_batch_size=64
)

# 高精度配置
HIGH_ACCURACY_CONFIG = HopRAGConfig(
    similarity_threshold=0.6,
    max_edges_per_node=15,
    max_hops=5,
    top_k_per_hop=25,
    llm_temperature=0.5
)
