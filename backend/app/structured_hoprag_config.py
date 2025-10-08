"""
Structured-HopRAG 配置模組
针对结构化法律文本优化的HopRAG系统
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum

class LegalLevel(Enum):
    """法律文档层级枚举 - 完整7层结构"""
    DOCUMENT = "document"                      # 法规（原law）
    DOCUMENT_COMPONENT = "document_component"  # 章（原chapter）
    BASIC_UNIT_HIERARCHY = "basic_unit_hierarchy"  # 节（原section）
    BASIC_UNIT = "basic_unit"                  # 条（原article）
    BASIC_UNIT_COMPONENT = "basic_unit_component"  # 项（原paragraph）
    ENUMERATION = "enumeration"                # 款/目（原subparagraph/item）
    
    @classmethod
    def get_hierarchy(cls) -> List[str]:
        """获取层级顺序（从高到低）"""
        return [
            cls.DOCUMENT.value,
            cls.DOCUMENT_COMPONENT.value,
            cls.BASIC_UNIT_HIERARCHY.value,
            cls.BASIC_UNIT.value,
            cls.BASIC_UNIT_COMPONENT.value,
            cls.ENUMERATION.value
        ]
    
    @classmethod
    def get_level_index(cls, level: str) -> int:
        """获取层级索引（用于计算aboutness权重）"""
        hierarchy = cls.get_hierarchy()
        return hierarchy.index(level) if level in hierarchy else -1

class EdgeType(Enum):
    """边类型枚举"""
    HIERARCHY = "hierarchy"          # 层级边（父子关系）
    REFERENCE = "reference"          # 引用边（准用、依第X条）
    SIMILAR_CONCEPT = "similar_concept"  # 相似概念边
    THEME = "theme"                  # 主题边（聚类）
    LLM_GENERATED = "llm_generated"  # LLM生成边（复杂情况）

@dataclass
class EdgePriority:
    """边类型优先级配置"""
    hierarchy: float = 1.0
    reference: float = 0.95
    theme: float = 0.85
    similar_concept: float = 0.8
    llm_generated: float = 0.7
    
    def get_priority(self, edge_type: str) -> float:
        """获取边类型的优先级权重"""
        return getattr(self, edge_type, 0.5)

@dataclass
class AboutnessWeights:
    """Aboutness权重配置（用于多层次嵌入聚合）"""
    # 从低层到高层，aboutness逐渐降低
    enumeration: float = 0.45        # 款/目：细节最丰富
    basic_unit_component: float = 0.40  # 项：细节丰富
    basic_unit: float = 0.30         # 条：中等细节
    basic_unit_hierarchy: float = 0.20  # 节：较宽泛
    document_component: float = 0.15  # 章：宽泛
    document: float = 0.10           # 法规：最宽泛
    
    def get_weight(self, level: str) -> float:
        """获取层级的默认aboutness权重"""
        return getattr(self, level, 0.3)

@dataclass
class LegalLogicTemplate:
    """法律逻辑模板"""
    name: str
    stages: List[str]
    keywords: List[str]
    
# 预定义的法律逻辑模板
LEGAL_LOGIC_TEMPLATES = {
    "侵权后果": LegalLogicTemplate(
        name="侵权后果",
        stages=["权利定义", "侵权行为", "民事责任", "刑事罚则"],
        keywords=["侵权", "违反", "后果", "责任", "罚则", "赔偿", "处罚"]
    ),
    "权利行使": LegalLogicTemplate(
        name="权利行使",
        stages=["权利定义", "行使方式", "限制条件", "例外情况"],
        keywords=["权利", "行使", "如何", "方式", "条件", "限制", "例外"]
    ),
    "合理使用": LegalLogicTemplate(
        name="合理使用",
        stages=["权利范围", "例外规定", "合理使用条款"],
        keywords=["合理使用", "例外", "不构成", "免责", "法定许可"]
    ),
    "申请程序": LegalLogicTemplate(
        name="申请程序",
        stages=["申请条件", "申请材料", "审查程序", "批准标准"],
        keywords=["申请", "程序", "如何", "材料", "审查", "批准"]
    ),
    "时效规定": LegalLogicTemplate(
        name="时效规定",
        stages=["起算时点", "期限长度", "中止中断", "法律效果"],
        keywords=["期限", "时效", "多久", "何时", "起算", "届满"]
    )
}

@dataclass
class StructuredHopRAGConfig:
    """Structured-HopRAG 系统配置"""
    
    # ========== 多层次嵌入配置 ==========
    enable_multi_level_embedding: bool = True
    aboutness_weights: AboutnessWeights = field(default_factory=AboutnessWeights)
    min_aboutness_threshold: float = 0.7  # 高相关阈值
    
    # ========== 规则边配置 ==========
    enable_rule_edges: bool = True
    
    # 层级边配置
    hierarchy_edge_enabled: bool = True
    hierarchy_weight_method: str = "cosine"  # cosine, fixed
    
    # 引用边配置
    reference_edge_enabled: bool = True
    reference_patterns: List[str] = field(default_factory=lambda: [
        r'準用.*?第\s*(\d+)\s*條',
        r'依.*?第\s*(\d+)\s*條',
        r'比照.*?第\s*(\d+)\s*條',
        r'適用.*?第\s*(\d+)\s*條'
    ])
    
    # 相似概念边配置
    similar_edge_enabled: bool = True
    similar_edge_threshold: float = 0.75
    similar_edge_levels: List[str] = field(default_factory=lambda: ["basic_unit"])  # 仅条级
    
    # 主题边配置
    theme_edge_enabled: bool = True
    theme_clustering_method: str = "kmeans"  # kmeans, dbscan
    theme_num_clusters: int = 10
    theme_levels: List[str] = field(default_factory=lambda: [
        "document_component", "basic_unit_hierarchy"
    ])
    
    # ========== LLM边配置（精简版）==========
    enable_llm_edges: bool = True
    llm_edge_only_complex: bool = True  # 仅针对复杂情况
    llm_edge_max_per_node: int = 2  # 每个节点最多2条LLM边（1 in + 1 out）
    llm_edge_levels: List[str] = field(default_factory=lambda: [
        "basic_unit", "basic_unit_component", "enumeration"
    ])  # 仅叶节点
    llm_complexity_threshold: float = 0.75  # 低于此阈值才用LLM
    
    # ========== 边权重和优先级 ==========
    edge_priority: EdgePriority = field(default_factory=EdgePriority)
    min_edge_weight: float = 0.6  # 低于此值的边将被删除
    max_edges_per_node: int = 15  # 每个节点最多保留的边数
    
    # ========== 检索配置（去LLM化）==========
    enable_llm_reasoning: bool = False  # 检索时不使用LLM推理
    enable_template_navigation: bool = True  # 启用法律逻辑模板导航
    enable_query_cache: bool = True  # 启用查询路径缓存
    
    max_hops: int = 3  # 最多3跳（相比原4跳更快）
    top_k_per_hop: int = 15  # 每跳保留15个节点
    initial_retrieve_k: int = 20  # 初始检索20个节点
    
    # ========== 遍历策略 ==========
    traversal_strategy: str = "priority_weighted"  # priority_weighted, similarity_only
    edge_weight_in_traversal: float = 0.6  # 边权重的影响比例
    query_similarity_weight: float = 0.4  # 查询相似度的影响比例
    
    # ========== Embedding配置 ==========
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 32
    
    # ========== LLM配置（仅索引时使用）==========
    llm_model: str = "gemini-2.0-flash-exp"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_max_retries: int = 3
    
    # ========== 关键词提取配置 ==========
    keyword_extraction_method: str = "tfidf"  # tfidf, textrank
    keyword_min_length: int = 2
    keyword_max_count: int = 10
    legal_term_boost: float = 1.5  # 法律术语权重提升
    
    # ========== 缓存配置 ==========
    cache_max_size: int = 1000  # 最多缓存1000条查询路径
    cache_ttl: int = 3600  # 缓存有效期（秒）
    
    # ========== 法律逻辑模板 ==========
    legal_templates: Dict[str, LegalLogicTemplate] = field(
        default_factory=lambda: LEGAL_LOGIC_TEMPLATES
    )
    
    # ========== 性能优化 ==========
    use_graph_embedding: bool = False  # GraphSAGE（可选，复杂度较高）
    edge_complexity_limit: str = "n_log_n"  # n_log_n, n_sqrt_n
    parallel_processing: bool = True
    num_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enable_multi_level_embedding': self.enable_multi_level_embedding,
            'enable_rule_edges': self.enable_rule_edges,
            'enable_llm_edges': self.enable_llm_edges,
            'enable_llm_reasoning': self.enable_llm_reasoning,
            'enable_template_navigation': self.enable_template_navigation,
            'max_hops': self.max_hops,
            'top_k_per_hop': self.top_k_per_hop,
            'edge_priority': {
                'hierarchy': self.edge_priority.hierarchy,
                'reference': self.edge_priority.reference,
                'theme': self.edge_priority.theme,
                'similar_concept': self.edge_priority.similar_concept,
                'llm_generated': self.edge_priority.llm_generated
            },
            'traversal_strategy': self.traversal_strategy,
            'enable_query_cache': self.enable_query_cache
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StructuredHopRAGConfig':
        """从字典创建配置"""
        # 简化版，仅处理基本字段
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if self.max_hops < 1:
            raise ValueError("max_hops must be at least 1")
        if self.top_k_per_hop < 1:
            raise ValueError("top_k_per_hop must be at least 1")
        if not 0.0 <= self.min_edge_weight <= 1.0:
            raise ValueError("min_edge_weight must be between 0.0 and 1.0")
        return True

# 预设配置方案
DEFAULT_CONFIG = StructuredHopRAGConfig()

# 快速配置（最小LLM使用）
FAST_CONFIG = StructuredHopRAGConfig(
    enable_llm_edges=False,  # 完全不使用LLM边
    enable_template_navigation=True,
    max_hops=2,
    top_k_per_hop=10
)

# 平衡配置（推荐）
BALANCED_CONFIG = StructuredHopRAGConfig(
    enable_llm_edges=True,
    llm_edge_only_complex=True,
    llm_edge_max_per_node=2,
    max_hops=3,
    top_k_per_hop=15
)

# 高精度配置（使用更多LLM边）
HIGH_ACCURACY_CONFIG = StructuredHopRAGConfig(
    enable_llm_edges=True,
    llm_edge_only_complex=False,
    llm_edge_max_per_node=3,
    max_hops=4,
    top_k_per_hop=20,
    min_edge_weight=0.5
)
