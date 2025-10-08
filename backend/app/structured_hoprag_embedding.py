"""
Structured-HopRAG 多层次嵌入模块
实现基于aboutness score的加权聚合
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .structured_hoprag_config import (
    StructuredHopRAGConfig, 
    LegalLevel,
    AboutnessWeights,
    DEFAULT_CONFIG
)

@dataclass
class MultiLevelNode:
    """多层次节点数据结构"""
    node_id: str
    level: str  # LegalLevel枚举值
    content: str
    
    # 嵌入向量
    direct_embedding: Optional[np.ndarray] = None  # 直接嵌入（叶节点）
    aggregated_embedding: Optional[np.ndarray] = None  # 聚合嵌入（父节点）
    final_embedding: Optional[np.ndarray] = None  # 最终使用的嵌入
    
    # 层级关系
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    
    # Aboutness相关
    aboutness_score: float = 0.0  # 相对于父节点的aboutness
    aboutness_weights: Dict[str, float] = None  # 子节点的aboutness权重
    
    # 元数据
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.aboutness_weights is None:
            self.aboutness_weights = {}
        if self.metadata is None:
            self.metadata = {}

class MultiLevelEmbedding:
    """多层次嵌入生成器"""
    
    def __init__(self, embedding_model, config: StructuredHopRAGConfig = DEFAULT_CONFIG):
        self.embedding_model = embedding_model
        self.config = config
        self.aboutness_weights = config.aboutness_weights
        
    def compute_multi_level_embeddings(
        self, 
        nodes: Dict[str, MultiLevelNode]
    ) -> Dict[str, MultiLevelNode]:
        """
        计算所有节点的多层次嵌入
        
        策略：
        1. 叶节点：直接嵌入
        2. 父节点：加权聚合子节点嵌入
        """
        print("🔮 开始计算多层次嵌入...")
        
        # Step 1: 识别叶节点和父节点
        leaf_nodes, parent_nodes = self._classify_nodes(nodes)
        
        # Step 2: 为叶节点生成直接嵌入
        self._compute_direct_embeddings(leaf_nodes)
        
        # Step 3: 自底向上聚合父节点嵌入
        self._compute_aggregated_embeddings(parent_nodes, nodes)
        
        # Step 4: 设置final_embedding
        for node in nodes.values():
            if node.aggregated_embedding is not None:
                node.final_embedding = node.aggregated_embedding
            else:
                node.final_embedding = node.direct_embedding
        
        print(f"✅ 多层次嵌入计算完成，共 {len(nodes)} 个节点")
        return nodes
    
    def _classify_nodes(
        self, 
        nodes: Dict[str, MultiLevelNode]
    ) -> Tuple[List[MultiLevelNode], List[MultiLevelNode]]:
        """分类叶节点和父节点"""
        leaf_nodes = []
        parent_nodes = []
        
        for node in nodes.values():
            if not node.children_ids or len(node.children_ids) == 0:
                leaf_nodes.append(node)
            else:
                parent_nodes.append(node)
        
        # 按层级排序父节点（从低到高，以便自底向上聚合）
        hierarchy = LegalLevel.get_hierarchy()
        parent_nodes.sort(
            key=lambda n: hierarchy.index(n.level) if n.level in hierarchy else 99,
            reverse=True  # 从低层到高层
        )
        
        print(f"  叶节点: {len(leaf_nodes)}, 父节点: {len(parent_nodes)}")
        return leaf_nodes, parent_nodes
    
    def _compute_direct_embeddings(self, leaf_nodes: List[MultiLevelNode]):
        """为叶节点计算直接嵌入"""
        print("  📝 计算叶节点直接嵌入...")
        
        # 批量编码
        contents = [node.content for node in leaf_nodes]
        
        if hasattr(self.embedding_model, 'encode'):
            embeddings = self.embedding_model.encode(contents)
        else:
            # 异步方法需要在外部处理
            embeddings = self.embedding_model.encode(contents)
        
        # 分配嵌入向量
        for i, node in enumerate(leaf_nodes):
            node.direct_embedding = embeddings[i]
            # 叶节点的aboutness默认为最大值
            node.aboutness_score = self.aboutness_weights.get_weight(node.level)
        
        print(f"  ✅ {len(leaf_nodes)} 个叶节点嵌入完成")
    
    def _compute_aggregated_embeddings(
        self, 
        parent_nodes: List[MultiLevelNode],
        all_nodes: Dict[str, MultiLevelNode]
    ):
        """自底向上聚合父节点嵌入"""
        print("  🔄 计算父节点聚合嵌入...")
        
        for parent in parent_nodes:
            # 获取所有子节点
            children = [
                all_nodes[child_id] 
                for child_id in parent.children_ids 
                if child_id in all_nodes
            ]
            
            if not children:
                # 没有子节点，使用直接嵌入
                if parent.content:
                    parent.direct_embedding = self.embedding_model.encode([parent.content])[0]
                    parent.aggregated_embedding = parent.direct_embedding
                continue
            
            # 计算aboutness权重
            aboutness_weights = self._calculate_aboutness_weights(
                parent, children, all_nodes
            )
            
            # 加权聚合
            aggregated = self._weighted_aggregation(
                children, aboutness_weights, all_nodes
            )
            
            parent.aggregated_embedding = aggregated
            parent.aboutness_weights = aboutness_weights
        
        print(f"  ✅ {len(parent_nodes)} 个父节点聚合完成")
    
    def _calculate_aboutness_weights(
        self,
        parent: MultiLevelNode,
        children: List[MultiLevelNode],
        all_nodes: Dict[str, MultiLevelNode]
    ) -> Dict[str, float]:
        """
        计算子节点的aboutness权重
        
        公式：w_i = aboutness_score(child_i) / Σ aboutness_score(all_children)
        
        aboutness_score通过cosine相似度隐式计算：
        - 如果parent有直接嵌入，使用cosine(child_embed, parent_embed)
        - 否则使用层级默认权重
        """
        weights = {}
        
        # 如果父节点有直接嵌入（例如有自己的文本内容）
        if parent.content:
            # 生成父节点的临时嵌入（用于计算相似度）
            parent_temp_embed = self.embedding_model.encode([parent.content])[0]
            
            # 计算每个子节点与父节点的相似度作为aboutness
            similarities = {}
            for child in children:
                child_embed = self._get_node_embedding(child, all_nodes)
                if child_embed is not None:
                    sim = self._cosine_similarity(child_embed, parent_temp_embed)
                    similarities[child.node_id] = max(sim, 0.0)  # 负值设为0
                else:
                    # 使用默认权重
                    similarities[child.node_id] = self.aboutness_weights.get_weight(child.level)
            
            # 归一化
            total = sum(similarities.values())
            if total > 0:
                weights = {k: v/total for k, v in similarities.items()}
            else:
                # 均等权重
                weights = {child.node_id: 1.0/len(children) for child in children}
        
        else:
            # 使用层级默认权重
            raw_weights = {
                child.node_id: self.aboutness_weights.get_weight(child.level)
                for child in children
            }
            total = sum(raw_weights.values())
            weights = {k: v/total for k, v in raw_weights.items()} if total > 0 else {}
        
        return weights
    
    def _weighted_aggregation(
        self,
        children: List[MultiLevelNode],
        weights: Dict[str, float],
        all_nodes: Dict[str, MultiLevelNode]
    ) -> np.ndarray:
        """
        加权聚合子节点嵌入
        
        公式：e_parent = Σ(w_i × e_child_i)
        """
        aggregated = None
        
        for child in children:
            child_embed = self._get_node_embedding(child, all_nodes)
            weight = weights.get(child.node_id, 0.0)
            
            if child_embed is not None and weight > 0:
                weighted_embed = child_embed * weight
                
                if aggregated is None:
                    aggregated = weighted_embed
                else:
                    aggregated += weighted_embed
        
        # 归一化（可选）
        if aggregated is not None:
            norm = np.linalg.norm(aggregated)
            if norm > 0:
                aggregated = aggregated / norm
        
        return aggregated
    
    def _get_node_embedding(
        self, 
        node: MultiLevelNode,
        all_nodes: Dict[str, MultiLevelNode]
    ) -> Optional[np.ndarray]:
        """获取节点的嵌入向量（优先使用已有的）"""
        if node.final_embedding is not None:
            return node.final_embedding
        elif node.aggregated_embedding is not None:
            return node.aggregated_embedding
        elif node.direct_embedding is not None:
            return node.direct_embedding
        else:
            return None
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def get_embedding_statistics(self, nodes: Dict[str, MultiLevelNode]) -> Dict[str, Any]:
        """获取嵌入统计信息"""
        stats = {
            "total_nodes": len(nodes),
            "direct_embeddings": sum(1 for n in nodes.values() if n.direct_embedding is not None),
            "aggregated_embeddings": sum(1 for n in nodes.values() if n.aggregated_embedding is not None),
            "final_embeddings": sum(1 for n in nodes.values() if n.final_embedding is not None),
        }
        
        # 按层级统计
        level_stats = {}
        for level in LegalLevel.get_hierarchy():
            level_nodes = [n for n in nodes.values() if n.level == level]
            level_stats[level] = {
                "count": len(level_nodes),
                "avg_aboutness": np.mean([n.aboutness_score for n in level_nodes]) if level_nodes else 0.0
            }
        
        stats["level_statistics"] = level_stats
        return stats
