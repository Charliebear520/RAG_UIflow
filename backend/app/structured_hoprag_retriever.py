"""
Structured-HopRAG 去LLM化检索器
基于预计算权重 + 法律逻辑模板导航
"""

import time
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict

from .structured_hoprag_config import (
    StructuredHopRAGConfig,
    EdgeType,
    LegalLogicTemplate,
    DEFAULT_CONFIG
)
from .structured_hoprag_embedding import MultiLevelNode

class QueryCache:
    """查询路径缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, query: str) -> Optional[List[str]]:
        """获取缓存的查询结果"""
        if query in self.cache:
            entry = self.cache[query]
            # 检查是否过期
            if time.time() - entry['timestamp'] < self.ttl:
                self.hits += 1
                return entry['results']
            else:
                del self.cache[query]
        
        self.misses += 1
        return None
    
    def set(self, query: str, results: List[str]):
        """设置缓存"""
        # 如果缓存满了，删除最旧的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[query] = {
            'results': results,
            'timestamp': time.time()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

class TemplateNavigator:
    """法律逻辑模板导航器"""
    
    def __init__(self, config: StructuredHopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.templates = config.legal_templates
    
    def match_query_template(self, query: str) -> Optional[LegalLogicTemplate]:
        """匹配查询的逻辑模板"""
        query_lower = query.lower()
        
        # 遍历所有模板，计算匹配分数
        best_template = None
        best_score = 0
        
        for template_name, template in self.templates.items():
            score = 0
            
            # 关键词匹配
            for keyword in template.keywords:
                if keyword in query_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_template = template
        
        # 需要至少匹配2个关键词
        if best_score >= 2:
            return best_template
        
        return None
    
    def get_template_path_nodes(
        self,
        template: LegalLogicTemplate,
        nodes: Dict[str, MultiLevelNode],
        initial_nodes: List[str]
    ) -> List[str]:
        """
        根据模板获取逻辑路径节点
        
        策略：
        1. 从初始节点开始
        2. 按模板stages顺序查找相关章节
        3. 跳过无关节点
        """
        path_nodes = []
        
        # 按stages顺序查找
        for stage in template.stages:
            stage_nodes = self._find_nodes_by_stage(stage, nodes)
            path_nodes.extend(stage_nodes)
        
        return path_nodes
    
    def _find_nodes_by_stage(
        self,
        stage: str,
        nodes: Dict[str, MultiLevelNode]
    ) -> List[str]:
        """根据stage查找节点"""
        matching_nodes = []
        
        for node_id, node in nodes.items():
            # 检查节点内容是否包含stage关键词
            if stage in node.content or stage in node.metadata.get('chapter_name', ''):
                matching_nodes.append(node_id)
        
        return matching_nodes[:5]  # 限制数量

class StructuredHopRAGRetriever:
    """Structured-HopRAG检索器（去LLM化）"""
    
    def __init__(
        self,
        graph: nx.DiGraph,
        nodes: Dict[str, MultiLevelNode],
        embedding_model,
        config: StructuredHopRAGConfig = DEFAULT_CONFIG
    ):
        self.graph = graph
        self.nodes = nodes
        self.embedding_model = embedding_model
        self.config = config
        
        # 初始化组件
        self.cache = QueryCache(
            max_size=config.cache_max_size,
            ttl=config.cache_ttl
        ) if config.enable_query_cache else None
        
        self.template_navigator = TemplateNavigator(config) \
            if config.enable_template_navigation else None
    
    async def retrieve(
        self,
        query: str,
        initial_nodes: List[str],
        k: int = 5
    ) -> Dict[int, List[str]]:
        """
        执行检索（去LLM化版本）
        
        流程：
        1. 检查缓存
        2. 尝试模板导航
        3. 基于权重的图遍历（无LLM推理）
        4. 缓存结果
        """
        print(f"🔍 Structured-HopRAG检索: '{query}'")
        start_time = time.time()
        
        # 1. 检查缓存
        if self.cache:
            cached_results = self.cache.get(query)
            if cached_results:
                print(f"  ⚡ 缓存命中！耗时: {time.time() - start_time:.3f}秒")
                return {0: cached_results[:k]}
        
        # 2. 尝试模板导航
        if self.template_navigator:
            template = self.template_navigator.match_query_template(query)
            if template:
                print(f"  📋 使用模板: {template.name}")
                template_results = self.template_navigator.get_template_path_nodes(
                    template, self.nodes, initial_nodes
                )
                if template_results:
                    # 与初始节点合并
                    combined_results = list(set(initial_nodes + template_results))
                    
                    # 缓存结果
                    if self.cache:
                        self.cache.set(query, combined_results)
                    
                    print(f"  ✅ 模板导航完成，耗时: {time.time() - start_time:.3f}秒")
                    return {0: combined_results[:k]}
        
        # 3. 基于权重的图遍历
        hop_results = await self._weight_based_traverse(query, initial_nodes)
        
        # 4. 缓存结果
        all_results = []
        for hop_nodes in hop_results.values():
            all_results.extend(hop_nodes)
        all_results = list(set(all_results))  # 去重
        
        if self.cache:
            self.cache.set(query, all_results)
        
        print(f"  ✅ 检索完成，耗时: {time.time() - start_time:.3f}秒")
        return hop_results
    
    async def _weight_based_traverse(
        self,
        query: str,
        initial_nodes: List[str]
    ) -> Dict[int, List[str]]:
        """
        基于权重的图遍历（无LLM推理）
        
        策略：
        1. 获取查询embedding
        2. 每跳获取邻居并按综合分数排序
        3. 综合分数 = 边权重 × 边优先级 × 查询相似度
        """
        # 获取查询embedding
        query_embedding = self._get_query_embedding(query)
        
        hop_results = {0: initial_nodes}
        visited = set(initial_nodes)
        current_nodes = initial_nodes
        
        for hop in range(1, self.config.max_hops + 1):
            print(f"  第 {hop} 跳...")
            
            candidates = []
            
            # 获取所有候选邻居
            for node_id in current_nodes:
                if node_id not in self.graph:
                    continue
                
                # 获取邻居
                for neighbor_id in self.graph.neighbors(node_id):
                    if neighbor_id in visited:
                        continue
                    
                    # 计算综合分数
                    score = self._calculate_neighbor_score(
                        node_id,
                        neighbor_id,
                        query_embedding
                    )
                    
                    candidates.append((neighbor_id, score))
            
            if not candidates:
                print(f"  第 {hop} 跳无新节点")
                break
            
            # 按分数排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 取top-k
            next_nodes = [node_id for node_id, _ in candidates[:self.config.top_k_per_hop]]
            
            hop_results[hop] = next_nodes
            visited.update(next_nodes)
            current_nodes = next_nodes
            
            print(f"  第 {hop} 跳找到 {len(next_nodes)} 个节点")
        
        return hop_results
    
    def _calculate_neighbor_score(
        self,
        from_node_id: str,
        to_node_id: str,
        query_embedding: Optional[np.ndarray]
    ) -> float:
        """
        计算邻居节点的综合分数
        
        综合分数 = (边权重 × 边优先级) × w1 + 查询相似度 × w2
        """
        # 1. 获取边数据
        edge_data = self.graph[from_node_id][to_node_id]
        edge_weight = edge_data.get('weight', 0.5)
        edge_type = edge_data.get('edge_type', EdgeType.LLM_GENERATED.value)
        
        # 2. 边优先级
        edge_priority = self.config.edge_priority.get_priority(edge_type)
        
        # 3. 边分数
        edge_score = edge_weight * edge_priority
        
        # 4. 查询相似度
        query_sim = 0.0
        if query_embedding is not None and to_node_id in self.nodes:
            to_node = self.nodes[to_node_id]
            if to_node.final_embedding is not None:
                query_sim = self._cosine_similarity(
                    query_embedding,
                    to_node.final_embedding
                )
        
        # 5. 综合分数
        if self.config.traversal_strategy == "priority_weighted":
            score = (edge_score * self.config.edge_weight_in_traversal + 
                    query_sim * self.config.query_similarity_weight)
        else:  # similarity_only
            score = query_sim
        
        return score
    
    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """获取查询embedding"""
        try:
            if hasattr(self.embedding_model, 'encode'):
                return self.embedding_model.encode([query])[0]
            else:
                return None
        except Exception as e:
            print(f"  ⚠️ 查询embedding失败: {e}")
            return None
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计"""
        stats = {
            'config': {
                'max_hops': self.config.max_hops,
                'top_k_per_hop': self.config.top_k_per_hop,
                'enable_template_navigation': self.config.enable_template_navigation,
                'enable_query_cache': self.config.enable_query_cache,
                'llm_reasoning_enabled': self.config.enable_llm_reasoning
            }
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats
