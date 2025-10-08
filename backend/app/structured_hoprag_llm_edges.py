"""
Structured-HopRAG 精简LLM边生成器
仅在规则边无法覆盖的复杂情况下使用LLM
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple

from .structured_hoprag_config import (
    StructuredHopRAGConfig,
    EdgeType,
    DEFAULT_CONFIG
)
from .structured_hoprag_embedding import MultiLevelNode

class LLMEdgeBuilder:
    """LLM边构建器（精简版）"""
    
    def __init__(self, llm_client, config: StructuredHopRAGConfig = DEFAULT_CONFIG):
        self.llm_client = llm_client
        self.config = config
        self.llm_call_count = 0
        
    async def build_llm_edges(
        self,
        nodes: Dict[str, MultiLevelNode],
        existing_edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        构建LLM边（精简版）
        
        策略：
        1. 只针对叶节点（basic_unit, basic_unit_component, enumeration）
        2. 检查每对节点是否已有规则边
        3. 如果无规则边且相似度<阈值（复杂情况），用LLM生成
        4. 每个节点最多1-2条LLM边
        """
        print("🤖 开始精简LLM边构建...")
        
        if not self.config.enable_llm_edges:
            print("  ⚠️ LLM边已禁用")
            return []
        
        # 筛选目标节点（仅叶节点）
        target_nodes = [
            n for n in nodes.values()
            if n.level in self.config.llm_edge_levels
        ]
        
        print(f"  📋 目标节点: {len(target_nodes)} 个")
        
        # 构建现有边的索引
        edge_index = self._build_edge_index(existing_edges)
        
        # 找出需要LLM处理的复杂节点对
        complex_pairs = self._find_complex_pairs(
            target_nodes, edge_index, nodes
        )
        
        print(f"  🔍 发现复杂节点对: {len(complex_pairs)} 对")
        
        # 为复杂节点对生成LLM边
        llm_edges = await self._generate_llm_edges_for_pairs(
            complex_pairs, nodes
        )
        
        print(f"  ✅ LLM边构建完成: {len(llm_edges)} 条")
        print(f"  📊 LLM调用次数: {self.llm_call_count}")
        
        return llm_edges
    
    def _build_edge_index(
        self,
        edges: List[Dict[str, Any]]
    ) -> Set[Tuple[str, str]]:
        """构建边索引（快速查找）"""
        index = set()
        
        for edge in edges:
            from_node = edge['from_node']
            to_node = edge['to_node']
            
            # 双向索引
            index.add((from_node, to_node))
            index.add((to_node, from_node))
        
        return index
    
    def _find_complex_pairs(
        self,
        nodes: List[MultiLevelNode],
        edge_index: Set[Tuple[str, str]],
        all_nodes: Dict[str, MultiLevelNode]
    ) -> List[Tuple[MultiLevelNode, MultiLevelNode]]:
        """
        找出需要LLM处理的复杂节点对
        
        判断标准：
        1. 两节点间无规则边
        2. embedding相似度 < 阈值（复杂情况）
        3. 但内容可能存在逻辑关联
        """
        complex_pairs = []
        
        for i, node_a in enumerate(nodes):
            # 限制每个节点的LLM边数量
            llm_edge_count = 0
            
            for node_b in nodes[i+1:]:
                # 1. 检查是否已有边
                if (node_a.node_id, node_b.node_id) in edge_index:
                    continue
                
                # 2. 计算相似度
                if node_a.final_embedding is None or node_b.final_embedding is None:
                    continue
                
                similarity = self._cosine_similarity(
                    node_a.final_embedding,
                    node_b.final_embedding
                )
                
                # 3. 复杂判断：相似度在中等范围（可能有隐含关联）
                if self.config.llm_edge_only_complex:
                    # 仅复杂情况：0.4 < sim < 0.75
                    if not (0.4 < similarity < self.config.llm_complexity_threshold):
                        continue
                else:
                    # 所有情况：sim < 0.75
                    if similarity >= self.config.llm_complexity_threshold:
                        continue
                
                # 4. 内容相关性初判（避免完全无关）
                if not self._has_potential_relevance(node_a, node_b):
                    continue
                
                complex_pairs.append((node_a, node_b))
                llm_edge_count += 1
                
                # 限制每个节点的LLM边数量
                if llm_edge_count >= self.config.llm_edge_max_per_node:
                    break
        
        return complex_pairs
    
    def _has_potential_relevance(
        self,
        node_a: MultiLevelNode,
        node_b: MultiLevelNode
    ) -> bool:
        """
        初判两节点是否有潜在关联
        
        简单方法：检查是否有共同的法律术语
        """
        common_terms = [
            '权利', '义务', '责任', '侵权', '赔偿', '罚则',
            '申请', '批准', '撤销', '无效', '处罚', '没收'
        ]
        
        content_a = node_a.content.lower()
        content_b = node_b.content.lower()
        
        # 至少有一个共同术语
        for term in common_terms:
            if term in content_a and term in content_b:
                return True
        
        return False
    
    async def _generate_llm_edges_for_pairs(
        self,
        pairs: List[Tuple[MultiLevelNode, MultiLevelNode]],
        all_nodes: Dict[str, MultiLevelNode]
    ) -> List[Dict[str, Any]]:
        """为复杂节点对生成LLM边"""
        llm_edges = []
        
        # 批量处理（控制并发）
        batch_size = 5
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            
            # 并发生成
            tasks = [
                self._generate_single_llm_edge(node_a, node_b)
                for node_a, node_b in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, dict) and result:
                    llm_edges.append(result)
        
        return llm_edges
    
    async def _generate_single_llm_edge(
        self,
        node_a: MultiLevelNode,
        node_b: MultiLevelNode
    ) -> Optional[Dict[str, Any]]:
        """
        为单对节点生成LLM边
        
        生成1个pseudo-query连接两个节点
        """
        prompt = self._build_llm_prompt(node_a, node_b)
        
        try:
            response = await self.llm_client.generate_async(prompt)
            self.llm_call_count += 1
            
            # 解析响应
            edge_data = self._parse_llm_response(response, node_a, node_b)
            
            return edge_data
            
        except Exception as e:
            print(f"  ⚠️ LLM边生成失败: {e}")
            return None
    
    def _build_llm_prompt(
        self,
        node_a: MultiLevelNode,
        node_b: MultiLevelNode
    ) -> str:
        """构建LLM prompt"""
        prompt = f"""你是一位法律专家。请分析以下两个法律条文是否存在逻辑关联。

条文A：
{node_a.content[:300]}

条文B：
{node_b.content[:300]}

任务：
1. 判断这两个条文是否有逻辑关联（如：因果关系、互补关系、例外关系、程序关联等）
2. 如果有关联，生成1个精炼的连接问题（pseudo-query），描述这种关联

要求：
- 如果无明显关联，返回 {{"relevant": false}}
- 如果有关联，返回 {{"relevant": true, "query": "连接问题", "relation_type": "关系类型"}}
- 问题应该精炼（15-30字）
- 关系类型如：因果、互补、例外、程序、引申等

请以JSON格式返回，不要包含其他文字。
"""
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        node_a: MultiLevelNode,
        node_b: MultiLevelNode
    ) -> Optional[Dict[str, Any]]:
        """解析LLM响应"""
        try:
            # 清理响应
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # 解析JSON
            data = json.loads(response)
            
            # 检查是否相关
            if not data.get('relevant', False):
                return None
            
            query = data.get('query', '')
            relation_type = data.get('relation_type', 'unknown')
            
            if not query:
                return None
            
            # 构建边
            edge = {
                'from_node': node_a.node_id,
                'to_node': node_b.node_id,
                'edge_type': EdgeType.LLM_GENERATED.value,
                'weight': 0.8,  # LLM边固定权重
                'directed': False,
                'metadata': {
                    'pseudo_query': query,
                    'relation_type': relation_type,
                    'generated_by': 'llm'
                }
            }
            
            return edge
            
        except Exception as e:
            print(f"  ⚠️ 解析LLM响应失败: {e}")
            return None
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def get_llm_statistics(self) -> Dict[str, Any]:
        """获取LLM使用统计"""
        return {
            'llm_calls': self.llm_call_count,
            'config': {
                'enabled': self.config.enable_llm_edges,
                'only_complex': self.config.llm_edge_only_complex,
                'max_per_node': self.config.llm_edge_max_per_node,
                'complexity_threshold': self.config.llm_complexity_threshold
            }
        }
