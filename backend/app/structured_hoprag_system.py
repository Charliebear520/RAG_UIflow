"""
Structured-HopRAG 主系统
针对结构化法律文本优化的HopRAG系统
集成：多层次嵌入 + 规则边 + 精简LLM边 + 去LLM化检索
"""

import asyncio
import time
import networkx as nx
from typing import Dict, List, Any, Optional
from datetime import datetime

from .structured_hoprag_config import (
    StructuredHopRAGConfig,
    LegalLevel,
    DEFAULT_CONFIG
)
from .structured_hoprag_embedding import MultiLevelNode, MultiLevelEmbedding
from .structured_hoprag_rule_edges import RuleEdgeBuilder
from .structured_hoprag_llm_edges import LLMEdgeBuilder
from .structured_hoprag_retriever import StructuredHopRAGRetriever

class StructuredHopRAGSystem:
    """Structured-HopRAG系统主类"""
    
    def __init__(
        self,
        llm_client,
        embedding_model,
        config: StructuredHopRAGConfig = DEFAULT_CONFIG
    ):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.config = config
        
        # 系统状态
        self.is_graph_built = False
        self.nodes: Dict[str, MultiLevelNode] = {}
        self.edges: List[Dict[str, Any]] = []
        self.graph: nx.DiGraph = nx.DiGraph()
        
        # 初始化模块
        self.multi_level_embedding = MultiLevelEmbedding(embedding_model, config)
        self.rule_edge_builder = RuleEdgeBuilder(config)
        self.llm_edge_builder = LLMEdgeBuilder(llm_client, config)
        self.retriever: Optional[StructuredHopRAGRetriever] = None
        
        # 统计信息
        self.build_stats = {}
        
    async def build_graph_from_multi_level_chunks(
        self,
        multi_level_chunks: Dict[str, Dict[str, List[Dict]]]
    ):
        """
        从多层次chunks构建Structured-HopRAG图
        
        完整流程：
        1. 创建节点（7层层级）
        2. 计算多层次嵌入
        3. 构建规则边
        4. 构建LLM边（精简）
        5. 构建NetworkX图
        """
        print("=" * 80)
        print("🏗️ 开始构建Structured-HopRAG知识图谱")
        print("=" * 80)
        
        overall_start = time.time()
        
        try:
            # Step 1: 创建节点
            print("\n【步骤1/5】创建多层次节点...")
            step_start = time.time()
            self.nodes = self._create_multi_level_nodes(multi_level_chunks)
            step_time = time.time() - step_start
            print(f"✅ 节点创建完成，共 {len(self.nodes)} 个节点，耗时: {step_time:.2f}秒")
            
            # Step 2: 计算多层次嵌入
            print("\n【步骤2/5】计算多层次嵌入...")
            step_start = time.time()
            self.nodes = self.multi_level_embedding.compute_multi_level_embeddings(self.nodes)
            embedding_stats = self.multi_level_embedding.get_embedding_statistics(self.nodes)
            step_time = time.time() - step_start
            print(f"✅ 嵌入计算完成，耗时: {step_time:.2f}秒")
            
            # Step 3: 构建规则边
            print("\n【步骤3/5】构建规则边...")
            step_start = time.time()
            rule_edges = self.rule_edge_builder.build_all_rule_edges(self.nodes)
            rule_stats = self.rule_edge_builder.get_edge_statistics()
            step_time = time.time() - step_start
            print(f"✅ 规则边构建完成，耗时: {step_time:.2f}秒")
            
            # Step 4: 构建LLM边（精简）
            print("\n【步骤4/5】构建LLM边（精简版）...")
            step_start = time.time()
            llm_edges = await self.llm_edge_builder.build_llm_edges(self.nodes, rule_edges)
            llm_stats = self.llm_edge_builder.get_llm_statistics()
            step_time = time.time() - step_start
            print(f"✅ LLM边构建完成，耗时: {step_time:.2f}秒")
            
            # Step 5: 合并边并构建图
            print("\n【步骤5/5】构建NetworkX图...")
            step_start = time.time()
            self.edges = rule_edges + llm_edges
            self._build_networkx_graph()
            step_time = time.time() - step_start
            print(f"✅ 图构建完成，耗时: {step_time:.2f}秒")
            
            # 初始化检索器
            self.retriever = StructuredHopRAGRetriever(
                self.graph,
                self.nodes,
                self.embedding_model,
                self.config
            )
            
            self.is_graph_built = True
            overall_time = time.time() - overall_start
            
            # 保存统计信息
            self.build_stats = {
                'total_time': overall_time,
                'nodes': len(self.nodes),
                'edges': len(self.edges),
                'embedding_stats': embedding_stats,
                'rule_edge_stats': rule_stats,
                'llm_edge_stats': llm_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n" + "=" * 80)
            print(f"🎉 Structured-HopRAG图谱构建成功！")
            print(f"📊 总耗时: {overall_time:.2f}秒 ({overall_time/60:.2f}分钟)")
            print(f"📈 节点: {len(self.nodes)}, 边: {len(self.edges)}")
            print(f"🤖 LLM调用: {llm_stats['llm_calls']} 次")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ 图谱构建失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_multi_level_nodes(
        self,
        multi_level_chunks: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[str, MultiLevelNode]:
        """创建多层次节点（7层完整层级）"""
        nodes = {}
        hierarchy = LegalLevel.get_hierarchy()
        
        for doc_id, levels in multi_level_chunks.items():
            # 按层级顺序创建节点
            for level in hierarchy:
                if level not in levels:
                    continue
                
                for idx, chunk in enumerate(levels[level]):
                    node_id = chunk.get('metadata', {}).get('id', 
                                                           f"{doc_id}_{level}_{idx}")
                    
                    node = MultiLevelNode(
                        node_id=node_id,
                        level=level,
                        content=chunk.get('content', ''),
                        metadata=chunk.get('metadata', {})
                    )
                    
                    # 设置父子关系
                    parent_id = chunk.get('metadata', {}).get('parent_id')
                    if parent_id:
                        node.parent_id = parent_id
                        # 更新父节点的children_ids
                        if parent_id in nodes:
                            nodes[parent_id].children_ids.append(node_id)
                    
                    nodes[node_id] = node
        
        return nodes
    
    def _build_networkx_graph(self):
        """构建NetworkX图"""
        self.graph = nx.DiGraph()
        
        # 添加节点
        for node_id, node in self.nodes.items():
            self.graph.add_node(
                node_id,
                level=node.level,
                content=node.content,
                embedding=node.final_embedding,
                metadata=node.metadata
            )
        
        # 添加边
        for edge in self.edges:
            from_node = edge['from_node']
            to_node = edge['to_node']
            
            # 过滤低权重边
            if edge['weight'] < self.config.min_edge_weight:
                continue
            
            self.graph.add_edge(
                from_node,
                to_node,
                edge_type=edge['edge_type'],
                weight=edge['weight'],
                directed=edge.get('directed', True),
                metadata=edge.get('metadata', {})
            )
        
        # 应用边数量限制
        self._apply_edge_limit()
    
    def _apply_edge_limit(self):
        """应用每个节点的最大边数限制"""
        for node_id in list(self.graph.nodes()):
            # 获取所有出边
            out_edges = list(self.graph.out_edges(node_id, data=True))
            
            # 如果超过限制，保留权重最高的边
            if len(out_edges) > self.config.max_edges_per_node:
                # 按权重排序
                out_edges.sort(key=lambda e: e[2].get('weight', 0), reverse=True)
                
                # 删除多余的边
                for _, to_node, _ in out_edges[self.config.max_edges_per_node:]:
                    self.graph.remove_edge(node_id, to_node)
    
    async def enhanced_retrieve(
        self,
        query: str,
        base_results: List[Dict],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Structured-HopRAG增强检索
        
        特点：
        - 无LLM推理（快速）
        - 模板导航（精准）
        - 查询缓存（高效）
        """
        if not self.is_graph_built:
            print("⚠️ 图谱未构建，返回基础结果")
            return base_results[:k]
        
        if not self.retriever:
            print("⚠️ 检索器未初始化")
            return base_results[:k]
        
        try:
            # 提取初始节点
            initial_nodes = []
            for result in base_results:
                node_id = result.get('node_id') or result.get('id')
                if node_id and node_id in self.nodes:
                    initial_nodes.append(node_id)
            
            if not initial_nodes:
                print("⚠️ 无有效初始节点")
                return base_results[:k]
            
            # 执行检索
            hop_results = await self.retriever.retrieve(
                query=query,
                initial_nodes=initial_nodes,
                k=k
            )
            
            # 合并所有跳的结果
            all_node_ids = []
            for hop, node_ids in hop_results.items():
                all_node_ids.extend(node_ids)
            
            # 去重
            all_node_ids = list(dict.fromkeys(all_node_ids))
            
            # 转换为结果格式
            enhanced_results = []
            for node_id in all_node_ids[:k]:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    result = {
                        'node_id': node_id,
                        'content': node.content,
                        'level': node.level,
                        'metadata': node.metadata,
                        'retrieval_method': 'structured_hoprag'
                    }
                    enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return base_results[:k]
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        if not self.is_graph_built:
            return {"error": "图谱未构建"}
        
        stats = {
            'graph_built': self.is_graph_built,
            'build_stats': self.build_stats,
            'graph_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph)
            },
            'config': self.config.to_dict()
        }
        
        if self.retriever:
            stats['retrieval_stats'] = self.retriever.get_retrieval_stats()
        
        return stats
    
    def export_graph_data(self) -> Dict[str, Any]:
        """导出图数据"""
        if not self.is_graph_built:
            return {"error": "图谱未构建"}
        
        return {
            'nodes': {
                node_id: {
                    'level': node.level,
                    'content': node.content,
                    'metadata': node.metadata,
                    'aboutness_score': node.aboutness_score
                }
                for node_id, node in self.nodes.items()
            },
            'edges': self.edges,
            'statistics': self.get_system_statistics()
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息（API调用）"""
        if not self.is_graph_built:
            return {}
        
        # 统计各层级节点数
        level_counts = {}
        for node in self.nodes.values():
            level_counts[node.level] = level_counts.get(node.level, 0) + 1
        
        # 统计边类型
        edge_type_counts = {}
        for edge in self.edges:
            edge_type = edge.get('type', 'unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'level_distribution': level_counts,
            'edge_type_distribution': edge_type_counts,
            'graph_built': self.is_graph_built,
            'networkx_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
            }
        }
    
    def reset(self):
        """重置系统"""
        self.is_graph_built = False
        self.nodes = {}
        self.edges = []
        self.graph = nx.DiGraph()
        self.build_stats = {}
        self.retriever = None
        print("✅ Structured-HopRAG系统已重置")
