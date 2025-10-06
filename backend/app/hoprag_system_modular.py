"""
HopRAG系統 - 模組化架構
按照原本設計的架構重新組織
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime

from .hoprag_config import HopRAGConfig, DEFAULT_CONFIG
from .hoprag_graph_builder import (
    PassageGraphBuilder, PseudoQueryGenerator, EdgeConnector, LegalNode
)
from .hoprag_hop_retriever import (
    HopRetriever, InitialRetriever, GraphTraverser, LLMReasoner, Algorithm1Traverser
)
from .hoprag_result_processor import (
    ResultProcessor, RelevanceFilter, ResultRanker
)

class HopRAGSystem:
    """HopRAG系統 - 模組化架構主類"""
    
    def __init__(self, llm_client, embedding_model, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        
        # 初始化模組
        self._initialize_modules()
        
        # 系統狀態
        self.is_graph_built = False
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, LegalNode] = {}
        self.edges: Dict[str, List[Dict[str, Any]]] = {}
        
    def _initialize_modules(self):
        """初始化所有模組"""
        print("🔧 初始化HopRAG模組...")
        
        # GraphBuilder模組
        self.pseudo_query_generator = PseudoQueryGenerator(self.llm_client, self.config)
        self.edge_connector = EdgeConnector(self.config)
        self.graph_builder = PassageGraphBuilder(self.config)
        
        # 設置GraphBuilder的組件
        self.graph_builder.set_components(
            self.pseudo_query_generator, 
            self.edge_connector, 
            self.embedding_model
        )
        
        # HopRetriever模組
        self.initial_retriever = InitialRetriever(self.config)
        self.graph_traverser = GraphTraverser(self.config)
        self.llm_reasoner = LLMReasoner(self.llm_client, self.config)
        self.algorithm1_traverser = Algorithm1Traverser(self.llm_client, self.embedding_model, self.config)
        self.hop_retriever = HopRetriever(self.config)
        
        # 設置HopRetriever的組件
        self.hop_retriever.set_llm_reasoner(self.llm_reasoner)
        self.hop_retriever.set_algorithm1_traverser(self.algorithm1_traverser)
        
        # ResultProcessor模組
        self.relevance_filter = RelevanceFilter(self.config)
        self.result_ranker = ResultRanker(self.config)
        self.result_processor = ResultProcessor(self.config)
        
        print("✅ HopRAG模組初始化完成")
    
    async def build_graph_from_multi_level_chunks(self, multi_level_chunks: Dict[str, Dict[str, List[Dict]]]):
        """從多層次chunks構建HopRAG圖"""
        print("🏗️ 開始構建HopRAG圖譜（模組化架構）...")
        
        try:
            # 使用GraphBuilder構建圖譜
            self.nodes, self.edges = await self.graph_builder.build_graph(multi_level_chunks)
            
            # 構建NetworkX圖
            self._build_networkx_graph()
            
            self.is_graph_built = True
            print("✅ HopRAG圖譜構建完成！")
            
        except Exception as e:
            print(f"❌ HopRAG圖譜構建失敗: {e}")
            raise
    
    def _build_networkx_graph(self):
        """構建NetworkX圖"""
        print("📊 構建NetworkX圖結構...")
        
        self.graph = nx.DiGraph()
        
        # 添加節點
        for node_id, node in self.nodes.items():
            node_attrs = {
                'node_type': node.node_type.value,
                'content': node.content,
                'contextualized_text': node.contextualized_text,
                'law_name': node.law_name,
                'article_number': node.article_number,
                'item_number': node.item_number,
                'parent_article_id': node.parent_article_id,
                'metadata': node.metadata,
                'incoming_questions': node.incoming_questions,
                'outgoing_questions': node.outgoing_questions
            }
            self.graph.add_node(node_id, **node_attrs)
        
        # 添加邊
        for from_node, edge_list in self.edges.items():
            for edge_data in edge_list:
                to_node = edge_data['to_node']
                edge_attrs = {
                    'pseudo_query': edge_data.get('pseudo_query', ''),
                    'similarity_score': edge_data.get('similarity_score', 0.0),
                    'edge_type': edge_data.get('edge_type', ''),
                    'outgoing_query_id': edge_data.get('outgoing_query_id', ''),
                    'incoming_query_id': edge_data.get('incoming_query_id', '')
                }
                self.graph.add_edge(from_node, to_node, **edge_attrs)
        
        print(f"✅ NetworkX圖構建完成：{self.graph.number_of_nodes()}個節點，{self.graph.number_of_edges()}條邊")
    
    async def enhanced_retrieve(self, query: str, base_results: List[Dict], k: int = 5) -> List[Dict[str, Any]]:
        """HopRAG增強檢索"""
        if not self.is_graph_built:
            print("⚠️ HopRAG圖譜未構建，返回基礎結果")
            return base_results[:k]
        
        print(f"🚀 開始HopRAG增強檢索，查詢: '{query}'")
        
        try:
            # Step 1: 使用HopRetriever進行多跳檢索
            hop_results = await self.hop_retriever.retrieve(
                query=query,
                base_results=base_results,
                graph=self.graph,
                nodes=self.nodes
            )
            
            # Step 2: 使用ResultProcessor處理結果
            enhanced_results = self.result_processor.process_results(
                base_results=base_results,
                hop_results=hop_results,
                nodes=self.nodes,
                query=query,
                k=k
            )
            
            print(f"✅ HopRAG增強檢索完成，返回 {len(enhanced_results)} 個結果")
            return enhanced_results
            
        except Exception as e:
            print(f"❌ HopRAG增強檢索失敗: {e}")
            # 返回基礎結果作為fallback
            return base_results[:k]
    
    async def enhanced_retrieve_with_algorithm1(self, query: str, k: int = 5, n_hop: int = 4) -> List[Dict[str, Any]]:
        """使用演算法1進行HopRAG增強檢索"""
        if not self.is_graph_built:
            print("⚠️ HopRAG圖譜未構建，無法使用演算法1")
            return []
        
        print(f"🚀 開始演算法1增強檢索，查詢: '{query}'")
        
        try:
            # 使用演算法1進行檢索
            algorithm1_results = await self.hop_retriever.retrieve_with_algorithm1(
                query=query,
                graph=self.graph,
                nodes=self.nodes,
                top_k=k,
                n_hop=n_hop
            )
            
            # 轉換為標準結果格式
            enhanced_results = []
            for node_id in algorithm1_results:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    result = {
                        'node_id': node_id,
                        'content': node.content,
                        'contextualized_text': node.contextualized_text,
                        'law_name': node.law_name,
                        'article_number': node.article_number,
                        'item_number': node.item_number,
                        'node_type': node.node_type.value,
                        'hop_level': 'algorithm1',
                        'hop_source': 'algorithm1_traversal',
                        'similarity_score': 1.0,  # 演算法1結果默認為最高分
                        'relevance_score': 1.0,
                        'rank': len(enhanced_results) + 1,
                        'metadata': node.metadata
                    }
                    enhanced_results.append(result)
            
            print(f"✅ 演算法1增強檢索完成，返回 {len(enhanced_results)} 個結果")
            return enhanced_results
            
        except Exception as e:
            print(f"❌ 演算法1增強檢索失敗: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """獲取圖統計信息"""
        if not self.is_graph_built:
            return {"error": "圖譜未構建"}
        
        # 統計節點類型
        article_count = sum(1 for node in self.nodes.values() if node.node_type.value == 'article')
        item_count = sum(1 for node in self.nodes.values() if node.node_type.value == 'item')
        
        # 統計邊類型
        edge_types = {}
        for edge_list in self.edges.values():
            for edge_data in edge_list:
                edge_type = edge_data.get('edge_type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # 統計偽查詢
        total_pseudo_queries = 0
        for node in self.nodes.values():
            total_pseudo_queries += len(node.incoming_questions) + len(node.outgoing_questions)
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "article_nodes": article_count,
            "item_nodes": item_count,
            "edge_type_distribution": edge_types,
            "total_pseudo_queries": total_pseudo_queries,
            "graph_built": self.is_graph_built,
            "config": self.config.to_dict(),
            "networkx_stats": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "is_connected": nx.is_weakly_connected(self.graph)
            }
        }
    
    def get_module_status(self) -> Dict[str, Any]:
        """獲取模組狀態"""
        return {
            "graph_builder": {
                "pseudo_query_generator": self.pseudo_query_generator is not None,
                "edge_connector": self.edge_connector is not None,
                "passage_graph_builder": self.graph_builder is not None
            },
            "hop_retriever": {
                "initial_retriever": self.initial_retriever is not None,
                "graph_traverser": self.graph_traverser is not None,
                "llm_reasoner": self.llm_reasoner is not None,
                "hop_retriever": self.hop_retriever is not None
            },
            "result_processor": {
                "relevance_filter": self.relevance_filter is not None,
                "result_ranker": self.result_ranker is not None,
                "result_processor": self.result_processor is not None
            },
            "system_status": {
                "graph_built": self.is_graph_built,
                "config_loaded": self.config is not None,
                "llm_client_available": self.llm_client is not None,
                "embedding_model_available": self.embedding_model is not None
            }
        }
    
    def update_config(self, new_config: HopRAGConfig):
        """更新配置"""
        print("🔧 更新HopRAG配置...")
        
        self.config = new_config
        
        # 重新初始化受影響的模組
        self._initialize_modules()
        
        print("✅ 配置更新完成")
    
    def export_graph_data(self) -> Dict[str, Any]:
        """導出圖數據"""
        if not self.is_graph_built:
            return {"error": "圖譜未構建"}
        
        return {
            "nodes": {
                node_id: {
                    "node_type": node.node_type.value,
                    "content": node.content,
                    "contextualized_text": node.contextualized_text,
                    "law_name": node.law_name,
                    "article_number": node.article_number,
                    "item_number": node.item_number,
                    "parent_article_id": node.parent_article_id,
                    "incoming_questions": node.incoming_questions,
                    "outgoing_questions": node.outgoing_questions,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": self.edges,
            "statistics": self.get_graph_statistics(),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_graph_data(self, graph_data: Dict[str, Any]):
        """導入圖數據"""
        print("📥 導入HopRAG圖數據...")
        
        try:
            # 重建節點
            self.nodes = {}
            for node_id, node_data in graph_data["nodes"].items():
                from .hoprag_graph_builder import LegalNode
                from .hoprag_config import NodeType
                
                node = LegalNode(
                    node_id=node_id,
                    node_type=NodeType(node_data["node_type"]),
                    content=node_data["content"],
                    contextualized_text=node_data["contextualized_text"],
                    law_name=node_data["law_name"],
                    article_number=node_data["article_number"],
                    item_number=node_data.get("item_number"),
                    parent_article_id=node_data.get("parent_article_id"),
                    incoming_questions=node_data.get("incoming_questions", []),
                    outgoing_questions=node_data.get("outgoing_questions", []),
                    metadata=node_data.get("metadata", {})
                )
                self.nodes[node_id] = node
            
            # 重建邊
            self.edges = graph_data["edges"]
            
            # 重建NetworkX圖
            self._build_networkx_graph()
            
            self.is_graph_built = True
            print("✅ 圖數據導入完成")
            
        except Exception as e:
            print(f"❌ 圖數據導入失敗: {e}")
            raise
    
    def reset_system(self):
        """重置系統"""
        print("🔄 重置HopRAG系統...")
        
        self.is_graph_built = False
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = {}
        
        # 重新初始化模組
        self._initialize_modules()
        
        print("✅ 系統重置完成")
