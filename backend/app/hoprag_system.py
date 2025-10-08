"""
HopRAG系統 - 後處理增強模式
基於論文HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation
"""

import json
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
from datetime import datetime

@dataclass
class PseudoQuery:
    """偽查詢數據結構"""
    query_id: str
    content: str
    query_type: str  # "incoming" 或 "outgoing"
    embedding: Optional[np.ndarray] = None
    similarity_threshold: float = 0.7

@dataclass
class LegalNode:
    """法律節點數據結構"""
    node_id: str
    node_type: str  # "basic_unit" 或 "basic_unit_component" (原 "article" 或 "item")
    content: str
    contextualized_text: str
    law_name: str
    article_number: str
    item_number: Optional[str] = None
    parent_article_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    # 偽查詢
    incoming_questions: List[str] = None
    outgoing_questions: List[str] = None
    pseudo_queries: Dict[str, List[PseudoQuery]] = None
    
    # 圖結構相關
    outgoing_edges: List[str] = None
    incoming_edges: List[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.incoming_questions is None:
            self.incoming_questions = []
        if self.outgoing_questions is None:
            self.outgoing_questions = []
        if self.pseudo_queries is None:
            self.pseudo_queries = {"incoming": [], "outgoing": []}
        if self.outgoing_edges is None:
            self.outgoing_edges = []
        if self.incoming_edges is None:
            self.incoming_edges = []

class PseudoQueryGenerator:
    """偽查詢生成器"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def generate_pseudo_queries_for_node(self, node: LegalNode) -> LegalNode:
        """為單個節點生成內向和外向問題"""
        print(f"🔍 為節點 {node.node_id} 生成偽查詢...")
        
        # 生成內向問題
        incoming_questions = await self._generate_incoming_questions(node)
        
        # 生成外向問題
        outgoing_questions = await self._generate_outgoing_questions(node)
        
        # 更新節點
        node.incoming_questions = incoming_questions
        node.outgoing_questions = outgoing_questions
        
        # 創建PseudoQuery對象
        node.pseudo_queries = {
            "incoming": [
                PseudoQuery(
                    query_id=f"{node.node_id}_in_{i}",
                    content=question,
                    query_type="incoming"
                ) for i, question in enumerate(incoming_questions)
            ],
            "outgoing": [
                PseudoQuery(
                    query_id=f"{node.node_id}_out_{i}",
                    content=question,
                    query_type="outgoing"
                ) for i, question in enumerate(outgoing_questions)
            ]
        }
        
        print(f"✅ 節點 {node.node_id} 偽查詢生成完成：{len(incoming_questions)}個內向，{len(outgoing_questions)}個外向")
        return node
    
    async def _generate_incoming_questions(self, node: LegalNode) -> List[str]:
        """生成內向問題 - 可以直接從該文本中找到答案的問題"""
        
        prompt = f"""
您是一位法律專家，需要為以下法律條文生成「內向問題」。

內向問題定義：可以直接從這段文本中找到完整答案的問題。

法律條文內容：
{node.contextualized_text}

請生成3-5個內向問題，要求：
1. 問題應該直接從上述條文中找到答案
2. 問題應該涵蓋條文的主要法律概念和規定
3. 問題應該清晰、具體，便於理解
4. 問題應該用繁體中文表達
5. 問題應該具有法律實務意義

請以JSON格式返回，格式如下：
{{
    "incoming_questions": [
        "問題1",
        "問題2", 
        "問題3",
        "問題4",
        "問題5"
    ]
}}

請確保JSON格式正確，不要包含任何其他文字。
"""
        
        try:
            response = await self.llm_client.generate_async(prompt)
            
            # 解析JSON響應
            if response.strip().startswith('{'):
                result = json.loads(response.strip())
                questions = result.get('incoming_questions', [])
                
                # 驗證問題質量
                validated_questions = self._validate_questions(questions, "incoming")
                return validated_questions[:5]  # 限制最多5個問題
            else:
                # 如果響應不是JSON格式，嘗試提取問題
                questions = self._extract_questions_from_text(response)
                return questions[:5]
                
        except Exception as e:
            print(f"❌ 生成內向問題失敗: {e}")
            # 返回默認問題
            return self._generate_default_incoming_questions(node)
    
    async def _generate_outgoing_questions(self, node: LegalNode) -> List[str]:
        """生成外向問題 - 由該文本引發但需要參考其他法條才能完整回答的問題"""
        
        prompt = f"""
您是一位法律專家，需要為以下法律條文生成「外向問題」。

外向問題定義：由這段文本引發，但需要參考其他法條才能完整回答的問題。

法律條文內容：
{node.contextualized_text}

請生成3-5個外向問題，要求：
1. 問題應該由上述條文引發，但答案需要參考其他相關法條
2. 問題應該具有邏輯關聯性，能夠引導到其他相關法律條文
3. 問題應該涵蓋條文的延伸法律概念和實務應用
4. 問題應該用繁體中文表達
5. 問題應該具有法律實務意義

請以JSON格式返回，格式如下：
{{
    "outgoing_questions": [
        "問題1",
        "問題2",
        "問題3", 
        "問題4",
        "問題5"
    ]
}}

請確保JSON格式正確，不要包含任何其他文字。
"""
        
        try:
            response = await self.llm_client.generate_async(prompt)
            
            # 解析JSON響應
            if response.strip().startswith('{'):
                result = json.loads(response.strip())
                questions = result.get('outgoing_questions', [])
                
                # 驗證問題質量
                validated_questions = self._validate_questions(questions, "outgoing")
                return validated_questions[:5]  # 限制最多5個問題
            else:
                # 如果響應不是JSON格式，嘗試提取問題
                questions = self._extract_questions_from_text(response)
                return questions[:5]
                
        except Exception as e:
            print(f"❌ 生成外向問題失敗: {e}")
            # 返回默認問題
            return self._generate_default_outgoing_questions(node)
    
    def _validate_questions(self, questions: List[str], question_type: str) -> List[str]:
        """驗證問題質量"""
        validated = []
        
        for question in questions:
            if isinstance(question, str) and len(question.strip()) > 10:
                # 基本驗證：問題長度、包含問號等
                question = question.strip()
                if question.endswith('？') or question.endswith('?'):
                    validated.append(question)
                else:
                    # 如果沒有問號，添加一個
                    validated.append(question + '？')
        
        return validated
    
    def _extract_questions_from_text(self, text: str) -> List[str]:
        """從文本中提取問題"""
        questions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ('？' in line or '?' in line) and len(line) > 10:
                # 清理行內容
                line = line.replace('- ', '').replace('* ', '').replace('1. ', '').replace('2. ', '').replace('3. ', '')
                questions.append(line)
        
        return questions[:5]
    
    def _generate_default_incoming_questions(self, node: LegalNode) -> List[str]:
        """生成默認內向問題"""
        article_num = node.article_number
        
        return [
            f"{article_num}的主要內容是什麼？",
            f"{article_num}規定了哪些法律要件？",
            f"根據{article_num}，相關的定義是什麼？",
            f"{article_num}的適用範圍是什麼？",
            f"{article_num}規定了什麼法律後果？"
        ]
    
    def _generate_default_outgoing_questions(self, node: LegalNode) -> List[str]:
        """生成默認外向問題"""
        article_num = node.article_number
        
        return [
            f"違反{article_num}會有什麼法律後果？",
            f"如何申請{article_num}規定的權利？",
            f"{article_num}與其他法條有什麼關聯？",
            f"在什麼情況下適用{article_num}？",
            f"{article_num}的實務操作程序是什麼？"
        ]

class HopRAGGraphDatabase:
    """HopRAG圖數據庫 - 基於NetworkX"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, LegalNode] = {}
        self.embedding_model = None
        self.similarity_threshold = 0.7
        self.max_edges_per_node = 10
        
    def add_node(self, node: LegalNode):
        """添加節點到圖數據庫"""
        self.nodes[node.node_id] = node
        
        # 添加到NetworkX圖
        node_attrs = {
            'node_type': node.node_type,
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
        
        self.graph.add_node(node.node_id, **node_attrs)
        
    def set_embedding_model(self, embedding_model):
        """設置embedding模型"""
        self.embedding_model = embedding_model
        
    async def build_graph_edges(self):
        """構建圖邊 - 邊匹配算法"""
        print("🔗 開始構建HopRAG圖邊...")
        
        # Step 1: 為所有偽查詢生成embedding
        await self._generate_pseudo_query_embeddings()
        
        # Step 2: 執行邊匹配算法
        await self._perform_edge_matching()
        
        # Step 3: 統計和驗證
        self._validate_graph_structure()
        
        print(f"✅ 圖邊構建完成！節點數: {self.graph.number_of_nodes()}, 邊數: {self.graph.number_of_edges()}")
    
    async def _generate_pseudo_query_embeddings(self):
        """為所有偽查詢生成embedding"""
        print("📊 生成偽查詢embedding向量...")
        
        all_queries = []
        query_mapping = {}
        
        # 收集所有偽查詢
        for node_id, node in self.nodes.items():
            for pseudo_query in node.pseudo_queries.get("outgoing", []):
                all_queries.append(pseudo_query.content)
                query_mapping[pseudo_query.content] = (node_id, "outgoing", pseudo_query)
                
            for pseudo_query in node.pseudo_queries.get("incoming", []):
                all_queries.append(pseudo_query.content)
                query_mapping[pseudo_query.content] = (node_id, "incoming", pseudo_query)
        
        if not all_queries:
            print("⚠️ 沒有偽查詢需要生成embedding")
            return
        
        # 批量生成embedding
        try:
            if hasattr(self.embedding_model, 'encode'):
                embeddings = self.embedding_model.encode(all_queries)
            else:
                # 如果是異步方法
                embeddings = await self.embedding_model.encode_async(all_queries)
            
            # 將embedding分配回偽查詢
            for i, query_content in enumerate(all_queries):
                node_id, query_type, pseudo_query = query_mapping[query_content]
                pseudo_query.embedding = embeddings[i]
                
        except Exception as e:
            print(f"❌ 生成偽查詢embedding失敗: {e}")
    
    async def _perform_edge_matching(self):
        """執行邊匹配算法"""
        print("🔗 執行邊匹配算法...")
        
        node_ids = list(self.nodes.keys())
        edge_count = 0
        
        for i, node_a in enumerate(node_ids):
            outgoing_queries_a = self.nodes[node_a].pseudo_queries.get("outgoing", [])
            
            for j, node_b in enumerate(node_ids):
                if i == j:
                    continue
                    
                incoming_queries_b = self.nodes[node_b].pseudo_queries.get("incoming", [])
                
                # 計算最佳相似度
                best_similarity = 0
                best_outgoing_query = None
                best_incoming_query = None
                
                for out_query in outgoing_queries_a:
                    for in_query in incoming_queries_b:
                        if out_query.embedding is not None and in_query.embedding is not None:
                            similarity = self._calculate_similarity(out_query, in_query)
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_outgoing_query = out_query
                                best_incoming_query = in_query
                
                # 如果相似度超過閾值，建立邊
                if best_similarity >= self.similarity_threshold:
                    edge_attr = {
                        'pseudo_query': f"{best_outgoing_query.content} -> {best_incoming_query.content}",
                        'similarity_score': best_similarity,
                        'outgoing_query_id': best_outgoing_query.query_id,
                        'incoming_query_id': best_incoming_query.query_id,
                        'edge_type': self._determine_edge_type(node_a, node_b)
                    }
                    
                    self.graph.add_edge(node_a, node_b, **edge_attr)
                    edge_count += 1
                    
                    # 限制每個節點的最大出邊數量
                    out_degree = self.graph.out_degree(node_a)
                    if out_degree >= self.max_edges_per_node:
                        break
            
            if i % 10 == 0:
                print(f"  處理進度: {i+1}/{len(node_ids)} 節點")
        
        print(f"  邊匹配完成，共建立 {edge_count} 條邊")
    
    def _calculate_similarity(self, query_a: PseudoQuery, query_b: PseudoQuery) -> float:
        """計算兩個偽查詢的相似度"""
        if query_a.embedding is None or query_b.embedding is None:
            return 0.0
        
        # 餘弦相似度
        similarity = np.dot(query_a.embedding, query_b.embedding) / (
            np.linalg.norm(query_a.embedding) * np.linalg.norm(query_b.embedding)
        )
        
        return float(similarity)
    
    def _determine_edge_type(self, from_node: str, to_node: str) -> str:
        """確定邊的類型 - 使用新的層級命名"""
        from_type = self.nodes[from_node].node_type
        to_type = self.nodes[to_node].node_type
        
        if from_type == 'basic_unit' and to_type == 'basic_unit':
            return 'basic_unit_to_basic_unit'
        elif from_type == 'basic_unit' and to_type == 'basic_unit_component':
            return 'basic_unit_to_component'
        elif from_type == 'basic_unit_component' and to_type == 'basic_unit':
            return 'component_to_basic_unit'
        else:
            return 'component_to_component'
    
    def _validate_graph_structure(self):
        """驗證圖結構"""
        print("📊 圖結構統計:")
        print(f"  節點總數: {self.graph.number_of_nodes()}")
        print(f"  邊總數: {self.graph.number_of_edges()}")
        
        # 統計節點類型
        basic_unit_count = sum(1 for node in self.nodes.values() if node.node_type == 'basic_unit')
        component_count = sum(1 for node in self.nodes.values() if node.node_type == 'basic_unit_component')
        print(f"  基本單元節點 (basic_unit): {basic_unit_count}")
        print(f"  基本單元組件節點 (basic_unit_component): {component_count}")
        
        # 統計邊類型
        edge_types = {}
        for from_node, to_node, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print("  邊類型分布:")
        for edge_type, count in edge_types.items():
            print(f"    {edge_type}: {count}")
    
    def get_neighbors_with_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """獲取節點的鄰居及其邊信息"""
        neighbors = []
        
        if node_id in self.graph:
            for neighbor_id in self.graph.successors(node_id):
                edge_data = self.graph[node_id][neighbor_id]
                neighbor_data = self.graph.nodes[neighbor_id]
                
                neighbors.append({
                    'node_id': neighbor_id,
                    'node_type': neighbor_data['node_type'],
                    'content': neighbor_data['content'],
                    'contextualized_text': neighbor_data['contextualized_text'],
                    'pseudo_query': edge_data.get('pseudo_query', ''),
                    'similarity_score': edge_data.get('similarity_score', 0.0),
                    'edge_type': edge_data.get('edge_type', '')
                })
        
        return neighbors
    
    def get_node_count(self) -> int:
        """獲取節點數量"""
        return self.graph.number_of_nodes()
    
    def get_edge_count(self) -> int:
        """獲取邊數量"""
        return self.graph.number_of_edges()
    
    def is_graph_built(self) -> bool:
        """檢查圖是否已構建"""
        return self.graph.number_of_nodes() > 0 and self.graph.number_of_edges() > 0

class HopRAGTraversalEngine:
    """HopRAG圖遍歷檢索引擎"""
    
    def __init__(self, graph_db: HopRAGGraphDatabase, llm_client):
        self.graph_db = graph_db
        self.llm_client = llm_client
        self.max_hops = 4
        self.top_k_per_hop = 20
        
    async def traverse_with_reasoning(self, query: str, initial_nodes: List[str], max_hops: int = None) -> Dict[int, List[str]]:
        """基於推理的圖遍歷"""
        if max_hops is None:
            max_hops = self.max_hops
            
        print(f"🔍 開始HopRAG檢索，查詢: '{query}'")
        
        hop_results = {0: initial_nodes}
        current_nodes = initial_nodes
        
        for hop in range(1, max_hops + 1):
            print(f"  第 {hop} 跳檢索...")
            next_nodes = []
            
            for node_id in current_nodes:
                # 獲取鄰居節點
                neighbors = self.graph_db.get_neighbors_with_edges(node_id)
                
                # 使用LLM推理判斷相關性
                relevant_neighbors = await self._filter_by_llm_reasoning(
                    query, node_id, neighbors
                )
                
                next_nodes.extend(relevant_neighbors)
            
            # 去重和限制數量
            next_nodes = list(set(next_nodes))[:self.top_k_per_hop]
            hop_results[hop] = next_nodes
            current_nodes = next_nodes
            
            if not current_nodes:
                break
        
        return hop_results
    
    async def _filter_by_llm_reasoning(self, query: str, current_node: str, neighbors: List[Dict[str, Any]]) -> List[str]:
        """使用LLM推理過濾鄰居節點"""
        relevant_neighbors = []
        
        for neighbor in neighbors:
            # 構建推理提示
            reasoning_prompt = self._build_reasoning_prompt(
                query, current_node, neighbor
            )
            
            try:
                # 調用LLM進行推理
                decision = await self.llm_client.generate_async(reasoning_prompt)
                
                # 解析決策
                if self._is_relevant_decision(decision):
                    relevant_neighbors.append(neighbor['node_id'])
                    
            except Exception as e:
                print(f"❌ LLM推理失敗: {e}")
                # 如果LLM失敗，基於相似度分數判斷
                if neighbor.get('similarity_score', 0) > 0.8:
                    relevant_neighbors.append(neighbor['node_id'])
        
        return relevant_neighbors
    
    def _build_reasoning_prompt(self, query: str, current_node: str, neighbor: Dict[str, Any]) -> str:
        """構建LLM推理提示"""
        current_content = self.graph_db.nodes[current_node].contextualized_text
        neighbor_content = neighbor['contextualized_text']
        pseudo_query = neighbor['pseudo_query']
        
        prompt = f"""
您是一個法律問答機器人。我需要評估一個輔助問題是否與主問題相關。

主問題: {query}

當前節點內容: {current_content}

輔助問題: {pseudo_query}

目標節點內容: {neighbor_content}

請評估輔助問題與主問題的相關性，只能返回以下三種結果之一：
1. "Completely Irrelevant" - 完全無關
2. "Indirectly relevant" - 間接相關  
3. "Relevant and Necessary" - 相關且必要

決策:"""
        
        return prompt
    
    def _is_relevant_decision(self, decision: str) -> bool:
        """判斷LLM決策是否表示相關"""
        decision_lower = decision.lower().strip()
        return ("relevant and necessary" in decision_lower or 
                "indirectly relevant" in decision_lower)

class HopRAGSystem:
    """完整的HopRAG系統"""
    
    def __init__(self, llm_client, embedding_model):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        
        # 初始化組件
        self.pseudo_query_generator = PseudoQueryGenerator(llm_client)
        self.graph_db = HopRAGGraphDatabase()
        self.graph_db.set_embedding_model(embedding_model)
        self.traversal_engine = HopRAGTraversalEngine(self.graph_db, llm_client)
        
        # 狀態
        self.is_graph_built = False
        
    async def build_graph_from_multi_level_chunks(self, multi_level_chunks: Dict[str, Dict[str, List[Dict]]]):
        """從多層次chunks構建HopRAG圖"""
        print("🏗️ 開始構建HopRAG圖譜...")
        
        # Step 1: 創建節點
        await self._create_nodes_from_chunks(multi_level_chunks)
        
        # Step 2: 生成偽查詢
        await self._generate_pseudo_queries()
        
        # Step 3: 構建圖邊
        await self.graph_db.build_graph_edges()
        
        self.is_graph_built = True
        print("✅ HopRAG圖譜構建完成！")
    
    async def _create_nodes_from_chunks(self, multi_level_chunks: Dict[str, Dict[str, List[Dict]]]):
        """從chunks創建節點"""
        print("📝 創建節點...")
        print(f"📊 多層次分塊數據結構: {list(multi_level_chunks.keys())}")
        
        for doc_id, levels in multi_level_chunks.items():
            print(f"📄 處理文檔: {doc_id}, 層次: {list(levels.keys())}")
            # 處理條級節點 (basic_unit層次)
            if 'basic_unit' in levels:
                print(f"📄 處理 {len(levels['basic_unit'])} 個basic_unit chunks")
                for chunk_idx, chunk in enumerate(levels['basic_unit']):
                    try:
                        # 確保chunk結構完整
                        if not isinstance(chunk, dict):
                            print(f"❌ Chunk {chunk_idx} 不是字典類型: {type(chunk)}")
                            continue
                        
                        # 確保metadata存在
                        if 'metadata' not in chunk:
                            chunk['metadata'] = {}
                        
                        # 確保content存在
                        if 'content' not in chunk:
                            print(f"❌ Chunk {chunk_idx} 沒有content欄位")
                            continue
                        
                        # 生成node_id，如果metadata中沒有id欄位
                        node_id = chunk['metadata'].get('id', f"{doc_id}_basic_unit_{chunk_idx}")
                    
                        basic_unit_node = LegalNode(
                            node_id=node_id,
                            node_type='basic_unit',
                            content=chunk['content'],
                            contextualized_text=chunk['content'],  # 使用content作為contextualized_text
                            law_name=chunk['metadata'].get('law_name', ''),
                            article_number=chunk['metadata'].get('article_label', ''),
                            metadata=chunk['metadata']
                        )
                        self.graph_db.add_node(basic_unit_node)
                        
                    except Exception as e:
                        print(f"❌ 創建basic_unit節點失敗 (chunk {chunk_idx}): {e}")
                        continue
            
            # 處理項級節點 (basic_unit_component層次)
            if 'basic_unit_component' in levels:
                print(f"📄 處理 {len(levels['basic_unit_component'])} 個basic_unit_component chunks")
                for chunk_idx, chunk in enumerate(levels['basic_unit_component']):
                    try:
                        # 確保chunk結構完整
                        if not isinstance(chunk, dict):
                            print(f"❌ Component Chunk {chunk_idx} 不是字典類型: {type(chunk)}")
                            continue
                        
                        # 確保metadata存在
                        if 'metadata' not in chunk:
                            chunk['metadata'] = {}
                        
                        # 確保content存在
                        if 'content' not in chunk:
                            print(f"❌ Component Chunk {chunk_idx} 沒有content欄位")
                            continue
                        
                        # 生成node_id，如果metadata中沒有id欄位
                        node_id = chunk['metadata'].get('id', f"{doc_id}_basic_unit_component_{chunk_idx}")
                        
                        component_node = LegalNode(
                            node_id=node_id,
                            node_type='basic_unit_component',
                            content=chunk['content'],
                            contextualized_text=chunk['content'],
                            law_name=chunk['metadata'].get('law_name', ''),
                            article_number=chunk['metadata'].get('article_label', ''),
                            item_number=chunk['metadata'].get('item_label', ''),
                            parent_article_id=chunk['metadata'].get('parent_article_id'),
                            metadata=chunk['metadata']
                        )
                        self.graph_db.add_node(component_node)
                        
                    except Exception as e:
                        print(f"❌ 創建basic_unit_component節點失敗 (chunk {chunk_idx}): {e}")
                        continue
        
        print(f"✅ 節點創建完成，共 {self.graph_db.get_node_count()} 個節點")
    
    async def _generate_pseudo_queries(self):
        """為所有節點生成偽查詢"""
        print("🤖 生成偽查詢...")
        
        nodes = list(self.graph_db.nodes.values())
        total_nodes = len(nodes)
        
        for i, node in enumerate(nodes):
            try:
                await self.pseudo_query_generator.generate_pseudo_queries_for_node(node)
                
                if (i + 1) % 10 == 0:
                    print(f"  進度: {i+1}/{total_nodes} 節點")
                    
            except Exception as e:
                print(f"❌ 節點 {node.node_id} 偽查詢生成失敗: {e}")
                continue
        
        print("✅ 偽查詢生成完成！")
    
    async def enhanced_retrieve(self, query: str, base_results: List[Dict], k: int = 5) -> List[Dict[str, Any]]:
        """HopRAG增強檢索"""
        if not self.is_graph_built:
            print("⚠️ HopRAG圖譜未構建，返回基礎結果")
            return base_results[:k]
        
        # 從基礎結果提取節點ID
        base_node_ids = []
        for result in base_results:
            # 嘗試從結果中提取node_id
            node_id = result.get('node_id') or result.get('id') or result.get('chunk_id')
            if node_id and node_id in self.graph_db.nodes:
                base_node_ids.append(node_id)
        
        if not base_node_ids:
            print("⚠️ 基礎結果中沒有找到有效的節點ID")
            return base_results[:k]
        
        # HopRAG多跳檢索
        hop_results = await self.traversal_engine.traverse_with_reasoning(
            query=query,
            initial_nodes=base_node_ids,
            max_hops=4
        )
        
        # 合併結果
        enhanced_results = self._merge_results(base_results, hop_results, query)
        
        return enhanced_results[:k]
    
    def _merge_results(self, base_results: List[Dict], hop_results: Dict[int, List[str]], query: str) -> List[Dict[str, Any]]:
        """合併基礎結果和HopRAG結果"""
        enhanced_results = []
        
        # 添加基礎結果
        for result in base_results:
            enhanced_result = result.copy()
            enhanced_result['hop_level'] = 0
            enhanced_result['hop_source'] = 'base_retrieval'
            enhanced_results.append(enhanced_result)
        
        # 添加HopRAG結果
        for hop_level, node_ids in hop_results.items():
            if hop_level == 0:  # 跳過基礎結果
                continue
                
            for node_id in node_ids:
                if node_id in self.graph_db.nodes:
                    node = self.graph_db.nodes[node_id]
                    
                    enhanced_result = {
                        'node_id': node_id,
                        'content': node.content,
                        'contextualized_text': node.contextualized_text,
                        'law_name': node.law_name,
                        'article_number': node.article_number,
                        'item_number': node.item_number,
                        'node_type': node.node_type,
                        'hop_level': hop_level,
                        'hop_source': 'hoprag_traversal',
                        'metadata': node.metadata
                    }
                    
                    enhanced_results.append(enhanced_result)
        
        # 去重（基於node_id）
        seen_nodes = set()
        unique_results = []
        for result in enhanced_results:
            node_id = result.get('node_id')
            if node_id and node_id not in seen_nodes:
                seen_nodes.add(node_id)
                unique_results.append(result)
        
        return unique_results
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """獲取圖統計信息"""
        if not self.is_graph_built:
            return {"error": "圖譜未構建"}
        
        return {
            "total_nodes": self.graph_db.get_node_count(),
            "total_edges": self.graph_db.get_edge_count(),
            "basic_unit_nodes": sum(1 for node in self.graph_db.nodes.values() if node.node_type == 'basic_unit'),
            "basic_unit_component_nodes": sum(1 for node in self.graph_db.nodes.values() if node.node_type == 'basic_unit_component'),
            "graph_built": self.is_graph_built,
            "similarity_threshold": self.graph_db.similarity_threshold,
            "max_edges_per_node": self.graph_db.max_edges_per_node
        }
