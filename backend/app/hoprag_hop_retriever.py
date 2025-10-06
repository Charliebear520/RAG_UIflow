"""
HopRAG多跳檢索器模組
包含InitialRetriever、GraphTraverser、LLMReasoner
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
import networkx as nx
from collections import Counter, deque
import numpy as np

from .hoprag_config import HopRAGConfig, DEFAULT_CONFIG
from .hoprag_graph_builder import LegalNode

class InitialRetriever:
    """初始檢索器"""
    
    def __init__(self, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        
    async def retrieve_initial_nodes(self, query: str, base_results: List[Dict[str, Any]]) -> List[str]:
        """從基礎檢索結果中提取初始節點"""
        print(f"🔍 初始檢索：查詢 '{query}'")
        
        initial_nodes = []
        for result in base_results:
            # 嘗試從結果中提取node_id
            node_id = result.get('node_id') or result.get('id') or result.get('chunk_id')
            if node_id:
                initial_nodes.append(node_id)
        
        # 限制初始節點數量
        initial_nodes = initial_nodes[:self.config.initial_retrieve_k]
        
        print(f"✅ 初始檢索完成，獲得 {len(initial_nodes)} 個初始節點")
        return initial_nodes
    
    def get_retrieval_stats(self, initial_nodes: List[str]) -> Dict[str, Any]:
        """獲取檢索統計信息"""
        return {
            "num_initial_nodes": len(initial_nodes),
            "retrieval_strategy": self.config.base_strategy,
            "max_initial_nodes": self.config.initial_retrieve_k
        }

class LLMReasoner:
    """LLM推理器"""
    
    def __init__(self, llm_client, config: HopRAGConfig = DEFAULT_CONFIG):
        self.llm_client = llm_client
        self.config = config
        
    async def reason_about_relevance(self, query: str, current_node: str, 
                                   neighbor: Dict[str, Any]) -> bool:
        """使用LLM推理判斷鄰居節點的相關性"""
        try:
            # 構建推理提示
            reasoning_prompt = self._build_reasoning_prompt(
                query, current_node, neighbor
            )
            
            # 調用LLM進行推理
            decision = await self.llm_client.generate_async(reasoning_prompt)
            
            # 解析決策
            return self._is_relevant_decision(decision)
            
        except Exception as e:
            print(f"❌ LLM推理失敗: {e}")
            # 如果LLM失敗，基於相似度分數判斷
            return neighbor.get('similarity_score', 0) > 0.8
    
    def _build_reasoning_prompt(self, query: str, current_node: str, neighbor: Dict[str, Any]) -> str:
        """構建LLM推理提示 - 按照Figure 8要求實現"""
        current_content = neighbor.get('current_content', '')
        neighbor_content = neighbor.get('contextualized_text', '')
        pseudo_query = neighbor.get('pseudo_query', '')
        
        prompt = f"""
您是一個法律問答機器人。我將提供您一個主問題，涉及多個法律條文信息，以及一個額外的輔助問題。您的任務是回答主問題，但由於主問題涉及很多您可能不知道的信息，您有機會使用輔助問題來收集您需要的信息。但是，輔助問題可能並不總是有用的，因此您需要評估輔助問題與主問題的關係，以確定是否使用它。

您需要評估輔助問題是否完全無關、間接相關，或相關且必要於回答主問題。您只能返回這三種結果之一。

請注意，主問題將涉及多個背景句子，這意味著回答主問題需要結合和推理多個信息片段。但是，您不知道哪些具體句子是回答主問題所必需的。您的任務是評估給定的輔助問題是否相關且必要、間接相關，或完全無關於回答主問題。

結果1：[完全無關]。在這種情況下，您確定即使沒有輔助問題的信息，您仍然可以回答主問題，或者輔助問題中的信息與主問題的答案完全無關。

結果2：[間接相關]。在這種情況下，您發現輔助問題與主問題相關，但其答案不是回答主問題所需的多個信息片段的一部分。輔助問題關注相關主題，但不提供回答主問題所需的關鍵信息。

結果3：[相關且必要]。在這種情況下，您發現輔助問題是主問題的子問題，這意味著如果不回答輔助問題，您將無法回答主問題。輔助問題提供的信息是回答主問題所必需的。

範例1：
主問題：張三違反著作權法第八條的重製權規定，會面臨什麼法律後果？
輔助問題：美國總統發表國情咨文的目的是什麼？
在這種情況下，經過仔細考慮，您發現輔助問題無助於回答主問題。輔助問題與主問題完全無關。您的回應應該是：
{{"Decision": "Completely Irrelevant"}}

範例2：
主問題：張三違反著作權法第八條的重製權規定，會面臨什麼法律後果？
輔助問題：這個法律後果對該地區的二次創作有什麼意義？
在這種情況下，您注意到主問題和輔助問題都涉及類似的主題，但輔助問題關注的是法律後果的意義，而主問題詢問的是具體的法律後果。經過仔細考慮，您發現輔助問題是相關的，但其答案不提供回答主問題所需的任何關鍵信息。您的回應應該是：
{{"Decision": "Indirectly relevant"}}

範例3：
主問題：張三違反著作權法第八條的重製權規定，會面臨什麼法律後果？
輔助問題：著作權法第八條規定了哪些權利？
在這種情況下，經過仔細考慮，您發現輔助問題確實與主問題相關。輔助問題是主問題的子問題，提供回答主問題所需的必要信息。如果不回答輔助問題，您將無法回答主問題。您的回應應該是：
{{"Decision": "Relevant and Necessary"}}

現在請嚴格遵循JSON格式，避免不必要的轉義、換行或空格。您還應該特別注意確保，除了JSON和列表格式本身使用雙引號(")外，其他所有雙引號的實例都應替換為單引號。例如，使用'著作權法'而不是"著作權法"。

主問題：{query}
輔助問題：{pseudo_query}

決策："""
        
        return prompt
    
    def _is_relevant_decision(self, decision: str) -> bool:
        """判斷LLM決策是否表示相關"""
        import json
        
        try:
            # 嘗試解析JSON格式
            if decision.strip().startswith('{'):
                result = json.loads(decision.strip())
                decision_value = result.get('Decision', '').lower()
                return ("relevant and necessary" in decision_value or 
                        "indirectly relevant" in decision_value)
        except:
            pass
        
        # 如果JSON解析失敗，使用原始邏輯
        decision_lower = decision.lower().strip()
        return ("relevant and necessary" in decision_lower or 
                "indirectly relevant" in decision_lower)
    
    async def batch_reason_about_relevance(self, query: str, current_node: str,
                                         neighbors: List[Dict[str, Any]]) -> List[bool]:
        """批量推理鄰居節點的相關性"""
        tasks = []
        for neighbor in neighbors:
            task = self.reason_about_relevance(query, current_node, neighbor)
            tasks.append(task)
        
        # 並行執行推理
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 處理異常結果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ 批量推理中出現異常: {result}")
                processed_results.append(False)
            else:
                processed_results.append(result)
        
        return processed_results

class GraphTraverser:
    """圖遍歷器"""
    
    def __init__(self, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.llm_reasoner = None
        
    def set_llm_reasoner(self, llm_reasoner: LLMReasoner):
        """設置LLM推理器"""
        self.llm_reasoner = llm_reasoner
        
    async def traverse_graph(self, query: str, initial_nodes: List[str], 
                           graph: nx.DiGraph, nodes: Dict[str, LegalNode]) -> Dict[int, List[str]]:
        """基於推理的圖遍歷"""
        print(f"🔍 開始HopRAG圖遍歷，查詢: '{query}'")
        
        hop_results = {0: initial_nodes}
        current_nodes = initial_nodes
        
        for hop in range(1, self.config.max_hops + 1):
            print(f"  第 {hop} 跳檢索...")
            next_nodes = []
            
            for node_id in current_nodes:
                # 獲取鄰居節點
                neighbors = self._get_neighbors_with_edges(node_id, graph, nodes)
                
                # 使用LLM推理判斷相關性
                if self.llm_reasoner and neighbors:
                    relevant_neighbors = await self._filter_by_llm_reasoning(
                        query, node_id, neighbors
                    )
                    next_nodes.extend(relevant_neighbors)
            
            # 去重和限制數量
            next_nodes = list(set(next_nodes))[:self.config.top_k_per_hop]
            hop_results[hop] = next_nodes
            current_nodes = next_nodes
            
            if not current_nodes:
                print(f"  第 {hop} 跳後無更多相關節點，停止遍歷")
                break
        
        # 統計信息
        total_nodes_found = sum(len(nodes) for nodes in hop_results.values())
        print(f"✅ 圖遍歷完成，共找到 {total_nodes_found} 個相關節點")
        
        return hop_results
    
    def _get_neighbors_with_edges(self, node_id: str, graph: nx.DiGraph, 
                                nodes: Dict[str, LegalNode]) -> List[Dict[str, Any]]:
        """獲取節點的鄰居及其邊信息"""
        neighbors = []
        
        if node_id in graph:
            for neighbor_id in graph.successors(node_id):
                if neighbor_id in nodes:
                    neighbor_node = nodes[neighbor_id]
                    edge_data = graph[node_id][neighbor_id]
                    
                    neighbor_info = {
                        'node_id': neighbor_id,
                        'node_type': neighbor_node.node_type.value,
                        'content': neighbor_node.content,
                        'contextualized_text': neighbor_node.contextualized_text,
                        'pseudo_query': edge_data.get('pseudo_query', ''),
                        'similarity_score': edge_data.get('similarity_score', 0.0),
                        'edge_type': edge_data.get('edge_type', '')
                    }
                    
                    neighbors.append(neighbor_info)
        
        return neighbors
    
    async def _filter_by_llm_reasoning(self, query: str, current_node: str, 
                                     neighbors: List[Dict[str, Any]]) -> List[str]:
        """使用LLM推理過濾鄰居節點"""
        if not self.llm_reasoner:
            # 如果沒有LLM推理器，基於相似度分數過濾
            relevant_neighbors = [
                neighbor['node_id'] for neighbor in neighbors 
                if neighbor.get('similarity_score', 0) > 0.7
            ]
            return relevant_neighbors
        
        # 批量推理
        relevance_decisions = await self.llm_reasoner.batch_reason_about_relevance(
            query, current_node, neighbors
        )
        
        # 篩選相關的鄰居
        relevant_neighbors = []
        for neighbor, is_relevant in zip(neighbors, relevance_decisions):
            if is_relevant:
                relevant_neighbors.append(neighbor['node_id'])
        
        return relevant_neighbors
    
    def get_traversal_stats(self, hop_results: Dict[int, List[str]]) -> Dict[str, Any]:
        """獲取遍歷統計信息"""
        total_nodes = sum(len(nodes) for nodes in hop_results.values())
        hop_counts = {f"hop_{hop}": len(nodes) for hop, nodes in hop_results.items()}
        
        return {
            "total_nodes_found": total_nodes,
            "max_hops_reached": max(hop_results.keys()) if hop_results else 0,
            "hop_distribution": hop_counts,
            "config_max_hops": self.config.max_hops,
            "config_top_k_per_hop": self.config.top_k_per_hop
        }

class HopRetriever:
    """多跳檢索器主類"""
    
    def __init__(self, config: HopRAGConfig = DEFAULT_CONFIG):
        self.config = config
        
        # 初始化組件
        self.initial_retriever = InitialRetriever(config)
        self.graph_traverser = GraphTraverser(config)
        self.algorithm1_traverser = None
        self.llm_reasoner = None
        
    def set_llm_reasoner(self, llm_reasoner: LLMReasoner):
        """設置LLM推理器"""
        self.llm_reasoner = llm_reasoner
        self.graph_traverser.set_llm_reasoner(llm_reasoner)
    
    def set_algorithm1_traverser(self, algorithm1_traverser):
        """設置演算法1遍歷器"""
        self.algorithm1_traverser = algorithm1_traverser
        
    async def retrieve(self, query: str, base_results: List[Dict[str, Any]], 
                      graph: nx.DiGraph, nodes: Dict[str, LegalNode]) -> Dict[int, List[str]]:
        """執行多跳檢索"""
        print(f"🚀 開始HopRAG多跳檢索，查詢: '{query}'")
        
        # Step 1: 初始檢索
        initial_nodes = await self.initial_retriever.retrieve_initial_nodes(query, base_results)
        
        if not initial_nodes:
            print("⚠️ 沒有找到初始節點")
            return {0: []}
        
        # Step 2: 圖遍歷
        hop_results = await self.graph_traverser.traverse_graph(
            query, initial_nodes, graph, nodes
        )
        
        # Step 3: 統計信息
        initial_stats = self.initial_retriever.get_retrieval_stats(initial_nodes)
        traversal_stats = self.graph_traverser.get_traversal_stats(hop_results)
        
        print(f"📊 檢索統計: {initial_stats}")
        print(f"📊 遍歷統計: {traversal_stats}")
        
        return hop_results
    
    def get_retrieval_summary(self, hop_results: Dict[int, List[str]]) -> Dict[str, Any]:
        """獲取檢索摘要"""
        total_nodes = sum(len(nodes) for nodes in hop_results.values())
        
        return {
            "total_nodes_retrieved": total_nodes,
            "hops_performed": len(hop_results) - 1,  # 減去初始跳
            "nodes_per_hop": {f"hop_{hop}": len(nodes) for hop, nodes in hop_results.items()},
            "config_used": {
                "max_hops": self.config.max_hops,
                "top_k_per_hop": self.config.top_k_per_hop,
                "base_strategy": self.config.base_strategy
            }
        }
    
    async def retrieve_with_algorithm1(self, query: str, graph: nx.DiGraph, 
                                     nodes: Dict[str, LegalNode], 
                                     top_k: int = 5, n_hop: int = 4) -> List[str]:
        """使用演算法1進行檢索"""
        if not self.algorithm1_traverser:
            print("❌ 演算法1遍歷器未設置，使用標準檢索")
            return await self.retrieve(query, [], graph, nodes)
        
        print(f"🚀 使用演算法1進行檢索")
        return await self.algorithm1_traverser.reasoning_augmented_traversal(
            query, graph, nodes, top_k, n_hop
        )

class Algorithm1Traverser:
    """演算法1：基於推理的圖遍歷 - 完整實現"""
    
    def __init__(self, llm_client, embedding_model, config: HopRAGConfig = DEFAULT_CONFIG):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.config = config
        
    async def reasoning_augmented_traversal(self, query: str, graph: nx.DiGraph, 
                                          nodes: Dict[str, LegalNode], 
                                          top_k: int, n_hop: int) -> List[str]:
        """
        演算法1：基於推理的圖遍歷 - 增強版（支持nhop分析）
        
        Args:
            query: 原始查詢 q
            graph: 圖結構 G
            nodes: 節點字典
            top_k: 最終返回的頂級結果數量
            n_hop: 圖遍歷的最大跳數
            
        Returns:
            C: 最終修剪後的相關上下文集合
        """
        print(f"🚀 開始演算法1：基於推理的圖遍歷")
        print(f"   查詢: '{query}'")
        print(f"   最大跳數: {n_hop}")
        print(f"   目標結果數: {top_k}")
        
        # 初始化nhop分析數據
        nhop_stats = {
            'llm_calls': 0,
            'queue_lengths': [],
            'new_nodes_per_hop': [],
            'total_nodes_visited': 0,
            'early_stop_triggered': False
        }
        
        # Step 1: v_q <- EMB(q) - 查詢嵌入
        v_q = await self._get_query_embedding(query)
        
        # Step 2: k_q <- NER(q) - 命名實體識別
        k_q = await self._extract_entities(query)
        
        # Step 3: C_queue <- Retrieve(v_q, k_q, G) - 初始檢索
        C_queue = await self._initial_retrieve(v_q, k_q, graph, nodes)
        
        # Step 4: C_count <- Counter(C_queue) - 計數器初始化
        C_count = Counter(C_queue)
        
        print(f"✅ 初始檢索完成，獲得 {len(C_queue)} 個初始節點")
        nhop_stats['queue_lengths'].append(len(C_queue))
        
        # Step 5-16: 廣度優先遍歷循環
        for i in range(1, n_hop + 1):
            print(f"  第 {i} 跳檢索...")
            
            # 記錄跳躍前的隊列長度
            initial_queue_length = len(C_queue)
            new_nodes_this_hop = 0
            
            # 創建當前跳的隊列副本
            current_queue = deque(C_queue)
            queue_size = len(current_queue)
            
            # Step 6: for j <- 1, 2, ..., |C_queue| do
            for j in range(queue_size):
                # Step 7: v_j <- C_queue.dequeue()
                v_j = current_queue.popleft()
                
                # Step 8: v_k <- Reason({<v_j, e_j,k, v_k>})
                v_k = await self._reason_next_node(query, v_j, graph, nodes)
                
                # 追蹤LLM呼叫次數
                if self.config.nhop_cost_tracking:
                    nhop_stats['llm_calls'] += 1
                
                if v_k is not None:
                    # Step 9-13: 更新計數器
                    if v_k not in C_count:
                        # Step 10-11: 新節點加入隊列和計數器
                        C_queue.append(v_k)
                        C_count[v_k] = 1
                        new_nodes_this_hop += 1
                    else:
                        # Step 13: 已存在節點計數加1
                        C_count[v_k] += 1
            
            # 記錄本跳統計
            nhop_stats['new_nodes_per_hop'].append(new_nodes_this_hop)
            nhop_stats['queue_lengths'].append(len(C_queue))
            nhop_stats['total_nodes_visited'] = len(C_count)
            
            print(f"    新增節點: {new_nodes_this_hop}, 隊列長度: {len(C_queue)}")
            
            # 早期停止檢查（基於論文觀察：第5跳隊列長度僅1.23）
            if (self.config.enable_nhop_analysis and 
                len(C_queue) <= self.config.queue_length_threshold and 
                i >= 2):  # 至少執行2跳
                print(f"  🛑 早期停止：隊列長度 {len(C_queue)} <= 閾值 {self.config.queue_length_threshold}")
                nhop_stats['early_stop_triggered'] = True
                break
            
            # 限制隊列大小，避免爆炸性增長
            if len(C_queue) > self.config.top_k_per_hop * 2:
                # 按計數排序，保留最重要的節點
                sorted_nodes = sorted(C_count.items(), key=lambda x: x[1], reverse=True)
                C_queue = deque([node_id for node_id, _ in sorted_nodes[:self.config.top_k_per_hop * 2]])
                print(f"    隊列截斷：保留前 {len(C_queue)} 個節點")
        
        # Step 17: C <- Prune(C_count, v_q, k_q, top_k) - 結果修剪
        C = await self._prune_results(C_count, v_q, k_q, top_k, nodes)
        
        # 輸出nhop分析結果
        if self.config.enable_nhop_analysis:
            self._print_nhop_analysis(nhop_stats, n_hop)
        
        print(f"✅ 演算法1完成，返回 {len(C)} 個最終結果")
        return C
    
    def _print_nhop_analysis(self, nhop_stats: Dict[str, Any], max_nhop: int):
        """打印nhop分析結果"""
        print(f"\n📊 nhop性能分析報告")
        print(f"=" * 50)
        
        # 基本統計
        print(f"🔢 基本統計:")
        print(f"   總LLM呼叫次數: {nhop_stats['llm_calls']}")
        print(f"   總訪問節點數: {nhop_stats['total_nodes_visited']}")
        print(f"   早期停止: {'是' if nhop_stats['early_stop_triggered'] else '否'}")
        
        # 隊列長度分析
        print(f"\n📈 隊列長度變化:")
        for i, length in enumerate(nhop_stats['queue_lengths']):
            hop_label = "初始" if i == 0 else f"第{i}跳"
            print(f"   {hop_label}: {length} 個節點")
        
        # 每跳新增節點分析
        print(f"\n🆕 每跳新增節點:")
        for i, new_nodes in enumerate(nhop_stats['new_nodes_per_hop']):
            print(f"   第{i+1}跳: {new_nodes} 個新節點")
        
        # 成本效益分析
        if nhop_stats['llm_calls'] > 0:
            efficiency = nhop_stats['total_nodes_visited'] / nhop_stats['llm_calls']
            print(f"\n💰 成本效益分析:")
            print(f"   節點/LLM呼叫比: {efficiency:.2f}")
            print(f"   平均每跳LLM呼叫: {nhop_stats['llm_calls'] / len(nhop_stats['new_nodes_per_hop']):.1f}")
        
        # 與論文數據對比
        print(f"\n📚 與論文數據對比:")
        print(f"   論文nhop=4平均LLM呼叫: ~38.53次")
        print(f"   當前LLM呼叫次數: {nhop_stats['llm_calls']}次")
        if nhop_stats['llm_calls'] > 0:
            ratio = nhop_stats['llm_calls'] / 38.53
            print(f"   相對論文比例: {ratio:.2f}x")
        
        # 建議
        print(f"\n💡 優化建議:")
        if nhop_stats['early_stop_triggered']:
            print(f"   ✅ 早期停止生效，節省了計算成本")
        if len(nhop_stats['queue_lengths']) > 1:
            final_queue = nhop_stats['queue_lengths'][-1]
            if final_queue <= 2:
                print(f"   ✅ 最終隊列長度({final_queue})符合論文觀察(≤1.23)")
            else:
                print(f"   ⚠️ 最終隊列長度({final_queue})較高，可考慮調整閾值")
        
        print(f"=" * 50)
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Step 1: v_q <- EMB(q) - 查詢嵌入"""
        try:
            if hasattr(self.embedding_model, 'encode_async'):
                embedding = await self.embedding_model.encode_async([query])
            else:
                embedding = self.embedding_model.encode([query])
            return embedding[0]
        except Exception as e:
            print(f"❌ 查詢嵌入失敗: {e}")
            return np.zeros(768)  # 返回零向量作為fallback
    
    async def _extract_entities(self, query: str) -> Set[str]:
        """Step 2: k_q <- NER(q) - 命名實體識別"""
        import jieba
        import re
        
        # 簡單的實體提取（可以後續改進為真正的NER）
        words = jieba.lcut(query)
        
        # 過濾停用詞，保留有意義的詞彙
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好', '自己', '這', '什麼', '如何', '為什麼', '哪些', '什麼時候', '哪裡', '多少', '怎麼', '是否', '能否', '可以', '應該', '必須', '需要', '要求'}
        
        entities = set()
        for word in words:
            word = word.strip()
            if (len(word) >= 2 and 
                word not in stop_words and 
                not re.match(r'^[0-9]+$', word) and
                not re.match(r'^[a-zA-Z]+$', word)):
                entities.add(word)
        
        return entities
    
    async def _initial_retrieve(self, v_q: np.ndarray, k_q: Set[str], 
                              graph: nx.DiGraph, nodes: Dict[str, LegalNode]) -> List[str]:
        """Step 3: C_queue <- Retrieve(v_q, k_q, G) - 初始檢索"""
        # 計算所有節點與查詢的相似度
        similarities = []
        
        for node_id, node in nodes.items():
            if hasattr(node, 'embedding') and node.embedding is not None:
                # 語義相似度
                semantic_sim = np.dot(v_q, node.embedding) / (
                    np.linalg.norm(v_q) * np.linalg.norm(node.embedding)
                )
                
                # 詞彙相似度（如果有關鍵詞）
                lexical_sim = 0.0
                if hasattr(node, 'keywords') and node.keywords and k_q:
                    intersection = len(node.keywords.intersection(k_q))
                    union = len(node.keywords.union(k_q))
                    if union > 0:
                        lexical_sim = intersection / union
                
                # 混合相似度
                mixed_sim = (semantic_sim + lexical_sim) / 2
                similarities.append((node_id, mixed_sim))
        
        # 按相似度排序，取前k個
        similarities.sort(key=lambda x: x[1], reverse=True)
        initial_nodes = [node_id for node_id, _ in similarities[:self.config.initial_retrieve_k]]
        
        return initial_nodes
    
    async def _reason_next_node(self, query: str, current_node: str, 
                              graph: nx.DiGraph, nodes: Dict[str, LegalNode]) -> Optional[str]:
        """Step 8: v_k <- Reason({<v_j, e_j,k, v_k>}) - LLM推理下一個節點"""
        if current_node not in graph:
            return None
        
        # 獲取當前節點的所有鄰居
        neighbors = list(graph.successors(current_node))
        if not neighbors:
            return None
        
        # 如果只有一個鄰居，直接返回
        if len(neighbors) == 1:
            return neighbors[0]
        
        # 構建推理提示
        current_node_data = nodes.get(current_node)
        if not current_node_data:
            return None
        
        # 收集鄰居信息
        neighbor_info = []
        for neighbor_id in neighbors:
            neighbor_data = nodes.get(neighbor_id)
            if neighbor_data:
                edge_data = graph[current_node][neighbor_id]
                neighbor_info.append({
                    'node_id': neighbor_id,
                    'content': neighbor_data.content[:200],  # 限制長度
                    'pseudo_query': edge_data.get('pseudo_query', ''),
                    'similarity_score': edge_data.get('similarity_score', 0.0)
                })
        
        # 構建LLM推理提示
        prompt = self._build_reasoning_prompt(query, current_node_data, neighbor_info)
        
        try:
            # 調用LLM進行推理
            response = await self.llm_client.generate_async(prompt)
            
            # 解析LLM響應，提取最相關的節點ID
            selected_node = self._parse_reasoning_response(response, neighbor_info)
            return selected_node
            
        except Exception as e:
            print(f"❌ LLM推理失敗: {e}")
            # Fallback: 返回相似度最高的鄰居
            best_neighbor = max(neighbor_info, key=lambda x: x['similarity_score'])
            return best_neighbor['node_id']
    
    def _build_reasoning_prompt(self, query: str, current_node: LegalNode, 
                              neighbor_info: List[Dict[str, Any]]) -> str:
        """構建LLM推理提示"""
        current_content = current_node.content[:300]  # 限制長度
        
        # 構建鄰居選項
        options = []
        for i, neighbor in enumerate(neighbor_info):
            options.append(f"選項{i+1}: {neighbor['node_id']}\n"
                          f"內容: {neighbor['content']}\n"
                          f"連接問題: {neighbor['pseudo_query']}\n"
                          f"相似度: {neighbor['similarity_score']:.3f}\n")
        
        prompt = f"""
您是一個法律問答機器人。我需要您幫助選擇最相關的下一個節點。

主問題: {query}

當前節點內容: {current_content}

可選的下一個節點:
{chr(10).join(options)}

請基於以下標準選擇最相關的節點：
1. 與主問題的邏輯關聯性
2. 連接問題的相關性
3. 內容的相關性

請只返回選項編號（如：選項1、選項2等），不要包含其他文字。
"""
        
        return prompt
    
    def _parse_reasoning_response(self, response: str, neighbor_info: List[Dict[str, Any]]) -> str:
        """解析LLM推理響應"""
        response = response.strip().lower()
        
        # 嘗試提取選項編號
        for i in range(len(neighbor_info)):
            if f"選項{i+1}" in response or f"option{i+1}" in response or str(i+1) in response:
                return neighbor_info[i]['node_id']
        
        # 如果無法解析，返回相似度最高的
        best_neighbor = max(neighbor_info, key=lambda x: x['similarity_score'])
        return best_neighbor['node_id']
    
    async def _prune_results(self, C_count: Counter, v_q: np.ndarray, k_q: Set[str], 
                           top_k: int, nodes: Dict[str, LegalNode]) -> List[str]:
        """
        Step 17: C <- Prune(C_count, v_q, k_q, top_k) - 結果修剪
        按照論文要求實現有用性度量：H_i = (SIM(v_i, q) + IMP(v_i, C_count)) / 2
        """
        print(f"🔍 開始修剪階段，候選節點數: {len(C_count)}")
        
        # 計算每個節點的有用性度量 H_i
        helpfulness_scores = []
        
        # 計算總訪問次數（用於IMP計算）
        total_visits = sum(C_count.values())
        
        for node_id, count in C_count.items():
            node = nodes.get(node_id)
            if not node:
                continue
            
            # 1. 計算 IMP(v_i, C_count) = C_count[v_i] / Σ C_count[v_j] (規範化訪問次數)
            imp_score = count / total_visits if total_visits > 0 else 0
            
            # 2. 計算 SIM(v_i, q) = 段落與查詢的平均詞彙+語義相似度
            sim_score = self._calculate_sim_score(node, v_q, k_q)
            
            # 3. 計算有用性度量 H_i = (SIM(v_i, q) + IMP(v_i, C_count)) / 2
            helpfulness_score = (sim_score + imp_score) / 2
            
            helpfulness_scores.append((node_id, helpfulness_score, sim_score, imp_score))
        
        # 按有用性分數排序，取前top_k個
        helpfulness_scores.sort(key=lambda x: x[1], reverse=True)
        pruned_results = [node_id for node_id, _, _, _ in helpfulness_scores[:top_k]]
        
        # 輸出修剪統計信息
        print(f"✅ 修剪完成，保留 {len(pruned_results)} 個節點")
        if helpfulness_scores:
            best_score = helpfulness_scores[0][1]
            worst_score = helpfulness_scores[-1][1]
            print(f"   有用性分數範圍: {worst_score:.3f} - {best_score:.3f}")
        
        return pruned_results
    
    def _calculate_sim_score(self, node: LegalNode, v_q: np.ndarray, k_q: Set[str]) -> float:
        """
        計算 SIM(v_i, q) = 段落與查詢的平均詞彙+語義相似度
        """
        # 語義相似度（餘弦相似度）
        semantic_score = 0.0
        if hasattr(node, 'embedding') and node.embedding is not None:
            semantic_score = np.dot(v_q, node.embedding) / (
                np.linalg.norm(v_q) * np.linalg.norm(node.embedding)
            )
        
        # 詞彙相似度（Jaccard相似度）
        lexical_score = 0.0
        if hasattr(node, 'keywords') and node.keywords and k_q:
            intersection = len(node.keywords.intersection(k_q))
            union = len(node.keywords.union(k_q))
            if union > 0:
                lexical_score = intersection / union
        
        # 平均詞彙+語義相似度
        sim_score = (semantic_score + lexical_score) / 2
        
        return sim_score
